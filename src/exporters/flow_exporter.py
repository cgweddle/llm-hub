"""
Flow Exporter ("eject")

Deconstructs a flow into a standalone, human-readable Python module: one file
per tool (verbatim script_code), one file per agent node (the compile_agent
recipe emitted as source, credentials read from env vars), a readable flow.py
orchestrator with each node's input statically resolved into explicit kwargs,
requirements.txt, README.md, and a __main__.py entrypoint.

Pure text generation — nothing is exec'd, no pydantic_ai import, so this
module is safe to import from the backend API process. Topology comes from
the same decompose_into_chains/compute_chain_levels used by the runtime;
input wiring mirrors flow_runner._build_node_input branch by branch, and the
agent module mirrors agent_runner.compile_agent.

Phase 1: tool-only flows and single-node agents. Multi-node or looping
agents raise FlowExportError.
"""

import io
import keyword
import os
import re
import sys
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from src.database.database import get_agent_by_id, get_tool_by_id
from src.factories.flow_to_script_factory import (
    Chain,
    compute_chain_levels,
    decompose_into_chains,
)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.prompt_template import resolve_system_prompt_template


class FlowExportError(ValueError):
    """A flow cannot be exported; the message says why. Maps to HTTP 422."""


# Versions pinned to deploy/flow-runner/requirements.txt so an exported
# module runs the same library versions as the hosted runner.
AGENT_BASE_PINS = ["pydantic[email]==2.12.5", "pydantic-ai==1.31.0"]
SDK_PINS = {"anthropic": "anthropic==0.77.0", "openai": "openai==2.12.0"}

# Runtime-infrastructure packages that must never leak into an export.
PACKAGE_DENYLIST = {
    "sqlalchemy", "psycopg2", "psycopg2-binary", "langfuse", "redis",
    "httpx", "python-dotenv", "celery", "podman-py",
}

API_KEY_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "lmstudio": "LMSTUDIO_API_KEY",
}


# ─── Emitted runtime helpers (included in flow.py only when used) ────────────

_HELPER_AS_TEXT = '''\
def _as_text(value):
    """Agents consume text: pass strings through, JSON-encode anything else."""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, default=str)
'''

_HELPER_PICK = '''\
def _pick(upstream, field):
    """Take one field from a dict output; fall back to the whole value."""
    if isinstance(upstream, dict) and field in upstream:
        return upstream[field]
    return upstream
'''

_HELPER_MERGE_AUTO = '''\
def _merge_auto(base, upstream, params):
    """Merge an unmapped upstream output into a tool's kwargs.

    Dict outputs merge key by key; anything else fills the tool's only
    parameter, or the first parameter not already provided.
    """
    if isinstance(upstream, dict):
        return {**base, **upstream}
    if len(params) == 1:
        return {**base, params[0]: upstream}
    unfilled = [p for p in params if p not in base]
    if unfilled:
        return {**base, unfilled[0]: upstream}
    raise TypeError(f"No free parameter to receive upstream value {upstream!r}")
'''


# ─── Small utilities ─────────────────────────────────────────────────────────

def _slugify(name: str, seen: Set[str], fallback: str = "item") -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", (name or "").strip().lower()).strip("_")
    if not slug:
        slug = fallback
    if slug[0].isdigit():
        slug = f"n_{slug}"
    if keyword.iskeyword(slug):
        slug = f"{slug}_"
    base, i = slug, 2
    while slug in seen:
        slug = f"{base}_{i}"
        i += 1
    seen.add(slug)
    return slug


def _py_string(s: str) -> str:
    """Render a string as readable Python source (triple-quoted when multiline)."""
    if "\n" not in s:
        return repr(s)
    body = s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    if body.endswith('"'):
        return repr(s)
    return f'"""{body}"""'


def _incoming_edges(graph_config: dict, node_id: str) -> List[dict]:
    return [
        e for e in graph_config.get("edges", [])
        if e.get("to_node") == node_id and not e.get("is_loop")
    ]


def _node_resource_id(config: dict) -> Optional[int]:
    return config.get("id") or config.get("tool_id") or config.get("agent_id")


def _tool_param_names(tool) -> List[str]:
    schema = tool.input_schema or {}
    props = schema.get("properties", schema) if isinstance(schema, dict) else {}
    return list(props.keys())


def _tool_is_async(tool) -> bool:
    return f"async def {tool.main_function}" in (tool.script_code or "")


# ─── Node model ──────────────────────────────────────────────────────────────

@dataclass
class _ToolModule:
    tool: Any
    module: str          # module name under tools/
    func: str            # main_function name
    alias: str           # name the function is imported as (collision-safe)


@dataclass
class _AgentModule:
    agent: Any
    sub_config: dict
    provider_config: dict
    tool_modules: List[_ToolModule]
    module: str          # module name under agents/
    run_func: str        # "run_<slug>"


@dataclass
class _Node:
    node_id: str
    node_type: str       # tool | agent | trigger
    config: dict
    var: str             # local variable holding the node's output
    tool_module: Optional[_ToolModule] = None
    agent_module: Optional[_AgentModule] = None


def _static_output_kind(node: _Node) -> Tuple[str, Optional[Set[str]]]:
    """Classify a node's output for static wiring.

    Returns (kind, known_fields): kind is "text" (agents/triggers),
    "dict" (fields known when the output_schema lists properties),
    "nondict", or "unknown".
    """
    if node.node_type in ("agent", "trigger"):
        return "text", None
    schema = node.tool_module.tool.output_schema or {}
    if not isinstance(schema, dict):
        return "unknown", None
    if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
        return "dict", set(schema["properties"].keys())
    type_str = str(schema.get("type") or "").strip()
    if type_str.lower().startswith(("dict", "mapping")):
        return "dict", None
    if not type_str or type_str in ("Any", "object"):
        return "unknown", None
    return "nondict", None


# ─── flow.py emitter ─────────────────────────────────────────────────────────

class _FlowEmitter:
    """Builds flow.py: static input wiring + level-ordered orchestration."""

    def __init__(self, flow, graph_config: dict, nodes: Dict[str, _Node]):
        self.flow = flow
        self.graph = graph_config
        self.nodes = nodes
        self.entry_point = graph_config.get("entry_point")
        self.helpers_used: Set[str] = set()
        self.needs_asyncio = False
        self.needs_json = False

    # ── per-node input expressions ──

    def _tool_call_args(self, node: _Node) -> str:
        """Render a tool node's kwargs, mirroring _build_node_input statically."""
        incoming = _incoming_edges(self.graph, node.node_id)
        tool = node.tool_module.tool
        params = _tool_param_names(tool)

        if not incoming:
            if node.node_id == self.entry_point:
                return "**initial_input"
            items = [("kw", k, repr(v)) for k, v in (node.config.get("input_values") or {}).items()]
            return self._render_args(items)

        # base = typed-in values, then each edge overlays in graph order
        items: List[tuple] = [
            ("kw", k, repr(v)) for k, v in (node.config.get("input_values") or {}).items()
        ]
        # Params known to be filled so far; None once a merge of unknown keys occurs.
        filled: Optional[Set[str]] = {k for _, k, _ in items}

        for edge in incoming:
            upstream = self.nodes[edge["from_node"]]
            kind, fields = _static_output_kind(upstream)
            mapping = edge.get("mapping")
            if mapping:
                for out_field, in_param in mapping.items():
                    if out_field == "":
                        expr = upstream.var
                    elif kind == "dict" and fields is not None and out_field in fields:
                        expr = f'{upstream.var}[{out_field!r}]'
                    elif kind in ("text", "nondict"):
                        expr = upstream.var
                    else:
                        self.helpers_used.add("_pick")
                        expr = f'_pick({upstream.var}, {out_field!r})'
                    items.append(("kw", in_param, expr))
                    if filled is not None:
                        filled.add(in_param)
            elif kind == "dict":
                items.append(("splat", upstream.var))
                if fields is not None and filled is not None:
                    filled.update(fields)
                else:
                    filled = None
            elif kind in ("text", "nondict") and filled is not None:
                if len(params) == 1:
                    target = params[0]
                else:
                    unfilled = [p for p in params if p not in filled]
                    if not unfilled:
                        raise FlowExportError(
                            f"Tool node '{node.node_id}' has no free parameter for the "
                            f"unmapped output of '{edge['from_node']}'. Add an edge "
                            f"mapping to say which parameter it should fill."
                        )
                    target = unfilled[0]
                items.append(("kw", target, upstream.var))
                filled.add(target)
            else:
                self.helpers_used.add("_merge_auto")
                items.append(("auto", upstream.var, params))
                filled = None
        return self._render_args(items)

    @staticmethod
    def _render_args(items: List[tuple]) -> str:
        """Render wiring items as call args: plain kwargs when possible,
        a dict literal (with ** merges / _merge_auto folds) otherwise."""
        if not items:
            return ""
        if all(kind == "kw" for kind, *_ in items):
            ordered: Dict[str, str] = {}
            for _, param, expr in items:
                ordered.pop(param, None)
                ordered[param] = expr        # later edges override earlier
            return ", ".join(f"{p}={e}" for p, e in ordered.items())
        # Fold into a dict expression preserving override order.
        parts: List[str] = []
        for item in items:
            if item[0] == "kw":
                parts.append(f"{item[1]!r}: {item[2]}")
            elif item[0] == "splat":
                parts.append(f"**{item[1]}")
            else:  # auto — fold everything so far through _merge_auto
                base = "{" + ", ".join(parts) + "}"
                parts = [f"**_merge_auto({base}, {item[1]}, params={item[2]!r})"]
        if len(parts) == 1 and parts[0].startswith("**"):
            return parts[0]
        return "**{" + ", ".join(parts) + "}"

    def _agent_input_expr(self, node: _Node) -> str:
        incoming = _incoming_edges(self.graph, node.node_id)
        if not incoming:
            if node.node_id == self.entry_point:
                self.helpers_used.add("_as_text")
                return "_as_text(initial_input)"
            return repr(node.config.get("input_value", "") or "")
        upstream = self.nodes[incoming[0]["from_node"]]
        kind, _ = _static_output_kind(upstream)
        if kind == "text":
            return upstream.var
        self.helpers_used.add("_as_text")
        return f"_as_text({upstream.var})"

    def _trigger_expr(self, node: _Node) -> str:
        incoming = _incoming_edges(self.graph, node.node_id)
        if incoming:
            return self.nodes[incoming[0]["from_node"]].var
        input_value = node.config.get("input_value")
        if node.node_id == self.entry_point:
            if input_value:
                return f"initial_input if initial_input else {input_value!r}"
            return "initial_input"
        return repr(node.config.get("input_value", ""))

    # ── per-node statements / gather expressions ──

    def _node_statement(self, node: _Node, indent: str = "    ") -> str:
        if node.node_type == "trigger":
            return f"{indent}{node.var} = {self._trigger_expr(node)}"
        if node.node_type == "agent":
            return f"{indent}{node.var} = await {node.agent_module.run_func}({self._agent_input_expr(node)})"
        args = self._tool_call_args(node)
        call = f"{node.tool_module.alias}({args})"
        if _tool_is_async(node.tool_module.tool):
            return f"{indent}{node.var} = await {call}"
        return f"{indent}{node.var} = {call}"

    def _gather_expr(self, node: _Node) -> str:
        """A single node as an awaitable inside asyncio.gather."""
        if node.node_type == "agent":
            return f"{node.agent_module.run_func}({self._agent_input_expr(node)})"
        args = self._tool_call_args(node)
        if _tool_is_async(node.tool_module.tool):
            return f"{node.tool_module.alias}({args})"
        self.needs_asyncio = True
        return f"asyncio.to_thread({node.tool_module.alias}{', ' if args else ''}{args})"

    # ── orchestration body ──

    def _level_comment(self, index: int, level: List[Chain], parallel: bool) -> str:
        names = ", ".join(
            self.nodes[n].config.get("name", n) for c in level for n in c.nodes
        )
        suffix = " (parallel)" if parallel else ""
        return f"    # ── Level {index}: {names}{suffix} ──"

    def emit(self) -> str:
        chains, _ = decompose_into_chains(self.graph)
        levels = compute_chain_levels(chains)
        chain_slugs: Set[str] = set()

        body: List[str] = []
        for index, level in enumerate(levels, start=1):
            awaitable = [c for c in level if not (len(c.nodes) == 1 and self.nodes[c.nodes[0]].node_type == "trigger")]
            parallel = len(awaitable) > 1
            body.append(self._level_comment(index, level, parallel))

            if not parallel:
                for chain in level:
                    for node_id in chain.nodes:
                        body.append(self._node_statement(self.nodes[node_id]))
                body.append("")
                continue

            # Triggers are plain assignments — set them before the gather.
            for chain in level:
                if chain not in awaitable:
                    body.append(self._node_statement(self.nodes[chain.nodes[0]]))

            self.needs_asyncio = True
            targets: List[str] = []
            calls: List[str] = []
            for chain in awaitable:
                tail = self.nodes[chain.tail]
                targets.append(tail.var)
                if len(chain.nodes) == 1:
                    calls.append(self._gather_expr(tail))
                else:
                    # Multi-node branch: keep its steps sequential and named
                    # inside a local coroutine, gather the coroutine.
                    fn = f"_branch_{_slugify(tail.config.get('name', chain.tail), chain_slugs)}"
                    body.append(f"    async def {fn}():")
                    for node_id in chain.nodes:
                        body.append(self._node_statement(self.nodes[node_id], indent="        "))
                    body.append(f"        return {tail.var}")
                    calls.append(f"{fn}()")
            body.append(f"    {', '.join(targets)} = await asyncio.gather(")
            for call in calls:
                body.append(f"        {call},")
            body.append("    )")
            body.append("")

        exit_points = self.graph.get("exit_points") or [chains[-1].tail]
        exit_vars = [self.nodes[n].var for n in exit_points if n in self.nodes]
        if len(exit_vars) == 1:
            body.append(f"    return {exit_vars[0]}")
        else:
            names = ", ".join(self.nodes[n].config.get("name", n) for n in exit_points)
            body.append(f"    # outputs: {names}")
            body.append(f"    return ({', '.join(exit_vars)})")

        return self._assemble(body)

    def _assemble(self, body: List[str]) -> str:
        self.needs_json = "_as_text" in self.helpers_used

        imports: List[str] = []
        if self.needs_asyncio:
            imports.append("import asyncio")
        if self.needs_json:
            imports.append("import json")

        tool_imports: List[str] = []
        seen_modules: Set[str] = set()
        for node in self.nodes.values():
            tm = node.tool_module
            if tm and tm.module not in seen_modules:
                seen_modules.add(tm.module)
                tool_imports.append(_import_line("tools", tm))
        agent_imports: List[str] = []
        for node in self.nodes.values():
            am = node.agent_module
            if am:
                agent_imports.append(f"from agents.{am.module} import {am.run_func}")

        helper_blocks = [
            block for name, block in (
                ("_as_text", _HELPER_AS_TEXT),
                ("_pick", _HELPER_PICK),
                ("_merge_auto", _HELPER_MERGE_AUTO),
            ) if name in self.helpers_used
        ]

        parts = [
            f'"""Standalone orchestrator for flow "{self.flow.name}".\n\n'
            "Exported from LLM Hub. Nodes run in dependency order; nodes at the\n"
            'same level run concurrently with asyncio.gather.\n"""'
        ]
        if imports:
            parts.append("\n".join(imports))
        if tool_imports or agent_imports:
            parts.append("\n".join(sorted(tool_imports) + sorted(agent_imports)))
        parts.extend(helper_blocks)
        parts.append("async def run_flow(initial_input=None):\n" + "\n".join(body))
        return "\n\n".join(parts) + "\n"


def _import_line(package: str, tm: _ToolModule) -> str:
    if tm.alias == tm.func:
        return f"from {package}.{tm.module} import {tm.func}"
    return f"from {package}.{tm.module} import {tm.func} as {tm.alias}"


# ─── Agent module emitter (compile_agent recipe as source) ───────────────────

def _get_path_description(path_config) -> str:
    if isinstance(path_config, str):
        return path_config
    return path_config.get("description", "")


def _emit_agent_module(am: _AgentModule) -> str:
    provider_config = am.provider_config
    provider = provider_config.get("provider")
    model = provider_config.get("model")
    base_url = provider_config.get("base_url")
    if not provider or not model:
        raise FlowExportError(
            f"LLM config '{provider_config.get('name')}' is missing a provider or model name"
        )

    if provider == "lmstudio":
        key_expr = 'os.environ.get("LMSTUDIO_API_KEY", "lm-studio")'
        base_url = base_url or "http://localhost:1234/v1"
        model_cls, provider_cls, module_suffix = "OpenAIChatModel", "OpenAIProvider", "openai"
    elif provider == "anthropic":
        key_expr = 'os.environ["ANTHROPIC_API_KEY"]'
        model_cls, provider_cls, module_suffix = "AnthropicModel", "AnthropicProvider", "anthropic"
    elif provider == "openai":
        key_expr = 'os.environ["OPENAI_API_KEY"]'
        model_cls, provider_cls, module_suffix = "OpenAIChatModel", "OpenAIProvider", "openai"
    else:
        raise FlowExportError(
            f"Unsupported provider '{provider}'. Supported: anthropic, openai, lmstudio"
        )

    sub_config = am.sub_config
    tool_records = [tm.tool for tm in am.tool_modules]
    system_prompt = sub_config.get("system_prompt", "You are a helpful assistant.")
    system_prompt = resolve_system_prompt_template(system_prompt, sub_config, tool_records)

    output_paths = sub_config.get("output_paths") or {}
    if output_paths:
        routing_lines = ["\n\nYou must choose one of the following output paths:"]
        for path_name, path_config in output_paths.items():
            routing_lines.append(f'- "{path_name.capitalize()}": {_get_path_description(path_config)}')
        system_prompt += "\n".join(routing_lines)

    lines: List[str] = [
        f'"""Agent "{am.agent.name}" — exported from LLM Hub agent id {am.agent.id}."""',
        "import os",
        "",
        "from pydantic_ai import Agent",
        f"from pydantic_ai.models.{module_suffix} import {model_cls}",
        f"from pydantic_ai.providers.{module_suffix} import {provider_cls}",
    ]
    if output_paths:
        lines.insert(2, "from pydantic import BaseModel")
    for tm in am.tool_modules:
        lines.append(_import_line("tools", tm))
    lines.append("")
    lines.append("")

    previous_output_paths = []
    if output_paths:
        class_to_path: Dict[str, str] = {}
        for path_name, path_config in output_paths.items():
            class_name = path_name.capitalize()
            if not class_name.isidentifier():
                raise FlowExportError(
                    f"Output path '{path_name}' of agent '{am.agent.name}' is not a valid class name"
                )
            class_to_path[class_name] = path_name
            if isinstance(path_config, dict) and path_config.get("return_behavior") == "previous_output":
                previous_output_paths.append(path_name)
            lines.append(f"class {class_name}(BaseModel):")
            description = _get_path_description(path_config)
            if description:
                lines.append(f'    """{description}"""')
            lines.append("    content: str")
            lines.append("")
        lines.append("")
        lines.append(f"CLASS_TO_PATH = {class_to_path!r}")
        if previous_output_paths:
            lines.append(f"PREVIOUS_OUTPUT_PATHS = {previous_output_paths!r}")
        lines.append("")

    lines.append(f"SYSTEM_PROMPT = {_py_string(system_prompt)}")
    user_prompt = (sub_config.get("user_prompt") or "").strip()
    if user_prompt:
        lines.append(f"USER_PROMPT = {_py_string(user_prompt)}")
    lines.append("")

    provider_args = [f"api_key={key_expr}"]
    if base_url:
        provider_args.append(f"base_url={base_url!r}")
    lines.append(f"model = {model_cls}(")
    lines.append(f"    {model!r},")
    lines.append(f"    provider={provider_cls}({', '.join(provider_args)}),")
    lines.append(")")

    agent_args = ["model=model", "system_prompt=SYSTEM_PROMPT"]
    if output_paths:
        union = " | ".join(p.capitalize() for p in output_paths)
        agent_args.append(f"output_type={union}")
    lines.append(f"agent = Agent({', '.join(agent_args)})")
    for tm in am.tool_modules:
        lines.append(f"agent.tool_plain({tm.alias})")
    lines.append("")
    lines.append("")

    lines.append(f"async def {am.run_func}(node_input: str) -> str:")
    if user_prompt:
        lines.append('    prompt = USER_PROMPT.replace("{input}", node_input).replace("{message_history}", "")')
    else:
        lines.append("    prompt = node_input")
    lines.append("    result = await agent.run(prompt)")
    lines.append("    output = result.output")
    if output_paths:
        if previous_output_paths:
            lines.append("    # Paths with return_behavior \"previous_output\" hand the input back unchanged.")
            lines.append("    if CLASS_TO_PATH.get(type(output).__name__) in PREVIOUS_OUTPUT_PATHS:")
            lines.append("        return node_input")
        lines.append('    if hasattr(output, "content"):')
        lines.append("        return str(output.content)")
        lines.append("    return str(output)")
    else:
        lines.append('    if hasattr(output, "model_dump"):')
        lines.append("        output = output.model_dump()")
        lines.append('    return str(output) if output else ""')
    return "\n".join(lines) + "\n"


# ─── Tool / requirements / README / __main__ emitters ────────────────────────

def _emit_tool_module(tm: _ToolModule) -> str:
    header = (
        f'# Exported from LLM Hub tool "{tm.tool.name}" (tool id {tm.tool.id}).\n'
        f"# Original user script, unmodified. Entry point: {tm.func}\n\n"
    )
    return header + (tm.tool.script_code or "").rstrip() + "\n"


def _emit_requirements(tools: List[Any], providers_used: Set[str]) -> str:
    packages: Set[str] = set()
    for tool in tools:
        for package in tool.required_packages or []:
            if package.split("==")[0].split("[")[0].lower() not in PACKAGE_DENYLIST:
                packages.add(package)
    lines = sorted(packages)
    if providers_used:
        lines += AGENT_BASE_PINS
        sdk_keys = {"openai" if p == "lmstudio" else p for p in providers_used}
        lines += sorted(SDK_PINS[k] for k in sdk_keys if k in SDK_PINS)
    return "\n".join(lines) + ("\n" if lines else "")


def _entry_description(entry_node: Optional[_Node]) -> Tuple[str, str]:
    """(input format description, example CLI arg) for the entry node."""
    if entry_node is None or entry_node.node_type in ("trigger", "agent"):
        default = (entry_node.config.get("input_value") if entry_node else "") or ""
        desc = "a text input"
        if default:
            desc += f' (optional — defaults to "{default}")'
        return desc, "'your input text'"
    params = _tool_param_names(entry_node.tool_module.tool)
    keys = ", ".join(f'"{p}": ...' for p in params) or '"param": ...'
    return (
        f"a JSON object of keyword arguments for {entry_node.tool_module.func}",
        f"'{{{keys}}}'".replace("...", "<value>"),
    )


def _emit_readme(flow, files: Dict[str, str], env_vars: List[str],
                 entry_node: Optional[_Node]) -> str:
    input_desc, example = _entry_description(entry_node)
    tree = "\n".join(f"  {path}" for path in sorted(files))
    env_section = ""
    if env_vars:
        rows = "\n".join(f"| `{v}` | API key for the corresponding provider |" for v in env_vars)
        env_section = (
            "\n## Credentials\n\n"
            "API keys are read from environment variables — nothing is embedded in this export.\n\n"
            "| Variable | Purpose |\n|---|---|\n" + rows + "\n"
        )
    description = f"\n{flow.description}\n" if flow.description else ""
    return f"""# {flow.name}

Standalone export of the LLM Hub flow "{flow.name}". Runs without LLM Hub —
only the packages in requirements.txt are needed.
{description}
## Files

```
{tree}
```

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```
{env_section}
## Run

From this directory (so the `tools/` and `agents/` packages are importable):

```bash
python __main__.py {example}
```

The flow input is {input_desc}.

## Notes

- Node inputs are wired exactly as they were on the platform (edge mappings
  inlined as explicit keyword arguments).
- Unlike the platform, tool calls have no per-node timeout here.
- Agent tools are registered as the raw Python functions; if a tool script
  lacks type hints or a docstring, the LLM sees a weaker schema than on
  the platform.
"""


def _emit_main(flow, entry_node: Optional[_Node]) -> str:
    input_desc, _ = _entry_description(entry_node)
    tool_entry = entry_node is not None and entry_node.node_type == "tool"
    lines = [
        f'"""Command-line entrypoint for flow "{flow.name}"."""',
        "import argparse",
        "import asyncio",
        "import json",
        "",
        "from flow import run_flow",
        "",
        "",
        "def main():",
        f"    parser = argparse.ArgumentParser(description={('Run flow ' + flow.name)!r})",
    ]
    if tool_entry:
        params = _tool_param_names(entry_node.tool_module.tool)
        lines += [
            f"    parser.add_argument('input', help={input_desc!r})",
            "    args = parser.parse_args()",
            "    try:",
            "        initial_input = json.loads(args.input)",
            "    except json.JSONDecodeError:",
            f"        parser.error('input must be a JSON object with keys: {', '.join(params)}')",
            "    if not isinstance(initial_input, dict):",
            f"        parser.error('input must be a JSON object with keys: {', '.join(params)}')",
        ]
    else:
        lines += [
            f"    parser.add_argument('input', nargs='?', default=None, help={input_desc!r})",
            "    args = parser.parse_args()",
            "    initial_input = args.input",
        ]
    lines += [
        "    result = asyncio.run(run_flow(initial_input))",
        "    if isinstance(result, str):",
        "        print(result)",
        "    else:",
        "        print(json.dumps(result, indent=2, default=str))",
        "",
        "",
        'if __name__ == "__main__":',
        "    main()",
    ]
    return "\n".join(lines) + "\n"


# ─── Resource loading and guards ─────────────────────────────────────────────

def _check_phase1(agent) -> dict:
    """Return the agent's single sub-node config; reject multi-node graphs."""
    graph = agent.graph_config or {}
    nodes = graph.get("nodes") or {}
    edges = graph.get("edges") or []
    if len(nodes) != 1 or any(e.get("is_loop") for e in edges):
        raise FlowExportError(
            f"Agent '{agent.name}' has {len(nodes)} sub-nodes"
            f"{' and a reflection loop' if any(e.get('is_loop') for e in edges) else ''}; "
            "export currently supports single-node agents only"
        )
    return next(iter(nodes.values()))


def _resolve_agent_provider(node_id: str, node_name: str, llm_config: dict,
                            agent_llms: Optional[Dict[str, str]]) -> dict:
    provider_name = (agent_llms or {}).get(node_id)
    if not provider_name:
        raise FlowExportError(
            f"No LLM selected for agent node '{node_name}'. Pick an LLM for the "
            "agent on the canvas before exporting."
        )
    for model_config in llm_config.get("models", []):
        if model_config.get("name") == provider_name:
            return model_config
    raise FlowExportError(f"LLM provider '{provider_name}' not found in config")


def _load_tool(session, tool_id, node_label: str):
    tool = get_tool_by_id(session, tool_id) if tool_id else None
    if tool is None:
        raise FlowExportError(f"Tool with ID {tool_id} not found for {node_label}")
    if not tool.script_code or not tool.main_function:
        raise FlowExportError(f"Tool '{tool.name}' has no script_code/main_function to export")
    return tool


# ─── Public API ──────────────────────────────────────────────────────────────

def export_flow(flow, session, llm_config: dict,
                agent_llms: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build the standalone module for a flow.

    Returns {relative_path: file_content}. Raises FlowExportError for
    anything that makes the flow unexportable (Phase-1 limits, missing
    records, unresolvable providers, unmappable wiring).
    """
    graph_config = flow.graph_config or {}
    if not graph_config.get("nodes"):
        raise FlowExportError("Flow has no nodes to export")

    module_slugs: Set[str] = set()
    var_slugs: Set[str] = set()
    func_aliases: Set[str] = set()
    tool_modules_by_id: Dict[int, _ToolModule] = {}
    nodes: Dict[str, _Node] = {}
    agent_modules: List[_AgentModule] = []
    providers_used: Set[str] = set()

    def tool_module_for(tool) -> _ToolModule:
        existing = tool_modules_by_id.get(tool.id)
        if existing:
            return existing
        module = _slugify(tool.name, module_slugs, fallback=f"tool_{tool.id}")
        alias = tool.main_function
        if alias in func_aliases:
            alias = f"{module}_{tool.main_function}"
        func_aliases.add(alias)
        tm = _ToolModule(tool=tool, module=module, func=tool.main_function, alias=alias)
        tool_modules_by_id[tool.id] = tm
        return tm

    for node_id, config in graph_config["nodes"].items():
        node_type = config.get("node_type") or "tool"
        var = _slugify(config.get("name") or node_id, var_slugs, fallback="node") + "_out"
        node = _Node(node_id=node_id, node_type=node_type, config=config, var=var)

        if node_type == "tool":
            tool = _load_tool(session, _node_resource_id(config), f"node '{node_id}'")
            node.tool_module = tool_module_for(tool)
        elif node_type == "agent":
            agent = get_agent_by_id(session, _node_resource_id(config))
            if agent is None:
                raise FlowExportError(f"Agent with ID {_node_resource_id(config)} not found for node '{node_id}'")
            sub_config = _check_phase1(agent)
            provider_config = _resolve_agent_provider(
                node_id, config.get("name", node_id), llm_config, agent_llms)
            providers_used.add(provider_config.get("provider"))
            agent_tool_modules = [
                tool_module_for(_load_tool(session, tid, f"agent '{agent.name}'"))
                for tid in (sub_config.get("tool_ids") or [])
            ]
            slug = _slugify(agent.name, module_slugs, fallback=f"agent_{agent.id}")
            node.agent_module = _AgentModule(
                agent=agent, sub_config=sub_config, provider_config=provider_config,
                tool_modules=agent_tool_modules, module=slug, run_func=f"run_{slug}",
            )
            agent_modules.append(node.agent_module)
        nodes[node_id] = node

    files: Dict[str, str] = {}
    if tool_modules_by_id:
        files["tools/__init__.py"] = ""
        for tm in tool_modules_by_id.values():
            files[f"tools/{tm.module}.py"] = _emit_tool_module(tm)
    if agent_modules:
        files["agents/__init__.py"] = ""
        for am in agent_modules:
            files[f"agents/{am.module}.py"] = _emit_agent_module(am)

    files["flow.py"] = _FlowEmitter(flow, graph_config, nodes).emit()

    entry_node = nodes.get(graph_config.get("entry_point"))
    all_tools = [tm.tool for tm in tool_modules_by_id.values()]
    files["requirements.txt"] = _emit_requirements(all_tools, providers_used)
    files["__main__.py"] = _emit_main(flow, entry_node)
    env_vars = sorted({API_KEY_ENV_VARS[p] for p in providers_used if p in API_KEY_ENV_VARS})
    files["README.md"] = _emit_readme(flow, files, env_vars, entry_node)
    return files


def export_flow_zip(flow, session, llm_config: dict,
                    agent_llms: Optional[Dict[str, str]] = None) -> Tuple[bytes, str]:
    """Zip the export under a <flow_slug>/ prefix. Returns (bytes, filename)."""
    files = export_flow(flow, session, llm_config, agent_llms)
    slug = _slugify(flow.name, set(), fallback=f"flow_{flow.id}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.writestr(f"{slug}/{path}", files[path])
    return buffer.getvalue(), f"{slug}.zip"
