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

Tool-only flows, single-node agents, and multi-node/looping agents are all
supported — a multi-node agent is exported as a folder (one file per
sub-agent, plus a graph.json) driven by a vendored copy of the agent graph
engine and message serializer, written once per export to agents/_runtime/.
"""

import io
import json
import keyword
import os
import re
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
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


# __file__ is always .../src/exporters/flow_exporter.py, wherever the
# process was started — never build these from a cwd-relative path.
_SRC_DIR = Path(__file__).resolve().parent.parent
_ENGINE_SOURCE = _SRC_DIR / "executors" / "agent_graph_engine.py"
_SERIALIZER_SOURCE = _SRC_DIR / "utils" / "message_serializer.py"


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

_HELPER_RUNTIME_PROMPTS = '''\
"""Build a sub-agent's user message from its prompt template."""
from agents._runtime.message_serializer import serialize_messages


def apply_prompt(user_prompt, node_input, messages):
    """Fill {input} and {message_history}; an empty template sends the input as-is."""
    if not user_prompt:
        return node_input
    result = user_prompt.replace("{input}", node_input)
    if messages:
        return result.replace("{message_history}", serialize_messages(messages))
    return result.replace("{message_history}", "")
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


def _double_quoted(s: str) -> str:
    """Render s as a double-quoted Python string literal (repr's escaping,
    forced onto a `"` delimiter so plain identifiers stay `"writer"`-style).
    Avoids json.dumps — its \\uXXXX surrogate-pair escapes for non-BMP
    characters would not round-trip as a Python string literal."""
    r = repr(s)
    if r[0] == '"':
        return r
    body = r[1:-1]
    out: List[str] = []
    i = 0
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt == "'":
                out.append("'")
            else:
                out.append(c)
                out.append(nxt)
            i += 2
            continue
        if c == '"':
            out.append('\\"')
        else:
            out.append(c)
        i += 1
    return '"' + "".join(out) + '"'


def _needs_docstring_escape(c: str) -> bool:
    return c in ("\\", '"') or c == "\x7f" or (ord(c) < 0x20 and c not in ("\n", "\t"))


def _docstring_text(s: str) -> str:
    """Escape text for safe interpolation inside a triple-quoted docstring."""
    if not any(_needs_docstring_escape(c) for c in s):
        return s
    out = []
    for c in s:
        if c == "\\":
            out.append("\\\\")
        elif c == '"':
            out.append('\\"')
        elif c == "\r":
            out.append("\\r")
        elif c in ("\n", "\t"):
            out.append(c)
        elif ord(c) < 0x20 or c == "\x7f":
            out.append(f"\\x{ord(c):02x}")
        else:
            out.append(c)
    return "".join(out)


def _comment_text(s: str) -> str:
    """Collapse embedded newlines so text stays a single '#' comment line."""
    return s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")


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
class _SubNode:
    node_id: str          # raw canvas node id — data only, never a file name
    module: str           # sub-agent's file name under agents/<slug>/agents/
    config: dict
    tool_modules: List[_ToolModule]


@dataclass
class _AgentModule:
    agent: Any
    sub_config: dict      # the agent's only node for single-node agents
    provider_config: dict
    tool_modules: List[_ToolModule]
    module: str           # file name (single-node) or folder name (multi-node)
    run_func: str         # "run_<slug>"
    graph_config: dict = field(default_factory=dict)
    sub_nodes: List[_SubNode] = field(default_factory=list)
    is_multi: bool = False


@dataclass
class _Node:
    node_id: str
    node_type: str       # tool | agent | trigger
    config: dict
    var: str             # local variable holding the node's output
    tool_module: Optional[_ToolModule] = None
    agent_module: Optional[_AgentModule] = None


def _classify_type_str(type_str: Optional[str]) -> str:
    """Classify a schema type string as "dict", "nondict", or "unknown"."""
    t = str(type_str or "").strip()
    if t.lower().startswith(("dict", "mapping")):
        return "dict"
    if not t or t in ("Any", "object"):
        return "unknown"
    return "nondict"


def _static_output_kind(node: _Node) -> Tuple[str, Optional[Dict[str, Optional[str]]]]:
    """Classify a node's output for static wiring.

    Returns (kind, known_fields): kind is "text" (agents/triggers),
    "dict" (fields known when the output_schema lists properties — a
    {name: type_string} map), "nondict", or "unknown".
    """
    if node.node_type in ("agent", "trigger"):
        return "text", None
    schema = node.tool_module.tool.output_schema or {}
    if not isinstance(schema, dict):
        return "unknown", None
    if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
        fields = {
            name: (prop.get("type") if isinstance(prop, dict) else None)
            for name, prop in schema["properties"].items()
        }
        return "dict", fields
    return _classify_type_str(schema.get("type")), None


# ─── flow.py emitter ─────────────────────────────────────────────────────────

class _FlowEmitter:
    """Builds flow.py: static input wiring + level-ordered orchestration."""

    def __init__(self, flow, graph_config: dict, nodes: Dict[str, _Node],
                 parallel: bool = True):
        self.flow = flow
        self.graph = graph_config
        self.nodes = nodes
        self.parallel = parallel
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

        def add_whole(expr, value_kind, value_fields, source_node):
            """Merge a whole-input value: an unmapped edge or a {field: ""} selection."""
            nonlocal filled
            if value_kind == "dict":
                items.append(("splat", expr))
                if value_fields is not None and filled is not None:
                    filled.update(value_fields)
                else:
                    filled = None
            elif value_kind in ("text", "nondict") and filled is not None:
                if len(params) == 1:
                    target = params[0]
                else:
                    unfilled = [p for p in params if p not in filled]
                    if not unfilled:
                        raise FlowExportError(
                            f"Tool node '{node.node_id}' has no free parameter for the "
                            f"unmapped output of '{source_node}'. Add an edge "
                            f"mapping to say which parameter it should fill."
                        )
                    target = unfilled[0]
                items.append(("kw", target, expr))
                filled.add(target)
            else:
                self.helpers_used.add("_merge_auto")
                items.append(("auto", expr, params))
                filled = None

        for edge in incoming:
            upstream = self.nodes[edge["from_node"]]
            kind, fields = _static_output_kind(upstream)
            mapping = edge.get("mapping")
            if mapping:
                for out_field, in_param in mapping.items():
                    if out_field == "":
                        expr, value_kind, value_fields = upstream.var, kind, fields
                    elif kind == "dict" and fields is not None and out_field in fields:
                        expr = f'{upstream.var}[{out_field!r}]'
                        value_kind = _classify_type_str(fields[out_field])
                        value_fields = None
                    elif kind in ("text", "nondict"):
                        expr, value_kind, value_fields = upstream.var, kind, None
                    else:
                        self.helpers_used.add("_pick")
                        expr = f'_pick({upstream.var}, {out_field!r})'
                        value_kind, value_fields = "unknown", None
                    if in_param == "":
                        add_whole(expr, value_kind, value_fields, edge["from_node"])
                    else:
                        items.append(("kw", in_param, expr))
                        if filled is not None:
                            filled.add(in_param)
            else:
                add_whole(upstream.var, kind, fields, edge["from_node"])
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

    def _selected_upstream_expr(self, edge: dict) -> Tuple[str, str]:
        """(expression, kind) for a whole-input edge, honoring its field
        selection ({out_field: ""} from an expanded dict output)."""
        upstream = self.nodes[edge["from_node"]]
        kind, fields = _static_output_kind(upstream)
        mapping = edge.get("mapping")
        out_field = next(iter(mapping)) if mapping else ""
        if not out_field or kind in ("text", "nondict"):
            return upstream.var, kind
        if kind == "dict" and fields is not None and out_field in fields:
            expr = f'{upstream.var}[{out_field!r}]'
            field_type = str(fields[out_field] or "")
            return expr, ("text" if field_type == "str" else _classify_type_str(field_type))
        self.helpers_used.add("_pick")
        return f'_pick({upstream.var}, {out_field!r})', "unknown"

    def _agent_input_expr(self, node: _Node) -> str:
        incoming = _incoming_edges(self.graph, node.node_id)
        if not incoming:
            if node.node_id == self.entry_point:
                self.helpers_used.add("_as_text")
                return "_as_text(initial_input)"
            return repr(node.config.get("input_value", "") or "")
        expr, kind = self._selected_upstream_expr(incoming[0])
        if kind == "text":
            return expr
        self.helpers_used.add("_as_text")
        return f"_as_text({expr})"

    def _trigger_expr(self, node: _Node) -> str:
        incoming = _incoming_edges(self.graph, node.node_id)
        if incoming:
            return self._selected_upstream_expr(incoming[0])[0]
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
            _comment_text(self.nodes[n].config.get("name", n))
            for c in level for n in c.nodes
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
            parallel = self.parallel and len(awaitable) > 1
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
            names = ", ".join(
                _comment_text(self.nodes[n].config.get("name", n)) for n in exit_points
            )
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

        if self.parallel:
            mode_line = ("Exported from LLM Hub. Nodes run in dependency order; nodes at the\n"
                         "same level run concurrently with asyncio.gather.")
        else:
            mode_line = "Exported from LLM Hub. Nodes run sequentially in dependency order."
        parts = [
            f'"""Standalone orchestrator for flow "{_docstring_text(self.flow.name)}".'
            f'\n\n{mode_line}\n"""'
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


def _resolve_provider_model(provider_config: dict) -> Tuple[str, Any, Optional[str], str, str, str]:
    """(key_expr, model, base_url, model_cls, provider_cls, module_suffix)
    for a provider config, dispatching on its "provider" field."""
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
    return key_expr, model, base_url, model_cls, provider_cls, module_suffix


def _output_path_routing_suffix(output_paths: dict) -> str:
    """Instructions appended to a system prompt when the agent must choose
    among output_paths: one "- "<Cap>": <description>" line per path."""
    if not output_paths:
        return ""
    routing_lines = ["\n\nYou must choose one of the following output paths:"]
    for path_name, path_config in output_paths.items():
        routing_lines.append(f'- "{path_name.capitalize()}": {_get_path_description(path_config)}')
    return "\n".join(routing_lines)


def _emit_single_node_agent(am: _AgentModule) -> str:
    key_expr, model, base_url, model_cls, provider_cls, module_suffix = (
        _resolve_provider_model(am.provider_config)
    )

    sub_config = am.sub_config
    tool_records = [tm.tool for tm in am.tool_modules]
    system_prompt = sub_config.get("system_prompt", "You are a helpful assistant.")
    system_prompt = resolve_system_prompt_template(system_prompt, sub_config, tool_records)

    output_paths = sub_config.get("output_paths") or {}
    system_prompt += _output_path_routing_suffix(output_paths)

    lines: List[str] = [
        f'"""Agent "{_docstring_text(am.agent.name)}" — exported from LLM Hub agent id '
        f'{am.agent.id}."""',
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
                lines.append(f'    """{_docstring_text(description)}"""')
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


def _emit_agent_files(am: _AgentModule) -> Dict[str, str]:
    if am.is_multi:
        return _emit_multi_node_agent(am)
    return {f"agents/{am.module}.py": _emit_single_node_agent(am)}


def _emit_runtime_files() -> Dict[str, str]:
    """agents/_runtime/: the vendored engine + serializer, plus apply_prompt.
    Written once per export, whenever any exported agent is_multi."""
    return {
        "agents/_runtime/__init__.py": "",
        "agents/_runtime/graph_engine.py": _ENGINE_SOURCE.read_text(encoding="utf-8"),
        "agents/_runtime/message_serializer.py": _SERIALIZER_SOURCE.read_text(encoding="utf-8"),
        "agents/_runtime/prompts.py": _HELPER_RUNTIME_PROMPTS,
    }


# ─── Multi-node agent folder emitter ──────────────────────────────────────────

# Names every sub-agent file defines or imports; a tool whose main function
# shares one of these would shadow it (e.g. a tool named "model" would
# replace the LLM model). Imported under an alias instead — see
# _sub_agent_tool_import.
_SUB_AGENT_RESERVED_NAMES = {
    "agent", "run", "model", "apply_prompt", "Agent", "BaseModel",
    "SYSTEM_PROMPT", "USER_PROMPT", "CLASS_TO_PATH", "_output_types",
}


def _sub_agent_tool_import(tm: _ToolModule) -> Tuple[str, str]:
    """(import line, local name) for a tool imported into a sub-agent file."""
    if tm.alias in _SUB_AGENT_RESERVED_NAMES:
        local_name = f"{tm.module}_{tm.func}"
        return f"from tools.{tm.module} import {tm.func} as {local_name}", local_name
    return _import_line("tools", tm), tm.alias


def _emit_multi_model(am: _AgentModule) -> str:
    key_expr, model, base_url, model_cls, provider_cls, module_suffix = (
        _resolve_provider_model(am.provider_config)
    )

    provider_args = [f"api_key={key_expr}"]
    if base_url:
        provider_args.append(f"base_url={base_url!r}")

    lines = [
        f'"""Shared model for agent "{_docstring_text(am.agent.name)}": every sub-agent '
        f'uses this one provider."""',
        "import os",
        "",
        f"from pydantic_ai.models.{module_suffix} import {model_cls}",
        f"from pydantic_ai.providers.{module_suffix} import {provider_cls}",
        "",
        f"model = {model_cls}(",
        f"    {model!r},",
        f"    provider={provider_cls}({', '.join(provider_args)}),",
        ")",
    ]
    return "\n".join(lines) + "\n"


def _emit_sub_agent(am: _AgentModule, sub: _SubNode) -> str:
    config = sub.config
    tool_records = [tm.tool for tm in sub.tool_modules]
    system_prompt = config.get("system_prompt", "You are a helpful assistant.")
    system_prompt = resolve_system_prompt_template(system_prompt, config, tool_records)

    output_paths = config.get("output_paths") or {}
    system_prompt += _output_path_routing_suffix(output_paths)

    sub_name = _docstring_text(config.get("name") or sub.node_id)
    lines: List[str] = [
        f'"""Sub-agent "{sub_name}" of agent "{_docstring_text(am.agent.name)}"."""',
    ]
    if output_paths:
        lines.append("from pydantic import BaseModel")
    lines.append("from pydantic_ai import Agent")
    lines.append("from agents._runtime.prompts import apply_prompt")
    lines.append(f"from agents.{am.module}.model import model")

    tool_locals: List[str] = []
    for tm in sub.tool_modules:
        import_line, local_name = _sub_agent_tool_import(tm)
        lines.append(import_line)
        tool_locals.append(local_name)
    lines.append("")

    class_to_path: Dict[str, str] = {}
    if output_paths:
        for path_name in output_paths:
            class_name = path_name.capitalize()
            if not class_name.isidentifier():
                raise FlowExportError(
                    f"Output path '{path_name}' of agent '{am.agent.name}' is not a valid class name"
                )
            class_to_path[class_name] = path_name

    lines.append(f"SYSTEM_PROMPT = {_py_string(system_prompt)}")
    user_prompt = (config.get("user_prompt") or "").strip()
    user_prompt_literal = _py_string(user_prompt) if user_prompt else '""'
    lines.append(f"USER_PROMPT = {user_prompt_literal}")
    lines.append("")

    if output_paths:
        lines.append("")
        lines.append("def _output_types():")
        for path_name, path_config in output_paths.items():
            class_name = path_name.capitalize()
            lines.append(f"    class {class_name}(BaseModel):")
            description = _get_path_description(path_config)
            if description:
                lines.append(f'        """{_docstring_text(description)}"""')
            lines.append("        content: str")
            lines.append("")
        union = " | ".join(class_to_path.keys())
        lines.append(f"    return {union}")
        lines.append("")
        lines.append("")
        lines.append(f"CLASS_TO_PATH = {class_to_path!r}")
        lines.append("")

    agent_args = ["model=model", "system_prompt=SYSTEM_PROMPT"]
    if output_paths:
        agent_args.append("output_type=_output_types()")
    lines.append(f"agent = Agent({', '.join(agent_args)})")
    for local_name in tool_locals:
        lines.append(f"agent.tool_plain({local_name})")
    lines.append("")
    lines.append("")

    lines.append("async def run(node_input, messages):")
    lines.append("    result = await agent.run(apply_prompt(USER_PROMPT, node_input, messages))")
    lines.append("    output = result.output")
    if output_paths:
        lines.append("    chosen_path = CLASS_TO_PATH.get(type(output).__name__)")
        lines.append('    if hasattr(output, "content"):')
        lines.append("        text = str(output.content)")
        lines.append('    elif hasattr(output, "model_dump"):')
        lines.append("        text = str(output.model_dump())")
        lines.append("    else:")
        lines.append("        text = str(output)")
        lines.append("    return (text, result.all_messages(), chosen_path)")
    else:
        lines.append('    if hasattr(output, "model_dump"):')
        lines.append("        output = output.model_dump()")
        lines.append('    return (str(output) if output else "", result.all_messages(), None)')

    return "\n".join(lines) + "\n"


def _emit_sub_agents_init(am: _AgentModule) -> str:
    sub_modules = sorted({sub.module for sub in am.sub_nodes})
    lines = [
        f"from agents.{am.module}.agents import {', '.join(sub_modules)}",
        "",
        "",
        "RUNNERS = {",
    ]
    for sub in am.sub_nodes:
        lines.append(f"    {_double_quoted(sub.node_id)}: {sub.module}.run,")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _trim_agent_graph(graph_config: dict) -> dict:
    """graph.json: only what the vendored engine reads to run the graph
    (node names, agent_type, output_paths, edges, entry/exit points, and
    max_loop_iterations). No prompts, tool_ids or llm_provider — those live
    in the sub-agent files."""
    nodes = {}
    for node_id, cfg in (graph_config.get("nodes") or {}).items():
        trimmed = {"name": cfg.get("name", node_id), "agent_type": cfg.get("agent_type")}
        if cfg.get("output_paths"):
            trimmed["output_paths"] = cfg["output_paths"]
        nodes[node_id] = trimmed
    result = {
        "nodes": nodes,
        "edges": graph_config.get("edges") or [],
        "entry_point": graph_config.get("entry_point"),
        "exit_points": graph_config.get("exit_points") or [],
    }
    if graph_config.get("max_loop_iterations") is not None:
        result["max_loop_iterations"] = graph_config["max_loop_iterations"]
    return result


def _emit_agent_orchestrator(am: _AgentModule) -> str:
    lines = [
        f'"""Agent "{_docstring_text(am.agent.name)}" — exported from LLM Hub agent id '
        f'{am.agent.id}. Runs the graph in graph.json."""',
        "import json",
        "from pathlib import Path",
        "",
        "from agents._runtime.graph_engine import run_graph",
        f"from agents.{am.module}.agents import RUNNERS",
        "",
        'GRAPH_CONFIG = json.loads((Path(__file__).parent / "graph.json").read_text(encoding="utf-8"))',
        "",
        "",
        "async def _run_node(node_id, node_config, node_input, messages):",
        "    output, msgs, chosen_path = await RUNNERS[node_id](node_input, messages)",
        "    return output, msgs, chosen_path, None",
        "",
        "",
        f"async def {am.run_func}(node_input: str) -> str:",
        '    return (await run_graph(GRAPH_CONFIG, node_input, _run_node))["result"]',
    ]
    return "\n".join(lines) + "\n"


def _emit_multi_node_agent(am: _AgentModule) -> Dict[str, str]:
    pkg = f"agents/{am.module}"
    files: Dict[str, str] = {
        f"{pkg}/model.py": _emit_multi_model(am),
        f"{pkg}/graph.json": json.dumps(_trim_agent_graph(am.graph_config), indent=2) + "\n",
        f"{pkg}/agent.py": _emit_agent_orchestrator(am),
        f"{pkg}/__init__.py": f"from agents.{am.module}.agent import {am.run_func}\n",
        f"{pkg}/agents/__init__.py": _emit_sub_agents_init(am),
    }
    for sub in am.sub_nodes:
        files[f"{pkg}/agents/{sub.module}.py"] = _emit_sub_agent(am, sub)
    return files


# ─── Tool / requirements / README / __main__ emitters ────────────────────────

def _emit_tool_module(tm: _ToolModule) -> str:
    header = (
        f'# Exported from LLM Hub tool "{_comment_text(tm.tool.name)}" (tool id {tm.tool.id}).\n'
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
                 entry_node: Optional[_Node], has_multi_agent: bool = False) -> str:
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
    multi_agent_section = ""
    if has_multi_agent:
        multi_agent_section = (
            "\n## Multi-node agents\n\n"
            "An agent with more than one node (or a reflection loop) is a folder under\n"
            "`agents/`: one file per sub-agent, a `graph.json` describing the node graph,\n"
            "and a shared `agents/_runtime/` (vendored graph engine + message serializer)\n"
            "used by every multi-node agent in this export.\n\n"
            "Unlike the platform, which retries a failed LLM call up to 3 more times with\n"
            "backoff, this export relies only on the provider SDK's built-in retries (2 per\n"
            "request) — a long provider outage that the platform would survive can fail an\n"
            "exported run.\n"
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
{multi_agent_section}
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
        f'"""Command-line entrypoint for flow "{_docstring_text(flow.name)}"."""',
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

def _validate_agent_graph(agent) -> dict:
    """Return the agent's graph_config; reject only what the vendored engine
    can't run. build_adjacency crashes with a bare KeyError on an unknown
    from_node and silently skips an unknown to_node, so both are rejected
    here with a message naming the agent and the bad edge."""
    graph = agent.graph_config or {}
    nodes = graph.get("nodes") or {}
    if not nodes:
        raise FlowExportError(f"Agent '{agent.name}' has no nodes to export")
    entry_point = graph.get("entry_point")
    if not entry_point or entry_point not in nodes:
        raise FlowExportError(
            f"Agent '{agent.name}' has an invalid entry_point '{entry_point}'"
        )
    for edge in graph.get("edges") or []:
        from_node = edge.get("from_node")
        to_node = edge.get("to_node")
        if not from_node or from_node not in nodes:
            raise FlowExportError(
                f"Agent '{agent.name}' has an edge from unknown node '{from_node}' "
                f"to '{to_node}'"
            )
        if not to_node or to_node not in nodes:
            raise FlowExportError(
                f"Agent '{agent.name}' has an edge from '{from_node}' to unknown "
                f"node '{to_node}'"
            )
    return graph


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
                agent_llms: Optional[Dict[str, str]] = None,
                parallel: bool = True) -> Dict[str, str]:
    """Build the standalone module for a flow.

    Returns {relative_path: file_content}. Raises FlowExportError for
    anything that makes the flow unexportable (Phase-1 limits, missing
    records, unresolvable providers, unmappable wiring).

    parallel=False emits a fully sequential flow.py: same-level branches
    run one after another instead of via asyncio.gather.
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
            agent_graph = _validate_agent_graph(agent)
            agent_nodes = agent_graph.get("nodes") or {}
            agent_edges = agent_graph.get("edges") or []
            is_multi = len(agent_nodes) > 1 or any(e.get("is_loop") for e in agent_edges)
            provider_config = _resolve_agent_provider(
                node_id, config.get("name", node_id), llm_config, agent_llms)
            providers_used.add(provider_config.get("provider"))
            slug = _slugify(agent.name, module_slugs, fallback=f"agent_{agent.id}")

            if is_multi:
                sub_name_slugs: Set[str] = set()
                sub_nodes: List[_SubNode] = []
                for sub_id, sub_cfg in agent_nodes.items():
                    sub_module = _slugify(sub_cfg.get("name") or "", sub_name_slugs, fallback="node")
                    sub_tool_ids_seen: Set[int] = set()
                    sub_tool_modules: List[_ToolModule] = []
                    for tid in (sub_cfg.get("tool_ids") or []):
                        tool = _load_tool(session, tid, f"agent '{agent.name}' sub-node '{sub_id}'")
                        if tool.id in sub_tool_ids_seen:
                            continue
                        sub_tool_ids_seen.add(tool.id)
                        sub_tool_modules.append(tool_module_for(tool))
                    sub_nodes.append(_SubNode(
                        node_id=sub_id, module=sub_module, config=sub_cfg,
                        tool_modules=sub_tool_modules,
                    ))
                node.agent_module = _AgentModule(
                    agent=agent, sub_config=next(iter(agent_nodes.values())),
                    provider_config=provider_config, tool_modules=[], module=slug,
                    run_func=f"run_{slug}", graph_config=agent_graph,
                    sub_nodes=sub_nodes, is_multi=True,
                )
            else:
                sub_config = next(iter(agent_nodes.values()))
                agent_tool_modules = [
                    tool_module_for(_load_tool(session, tid, f"agent '{agent.name}'"))
                    for tid in (sub_config.get("tool_ids") or [])
                ]
                node.agent_module = _AgentModule(
                    agent=agent, sub_config=sub_config, provider_config=provider_config,
                    tool_modules=agent_tool_modules, module=slug, run_func=f"run_{slug}",
                    graph_config=agent_graph, sub_nodes=[], is_multi=False,
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
            files.update(_emit_agent_files(am))
        if any(am.is_multi for am in agent_modules):
            files.update(_emit_runtime_files())

    files["flow.py"] = _FlowEmitter(flow, graph_config, nodes, parallel=parallel).emit()

    entry_node = nodes.get(graph_config.get("entry_point"))
    all_tools = [tm.tool for tm in tool_modules_by_id.values()]
    files["requirements.txt"] = _emit_requirements(all_tools, providers_used)
    files["__main__.py"] = _emit_main(flow, entry_node)
    env_vars = sorted({API_KEY_ENV_VARS[p] for p in providers_used if p in API_KEY_ENV_VARS})
    has_multi_agent = any(am.is_multi for am in agent_modules)
    files["README.md"] = _emit_readme(flow, files, env_vars, entry_node, has_multi_agent=has_multi_agent)
    return files


def export_flow_zip(flow, session, llm_config: dict,
                    agent_llms: Optional[Dict[str, str]] = None,
                    parallel: bool = True) -> Tuple[bytes, str]:
    """Zip the export under a <flow_slug>/ prefix. Returns (bytes, filename)."""
    files = export_flow(flow, session, llm_config, agent_llms, parallel=parallel)
    slug = _slugify(flow.name, set(), fallback=f"flow_{flow.id}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.writestr(f"{slug}/{path}", files[path])
    return buffer.getvalue(), f"{slug}.zip"
