"""
Flow Runner

Executes a flow by generating an orchestrator from its graph_config
(`src.factories.flow_to_script_factory.generate_orchestrator_code`), exec-ing
that source, and running the generated async `orchestrate(orchestrator, ...)`.

The generated code calls back into two module-level helpers defined here:
`_run_stage(orchestrator, node_id, stage_input)` and
`_resolve_node_input(orchestrator, node_id, upstream_map)`. The `orchestrator`
object is a `FlowRunContext` — a plain handle the generated code passes around.

Design (see claude_plans/item1-flow-runner.md):
- Chains at the same topological level run concurrently via `asyncio.gather`
  (parallel=True). Each stage opens its OWN DB session because
  create/update_execution commit, and a shared Session is not safe under gather.
- Input is state-authoritative: `_run_stage` re-derives each node's input from
  `ctx.state` + incoming-edge mappings, ignoring the passed `stage_input` except
  as the entry-point seed. The generated code's argument passing therefore drives
  ORDERING; data flows through `ctx.state`.
- `sequence` is precomputed from graph topology so the execution tree renders in a
  stable order regardless of which concurrent stage finishes first.
- Tool nodes are compiled in-process via tool_runner.compile_tool (native
  objects, no serialization); agent nodes reuse agent_executor. This module
  therefore executes user tool code and must never be driven from inside the
  backend API process — only the local flow child, the hosted flow-runner
  container, or pytest.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

from src.database.database_setup import DatabaseManager, Flow, Execution
from src.database.database import (
    create_execution,
    update_execution,
    get_execution_by_id,
    get_tool_by_id,
    get_agent_by_id,
)
from src.runners.tool_runner import compile_tool
from src.runners.agent_runner import BuiltAgent, compile_agent, resolve_provider_config
from src.factories.flow_to_script_factory import (
    generate_orchestrator_code,
    decompose_into_chains,
    compute_chain_levels,
)

logger = logging.getLogger(__name__)

TOOL_TIMEOUT_SECONDS = int(os.environ.get("TOOL_TIMEOUT_SECONDS", "300"))


def _json_safe(value: Any) -> Any:
    """Return `value` if it round-trips through JSON, else its str() form.

    Execution records store input/output in JSON columns. Tool outputs can be
    native Python objects (numpy arrays, DataFrames) that aren't serializable —
    this keeps recording from crashing the flow. The live value kept in
    `ctx.state` is always the raw object; only the DB copy is coerced.
    """
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _preview(value: Any, limit: int = 1000) -> Optional[str]:
    """Bounded human-readable preview of a node input/output for JSON columns.

    Full values are never persisted (they can be arbitrarily large); the raw
    object lives in ctx.state. DataFrames are duck-typed — pandas is not a
    dependency of this module or the flow-runner image. Never raises.
    """
    if value is None:
        return None
    if hasattr(value, "head") and hasattr(value, "to_string"):
        try:
            return str(value.head(20))[:limit]
        except Exception:
            pass
    try:
        s = value if isinstance(value, str) else json.dumps(value, default=str)
    except Exception:
        s = str(value)
    return s[:limit]


# ─── Context object (the `orchestrator` handle passed to generated code) ──────

@dataclass
class FlowRunContext:
    flow_id: int
    user_id: int
    graph_config: dict
    llm_config: dict
    agent_llms: Dict[str, str]
    conda_env: Optional[str]
    root_execution_id: int
    entry_point: str
    functions_by_node: Dict[str, Callable]      # tool node_id -> executable fn
    tool_schemas: Dict[str, dict]               # tool node_id -> input_schema
    agents_by_node: Dict[str, Dict[str, BuiltAgent]]  # agent node_id -> sub-node id -> BuiltAgent
    agent_graphs: Dict[str, dict]               # agent node_id -> agent graph_config at prepare time
    sequence_by_node: Dict[str, int]            # node_id -> precomputed sequence
    state: Dict[str, Any] = field(default_factory=dict)        # node_id -> raw output
    child_rows: Dict[str, int] = field(default_factory=dict)   # node_id -> execution row id
    fingerprints: Dict[str, dict] = field(default_factory=dict)  # node_id -> change-detection snapshot


# ─── Input resolution (state-authoritative; ports FlowExecutor._apply_mapping) ─

def _incoming_edges(graph_config: dict, node_id: str) -> List[dict]:
    return [
        e for e in graph_config.get("edges", [])
        if e.get("to_node") == node_id and not e.get("is_loop")
    ]


def _stringify_for_agent(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, default=str)


def _select_output_field(upstream: Any, mapping: Optional[dict]) -> Any:
    """Apply an edge's field selection to a whole-input target.

    An expanded dict output connected to a generic input (agent or
    schema-less tool) stores {out_field: ""}: pick that key from the
    upstream dict; fall back to the whole value."""
    if not mapping:
        return upstream
    out_field = next(iter(mapping))
    if out_field and isinstance(upstream, dict) and out_field in upstream:
        return upstream[out_field]
    return upstream


def _merge_into_base(base: dict, value: Any, props: dict) -> bool:
    """Merge an unmapped/whole-input value into a tool's kwargs.

    Dict values merge key by key; anything else fills the tool's only
    parameter or the first parameter not already provided. Returns False
    when there is nowhere to put a non-dict value — the caller then uses
    the value as the node's entire input."""
    if isinstance(value, dict):
        base.update(value)
        return True
    param_names = list(props.keys())
    unfilled = [p for p in param_names if p not in base]
    if len(param_names) == 1:
        base[param_names[0]] = value
        return True
    if unfilled:
        base[unfilled[0]] = value
        return True
    return False


def _build_node_input(ctx: FlowRunContext, node_id: str, stage_input: Any) -> Any:
    """Derive a node's real input from ctx.state + incoming-edge mappings.

    `stage_input` is used only as the seed for a node with no incoming edges
    (the entry_point). Every other node reads its upstream outputs from
    ctx.state and applies each edge's mapping / the node's typed-in values.
    """
    node_config = ctx.graph_config["nodes"][node_id]
    node_type = node_config.get("node_type") or "tool"
    incoming = _incoming_edges(ctx.graph_config, node_id)

    # Source nodes (no incoming edges).
    if not incoming:
        if node_id == ctx.entry_point:
            if node_type == "trigger":
                iv = node_config.get("input_value")
                return iv if iv else stage_input
            return stage_input  # tool/agent entry: initial_input as-is
        # Non-entry source node: run on its own typed-in values only.
        if node_type == "trigger":
            return node_config.get("input_value", "")
        if node_type == "agent":
            return node_config.get("input_value", "") or ""
        return dict(node_config.get("input_values", {}))

    # Agents consume text: take the first upstream output (or its selected
    # field, for edges from an expanded dict output), coerce to text.
    if node_type == "agent":
        edge = incoming[0]
        upstream = _select_output_field(ctx.state.get(edge["from_node"]), edge.get("mapping"))
        return _stringify_for_agent(upstream)

    if node_type == "trigger":  # unusual, but be safe
        edge = incoming[0]
        return _select_output_field(ctx.state.get(edge["from_node"]), edge.get("mapping"))

    # Tool target: base = typed-in values, overlay edge-mapped upstream values.
    base = dict(node_config.get("input_values", {}))
    schema = ctx.tool_schemas.get(node_id) or {}
    props = schema.get("properties", schema) if isinstance(schema, dict) else {}

    for edge in incoming:
        upstream = ctx.state.get(edge["from_node"])
        mapping = edge.get("mapping")
        if mapping:
            for out_field, in_param in mapping.items():
                if out_field == "" or not (isinstance(upstream, dict) and out_field in upstream):
                    value = upstream                     # whole output / scalar upstream
                else:
                    value = upstream[out_field]
                if in_param == "":
                    # Generic whole-input target: merge the selected value.
                    if not _merge_into_base(base, value, props):
                        return value                     # nothing to map into
                else:
                    base[in_param] = value
        elif not _merge_into_base(base, upstream, props):
            return upstream                              # nothing to map into
    return base


def _resolve_node_input(orchestrator: FlowRunContext, node_id: str, upstream_map=None) -> Any:
    """Called by generated code for chain heads. `upstream_map` is an ordering
    hint only — the real input is re-derived from ctx.state + edges."""
    return _build_node_input(orchestrator, node_id, None)


# ─── Per-node execution + recording ──────────────────────────────────────────

async def _execute_node_async(ctx: FlowRunContext, node_id: str, node_input: Any) -> Any:
    """Run a single node with an already-resolved input; record + store state.

    Opens its own DB session (session-per-stage, safe under gather). Raises on
    failure after recording, so the orchestrator fails fast.
    """
    node_config = ctx.graph_config["nodes"][node_id]
    node_type = node_config.get("node_type") or "tool"
    session = DatabaseManager().get_session()
    try:
        exec_kwargs = dict(
            parent_id=ctx.root_execution_id,
            user_id=ctx.user_id,
            execution_type=node_type,
            node_id=node_id,
            name=node_config.get("name", node_id),
            sequence=ctx.sequence_by_node.get(node_id),
            status="running",
            started_at=datetime.now(),
        )
        if node_type == "tool":
            exec_kwargs["tool_id"] = node_config.get("id") or node_config.get("tool_id")
            exec_kwargs["input_data"] = _preview(node_input)
        elif node_type == "agent":
            exec_kwargs["agent_id"] = node_config.get("id")
            input_text = _stringify_for_agent(node_input)
            exec_kwargs["input_data"] = _preview(input_text)
        else:  # trigger
            exec_kwargs["input_data"] = _preview(node_input)

        # Resume reuses the node's original row (one tree per logical run; the
        # flowrunner DB role has no DELETE grant, so update-in-place is the
        # only dedup mechanism available in hosted mode).
        reuse_id = ctx.child_rows.get(node_id)
        if reuse_id is not None:
            child = update_execution(
                session, reuse_id,
                status="running",
                error_message=None,
                started_at=exec_kwargs["started_at"],
                completed_at=None,
                sequence=exec_kwargs["sequence"],
                input_data=exec_kwargs["input_data"],
                output_data=None,
            )
        else:
            child = create_execution(session, **exec_kwargs)
            ctx.child_rows[node_id] = child.id

        try:
            if node_type == "tool":
                output = await _run_tool_node(ctx, node_id, node_input)
            elif node_type == "agent":
                output = await _run_agent_node(ctx, session, node_id, node_config, input_text, child)
            else:  # trigger
                output = node_input

            update_execution(
                session, child.id,
                status="completed",
                completed_at=datetime.now(),
                output_data=_preview(output),
            )
            ctx.state[node_id] = output
            return output

        except Exception as e:
            update_execution(
                session, child.id,
                status="failed",
                error_message=f"{type(e).__name__}: {e}",
                completed_at=datetime.now(),
            )
            logger.exception("Node %s failed", node_id)
            raise
    finally:
        session.close()


async def _run_tool_node(ctx: FlowRunContext, node_id: str, node_input: Any) -> Any:
    """Run a compiled tool function in a worker thread under the global timeout."""
    func = ctx.functions_by_node.get(node_id)
    if func is None:
        raise ValueError(f"No executable function prepared for tool node '{node_id}'")
    if not isinstance(node_input, dict):
        raise TypeError(
            f"Tool node '{node_id}' expected a dict of kwargs, got {type(node_input).__name__}"
        )
    try:
        # Off the event loop so concurrent stages in the same wave stay
        # responsive. Soft timeout: the thread survives, its result is discarded.
        return await asyncio.wait_for(
            asyncio.to_thread(func, **node_input), timeout=TOOL_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        raise TimeoutError(
            f"Tool node '{node_id}' timed out after {TOOL_TIMEOUT_SECONDS}s"
        ) from None


async def _run_agent_node(ctx: FlowRunContext, session: Session, node_id: str,
                          node_config: dict, input_text: str, child) -> str:
    """Run an agent node's pre-built sub-agents via AgentExecutor (async)."""
    from src.executors.agent_executor import AgentExecutor

    built_agents = ctx.agents_by_node.get(node_id)
    graph_config = ctx.agent_graphs.get(node_id)
    if built_agents is None or graph_config is None:
        raise ValueError(f"No compiled agent for flow node '{node_id}'")

    executor = AgentExecutor(session)
    return await executor.execute_agent_node(
        graph_config, input_text, built_agents, parent_execution=child
    )


# ─── Runtime helper called by the generated orchestrator ─────────────────────

async def _run_stage(orchestrator: FlowRunContext, node_id: str, stage_input: Any) -> Any:
    """Entry point the generated code awaits for every node.

    Nodes whose output is already in ctx.state short-circuit — this is what
    makes resume a plain re-run of the generated orchestrate: retained
    upstream results return instantly, only missing nodes execute.
    """
    if node_id in orchestrator.state:
        return orchestrator.state[node_id]
    node_input = _build_node_input(orchestrator, node_id, stage_input)
    return await _execute_node_async(orchestrator, node_id, node_input)


# ─── Resume support: change detection over the current graph ─────────────────

def _compute_fingerprints(session: Session, graph_config: dict) -> Dict[str, dict]:
    """Per-node snapshot used by resume() to detect what changed since run().

    Covers the node's own config, its incoming (non-loop) edges, and the
    updated_at of the tool/agent it references. A node whose fingerprint
    differs — or is new — gets its cached output invalidated, along with
    everything downstream of it.
    """
    fps: Dict[str, dict] = {}
    for node_id, node_config in graph_config.get("nodes", {}).items():
        node_type = node_config.get("node_type") or "tool"
        resource_updated = None
        if node_type == "tool":
            tool_id = node_config.get("id") or node_config.get("tool_id")
            tool = get_tool_by_id(session, tool_id) if tool_id else None
            if tool is not None and tool.updated_at:
                resource_updated = tool.updated_at.isoformat()
        elif node_type == "agent":
            agent_id = node_config.get("id") or node_config.get("agent_id")
            agent = get_agent_by_id(session, agent_id) if agent_id else None
            if agent is not None and agent.updated_at:
                resource_updated = agent.updated_at.isoformat()
        fps[node_id] = {
            "config": json.dumps(node_config, sort_keys=True, default=str),
            "in_edges": sorted(
                json.dumps(e, sort_keys=True, default=str)
                for e in _incoming_edges(graph_config, node_id)
            ),
            "resource_updated_at": resource_updated,
        }
    return fps


def _with_descendants(graph_config: dict, seeds: set) -> set:
    """Expand `seeds` with every node reachable from them via non-loop edges.

    Cached descendants of an invalidated node were computed from its OLD
    output, so they must re-run too — without this, ctx.state loses its
    ancestor-closure property and resume could serve stale results.
    """
    out_edges: Dict[str, List[str]] = {}
    for e in graph_config.get("edges", []) or []:
        if e.get("is_loop"):
            continue
        out_edges.setdefault(e["from_node"], []).append(e["to_node"])
    result = set(seeds)
    stack = list(seeds)
    while stack:
        for nxt in out_edges.get(stack.pop(), []):
            if nxt not in result:
                result.add(nxt)
                stack.append(nxt)
    return result


def _sweep_running_children(session: Session, root_id: int) -> None:
    """Mark child rows stuck at 'running' as failed.

    When one parallel chain raises, asyncio cancels sibling stages with
    CancelledError — a BaseException that bypasses the per-node failure
    recording — so without this sweep those rows would stay 'running' forever.
    """
    rows = (
        session.query(Execution)
        .filter(Execution.parent_id == root_id, Execution.status == "running")
        .all()
    )
    for row in rows:
        update_execution(
            session, row.id,
            status="failed",
            error_message="interrupted by sibling failure",
            completed_at=datetime.now(),
        )


# ─── Sync/async bridge ───────────────────────────────────────────────────────

def _run_async(coro):
    """Run an async coroutine from a sync caller.

    FastAPI sync `def` endpoints and Celery workers have no running loop, so
    asyncio.run works. The ThreadPoolExecutor fallback covers the unlikely case
    of being called while a loop is already running.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return pool.submit(asyncio.run, coro).result()


# ─── Public API ──────────────────────────────────────────────────────────────

class FlowRunner:
    """Codegen-based flow execution engine (replaces FlowExecutor)."""

    def __init__(self, session: Session, flow_id: int, user_id: int,
                 llm_config: Optional[Dict] = None,
                 agent_llms: Optional[Dict[str, str]] = None):
        self.session = session
        self.flow_id = flow_id
        self.user_id = user_id
        self.llm_config = llm_config or {"models": []}
        self.agent_llms = agent_llms or {}

        flow = session.query(Flow).filter(Flow.id == flow_id).first()
        if not flow:
            raise ValueError(f"Flow {flow_id} not found")
        self.flow = flow
        self.flow_name = flow.name
        self.graph_config = flow.graph_config

        # Checkpoint state retained across a failure for resume().
        self.ctx: Optional[FlowRunContext] = None
        self.root_execution_id: Optional[int] = None
        self.conda_env: Optional[str] = None
        self._initial_input: Any = None

    # -- helpers ------------------------------------------------------------

    def _prepare_tools(self):
        """Compile each tool node's main function and capture its input_schema."""
        functions: Dict[str, Callable] = {}
        schemas: Dict[str, dict] = {}
        for node_name, node_info in self.graph_config["nodes"].items():
            if node_info.get("node_type") in ("agent", "trigger"):
                continue
            tool_id = node_info.get("id") or node_info.get("tool_id")
            tool = get_tool_by_id(self.session, tool_id)
            if not tool:
                raise ValueError(f"Tool with ID {tool_id} not found for node '{node_name}'")
            functions[node_name] = compile_tool(tool)
            schemas[node_name] = tool.input_schema
        return functions, schemas

    def _prepare_agents(self):
        """Compile every agent node's sub-agents and capture its graph_config.

        One provider per flow agent node (from agent_llms); credentials are
        bound into each BuiltAgent's model object at compile time, so no
        env-var state is shared between agents.
        """
        agents: Dict[str, Dict[str, BuiltAgent]] = {}
        graphs: Dict[str, dict] = {}
        for node_name, node_info in self.graph_config["nodes"].items():
            if node_info.get("node_type") != "agent":
                continue
            agent_id = node_info.get("id") or node_info.get("agent_id")
            agent = get_agent_by_id(self.session, agent_id)
            if not agent:
                raise ValueError(f"Agent with ID {agent_id} not found for node '{node_name}'")
            if not agent.graph_config:
                raise ValueError(f"Agent {agent_id} has no graph_config")
            llm_provider = self.agent_llms.get(node_name)
            if not llm_provider:
                raise ValueError(f"No LLM selected for agent node '{node_name}'")
            provider_config = resolve_provider_config(self.llm_config, llm_provider)

            sub_agents: Dict[str, BuiltAgent] = {}
            for sub_node_id, sub_config in agent.graph_config.get("nodes", {}).items():
                tool_ids = sub_config.get("tool_ids", [])
                tool_records = [t for t in (get_tool_by_id(self.session, tid) for tid in tool_ids) if t]
                sub_agents[sub_node_id] = compile_agent(sub_config, tool_records, provider_config)
            agents[node_name] = sub_agents
            graphs[node_name] = agent.graph_config
        return agents, graphs

    def _compute_sequence(self) -> Dict[str, int]:
        """Precompute a stable node_id -> sequence from graph topology."""
        chains, _ = decompose_into_chains(self.graph_config)
        levels = compute_chain_levels(chains)
        seq: Dict[str, int] = {}
        i = 0
        for level in levels:
            for chain in level:
                for node in chain.nodes:
                    seq[node] = i
                    i += 1
        return seq

    def _build_context(self, root_id: int, conda_env: Optional[str]) -> FlowRunContext:
        functions, schemas = self._prepare_tools()
        agents, agent_graphs = self._prepare_agents()
        return FlowRunContext(
            flow_id=self.flow_id,
            user_id=self.user_id,
            graph_config=self.graph_config,
            llm_config=self.llm_config,
            agent_llms=self.agent_llms,
            conda_env=conda_env,
            root_execution_id=root_id,
            entry_point=self.graph_config["entry_point"],
            functions_by_node=functions,
            tool_schemas=schemas,
            agents_by_node=agents,
            agent_graphs=agent_graphs,
            sequence_by_node=self._compute_sequence(),
        )

    # -- public methods -----------------------------------------------------

    def run(self, initial_input: Any, conda_env: Optional[str] = None,
            execution_id: Optional[int] = None, parallel: bool = True) -> dict:
        """Execute the flow from its entry point.

        Returns {flow_id, execution_id, status, final_output, error?} where
        final_output is a bounded preview (raw outputs stay in ctx.state).
        `execution_id` reuses a pre-created row (Celery/container path).
        On failure the runner retains everything resume() needs in memory.
        """
        root_id = None
        ctx = None
        self._initial_input = initial_input
        self.conda_env = conda_env
        try:
            if execution_id is not None:
                existing = get_execution_by_id(self.session, execution_id)
                if not existing:
                    raise ValueError(f"Execution {execution_id} not found")
                update_execution(self.session, execution_id, status="running", started_at=datetime.now())
                root_id = execution_id
            else:
                root = create_execution(
                    self.session,
                    user_id=self.user_id,
                    flow_id=self.flow_id,
                    execution_type="flow",
                    name=self.flow.name,
                    input_data=_json_safe(initial_input),
                    status="running",
                    started_at=datetime.now(),
                )
                root_id = root.id

            ctx = self._build_context(root_id, conda_env)
            ctx.fingerprints = _compute_fingerprints(self.session, self.graph_config)
            self.ctx = ctx
            self.root_execution_id = root_id
            code = generate_orchestrator_code(self.graph_config, parallel=parallel)
            namespace: Dict[str, Any] = {}
            exec(code, namespace)
            final_output = _run_async(namespace["orchestrate"](ctx, initial_input))

            update_execution(
                self.session, root_id,
                status="completed",
                completed_at=datetime.now(),
            )
            return {
                "flow_id": self.flow_id,
                "execution_id": root_id,
                "status": "completed",
                "final_output": _preview(final_output),
            }
        except Exception as e:
            logger.error("Flow execution failed: %s", e)
            if root_id is not None:
                if ctx is not None:
                    _sweep_running_children(self.session, root_id)
                update_execution(
                    self.session, root_id,
                    status="failed",
                    error_message=str(e),
                    completed_at=datetime.now(),
                )
            return {
                "flow_id": self.flow_id,
                "execution_id": root_id,
                "status": "failed",
                "final_output": None,
                "error": str(e),
            }

    def resume(self) -> dict:
        """Resume a failed run in place, honoring edits made since the failure.

        Nothing is read from persisted or client-supplied I/O: cached raw
        outputs live in the retained ctx.state. The flow's current graph and
        tools/agents are re-fetched on a fresh session; any node whose config,
        incoming edges, or underlying tool/agent changed — plus everything
        downstream of it — is invalidated and re-runs. Unchanged completed
        nodes short-circuit in _run_stage. The original root execution row is
        reused; re-run nodes update their original child rows in place.
        """
        if self.ctx is None or self.root_execution_id is None:
            raise ValueError("No live run to resume")
        ctx = self.ctx
        root_id = self.root_execution_id
        # Fresh session: the construction-time session is request-scoped and
        # closed by resume time, and a fresh one guarantees edited tools and
        # agents are re-fetched instead of served from a stale identity map.
        session = DatabaseManager().get_session()
        self.session = session
        try:
            flow = session.query(Flow).filter(Flow.id == self.flow_id).first()
            if not flow:
                raise ValueError(f"Flow {self.flow_id} not found")
            self.flow = flow
            self.flow_name = flow.name
            self.graph_config = flow.graph_config

            functions, schemas = self._prepare_tools()
            agents, agent_graphs = self._prepare_agents()
            new_fps = _compute_fingerprints(session, self.graph_config)
            new_nodes = set(self.graph_config.get("nodes", {}))

            changed = {n for n in new_nodes if ctx.fingerprints.get(n) != new_fps[n]}
            invalid = _with_descendants(self.graph_config, changed)
            for node_id in invalid | (set(ctx.state) - new_nodes):
                ctx.state.pop(node_id, None)

            failed_nodes = {
                row.node_id
                for row in session.query(Execution)
                .filter(Execution.parent_id == root_id, Execution.status == "failed")
                .all()
                if row.node_id in new_nodes
            }
            resumed_from = sorted(changed | failed_nodes)

            ctx.graph_config = self.graph_config
            ctx.entry_point = self.graph_config["entry_point"]
            ctx.functions_by_node = functions
            ctx.tool_schemas = schemas
            ctx.agents_by_node = agents
            ctx.agent_graphs = agent_graphs
            ctx.sequence_by_node = self._compute_sequence()
            ctx.fingerprints = new_fps

            existing = get_execution_by_id(session, root_id)
            meta = dict(existing.execution_metadata or {}) if existing else {}
            meta["resumed_from_nodes"] = resumed_from
            meta["resume_count"] = meta.get("resume_count", 0) + 1
            update_execution(
                session, root_id,
                status="running",
                error_message=None,
                completed_at=None,
                execution_metadata=meta,
            )

            code = generate_orchestrator_code(self.graph_config)
            namespace: Dict[str, Any] = {}
            exec(code, namespace)
            final_output = _run_async(namespace["orchestrate"](ctx, self._initial_input))

            update_execution(
                session, root_id,
                status="completed",
                completed_at=datetime.now(),
            )
            return {
                "flow_id": self.flow_id,
                "execution_id": root_id,
                "status": "completed",
                "final_output": _preview(final_output),
            }
        except Exception as e:
            logger.error("Flow resume failed: %s", e)
            _sweep_running_children(session, root_id)
            update_execution(
                session, root_id,
                status="failed",
                error_message=str(e),
                completed_at=datetime.now(),
            )
            return {
                "flow_id": self.flow_id,
                "execution_id": root_id,
                "status": "failed",
                "final_output": None,
                "error": str(e),
            }
        finally:
            session.close()
