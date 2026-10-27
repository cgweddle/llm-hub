"""
Pure agent-graph traversal engine.

Named-inputs readiness engine: each node accumulates a dict of named inputs (one value per
`{name}` placeholder in its user_prompt) rather than a single input string.
Edges deliver their value to a specific named input (`target_input`, legacy
edges with none target `"input"`), and a node only runs once every one of its
forward (non-loop) incoming edges has fired. When nothing else is ready, the
earliest-in-topological-order waiting node is unstuck and runs with its still
missing inputs left absent (callers read them with `.get(name, "")`). Loop
edges never gate readiness; after a node runs, the "fired" mark is cleared for
any forward edge whose sender shares the node's strongly-connected component
(loop refinement), forcing a fresh value before that edge can fire again.

Every node that is ready at the same time runs concurrently (`parallel=True`,
the default). Results are applied one at a time, in forward-topological
order (then node id) when several finish together, so shared state is never
touched mid-await by another task. A node re-triggered while its current run
is still in flight is not started twice — it runs exactly once more after
that run completes, with whatever inputs have arrived by then. `parallel`
can be set to False to run one node at a time instead (the live runner
always passes True; a standalone export can opt into sequential output).

A node's `{message_history}` is the latest conversation of every node that
delivered one of its current inputs (forward or loop edge), concatenated in
forward-topological order then edge order. A node with no senders (the entry
node, or one whose inputs never arrived) gets an empty list.

This module is deliberately free of side effects — no database, no LangFuse,
no PydanticAI imports. Node execution and structural recording are supplied as
callables, so the same traversal drives both the live executor (which wraps
each run with tracing + DB writes) and a standalone flow export (which vendors
this file verbatim and supplies plain ``agent.run`` callables).

Injected callables:

``run_node(node_id, node_config, inputs, predecessor_messages)``
    async → ``(output, messages, chosen_path, trace_id)``. Runs one node.
    ``inputs`` is that node's accumulated named-inputs dict, snapshotted when
    the node started. ``predecessor_messages`` is the node's message history
    (see above), also snapshotted at start. ``messages`` is that run's full
    history (or None to leave it unchanged), ``chosen_path`` is the selected
    output-path name (or None), ``trace_id`` is an opaque observability
    handle passed on to ``record_node`` (or None).

``record_node(node_id, node_config, inputs, output, chosen_path, trace_id)``
    Optional. Records the completed node (e.g. a structural execution row).
    Called after ``previous_output`` has been applied, so ``output`` is
    final. Never called for a cancelled node (fail-fast cancels siblings
    before they complete).
"""

import asyncio
import logging
from collections import deque
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RunNode = Callable[
    [str, Dict[str, Any], Dict[str, str], List],
    Awaitable[Tuple[str, Optional[List], Optional[str], Optional[str]]],
]
RecordNode = Callable[[str, Dict[str, Any], Dict[str, str], str, Optional[str], Optional[str]], None]


def _preview(value: Any, limit: int = 200) -> Any:
    """Truncate a value's string form for the informational execution trace."""
    s = str(value)
    return s[:limit] + "..." if len(s) > limit else value


def _target_input(edge: Dict[str, Any]) -> str:
    return edge.get("target_input") or "input"


def _build_graph_structures(nodes_config: Dict[str, Any], edges: List[Dict[str, Any]]):
    """outgoing: node -> [(edge_index, edge)]; forward_in: node -> [(edge_index, edge)]
    over non-loop incoming edges only. Edges are tracked by list index, never
    collapsed by a (from, to) pair."""
    outgoing: Dict[str, list] = {n: [] for n in nodes_config}
    forward_in: Dict[str, list] = {n: [] for n in nodes_config}
    for idx, edge in enumerate(edges):
        from_node, to_node = edge.get("from_node"), edge.get("to_node")
        if from_node not in outgoing or to_node not in outgoing:
            continue
        outgoing[from_node].append((idx, edge))
        if not edge.get("is_loop"):
            forward_in[to_node].append((idx, edge))
    return outgoing, forward_in


def _topo_rank(nodes_config: Dict[str, Any], forward_in: Dict[str, list]) -> Dict[str, int]:
    """Topological order over forward (non-loop) edges only — these form a
    DAG, since cycles are always marked is_loop."""
    forward_out: Dict[str, list] = {n: [] for n in nodes_config}
    indegree: Dict[str, int] = {n: len(in_edges) for n, in_edges in forward_in.items()}
    for n, in_edges in forward_in.items():
        for _, edge in in_edges:
            forward_out[edge["from_node"]].append(n)

    order = []
    remaining = dict(indegree)
    queue = deque(n for n in nodes_config if indegree.get(n, 0) == 0)
    seen = set()
    while queue:
        n = queue.popleft()
        if n in seen:
            continue
        seen.add(n)
        order.append(n)
        for succ in forward_out.get(n, []):
            remaining[succ] -= 1
            if remaining[succ] == 0:
                queue.append(succ)
    for n in nodes_config:
        if n not in seen:
            order.append(n)
    return {n: i for i, n in enumerate(order)}


def strongly_connected(nodes_config: Dict[str, Any], edges: List[Dict[str, Any]]):
    """component_id: node -> component id, component_size: id -> size.
    Computed over ALL edges (forward + loop) — a node's loop membership
    reflects the whole graph's topology, not just the forward DAG.

    Not private: also used by validate.agent_graph_inputs for the
    shared-input exclusivity check, which needs the same notion of "inside
    a loop" (component size > 1)."""
    adjacency: Dict[str, list] = {n: [] for n in nodes_config}
    for edge in edges:
        from_node, to_node = edge.get("from_node"), edge.get("to_node")
        if from_node in adjacency and to_node in adjacency:
            adjacency[from_node].append(to_node)

    def reachable(start: str):
        seen = {start}
        stack = [start]
        while stack:
            n = stack.pop()
            for nxt in adjacency.get(n, []):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    reach = {n: reachable(n) for n in nodes_config}
    component_id: Dict[str, int] = {}
    next_id = 0
    for n in nodes_config:
        if n in component_id:
            continue
        component = {m for m in reach[n] if n in reach[m]}
        for m in component:
            component_id[m] = next_id
        next_id += 1

    component_size: Dict[int, int] = {}
    for cid in component_id.values():
        component_size[cid] = component_size.get(cid, 0) + 1
    return component_id, component_size


async def run_graph(
    graph_config: Dict[str, Any],
    inputs: Dict[str, Any],
    run_node: RunNode,
    record_node: Optional[RecordNode] = None,
    max_loop_iterations: int = None,
    parallel: bool = True,
) -> Dict[str, Any]:
    """Execute an agent graph via the lazy-readiness traversal (see module
    docstring).

    Side effects (tracing, DB writes) are the responsibility of the injected
    ``run_node`` / ``record_node`` callables; this function only walks the
    graph. Every node ready at once runs concurrently unless
    ``parallel=False``.
    """
    nodes_config = graph_config.get("nodes", {})
    entry_point = graph_config.get("entry_point")
    exit_points = graph_config.get("exit_points", [])
    edges = graph_config.get("edges", [])

    if max_loop_iterations is None:
        max_loop_iterations = graph_config.get("max_loop_iterations", 5)

    if not entry_point or not nodes_config:
        raise ValueError("Invalid graph_config: missing entry_point or nodes")

    outgoing, forward_in = _build_graph_structures(nodes_config, edges)
    topo_rank = _topo_rank(nodes_config, forward_in)
    component_id, component_size = strongly_connected(nodes_config, edges)

    logger.info(
        f"Graph execution: {len(nodes_config)} nodes, {len(edges)} edges, "
        f"entry: {entry_point}, exits: {exit_points}"
    )

    node_inputs: Dict[str, Dict[str, Any]] = {n: {} for n in nodes_config}
    node_inputs[entry_point] = dict(inputs)

    fired: set = set()
    waiting: set = set()
    outputs: Dict[str, Any] = {}
    loop_counts: Dict[int, int] = {}
    execution_trace = []
    latest_messages: Dict[str, List] = {}
    delivered_by: Dict[str, Dict[str, int]] = {}

    running: Dict[str, "asyncio.Task"] = {}
    rerun: set = set()
    queue = deque([entry_point])

    def history_for(node_id: str) -> List:
        senders = []
        seen_nodes = set()
        for idx in delivered_by.get(node_id, {}).values():
            from_node = edges[idx].get("from_node")
            if from_node in seen_nodes:
                continue
            seen_nodes.add(from_node)
            senders.append((topo_rank.get(from_node, 0), idx, from_node))
        senders.sort()
        history: List = []
        for _, _, from_node in senders:
            history.extend(latest_messages.get(from_node, []))
        return history

    async def execute(node_id: str, snapshot_inputs: Dict[str, Any], snapshot_history: List):
        node_config = nodes_config[node_id]
        try:
            output, node_messages, chosen_path, trace_id = await run_node(
                node_id, node_config, snapshot_inputs, snapshot_history
            )
        except Exception as e:
            logger.error(f"Sub-agent {node_id} failed: {e}")
            raise RuntimeError(
                f"Agent node '{node_config.get('name', node_id)}' failed: {e}"
            ) from e
        return snapshot_inputs, output, node_messages, chosen_path, trace_id

    def apply_result(node_id: str, result: Tuple) -> None:
        snapshot_inputs, output, node_messages, chosen_path, trace_id = result
        node_config = nodes_config[node_id]

        if chosen_path:
            path_config = node_config.get("output_paths", {}).get(chosen_path, {})
            if isinstance(path_config, dict) and path_config.get("return_behavior") == "previous_output":
                return_input = path_config.get("return_input") or "input"
                output = snapshot_inputs.get(return_input, "")

        outputs[node_id] = output
        if node_messages is not None:
            latest_messages[node_id] = node_messages

        execution_trace.append({
            "node": node_id,
            "name": node_config.get("name", node_id),
            "agent_type": node_config.get("agent_type"),
            "input": _preview(snapshot_inputs),
            "output": _preview(output),
            "status": "completed",
        })

        if record_node is not None:
            record_node(node_id, node_config, snapshot_inputs, output, chosen_path, trace_id)

        logger.debug(f"Sub-agent {node_id} completed")

        cid = component_id.get(node_id)
        for idx, edge in forward_in[node_id]:
            sender_cid = component_id.get(edge.get("from_node"))
            if sender_cid == cid and component_size.get(cid, 1) > 1:
                fired.discard(idx)

        is_exit = node_id in exit_points
        for idx, edge in outgoing[node_id]:
            to_node = edge.get("to_node")
            is_loop_edge = bool(edge.get("is_loop"))

            if is_exit and not is_loop_edge:
                continue

            edge_path = edge.get("output_path")
            if chosen_path is not None and edge_path is not None and edge_path != chosen_path:
                continue

            if is_loop_edge:
                loop_counts[idx] = loop_counts.get(idx, 0) + 1
                if loop_counts[idx] > max_loop_iterations:
                    logger.warning(
                        f"Loop limit reached for edge {node_id} -> {to_node}, skipping"
                    )
                    continue

            node_inputs[to_node][_target_input(edge)] = output
            delivered_by.setdefault(to_node, {})[_target_input(edge)] = idx

            if is_loop_edge:
                queue.append(to_node)
            else:
                fired.add(idx)
                if all(i in fired for i, _ in forward_in[to_node]):
                    waiting.discard(to_node)
                    queue.append(to_node)
                else:
                    waiting.add(to_node)

    def start_ready() -> None:
        while queue and (parallel or not running):
            n = queue.popleft()
            if n in running:
                rerun.add(n)
                continue
            for idx, edge in forward_in[n]:
                if idx not in fired:
                    node_inputs[n].setdefault(_target_input(edge), "")
            snapshot_inputs = dict(node_inputs[n])
            snapshot_history = history_for(n)
            running[n] = asyncio.create_task(execute(n, snapshot_inputs, snapshot_history))

    while True:
        start_ready()
        if running:
            done, _ = await asyncio.wait(running.values(), return_when=asyncio.FIRST_COMPLETED)
            finished = sorted(
                (n for n, task in running.items() if task in done),
                key=lambda n: (topo_rank.get(n, 0), n),
            )
            for n in finished:
                task = running.pop(n)
                error = task.exception()
                if error is not None:
                    remaining = list(running.values())
                    for other in remaining:
                        other.cancel()
                    if remaining:
                        await asyncio.gather(*remaining, return_exceptions=True)
                    raise error
                apply_result(n, task.result())
                if n in rerun:
                    rerun.discard(n)
                    queue.append(n)
            continue
        if not waiting:
            break
        n = min(waiting, key=lambda node_id: topo_rank.get(node_id, 0))
        waiting.discard(n)
        queue.append(n)

    final_outputs = {
        node_id: outputs.get(node_id, "")
        for node_id in exit_points
        if node_id in outputs
    }
    main_result = final_outputs.get(exit_points[0], "") if exit_points else ""

    return {
        "status": "completed",
        "result": main_result,
        "all_outputs": final_outputs,
        "execution_trace": execution_trace,
        "sub_agent_count": len(nodes_config),
    }
