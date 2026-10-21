"""
Pure agent-graph traversal engine.

BFS over an agent ``graph_config`` with cycle support: output-path routing,
per-edge loop-iteration limits, ``previous_output`` return behavior, and
message-history threading between nodes.

This module is deliberately free of side effects — no database, no LangFuse,
no PydanticAI imports. Node execution and structural recording are supplied as
callables, so the same traversal drives both the live executor (which wraps
each run with tracing + DB writes) and a standalone flow export (which vendors
this file verbatim and supplies plain ``agent.run`` callables).

Injected callables:

``run_node(node_id, node_config, node_input, predecessor_messages)``
    async → ``(output, messages, chosen_path, trace_id)``. Runs one node.
    ``messages`` is that run's full history (or None to leave it unchanged),
    ``chosen_path`` is the selected output-path name (or None), ``trace_id`` is
    an opaque observability handle passed on to ``record_node`` (or None).

``record_node(node_id, node_config, node_input, output, chosen_path, trace_id)``
    Optional. Records the completed node (e.g. a structural execution row).
    Called after ``previous_output`` has been applied, so ``output`` is final.
"""

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RunNode = Callable[
    [str, Dict[str, Any], str, Optional[List]],
    Awaitable[Tuple[str, Optional[List], Optional[str], Optional[str]]],
]
RecordNode = Callable[[str, Dict[str, Any], str, str, Optional[str], Optional[str]], None]


def build_adjacency(graph_config: Dict[str, Any]):
    """Build (adjacency, loop_edges, edge_output_paths) from a graph_config.

    - adjacency:          node_id → [successor node_ids]
    - loop_edges:         set of (from, to) pairs marked is_loop
    - edge_output_paths:  (from, to) → output_path name for path-conditioned edges
    """
    nodes_config = graph_config.get("nodes", {})
    adjacency: Dict[str, list] = {node_id: [] for node_id in nodes_config}
    loop_edges: set = set()
    edge_output_paths: Dict[tuple, str] = {}
    for edge in graph_config.get("edges", []):
        from_node = edge.get("from_node")
        to_node = edge.get("to_node")
        if from_node and to_node:
            adjacency[from_node].append(to_node)
            if edge.get("is_loop", False):
                loop_edges.add((from_node, to_node))
            if edge.get("output_path"):
                edge_output_paths[(from_node, to_node)] = edge["output_path"]
    return adjacency, loop_edges, edge_output_paths


def _preview(value: Any, limit: int = 200) -> Any:
    """Truncate a value's string form for the informational execution trace."""
    return str(value)[:limit] + "..." if len(str(value)) > limit else value


async def run_graph(
    graph_config: Dict[str, Any],
    input_data: str,
    run_node: RunNode,
    record_node: Optional[RecordNode] = None,
    max_loop_iterations: int = None,
) -> Dict[str, Any]:
    """Execute an agent graph via BFS traversal with cycle support.

    Side effects (tracing, DB writes) are the responsibility of the injected
    ``run_node`` / ``record_node`` callables; this function only walks the graph.
    """
    nodes_config = graph_config.get("nodes", {})
    entry_point = graph_config.get("entry_point")
    exit_points = graph_config.get("exit_points", [])

    if max_loop_iterations is None:
        max_loop_iterations = graph_config.get("max_loop_iterations", 5)

    if not entry_point or not nodes_config:
        raise ValueError("Invalid graph_config: missing entry_point or nodes")

    adjacency, loop_edges, edge_output_paths = build_adjacency(graph_config)

    logger.info(
        f"Graph execution: {len(nodes_config)} nodes, "
        f"{len(graph_config.get('edges', []))} edges, "
        f"entry: {entry_point}, exits: {exit_points}"
    )

    execution_trace = []
    node_outputs: Dict[str, Any] = {}
    messages: List = []  # Accumulating message history across all nodes
    loop_counts: Dict[tuple, int] = {}

    current_nodes = [entry_point]

    while current_nodes:
        next_nodes = []

        for node_id in current_nodes:
            if node_id not in nodes_config:
                logger.warning(f"Node {node_id} not found in config, skipping")
                continue

            node_config = nodes_config[node_id]
            node_input = node_outputs.get(node_id + "_input", input_data)
            predecessor_msgs = messages if messages else None

            try:
                output, node_messages, chosen_path, trace_id = await run_node(
                    node_id, node_config, node_input, predecessor_msgs
                )
            except Exception as e:
                logger.error(f"Sub-agent {node_id} failed: {e}")
                raise RuntimeError(
                    f"Agent node '{node_config.get('name', node_id)}' failed: {e}"
                ) from e

            # Apply return behavior for the chosen path
            if chosen_path:
                path_config = node_config.get("output_paths", {}).get(chosen_path, {})
                if isinstance(path_config, dict) and path_config.get("return_behavior") == "previous_output":
                    output = node_input

            node_outputs[node_id] = output
            if node_messages is not None:
                messages = node_messages

            execution_trace.append({
                "node": node_id,
                "name": node_config.get("name", node_id),
                "agent_type": node_config.get("agent_type"),
                "input": _preview(node_input),
                "output": _preview(output),
                "status": "completed",
            })

            if record_node is not None:
                record_node(node_id, node_config, node_input, output, chosen_path, trace_id)

            logger.debug(f"Sub-agent {node_id} completed")

            # Traverse successors. Exit points skip forward edges but still
            # follow loop edges back.
            is_exit = node_id in exit_points
            for successor in adjacency.get(node_id, []):
                edge_key = (node_id, successor)
                is_loop_edge = edge_key in loop_edges

                if is_exit and not is_loop_edge:
                    continue

                # Filter by output_path when the node chose a specific path
                edge_path = edge_output_paths.get(edge_key)
                if chosen_path is not None and edge_path is not None:
                    if edge_path != chosen_path:
                        continue

                # Bound loop edges by the iteration limit
                if is_loop_edge:
                    loop_counts[edge_key] = loop_counts.get(edge_key, 0) + 1
                    if loop_counts[edge_key] > max_loop_iterations:
                        logger.warning(
                            f"Loop limit reached for edge {node_id} -> {successor}, skipping"
                        )
                        continue

                node_outputs[successor + "_input"] = output
                if successor not in next_nodes:
                    next_nodes.append(successor)

        current_nodes = next_nodes

    final_outputs = {
        node_id: node_outputs.get(node_id, "")
        for node_id in exit_points
        if node_id in node_outputs
    }
    main_result = final_outputs.get(exit_points[0], "") if exit_points else ""

    return {
        "status": "completed",
        "result": main_result,
        "all_outputs": final_outputs,
        "execution_trace": execution_trace,
        "sub_agent_count": len(nodes_config),
    }
