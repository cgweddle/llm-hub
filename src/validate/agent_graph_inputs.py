"""
Agent-graph named-input validation (claude_plans/named-agent-inputs.md, plan
step 2). Shared by the agent create/update endpoints, the flow save
endpoint, and ``FlowRunner._prepare_agents``.

    validate_agent_graph(graph_config: dict) -> list[str]
        Validates ONE agent's own graph_config (the structure used both by
        a standalone agent record and by a flow's agent_graphs[node_id]).
        Returns a list of human-readable problem strings; [] means valid.
        Checks:
          - every sub-node has at least one input, i.e.
            extract_prompt_inputs(node["user_prompt"]) is non-empty
            (decision 4)
          - every edge's target_input (missing -> "input", decision 5) is
            one of the target sub-node's own inputs
          - no two edges are fully identical: same from_node, output_path,
            to_node, and target_input (decision 10)
          - edges that share a (to_node, target_input) pair must be
            pairwise "exclusive" per the exclusivity algorithm below
            (decision 11) — otherwise rejected
          - every output path with return_behavior == "previous_output"
            has a return_input that is one of that sub-node's OWN inputs;
            a missing return_input legacy-reads as "input" (decision 9),
            so it's also a problem when "input" isn't one of that node's
            inputs

    validate_flow_agent_edges(flow_graph_config: dict,
                               agent_graphs_by_node_id: dict[str, dict]) -> list[str]
        Validates the FLOW-canvas edges feeding every agent node named in
        ``agent_graphs_by_node_id`` (node_id -> that agent's own
        graph_config, used to compute agent_external_inputs). Flow edges
        reuse the tool-edge ``mapping`` shape: ``{out_field_or_"": input_name}``;
        missing mapping, or ``{field: ""}``, targets "input" (same contract
        as ``agent_target_input()`` below, used by flow_runner.py and the
        exporter, plan step 5).
        Only non-loop edges whose to_node is a key of
        ``agent_graphs_by_node_id`` are examined. Returns problem strings for:
          - a flow edge whose resolved input name is not in
            agent_external_inputs(agent_graphs_by_node_id[to_node])
          - two (or more) flow edges resolving to the same input of the
            same agent node (flow edges have no output paths, so decision 11's
            exclusivity proof never applies on the canvas — always rejected)

Exclusivity algorithm ("Shared-input exclusivity check"):

    node_req[entry] = {}
    node_req[n]     = intersection(edge_req[e] for e in forward_in[n])
    edge_req[e]     = node_req[e.from_node] | ({e.from_node: e.output_path} if e.output_path else {})

    exclusive(e1, e2) = exists node d with
                        d in edge_req[e1] and d in edge_req[e2]
                        and edge_req[e1][d] != edge_req[e2][d]
                        and d is not inside a loop (its SCC has size 1)

Every pair of edges sharing a (to_node, target_input) must be exclusive.
Known accepted limitation: a deciding node inside a loop can never prove
exclusivity, even when its paths are in practice exclusive.
"""

import os
import sys
from typing import Any, Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from executors.agent_graph_engine import strongly_connected
from utils.prompt_template import extract_prompt_inputs, agent_external_inputs


def agent_target_input(edge: Dict[str, Any]) -> str:
    """Flow-edge mapping shape -> the agent input it targets. Reuses the
    tool-edge `mapping` shape ({out_field_or_"": input_name}); missing
    mapping, or a mapping whose value is "", targets "input" (decision 5)."""
    mapping = edge.get("mapping")
    if not mapping:
        return "input"
    value = next(iter(mapping.values()))
    return value or "input"


def _node_inputs_map(nodes_config: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        node_id: extract_prompt_inputs(node_cfg.get("user_prompt", ""))
        for node_id, node_cfg in nodes_config.items()
    }


def _build_forward_in(nodes_config: Dict[str, Any], edges: List[Dict[str, Any]]):
    forward_in: Dict[str, list] = {n: [] for n in nodes_config}
    for edge in edges:
        to_node = edge.get("to_node")
        if to_node in forward_in and not edge.get("is_loop"):
            forward_in[to_node].append(edge)
    return forward_in


def _compute_edge_req(nodes_config: Dict[str, Any], edges: List[Dict[str, Any]]):
    forward_in = _build_forward_in(nodes_config, edges)
    node_req_cache: Dict[str, dict] = {}
    edge_req_cache: Dict[int, dict] = {}

    def node_req(node_id):
        if node_id in node_req_cache:
            return node_req_cache[node_id]
        incoming = forward_in.get(node_id, [])
        if not incoming:
            result: Dict[str, str] = {}
        else:
            reqs = [edge_req(e) for e in incoming]
            result = dict(reqs[0])
            for r in reqs[1:]:
                result = {k: v for k, v in result.items() if k in r and r[k] == v}
        node_req_cache[node_id] = result
        return result

    def edge_req(edge):
        key = id(edge)
        if key in edge_req_cache:
            return edge_req_cache[key]
        base = dict(node_req(edge.get("from_node")))
        if edge.get("output_path"):
            base[edge.get("from_node")] = edge.get("output_path")
        edge_req_cache[key] = base
        return base

    return edge_req


def _compute_loop_nodes(nodes_config: Dict[str, Any], edges: List[Dict[str, Any]]) -> set:
    """Node ids whose strongly-connected component (over ALL edges) has
    size > 1 — "inside a loop" for the exclusivity check. Reuses the agent
    graph engine's own SCC computation (same notion of "loop" as decision 13's
    loop refinement)."""
    component_id, component_size = strongly_connected(nodes_config, edges)
    return {n for n, cid in component_id.items() if component_size[cid] > 1}


def _exclusive(e1: Dict[str, Any], e2: Dict[str, Any], edge_req, loop_nodes: set) -> bool:
    r1, r2 = edge_req(e1), edge_req(e2)
    for d, v in r1.items():
        if d in r2 and r2[d] != v and d not in loop_nodes:
            return True
    return False


def validate_agent_graph(graph_config: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    nodes_config = graph_config.get("nodes", {}) or {}
    edges = graph_config.get("edges", []) or []

    node_inputs = _node_inputs_map(nodes_config)

    for node_id, inputs in node_inputs.items():
        if not inputs:
            problems.append(
                f"Sub-agent '{node_id}' has no inputs — its user_prompt has no "
                "{name} placeholders."
            )

    for edge in edges:
        to_node = edge.get("to_node")
        if to_node not in nodes_config:
            continue
        target = edge.get("target_input") or "input"
        if target not in node_inputs.get(to_node, []):
            problems.append(
                f"Edge {edge.get('from_node')} -> {to_node} targets unknown input "
                f"'{target}' on sub-agent '{to_node}'."
            )

    edge_keys: Dict[tuple, int] = {}
    for edge in edges:
        key = (
            edge.get("from_node"),
            edge.get("output_path"),
            edge.get("to_node"),
            edge.get("target_input") or "input",
        )
        edge_keys[key] = edge_keys.get(key, 0) + 1
    for (from_node, output_path, to_node, target_input), count in edge_keys.items():
        if count > 1:
            problems.append(
                f"Edges {from_node} -> {to_node} (output_path={output_path!r}, "
                f"target_input={target_input!r}) are identical duplicates."
            )

    edge_req = _compute_edge_req(nodes_config, edges)
    loop_nodes = _compute_loop_nodes(nodes_config, edges)

    groups: Dict[tuple, list] = {}
    for edge in edges:
        to_node = edge.get("to_node")
        target = edge.get("target_input") or "input"
        groups.setdefault((to_node, target), []).append(edge)

    for (to_node, target), group_edges in groups.items():
        if len(group_edges) < 2:
            continue
        for i in range(len(group_edges)):
            for j in range(i + 1, len(group_edges)):
                e1, e2 = group_edges[i], group_edges[j]
                if not _exclusive(e1, e2, edge_req, loop_nodes):
                    problems.append(
                        f"'{to_node}.{target}' is fed by {e1.get('from_node')}->{to_node} "
                        f"and {e2.get('from_node')}->{to_node}, which can both fire. "
                        "Use separate inputs."
                    )

    for node_id, node_cfg in nodes_config.items():
        for path_name, path_cfg in (node_cfg.get("output_paths") or {}).items():
            if not isinstance(path_cfg, dict):
                continue
            if path_cfg.get("return_behavior") != "previous_output":
                continue
            return_input = path_cfg.get("return_input") or "input"
            if return_input not in node_inputs.get(node_id, []):
                problems.append(
                    f"Output path '{path_name}' on sub-agent '{node_id}' has "
                    f"return_input '{return_input}', which is not one of its own inputs."
                )

    return problems


def validate_flow_agent_edges(
    flow_graph_config: Dict[str, Any],
    agent_graphs_by_node_id: Dict[str, dict],
) -> List[str]:
    problems: List[str] = []
    edges = flow_graph_config.get("edges", []) or []

    for agent_node_id, agent_graph in agent_graphs_by_node_id.items():
        external = set(agent_external_inputs(agent_graph))
        incoming = [
            edge for edge in edges
            if edge.get("to_node") == agent_node_id and not edge.get("is_loop")
        ]

        seen: Dict[str, list] = {}
        for edge in incoming:
            target = agent_target_input(edge)
            if target not in external:
                problems.append(
                    f"Flow edge from '{edge.get('from_node')}' targets unknown input "
                    f"'{target}' on agent node '{agent_node_id}'."
                )
                continue
            seen.setdefault(target, []).append(edge)

        for target, group_edges in seen.items():
            if len(group_edges) > 1:
                sources = ", ".join(e.get("from_node", "?") for e in group_edges)
                problems.append(
                    f"Agent node '{agent_node_id}' input '{target}' is fed by multiple "
                    f"flow edges ({sources}); flow edges have no output paths so this "
                    "can't be made exclusive."
                )

    return problems
