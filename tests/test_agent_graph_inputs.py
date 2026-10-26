"""
Tests for the agent-graph input validation module
(claude_plans/named-agent-inputs.md, "Shared-input exclusivity check" + plan
step 2). This module does not exist yet — these tests pin the API the
implementer must create at ``src/validate/agent_graph_inputs.py``:

    validate_agent_graph(graph_config: dict) -> list[str]
        Validates ONE agent's own graph_config (the structure used both by
        a standalone agent record and by a flow's agent_graphs[node_id]).
        Returns a list of human-readable problem strings; [] means valid.
        Checks (see plan step 2):
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
        as ``agent_target_input()`` in src/runners/flow_runner.py, plan step 5).
        Only non-loop edges whose to_node is a key of
        ``agent_graphs_by_node_id`` are examined. Returns problem strings for:
          - a flow edge whose resolved input name is not in
            agent_external_inputs(agent_graphs_by_node_id[to_node])
          - two (or more) flow edges resolving to the same input of the
            same agent node (flow edges have no output paths, so decision 11's
            exclusivity proof never applies on the canvas — always rejected)

Exclusivity algorithm under test (plan, "Shared-input exclusivity check"):

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from validate.agent_graph_inputs import (
    validate_agent_graph,
    validate_flow_agent_edges,
)


# ─── validate_agent_graph ─────────────────────────────────────────────────


class TestNoInputSubAgent:
    def test_node_with_no_placeholders_is_rejected(self):
        graph_config = {
            "nodes": {
                "b": {"name": "B", "user_prompt": "Just text, no placeholders at all."},
            },
            "edges": [],
            "entry_point": "b",
            "exit_points": ["b"],
        }
        problems = validate_agent_graph(graph_config)
        assert len(problems) >= 1
        assert any("input" in p.lower() for p in problems)

    def test_empty_prompt_is_not_a_violation(self):
        """Empty user_prompt implicitly means {input} (decision 2)."""
        graph_config = {
            "nodes": {"b": {"name": "B", "user_prompt": ""}},
            "edges": [],
            "entry_point": "b",
            "exit_points": ["b"],
        }
        assert validate_agent_graph(graph_config) == []


class TestUnknownTargetInput:
    def test_edge_targets_input_not_on_destination_node(self):
        graph_config = {
            "nodes": {
                "a": {"name": "A", "user_prompt": "{input}"},
                "b": {"name": "B", "user_prompt": "{foo}"},
            },
            "edges": [{"from_node": "a", "to_node": "b", "target_input": "bar"}],
            "entry_point": "a",
            "exit_points": ["b"],
        }
        problems = validate_agent_graph(graph_config)
        assert len(problems) >= 1
        assert any("bar" in p for p in problems)

    def test_edge_targeting_a_real_input_is_fine(self):
        graph_config = {
            "nodes": {
                "a": {"name": "A", "user_prompt": "{input}"},
                "b": {"name": "B", "user_prompt": "{foo}"},
            },
            "edges": [{"from_node": "a", "to_node": "b", "target_input": "foo"}],
            "entry_point": "a",
            "exit_points": ["b"],
        }
        assert validate_agent_graph(graph_config) == []


class TestIdenticalEdges:
    def test_fully_identical_edges_rejected(self):
        graph_config = {
            "nodes": {
                "a": {"name": "A", "user_prompt": "{input}"},
                "b": {"name": "B", "user_prompt": "{input}"},
            },
            "edges": [
                {"from_node": "a", "to_node": "b"},
                {"from_node": "a", "to_node": "b"},
            ],
            "entry_point": "a",
            "exit_points": ["b"],
        }
        problems = validate_agent_graph(graph_config)
        assert len(problems) >= 1
        assert any("identical" in p.lower() or "duplicate" in p.lower() for p in problems)

    def test_edges_differing_by_output_path_not_identical(self):
        """Decision 10: several edges from one source into DIFFERENT inputs
        of the same target are fine; these two differ by output_path and
        target a different input each, so no "identical edges" problem."""
        graph_config = {
            "nodes": {
                "router": {"name": "Router", "user_prompt": "{input}",
                           "output_paths": {"a": {"description": "a"},
                                             "b": {"description": "b"}}},
                "final": {"name": "Final", "user_prompt": "{x} {y}"},
            },
            "edges": [
                {"from_node": "router", "to_node": "final", "output_path": "a", "target_input": "x"},
                {"from_node": "router", "to_node": "final", "output_path": "b", "target_input": "y"},
            ],
            "entry_point": "router",
            "exit_points": ["final"],
        }
        assert validate_agent_graph(graph_config) == []


class TestSharedInputExclusivity:
    def test_direct_siblings_with_different_output_paths_allowed(self):
        """Router's two branches feed the SAME input, but router's chosen
        path differs between them -> provably exclusive."""
        graph_config = {
            "nodes": {
                "router": {"name": "Router", "user_prompt": "{input}",
                           "output_paths": {"approve": {"description": "ok"},
                                             "reject": {"description": "no"}}},
                "final": {"name": "Final", "user_prompt": "{text}"},
            },
            "edges": [
                {"from_node": "router", "to_node": "final", "output_path": "approve", "target_input": "text"},
                {"from_node": "router", "to_node": "final", "output_path": "reject", "target_input": "text"},
            ],
            "entry_point": "router",
            "exit_points": ["final"],
        }
        assert validate_agent_graph(graph_config) == []

    def test_exclusive_through_an_intermediate_node_allowed(self):
        """Router -> A -> Final.text and Router -> B -> Final.text: A and B
        each inherit a single, differing output_path requirement from
        Router, so they're provably exclusive even though neither edge
        INTO Final itself carries an output_path."""
        graph_config = {
            "nodes": {
                "router": {"name": "Router", "user_prompt": "{input}",
                           "output_paths": {"approve": {"description": "ok"},
                                             "reject": {"description": "no"}}},
                "a": {"name": "A", "user_prompt": "{input}"},
                "b": {"name": "B", "user_prompt": "{input}"},
                "final": {"name": "Final", "user_prompt": "{text}"},
            },
            "edges": [
                {"from_node": "router", "to_node": "a", "output_path": "approve"},
                {"from_node": "router", "to_node": "b", "output_path": "reject"},
                {"from_node": "a", "to_node": "final", "target_input": "text"},
                {"from_node": "b", "to_node": "final", "target_input": "text"},
            ],
            "entry_point": "router",
            "exit_points": ["final"],
        }
        assert validate_agent_graph(graph_config) == []

    def test_independent_sources_rejected(self):
        """Two edges into the same input with no common deciding ancestor
        can both fire (nothing proves exclusivity) -> rejected."""
        graph_config = {
            "nodes": {
                "entry": {"name": "Entry", "user_prompt": "{input}"},
                "src_a": {"name": "SrcA", "user_prompt": "{input}"},
                "src_b": {"name": "SrcB", "user_prompt": "{input}"},
                "final": {"name": "Final", "user_prompt": "{text}"},
            },
            "edges": [
                {"from_node": "entry", "to_node": "src_a"},
                {"from_node": "entry", "to_node": "src_b"},
                {"from_node": "src_a", "to_node": "final", "target_input": "text"},
                {"from_node": "src_b", "to_node": "final", "target_input": "text"},
            ],
            "entry_point": "entry",
            "exit_points": ["final"],
        }
        problems = validate_agent_graph(graph_config)
        assert len(problems) >= 1
        assert any("text" in p and "final" in p.lower() for p in problems)

    def test_deciding_node_inside_a_loop_rejected(self):
        """Known accepted limitation: Critic's approve/reject both feed
        Publish.text, but Critic is inside a loop (SCC with Writer), so its
        differing choices can never be used as exclusivity proof."""
        graph_config = {
            "nodes": {
                "writer": {"name": "Writer", "user_prompt": "{input}"},
                "critic": {"name": "Critic", "user_prompt": "{draft}",
                           "output_paths": {
                               "revise": {"description": "needs work"},
                               "approve": {"description": "ok"},
                               "reject": {"description": "no"},
                           }},
                "publish": {"name": "Publish", "user_prompt": "{text}"},
            },
            "edges": [
                {"from_node": "writer", "to_node": "critic", "target_input": "draft"},
                {"from_node": "critic", "to_node": "writer", "is_loop": True, "output_path": "revise"},
                {"from_node": "critic", "to_node": "publish", "output_path": "approve", "target_input": "text"},
                {"from_node": "critic", "to_node": "publish", "output_path": "reject", "target_input": "text"},
            ],
            "entry_point": "writer",
            "exit_points": ["publish"],
        }
        problems = validate_agent_graph(graph_config)
        assert len(problems) >= 1
        assert any("text" in p and "publish" in p.lower() for p in problems)


class TestBadReturnInput:
    def test_return_input_not_among_nodes_inputs_rejected(self):
        graph_config = {
            "nodes": {
                "n": {"name": "N", "user_prompt": "{draft}",
                      "output_paths": {
                          "approve": {"description": "ok",
                                      "return_behavior": "previous_output",
                                      "return_input": "nonexistent"},
                      }},
            },
            "edges": [],
            "entry_point": "n",
            "exit_points": ["n"],
        }
        problems = validate_agent_graph(graph_config)
        assert len(problems) >= 1
        assert any("nonexistent" in p for p in problems)

    def test_return_input_matching_an_input_is_fine(self):
        graph_config = {
            "nodes": {
                "n": {"name": "N", "user_prompt": "{draft}",
                      "output_paths": {
                          "approve": {"description": "ok",
                                      "return_behavior": "previous_output",
                                      "return_input": "draft"},
                      }},
            },
            "edges": [],
            "entry_point": "n",
            "exit_points": ["n"],
        }
        assert validate_agent_graph(graph_config) == []

    def test_missing_return_input_legacy_reads_as_input(self):
        """Decision 9: a missing return_input defaults to "input" — a
        problem when that node has no "input" among its own inputs."""
        graph_config = {
            "nodes": {
                "n": {"name": "N", "user_prompt": "{draft}",
                      "output_paths": {
                          "approve": {"description": "ok",
                                      "return_behavior": "previous_output"},
                      }},
            },
            "edges": [],
            "entry_point": "n",
            "exit_points": ["n"],
        }
        problems = validate_agent_graph(graph_config)
        assert len(problems) >= 1

    def test_missing_return_input_fine_when_node_uses_input(self):
        graph_config = {
            "nodes": {
                "n": {"name": "N", "user_prompt": "{input}",
                      "output_paths": {
                          "approve": {"description": "ok",
                                      "return_behavior": "previous_output"},
                      }},
            },
            "edges": [],
            "entry_point": "n",
            "exit_points": ["n"],
        }
        assert validate_agent_graph(graph_config) == []


class TestValidGraphsReturnNoProblems:
    def test_simple_single_node(self):
        graph_config = {
            "nodes": {"main": {"name": "Main", "user_prompt": "{input}"}},
            "edges": [],
            "entry_point": "main",
            "exit_points": ["main"],
        }
        assert validate_agent_graph(graph_config) == []

    def test_reflection_loop_with_named_inputs(self):
        graph_config = {
            "nodes": {
                "writer": {"name": "Writer", "user_prompt": "{plan} {feedback}"},
                "critic": {"name": "Critic", "user_prompt": "{draft}",
                           "output_paths": {
                               "revise": {"description": "needs work"},
                               "approve": {"description": "ok",
                                           "return_behavior": "previous_output",
                                           "return_input": "draft"},
                           }},
            },
            "edges": [
                {"from_node": "writer", "to_node": "critic", "target_input": "draft"},
                {"from_node": "critic", "to_node": "writer", "is_loop": True,
                 "output_path": "revise", "target_input": "feedback"},
            ],
            "entry_point": "writer",
            "exit_points": ["critic"],
            "max_loop_iterations": 5,
        }
        assert validate_agent_graph(graph_config) == []


# ─── validate_flow_agent_edges ─────────────────────────────────────────────


def single_input_agent_graph(input_name="draft"):
    return {
        "nodes": {"main": {"name": "Main", "user_prompt": f"{{{input_name}}}"}},
        "edges": [],
        "entry_point": "main",
        "exit_points": ["main"],
    }


class TestValidateFlowAgentEdges:
    def test_flow_edge_into_missing_agent_input_rejected(self):
        flow_graph_config = {
            "nodes": {
                "T": {"node_type": "tool", "id": 1, "name": "T"},
                "AG": {"node_type": "agent", "id": 5, "name": "AG"},
            },
            "edges": [{"from_node": "T", "to_node": "AG", "mapping": {"": "notes"}}],
            "entry_point": "T",
            "exit_points": ["AG"],
        }
        agent_graphs = {"AG": single_input_agent_graph("draft")}
        problems = validate_flow_agent_edges(flow_graph_config, agent_graphs)
        assert len(problems) >= 1
        assert any("notes" in p for p in problems)

    def test_two_flow_edges_into_same_agent_input_rejected(self):
        flow_graph_config = {
            "nodes": {
                "T1": {"node_type": "tool", "id": 1, "name": "T1"},
                "T2": {"node_type": "tool", "id": 2, "name": "T2"},
                "AG": {"node_type": "agent", "id": 5, "name": "AG"},
            },
            "edges": [
                {"from_node": "T1", "to_node": "AG", "mapping": {"": "draft"}},
                {"from_node": "T2", "to_node": "AG", "mapping": {"": "draft"}},
            ],
            "entry_point": "T1",
            "exit_points": ["AG"],
        }
        agent_graphs = {"AG": single_input_agent_graph("draft")}
        problems = validate_flow_agent_edges(flow_graph_config, agent_graphs)
        assert len(problems) >= 1
        assert any("draft" in p for p in problems)

    def test_single_valid_flow_edge_no_problems(self):
        flow_graph_config = {
            "nodes": {
                "T": {"node_type": "tool", "id": 1, "name": "T"},
                "AG": {"node_type": "agent", "id": 5, "name": "AG"},
            },
            "edges": [{"from_node": "T", "to_node": "AG", "mapping": {"": "draft"}}],
            "entry_point": "T",
            "exit_points": ["AG"],
        }
        agent_graphs = {"AG": single_input_agent_graph("draft")}
        assert validate_flow_agent_edges(flow_graph_config, agent_graphs) == []

    def test_legacy_edge_with_no_mapping_targets_input(self):
        flow_graph_config = {
            "nodes": {
                "T": {"node_type": "tool", "id": 1, "name": "T"},
                "AG": {"node_type": "agent", "id": 5, "name": "AG"},
            },
            "edges": [{"from_node": "T", "to_node": "AG"}],
            "entry_point": "T",
            "exit_points": ["AG"],
        }
        agent_graphs = {"AG": single_input_agent_graph("input")}
        assert validate_flow_agent_edges(flow_graph_config, agent_graphs) == []

    def test_legacy_edge_with_no_mapping_into_renamed_input_rejected(self):
        flow_graph_config = {
            "nodes": {
                "T": {"node_type": "tool", "id": 1, "name": "T"},
                "AG": {"node_type": "agent", "id": 5, "name": "AG"},
            },
            "edges": [{"from_node": "T", "to_node": "AG"}],
            "entry_point": "T",
            "exit_points": ["AG"],
        }
        agent_graphs = {"AG": single_input_agent_graph("draft")}
        problems = validate_flow_agent_edges(flow_graph_config, agent_graphs)
        assert len(problems) >= 1

    def test_loop_edges_into_agent_node_ignored(self):
        """Only non-loop (forward) flow edges feed an agent's named inputs;
        this function isn't expected to examine loop edges into agent nodes
        (flows don't currently model agent-to-agent loops on the canvas)."""
        flow_graph_config = {
            "nodes": {
                "T": {"node_type": "tool", "id": 1, "name": "T"},
                "AG": {"node_type": "agent", "id": 5, "name": "AG"},
            },
            "edges": [
                {"from_node": "T", "to_node": "AG", "mapping": {"": "draft"}},
                {"from_node": "AG", "to_node": "T", "is_loop": True},
            ],
            "entry_point": "T",
            "exit_points": ["AG"],
        }
        agent_graphs = {"AG": single_input_agent_graph("draft")}
        assert validate_flow_agent_edges(flow_graph_config, agent_graphs) == []

    def test_multiple_agent_nodes_validated_independently(self):
        flow_graph_config = {
            "nodes": {
                "T": {"node_type": "tool", "id": 1, "name": "T"},
                "AG1": {"node_type": "agent", "id": 5, "name": "AG1"},
                "AG2": {"node_type": "agent", "id": 6, "name": "AG2"},
            },
            "edges": [
                {"from_node": "T", "to_node": "AG1", "mapping": {"": "draft"}},
                {"from_node": "T", "to_node": "AG2", "mapping": {"": "bogus"}},
            ],
            "entry_point": "T",
            "exit_points": ["AG1", "AG2"],
        }
        agent_graphs = {
            "AG1": single_input_agent_graph("draft"),
            "AG2": single_input_agent_graph("draft"),
        }
        problems = validate_flow_agent_edges(flow_graph_config, agent_graphs)
        assert len(problems) == 1
        assert "bogus" in problems[0]
