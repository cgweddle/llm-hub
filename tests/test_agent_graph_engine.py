"""
Unit tests for the pure agent-graph traversal engine.

These drive agent_graph_engine.run_graph directly with plain injected
callables — no AgentExecutor, no database, no LangFuse — which is exactly how
a standalone flow export would vendor and use it. They lock in the traversal
contract: BFS ordering, output-path routing, loop caps, previous_output, and
that record_node is optional (pure traversal with zero side effects).

Run with: pytest tests/test_agent_graph_engine.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from executors.agent_graph_engine import build_adjacency, run_graph


def make_run_node(script):
    """Build a run_node whose behavior is scripted per node_id.

    script[node_id] is a callable(node_input, call_index) -> (output, messages,
    chosen_path). trace_id is always None (no observability in a pure run).
    Records the visit order and inputs into `visited`.
    """
    visited = []
    counts = {}

    async def run_node(node_id, node_config, node_input, predecessor_messages):
        idx = counts.get(node_id, 0)
        counts[node_id] = idx + 1
        visited.append((node_id, node_input, predecessor_messages))
        output, messages, chosen_path = script[node_id](node_input, idx)
        return output, messages, chosen_path, None

    run_node.visited = visited
    return run_node


REFLECTION_GRAPH = {
    "nodes": {
        "writer": {"agent_type": "pydanticai", "name": "Writer"},
        "reviewer": {"agent_type": "pydanticai", "name": "Reviewer"},
        "formatter": {"agent_type": "pydanticai", "name": "Formatter"},
    },
    "edges": [
        {"from_node": "writer", "to_node": "reviewer", "is_loop": False},
        {"from_node": "reviewer", "to_node": "writer", "output_path": "revise", "is_loop": True},
        {"from_node": "reviewer", "to_node": "formatter", "output_path": "approve", "is_loop": False},
    ],
    "entry_point": "writer",
    "exit_points": ["formatter"],
    "max_loop_iterations": 5,
}


class TestBuildAdjacency:
    def test_edge_metadata_extracted(self):
        adjacency, loop_edges, edge_output_paths = build_adjacency(REFLECTION_GRAPH)
        assert adjacency["writer"] == ["reviewer"]
        assert set(adjacency["reviewer"]) == {"writer", "formatter"}
        assert loop_edges == {("reviewer", "writer")}
        assert edge_output_paths == {
            ("reviewer", "writer"): "revise",
            ("reviewer", "formatter"): "approve",
        }


class TestRunGraph:
    @pytest.mark.asyncio
    async def test_linear_two_node(self):
        graph = {
            "nodes": {"a": {"name": "A"}, "b": {"name": "B"}},
            "edges": [{"from_node": "a", "to_node": "b"}],
            "entry_point": "a",
            "exit_points": ["b"],
        }
        run_node = make_run_node({
            "a": lambda inp, i: ("out-a", [], None),
            "b": lambda inp, i: (f"out-b({inp})", [], None),
        })
        result = await run_graph(graph, "seed", run_node)
        assert result["status"] == "completed"
        assert result["result"] == "out-b(out-a)"
        assert [v[0] for v in run_node.visited] == ["a", "b"]
        assert run_node.visited[0][1] == "seed"      # entry gets the graph input
        assert run_node.visited[1][1] == "out-a"     # b gets a's output

    @pytest.mark.asyncio
    async def test_record_node_optional(self):
        """A pure traversal with record_node=None must run with zero effects."""
        graph = {
            "nodes": {"only": {"name": "Only"}},
            "edges": [],
            "entry_point": "only",
            "exit_points": ["only"],
        }
        run_node = make_run_node({"only": lambda inp, i: ("done", [], None)})
        result = await run_graph(graph, "hi", run_node, record_node=None)
        assert result["result"] == "done"

    @pytest.mark.asyncio
    async def test_record_node_receives_final_output_and_trace(self):
        recorded = []

        def record_node(node_id, node_config, node_input, output, chosen_path, trace_id):
            recorded.append((node_id, node_input, output, chosen_path, trace_id))

        graph = {
            "nodes": {"n": {"name": "N", "output_paths": {
                "keep": {"return_behavior": "previous_output"}}}},
            "edges": [],
            "entry_point": "n",
            "exit_points": ["n"],
        }
        # Node emits "generated" but chooses the previous_output path "keep",
        # so the engine must hand record_node the INPUT, not "generated".
        run_node = make_run_node({"n": lambda inp, i: ("generated", [], "keep")})
        result = await run_graph(graph, "original-input", run_node, record_node=record_node)
        assert result["result"] == "original-input"
        assert recorded == [("n", "original-input", "original-input", "keep", None)]

    @pytest.mark.asyncio
    async def test_output_path_approve_exits(self):
        run_node = make_run_node({
            "writer": lambda inp, i: ("draft", [], None),
            "reviewer": lambda inp, i: ("looks good", [], "approve"),
            "formatter": lambda inp, i: ("formatted", [], None),
        })
        result = await run_graph(REFLECTION_GRAPH, "topic", run_node)
        assert result["result"] == "formatted"
        assert [v[0] for v in run_node.visited] == ["writer", "reviewer", "formatter"]

    @pytest.mark.asyncio
    async def test_output_path_revise_loops_then_approves(self):
        run_node = make_run_node({
            "writer": lambda inp, i: (f"draft-{i}", [], None),
            # First review revises (loops back), second approves.
            "reviewer": lambda inp, i: ("final", [], "approve") if i >= 1
                                       else ("needs work", [], "revise"),
            "formatter": lambda inp, i: ("formatted", [], None),
        })
        result = await run_graph(REFLECTION_GRAPH, "topic", run_node)
        assert result["result"] == "formatted"
        assert [v[0] for v in run_node.visited] == [
            "writer", "reviewer", "writer", "reviewer", "formatter"]

    @pytest.mark.asyncio
    async def test_max_loop_iterations_caps_runaway(self):
        graph = {
            "nodes": {"a": {"name": "A"}, "b": {"name": "B"}},
            "edges": [
                {"from_node": "a", "to_node": "b", "is_loop": False},
                {"from_node": "b", "to_node": "a", "is_loop": True},
            ],
            "entry_point": "a",
            "exit_points": ["b"],
            "max_loop_iterations": 2,
        }
        # b always tries to loop back; the cap must stop it.
        run_node = make_run_node({
            "a": lambda inp, i: ("a-out", [], None),
            "b": lambda inp, i: ("b-out", [], None),
        })
        result = await run_graph(graph, "start", run_node)
        assert result["status"] == "completed"
        # a: initial + 2 loop-backs = 3 runs; b: 3 runs
        b_runs = sum(1 for v in run_node.visited if v[0] == "b")
        assert b_runs == 3

    @pytest.mark.asyncio
    async def test_message_history_threads_forward(self):
        graph = {
            "nodes": {"a": {"name": "A"}, "b": {"name": "B"}},
            "edges": [{"from_node": "a", "to_node": "b"}],
            "entry_point": "a",
            "exit_points": ["b"],
        }
        run_node = make_run_node({
            "a": lambda inp, i: ("out-a", [{"role": "assistant", "content": "hi"}], None),
            "b": lambda inp, i: ("out-b", [], None),
        })
        await run_graph(graph, "seed", run_node)
        # a has no predecessor history; b receives a's all_messages().
        assert run_node.visited[0][2] is None
        assert run_node.visited[1][2] == [{"role": "assistant", "content": "hi"}]

    @pytest.mark.asyncio
    async def test_invalid_graph_raises(self):
        with pytest.raises(ValueError, match="missing entry_point or nodes"):
            await run_graph({"nodes": {}, "entry_point": None}, "x",
                            make_run_node({}))

    @pytest.mark.asyncio
    async def test_node_failure_wrapped_with_name(self):
        graph = {
            "nodes": {"boom": {"name": "Boomer"}},
            "edges": [],
            "entry_point": "boom",
            "exit_points": ["boom"],
        }

        async def run_node(node_id, node_config, node_input, predecessor_messages):
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError, match="Agent node 'Boomer' failed: kaboom"):
            await run_graph(graph, "x", run_node)
