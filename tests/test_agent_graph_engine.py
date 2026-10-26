"""
Unit tests for the pure agent-graph traversal engine, named-inputs version
(claude_plans/named-agent-inputs.md, "Engine algorithm").

These drive agent_graph_engine.run_graph directly with plain injected
callables — no AgentExecutor, no database, no LangFuse — which is exactly how
a standalone flow export would vendor and use it.

New public contract under test (not yet implemented):

    run_graph(graph_config, inputs: dict, run_node, record_node=None,
              max_loop_iterations=None) -> dict

    run_node(node_id, node_config, inputs: dict, predecessor_messages)
        async -> (output, messages, chosen_path, trace_id)
        `inputs` is the node's accumulated NAMED inputs dict (not a single
        string). Missing inputs are omitted or empty string, never raise.

    record_node(node_id, node_config, inputs: dict, output, chosen_path, trace_id)
        Same `inputs` dict shape.

Readiness/loop-refinement rules under test (decisions 12-13):
  - A node runs once every one of its forward (non-loop) incoming edges has
    delivered a value ("fired"). Loop edges never block and always enqueue
    their target immediately, bypassing the forward-readiness bookkeeping.
  - When nothing else is ready to run, the earliest (in forward topological
    order) waiting node is unstuck and runs with its still-missing inputs
    resolved to "".
  - After a node runs, its forward-edges' "fired" marks are cleared for any
    sender that shares its loop (its strongly-connected component over
    forward+loop edges, size > 1) — forcing a fresh value before it can
    re-fire that edge. Senders outside the loop keep their fired mark. The
    delivered VALUES are never cleared, only the readiness bookkeeping.

Run with: pytest tests/test_agent_graph_engine.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from executors.agent_graph_engine import run_graph


def make_run_node(script):
    """Build a run_node whose behavior is scripted per node_id.

    script[node_id] is a callable(inputs_dict, call_index) -> (output,
    messages, chosen_path). trace_id is always None (no observability in a
    pure run). Records (node_id, inputs-snapshot, predecessor_messages) into
    `visited`, snapshotting `inputs` at call time since the engine may go on
    to mutate the live dict for later iterations.
    """
    visited = []
    counts = {}

    async def run_node(node_id, node_config, inputs, predecessor_messages):
        idx = counts.get(node_id, 0)
        counts[node_id] = idx + 1
        visited.append((node_id, dict(inputs), predecessor_messages))
        output, messages, chosen_path = script[node_id](inputs, idx)
        return output, messages, chosen_path, None

    run_node.visited = visited
    run_node.call_count = counts
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


class TestRunGraphBasics:
    @pytest.mark.asyncio
    async def test_linear_two_node_legacy_edge_targets_input(self):
        """An edge with no target_input delivers to the "input" key
        (decision 5 — the one-time legacy read, not a runtime fallback)."""
        graph = {
            "nodes": {"a": {"name": "A"}, "b": {"name": "B"}},
            "edges": [{"from_node": "a", "to_node": "b"}],
            "entry_point": "a",
            "exit_points": ["b"],
        }
        run_node = make_run_node({
            "a": lambda inp, i: ("out-a", [], None),
            "b": lambda inp, i: (f"out-b({inp.get('input')})", [], None),
        })
        result = await run_graph(graph, {"input": "seed"}, run_node)
        assert result["status"] == "completed"
        assert result["result"] == "out-b(out-a)"
        assert [v[0] for v in run_node.visited] == ["a", "b"]
        assert run_node.visited[0][1] == {"input": "seed"}
        assert run_node.visited[1][1] == {"input": "out-a"}

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
        result = await run_graph(graph, {"input": "hi"}, run_node, record_node=None)
        assert result["result"] == "done"

    @pytest.mark.asyncio
    async def test_record_node_receives_final_output_and_dict_inputs(self):
        recorded = []

        def record_node(node_id, node_config, inputs, output, chosen_path, trace_id):
            recorded.append((node_id, dict(inputs), output, chosen_path, trace_id))

        graph = {
            "nodes": {"n": {"name": "N", "output_paths": {
                "keep": {"return_behavior": "previous_output"}}}},
            "edges": [],
            "entry_point": "n",
            "exit_points": ["n"],
        }
        # Node emits "generated" but chooses the previous_output path "keep",
        # so the engine must hand record_node the INPUT, not "generated".
        # No return_input given -> legacy default reads inputs["input"].
        run_node = make_run_node({"n": lambda inp, i: ("generated", [], "keep")})
        result = await run_graph(graph, {"input": "original-input"}, run_node,
                                  record_node=record_node)
        assert result["result"] == "original-input"
        assert recorded == [("n", {"input": "original-input"}, "original-input", "keep", None)]

    @pytest.mark.asyncio
    async def test_return_input_names_the_input_to_return(self):
        """Decision 9: return_input picks which named input previous_output
        hands back, independent of the legacy "input" default."""
        graph = {
            "nodes": {"n": {"name": "N", "output_paths": {
                "keep": {"return_behavior": "previous_output", "return_input": "original"}}}},
            "edges": [],
            "entry_point": "n",
            "exit_points": ["n"],
        }
        run_node = make_run_node({"n": lambda inp, i: ("generated", [], "keep")})
        result = await run_graph(
            graph, {"original": "original-input", "other": "ignored"}, run_node)
        assert result["result"] == "original-input"

    @pytest.mark.asyncio
    async def test_invalid_graph_raises(self):
        with pytest.raises(ValueError, match="missing entry_point or nodes"):
            await run_graph({"nodes": {}, "entry_point": None}, {"input": "x"},
                            make_run_node({}))

    @pytest.mark.asyncio
    async def test_node_failure_wrapped_with_name(self):
        graph = {
            "nodes": {"boom": {"name": "Boomer"}},
            "edges": [],
            "entry_point": "boom",
            "exit_points": ["boom"],
        }

        async def run_node(node_id, node_config, inputs, predecessor_messages):
            raise RuntimeError("kaboom")

        with pytest.raises(RuntimeError, match="Agent node 'Boomer' failed: kaboom"):
            await run_graph(graph, {"input": "x"}, run_node)

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
        await run_graph(graph, {"input": "seed"}, run_node)
        assert run_node.visited[0][2] is None
        assert run_node.visited[1][2] == [{"role": "assistant", "content": "hi"}]


class TestOutputPathRouting:
    @pytest.mark.asyncio
    async def test_output_path_approve_exits(self):
        run_node = make_run_node({
            "writer": lambda inp, i: ("draft", [], None),
            "reviewer": lambda inp, i: ("looks good", [], "approve"),
            "formatter": lambda inp, i: ("formatted", [], None),
        })
        result = await run_graph(REFLECTION_GRAPH, {"input": "topic"}, run_node)
        assert result["result"] == "formatted"
        assert [v[0] for v in run_node.visited] == ["writer", "reviewer", "formatter"]

    @pytest.mark.asyncio
    async def test_output_path_revise_loops_then_approves(self):
        """A loop that exits through a path: formatter runs once, after the
        LATE approve, not blocked by the earlier revise iteration."""
        run_node = make_run_node({
            "writer": lambda inp, i: (f"draft-{i}", [], None),
            "reviewer": lambda inp, i: ("final", [], "approve") if i >= 1
                                       else ("needs work", [], "revise"),
            "formatter": lambda inp, i: ("formatted", [], None),
        })
        result = await run_graph(REFLECTION_GRAPH, {"input": "topic"}, run_node)
        assert result["result"] == "formatted"
        assert [v[0] for v in run_node.visited] == [
            "writer", "reviewer", "writer", "reviewer", "formatter"]
        assert run_node.call_count["formatter"] == 1

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
        run_node = make_run_node({
            "a": lambda inp, i: ("a-out", [], None),
            "b": lambda inp, i: ("b-out", [], None),
        })
        result = await run_graph(graph, {"input": "start"}, run_node)
        assert result["status"] == "completed"
        # a: initial + 2 loop-backs = 3 runs; b: 3 runs
        assert run_node.call_count["b"] == 3
        assert run_node.call_count["a"] == 3

    @pytest.mark.asyncio
    async def test_two_edges_same_pair_different_output_path_and_input(self):
        """Decision 10 / engine note: edges are tracked as individual
        objects (by list index), never collapsed by a (from, to) pair — two
        edges between the same nodes keep separate output_path and
        target_input. The naive old (from, to)-keyed representation would
        make the second edge's output_path silently win for BOTH edges."""
        graph = {
            "nodes": {
                "a": {"name": "A", "output_paths": {
                    "p1": {"description": "one"}, "p2": {"description": "two"}}},
                "b": {"name": "B"},
            },
            "edges": [
                {"from_node": "a", "to_node": "b", "output_path": "p1", "target_input": "x"},
                {"from_node": "a", "to_node": "b", "output_path": "p2", "target_input": "y"},
            ],
            "entry_point": "a",
            "exit_points": ["b"],
        }
        run_node = make_run_node({
            "a": lambda inp, i: ("A-out", [], "p1"),
            "b": lambda inp, i: ("B-out", [], None),
        })
        result = await run_graph(graph, {"input": "seed"}, run_node)
        assert result["status"] == "completed"
        assert run_node.call_count["b"] == 1
        b_inputs = run_node.visited[-1][1]
        assert b_inputs.get("x") == "A-out"
        assert b_inputs.get("y", "") == ""


class TestSkipEdgeMerging:
    @pytest.mark.asyncio
    async def test_writer_waits_for_both_direct_and_relayed_input(self):
        """Planner -> Writer.plan (direct) and Planner -> Researcher ->
        Writer.research (relayed): Writer must run exactly ONCE, after BOTH
        have arrived — not once per level with incomplete inputs."""
        graph = {
            "nodes": {
                "planner": {"name": "Planner"},
                "researcher": {"name": "Researcher"},
                "writer": {"name": "Writer"},
            },
            "edges": [
                {"from_node": "planner", "to_node": "writer", "target_input": "plan"},
                {"from_node": "planner", "to_node": "researcher"},
                {"from_node": "researcher", "to_node": "writer", "target_input": "research"},
            ],
            "entry_point": "planner",
            "exit_points": ["writer"],
        }
        run_node = make_run_node({
            "planner": lambda inp, i: ("P", [], None),
            "researcher": lambda inp, i: (f"R({inp.get('input')})", [], None),
            "writer": lambda inp, i: (f"W(plan={inp.get('plan')},research={inp.get('research')})",
                                       [], None),
        })
        result = await run_graph(graph, {"input": "go"}, run_node)
        assert run_node.call_count["writer"] == 1
        writer_inputs = run_node.visited[-1][1]
        assert writer_inputs == {"plan": "P", "research": "R(P)"}
        assert result["result"] == "W(plan=P,research=R(P))"


class TestRouterSkipsDownstream:
    @pytest.mark.asyncio
    async def test_approve_skips_reviser_writer_gets_empty_notes(self):
        """Router picks approve -> Reviser never runs; Writer still runs
        (unstuck by the lazy-readiness rule) with notes resolved to ""."""
        graph = {
            "nodes": {
                "router": {"name": "Router", "output_paths": {
                    "approve": {"description": "ok"}, "revise": {"description": "no"}}},
                "reviser": {"name": "Reviser"},
                "writer": {"name": "Writer"},
            },
            "edges": [
                {"from_node": "router", "to_node": "writer"},
                {"from_node": "router", "to_node": "reviser", "output_path": "revise"},
                {"from_node": "reviser", "to_node": "writer", "target_input": "notes"},
            ],
            "entry_point": "router",
            "exit_points": ["writer"],
        }
        run_node = make_run_node({
            "router": lambda inp, i: ("approved", [], "approve"),
            "reviser": lambda inp, i: ("SHOULD NOT RUN", [], None),
            "writer": lambda inp, i: (f"notes={inp.get('notes', 'MISSING')!r}", [], None),
        })
        result = await run_graph(graph, {"input": "go"}, run_node)
        assert run_node.call_count.get("reviser", 0) == 0
        assert run_node.call_count["writer"] == 1
        writer_inputs = run_node.visited[-1][1]
        assert writer_inputs.get("notes", "") == ""
        assert writer_inputs.get("input") == "approved"
        assert result["result"] == "notes=''"


class TestReflectionLoopPreservesOutOfLoopInput:
    @pytest.mark.asyncio
    async def test_plan_persists_feedback_updates(self):
        """Planner -> Writer.plan, Writer -> Critic, Critic -(revise, loop)->
        Writer.feedback. Writer keeps `plan` across iterations (planner is
        outside the loop) while `feedback` changes every iteration."""
        graph = {
            "nodes": {
                "planner": {"name": "Planner"},
                "writer": {"name": "Writer"},
                "critic": {"name": "Critic", "output_paths": {
                    "revise": {"description": "no"}, "approve": {"description": "ok"}}},
            },
            "edges": [
                {"from_node": "planner", "to_node": "writer", "target_input": "plan"},
                {"from_node": "writer", "to_node": "critic"},
                {"from_node": "critic", "to_node": "writer", "is_loop": True,
                 "output_path": "revise", "target_input": "feedback"},
            ],
            "entry_point": "planner",
            "exit_points": ["critic"],
            "max_loop_iterations": 5,
        }
        run_node = make_run_node({
            "planner": lambda inp, i: ("TopicPlan", [], None),
            "writer": lambda inp, i: (f"draft{i + 1}(plan={inp.get('plan')},"
                                       f"feedback={inp.get('feedback', '')!r})", [], None),
            "critic": lambda inp, i: (("needs more work", [], "revise") if i == 0
                                       else ("approved!", [], "approve")),
        })
        result = await run_graph(graph, {"input": "go"}, run_node)

        writer_calls = [v for v in run_node.visited if v[0] == "writer"]
        assert len(writer_calls) == 2
        assert writer_calls[0][1]["plan"] == "TopicPlan"
        assert writer_calls[0][1].get("feedback", "") == ""
        assert writer_calls[1][1]["plan"] == "TopicPlan"
        assert writer_calls[1][1]["feedback"] == "needs more work"
        assert result["result"] == "approved!"


class TestLoopRefinementAcrossSCC:
    @pytest.mark.asyncio
    async def test_critic_waits_for_fresh_summary_each_iteration(self):
        """Writer -> Critic.draft, Writer -> Summarizer -> Critic.summary,
        Critic -(revise, loop)-> Writer.feedback. Writer, Summarizer and
        Critic are mutually reachable (writer -> critic -> writer is a
        2-cycle, and critic can reach summarizer via writer), so they share
        one strongly-connected component: after Critic runs, BOTH its
        forward inputs (draft AND summary) must arrive fresh before it can
        run again — it must not fire on a stale summary paired with a new
        draft, and it must run exactly once per round."""
        graph = {
            "nodes": {
                "writer": {"name": "Writer"},
                "summarizer": {"name": "Summarizer"},
                "critic": {"name": "Critic", "output_paths": {
                    "revise": {"description": "no"}, "approve": {"description": "ok"}}},
            },
            "edges": [
                {"from_node": "writer", "to_node": "critic", "target_input": "draft"},
                {"from_node": "writer", "to_node": "summarizer"},
                {"from_node": "summarizer", "to_node": "critic", "target_input": "summary"},
                {"from_node": "critic", "to_node": "writer", "is_loop": True,
                 "output_path": "revise", "target_input": "feedback"},
            ],
            "entry_point": "writer",
            "exit_points": ["critic"],
            "max_loop_iterations": 5,
        }
        run_node = make_run_node({
            "writer": lambda inp, i: (f"draft{i + 1}", [], None),
            "summarizer": lambda inp, i: (f"sum-of-{inp.get('input')}", [], None),
            "critic": lambda inp, i: (("needs work", [], "revise") if i == 0
                                       else ("approved", [], "approve")),
        })
        await run_graph(graph, {"input": "go"}, run_node)

        critic_calls = [v for v in run_node.visited if v[0] == "critic"]
        assert len(critic_calls) == 2
        assert critic_calls[0][1] == {"draft": "draft1", "summary": "sum-of-draft1"}
        assert critic_calls[1][1] == {"draft": "draft2", "summary": "sum-of-draft2"}
        # Never a mismatched pairing (stale summary with a fresh draft).
        for call in critic_calls:
            draft_n = call[1]["draft"].replace("draft", "")
            summary_n = call[1]["summary"].replace("sum-of-draft", "")
            assert draft_n == summary_n
