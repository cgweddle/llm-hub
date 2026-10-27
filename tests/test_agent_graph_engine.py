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

import asyncio
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
        """Updated for parallel-agent-execution decision 2: history is no
        longer a single replaced `messages` variable ("most recently
        finished node"). It is now per-sender (`latest_messages[sender]`),
        concatenated in forward-topological/edge order for each of a node's
        CURRENT senders. A node with zero senders (the entry node) gets an
        empty list, not None — there is nothing to concatenate."""
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
        assert run_node.visited[0][2] == []
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


# ============================================================================
# Parallel sub-agent execution (claude_plans/parallel-agent-execution.md).
#
# New public contract under test (not yet implemented):
#
#     run_graph(graph_config, inputs, run_node, record_node=None,
#               max_loop_iterations=None, parallel=True)
#
# Every ready node runs concurrently (asyncio tasks); results are applied
# one at a time as tasks complete, in forward-topological-order (then node
# id) when several finish in the same batch. `parallel=False` reproduces
# today's one-at-a-time behavior exactly.
#
# These tests drive real concurrency with asyncio.Event-gated fake run_node
# coroutines instead of sleep-based timing, so ordering is deterministic.
# Every blocking wait is wrapped in asyncio.wait_for with a short timeout so
# a still-sequential engine fails fast (TimeoutError) instead of hanging the
# suite.
# ============================================================================

_GATE_TIMEOUT = 0.5


class TestConcurrentSiblingsOverlap:
    @pytest.mark.asyncio
    async def test_two_researchers_fed_by_one_planner_overlap_in_time(self):
        """Planner -> Researcher A, Researcher B. Both must be STARTED
        before either FINISHES: each researcher signals its own "started"
        event immediately on entry, then waits for the other's "started"
        event before returning. Under today's one-node-at-a-time queue,
        this deadlocks (A waits forever for B's start, since B is never
        even popped from the queue until A's run() call returns) and the
        wait_for below times out."""
        a_started = asyncio.Event()
        b_started = asyncio.Event()

        async def run_node(node_id, node_config, inputs, predecessor_messages):
            if node_id == "planner":
                return ("plan", [], None, None)
            if node_id == "researcher_a":
                a_started.set()
                await asyncio.wait_for(b_started.wait(), timeout=_GATE_TIMEOUT)
                return ("A-out", [], None, None)
            if node_id == "researcher_b":
                b_started.set()
                await asyncio.wait_for(a_started.wait(), timeout=_GATE_TIMEOUT)
                return ("B-out", [], None, None)
            raise AssertionError(f"unexpected node {node_id}")

        graph = {
            "nodes": {
                "planner": {"name": "Planner"},
                "researcher_a": {"name": "Researcher A"},
                "researcher_b": {"name": "Researcher B"},
            },
            "edges": [
                {"from_node": "planner", "to_node": "researcher_a"},
                {"from_node": "planner", "to_node": "researcher_b"},
            ],
            "entry_point": "planner",
            "exit_points": ["researcher_a", "researcher_b"],
        }

        result = await run_graph(graph, {"input": "go"}, run_node, parallel=True)
        assert result["status"] == "completed"
        assert result["all_outputs"] == {"researcher_a": "A-out", "researcher_b": "B-out"}


class TestCompletionOrderDeterminism:
    """Planner -> {Researcher A, Researcher B} -> Writer. Writer must see
    identical inputs, output and recorded order no matter which researcher's
    LLM call happens to return first."""

    def _build(self, a_delay, b_delay, recorded):
        async def run_node(node_id, node_config, inputs, predecessor_messages):
            if node_id == "planner":
                return ("plan", [], None, None)
            if node_id == "researcher_a":
                await asyncio.sleep(a_delay)
                return ("A-out", [{"role": "assistant", "content": "a-said"}], None, None)
            if node_id == "researcher_b":
                await asyncio.sleep(b_delay)
                return ("B-out", [{"role": "assistant", "content": "b-said"}], None, None)
            if node_id == "writer":
                recorded.append(("writer", dict(inputs), list(predecessor_messages or [])))
                return (f"W(a={inputs.get('a')},b={inputs.get('b')})", [], None, None)
            raise AssertionError(f"unexpected node {node_id}")

        graph = {
            "nodes": {
                "planner": {"name": "Planner"},
                "researcher_a": {"name": "Researcher A"},
                "researcher_b": {"name": "Researcher B"},
                "writer": {"name": "Writer"},
            },
            "edges": [
                {"from_node": "planner", "to_node": "researcher_a"},
                {"from_node": "planner", "to_node": "researcher_b"},
                {"from_node": "researcher_a", "to_node": "writer", "target_input": "a"},
                {"from_node": "researcher_b", "to_node": "writer", "target_input": "b"},
            ],
            "entry_point": "planner",
            "exit_points": ["writer"],
        }
        return graph, run_node

    @pytest.mark.asyncio
    async def test_b_finishes_first_then_a(self):
        recorded = []
        graph, run_node = self._build(a_delay=0.05, b_delay=0.0, recorded=recorded)
        result = await run_graph(graph, {"input": "go"}, run_node, parallel=True)
        assert result["result"] == "W(a=A-out,b=B-out)"
        assert len(recorded) == 1
        assert recorded[0][1] == {"a": "A-out", "b": "B-out"}

    @pytest.mark.asyncio
    async def test_a_finishes_first_then_b(self):
        recorded = []
        graph, run_node = self._build(a_delay=0.0, b_delay=0.05, recorded=recorded)
        result = await run_graph(graph, {"input": "go"}, run_node, parallel=True)
        assert result["result"] == "W(a=A-out,b=B-out)"
        assert len(recorded) == 1
        assert recorded[0][1] == {"a": "A-out", "b": "B-out"}

    @pytest.mark.asyncio
    async def test_both_orders_give_identical_writer_state(self):
        """The two completion orders above must agree byte-for-byte on
        Writer's inputs, output and history — order of completion must not
        leak into recorded state."""
        recorded_b_first, recorded_a_first = [], []
        graph1, run_node1 = self._build(a_delay=0.05, b_delay=0.0, recorded=recorded_b_first)
        graph2, run_node2 = self._build(a_delay=0.0, b_delay=0.05, recorded=recorded_a_first)

        result1 = await run_graph(graph1, {"input": "go"}, run_node1, parallel=True)
        result2 = await run_graph(graph2, {"input": "go"}, run_node2, parallel=True)

        assert result1["result"] == result2["result"]
        assert recorded_b_first[0][1] == recorded_a_first[0][1]
        assert recorded_b_first[0][2] == recorded_a_first[0][2]


class TestRerunDedupAndInputSnapshot:
    @pytest.mark.asyncio
    async def test_loop_retrigger_while_running_is_deduped_and_reruns_with_fresh_input(self):
        """start -> x (forward, target "a") and start -> z (forward,
        default "input"), z -> x (is_loop, target "b"). z finishes fast and
        fires its loop edge into x WHILE x's own first run is still pending
        (x awaits z's completion event internally) — exercising three rules
        at once:

        - x must not be started a second time while its first run is still
          in flight (the engine's `rerun` bookkeeping, not a second task);
        - x's FIRST run's inputs were snapshotted when it started, so it
          must NOT see "b" (z hasn't fired yet at that point) — proving
          inputs are snapshotted at start, not read live;
        - once x's first run finishes, it must run exactly once MORE (the
          deferred retrigger), this time seeing "b" = z's delivered value —
          proving a value that arrived mid-run is not lost, and is used on
          the very next run (loop refinement only clears what a run's
          snapshot actually used; a mid-run arrival was never in that
          snapshot, so it survives untouched).

        Under today's engine (no asyncio concurrency, no `parallel` kwarg),
        x's first run can never observe z's completion (z hasn't even been
        popped from the queue yet, since the current loop fully awaits each
        node one at a time) — the wait_for below times out.
        """
        z_done = asyncio.Event()
        x_calls = []
        x_in_flight = False

        async def run_node(node_id, node_config, inputs, predecessor_messages):
            nonlocal x_in_flight
            if node_id == "start":
                return ("S", [], None, None)
            if node_id == "z":
                z_done.set()
                return ("Z-out", [], None, None)
            if node_id == "x":
                assert not x_in_flight, "x must never run twice concurrently"
                x_in_flight = True
                try:
                    idx = len(x_calls)
                    x_calls.append(dict(inputs))
                    if idx == 0:
                        await asyncio.wait_for(z_done.wait(), timeout=_GATE_TIMEOUT)
                        return (f"x1(a={inputs.get('a')},b={inputs.get('b', '')})", [], None, None)
                    return (f"x2(a={inputs.get('a')},b={inputs.get('b', '')})", [], None, None)
                finally:
                    x_in_flight = False
            raise AssertionError(f"unexpected node {node_id}")

        graph = {
            "nodes": {"start": {"name": "Start"}, "x": {"name": "X"}, "z": {"name": "Z"}},
            "edges": [
                {"from_node": "start", "to_node": "x", "target_input": "a"},
                {"from_node": "start", "to_node": "z"},
                {"from_node": "z", "to_node": "x", "is_loop": True, "target_input": "b"},
            ],
            "entry_point": "start",
            "exit_points": ["x"],
            "max_loop_iterations": 5,
        }

        result = await run_graph(graph, {"input": "go"}, run_node, parallel=True)

        assert len(x_calls) == 2, "x should run once (blocked), then exactly once more (deduped retrigger)"
        assert x_calls[0] == {"a": "S"}  # "b" not yet arrived -> snapshotted without it
        assert x_calls[1] == {"a": "S", "b": "Z-out"}  # mid-run arrival used on the next run
        assert result["result"] == "x2(a=S,b=Z-out)"


class TestUnstickWaitsForRunningSibling:
    @pytest.mark.asyncio
    async def test_unstick_does_not_fire_while_a_sibling_could_still_deliver(self):
        """start -> fast (forward) and start -> slow (forward); fast -> join
        (target "from_fast") and slow -> join (target "from_slow"). `join`
        needs both. `slow` is genuinely still executing (a real
        asyncio.sleep, not an immediately-resolved coroutine) when `fast`
        completes and leaves `join` partially fired. The engine must NOT
        treat `join` as unstickable while `slow` — a running sibling that
        can still deliver the missing input — hasn't finished. If it
        wrongly did, `join` would run twice: once early with
        `from_slow=""`, and once more (legitimately re-triggered) once
        `slow` actually delivers."""
        join_calls = []

        async def run_node(node_id, node_config, inputs, predecessor_messages):
            if node_id == "start":
                return ("S", [], None, None)
            if node_id == "fast":
                return ("fast-out", [], None, None)
            if node_id == "slow":
                await asyncio.sleep(0.05)
                return ("slow-out", [], None, None)
            if node_id == "join":
                join_calls.append(dict(inputs))
                return (f"J(from_fast={inputs.get('from_fast')},from_slow={inputs.get('from_slow', '')})",
                        [], None, None)
            raise AssertionError(f"unexpected node {node_id}")

        graph = {
            "nodes": {
                "start": {"name": "Start"}, "fast": {"name": "Fast"},
                "slow": {"name": "Slow"}, "join": {"name": "Join"},
            },
            "edges": [
                {"from_node": "start", "to_node": "fast"},
                {"from_node": "start", "to_node": "slow"},
                {"from_node": "fast", "to_node": "join", "target_input": "from_fast"},
                {"from_node": "slow", "to_node": "join", "target_input": "from_slow"},
            ],
            "entry_point": "start",
            "exit_points": ["join"],
        }

        result = await run_graph(graph, {"input": "go"}, run_node, parallel=True)

        assert len(join_calls) == 1, "join must run exactly once, after BOTH inputs arrived"
        assert join_calls[0] == {"from_fast": "fast-out", "from_slow": "slow-out"}
        assert result["result"] == "J(from_fast=fast-out,from_slow=slow-out)"


class TestFailFastCancelsSiblings:
    @pytest.mark.asyncio
    async def test_failure_cancels_running_sibling_and_cancelled_sibling_is_never_recorded(self):
        """start -> boom (forward, fails quickly) and start -> slow
        (forward, long-running). boom's failure must cancel slow's still-
        running task and re-raise — slow must never reach record_node (rows
        are only written on completion, per decision 4)."""
        slow_cancelled = asyncio.Event()
        recorded = []

        def record_node(node_id, node_config, inputs, output, chosen_path, trace_id):
            recorded.append(node_id)

        async def run_node(node_id, node_config, inputs, predecessor_messages):
            if node_id == "start":
                return ("S", [], None, None)
            if node_id == "boom":
                await asyncio.sleep(0.01)
                raise RuntimeError("kaboom")
            if node_id == "slow":
                try:
                    await asyncio.sleep(5)
                    return ("slow-out", [], None, None)
                except asyncio.CancelledError:
                    slow_cancelled.set()
                    raise
            raise AssertionError(f"unexpected node {node_id}")

        graph = {
            "nodes": {"start": {"name": "Start"}, "boom": {"name": "Boom"}, "slow": {"name": "Slow"}},
            "edges": [
                {"from_node": "start", "to_node": "boom"},
                {"from_node": "start", "to_node": "slow"},
            ],
            "entry_point": "start",
            "exit_points": ["boom", "slow"],
        }

        with pytest.raises(RuntimeError, match="kaboom"):
            await run_graph(graph, {"input": "go"}, run_node, record_node=record_node, parallel=True)

        await asyncio.wait_for(slow_cancelled.wait(), timeout=_GATE_TIMEOUT)
        assert "slow" not in recorded
        assert "boom" not in recorded  # boom itself failed -> never reaches record_node either


class TestFailFastTiebreak:
    @pytest.mark.asyncio
    async def test_two_same_batch_failures_report_the_earlier_topo_rank_first(self):
        """start fans out to "zeta" and "alpha" (dict insertion order —
        zeta first, alpha second — so topo_rank(zeta) < topo_rank(alpha)
        despite "alpha" sorting first alphabetically). Both fail WITHOUT
        awaiting anything, so both land in the SAME asyncio.wait()
        completion batch. The engine must raise the earlier-topo-rank
        node's error (zeta), not alpha's, and not whichever happened to be
        iterated first by accident."""
        graph = {
            "nodes": {
                "start": {"name": "Start"},
                "zeta": {"name": "Zeta"},
                "alpha": {"name": "Alpha"},
            },
            "edges": [
                {"from_node": "start", "to_node": "zeta"},
                {"from_node": "start", "to_node": "alpha"},
            ],
            "entry_point": "start",
            "exit_points": ["zeta", "alpha"],
        }

        async def run_node(node_id, node_config, inputs, predecessor_messages):
            if node_id == "start":
                return ("S", [], None, None)
            if node_id == "zeta":
                raise RuntimeError("zeta-boom")
            if node_id == "alpha":
                raise RuntimeError("alpha-boom")
            raise AssertionError(f"unexpected node {node_id}")

        with pytest.raises(RuntimeError, match="zeta-boom"):
            await run_graph(graph, {"input": "go"}, run_node, parallel=True)


class TestParallelFalseIsSequential:
    @pytest.mark.asyncio
    async def test_parallel_false_never_overlaps_and_matches_sequential_order(self):
        concurrent = 0
        max_concurrent = 0
        visited_order = []

        async def run_node(node_id, node_config, inputs, predecessor_messages):
            nonlocal concurrent, max_concurrent
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            visited_order.append(node_id)
            try:
                await asyncio.sleep(0.01)
                return (f"{node_id}-out", [], None, None)
            finally:
                concurrent -= 1

        graph = {
            "nodes": {
                "planner": {"name": "Planner"},
                "researcher_a": {"name": "Researcher A"},
                "researcher_b": {"name": "Researcher B"},
                "writer": {"name": "Writer"},
            },
            "edges": [
                {"from_node": "planner", "to_node": "researcher_a"},
                {"from_node": "planner", "to_node": "researcher_b"},
                {"from_node": "researcher_a", "to_node": "writer", "target_input": "a"},
                {"from_node": "researcher_b", "to_node": "writer", "target_input": "b"},
            ],
            "entry_point": "planner",
            "exit_points": ["writer"],
        }

        result = await run_graph(graph, {"input": "go"}, run_node, parallel=False)

        assert max_concurrent == 1, "parallel=False must never run two nodes at once"
        assert visited_order == ["planner", "researcher_a", "researcher_b", "writer"]
        assert result["result"] == "writer-out"


class TestMessageHistoryPerSender:
    """Decision 2: {message_history} = the direct senders' latest
    conversations, concatenated in forward-topological order then edge
    order, regardless of completion order. Replaces the old single
    `messages = node_messages` ("most recently finished node") model."""

    @pytest.mark.asyncio
    async def test_writer_gets_both_senders_histories_regardless_of_completion_order(self):
        async def make_run_node_with_delays(a_delay, b_delay, captured):
            async def run_node(node_id, node_config, inputs, predecessor_messages):
                if node_id == "planner":
                    return ("plan", [], None, None)
                if node_id == "researcher_a":
                    await asyncio.sleep(a_delay)
                    return ("A-out", [{"role": "assistant", "content": "a-conversation"}], None, None)
                if node_id == "researcher_b":
                    await asyncio.sleep(b_delay)
                    return ("B-out", [{"role": "assistant", "content": "b-conversation"}], None, None)
                if node_id == "writer":
                    captured.append(list(predecessor_messages or []))
                    return ("done", [], None, None)
                raise AssertionError(f"unexpected node {node_id}")
            return run_node

        graph = {
            "nodes": {
                "planner": {"name": "Planner"},
                "researcher_a": {"name": "Researcher A"},
                "researcher_b": {"name": "Researcher B"},
                "writer": {"name": "Writer"},
            },
            "edges": [
                {"from_node": "planner", "to_node": "researcher_a"},
                {"from_node": "planner", "to_node": "researcher_b"},
                {"from_node": "researcher_a", "to_node": "writer", "target_input": "a"},
                {"from_node": "researcher_b", "to_node": "writer", "target_input": "b"},
            ],
            "entry_point": "planner",
            "exit_points": ["writer"],
        }

        expected = [
            {"role": "assistant", "content": "a-conversation"},
            {"role": "assistant", "content": "b-conversation"},
        ]

        captured_b_first = []
        run_node_b_first = await make_run_node_with_delays(0.05, 0.0, captured_b_first)
        await run_graph(graph, {"input": "go"}, run_node_b_first, parallel=True)
        assert captured_b_first == [expected]

        captured_a_first = []
        run_node_a_first = await make_run_node_with_delays(0.0, 0.05, captured_a_first)
        await run_graph(graph, {"input": "go"}, run_node_a_first, parallel=True)
        assert captured_a_first == [expected]

    @pytest.mark.asyncio
    async def test_input_that_never_arrived_contributes_nothing_to_history(self):
        """Router picks "approve" -> Reviser never runs (same shape as
        TestRouterSkipsDownstream). Writer is unstuck with "notes" missing,
        so Reviser never contributes a conversation — Writer's history must
        contain only the Router's, not an empty/None placeholder for
        Reviser."""
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

        async def run_node(node_id, node_config, inputs, predecessor_messages):
            if node_id == "router":
                return ("approved", [{"role": "assistant", "content": "router-said"}], "approve", None)
            if node_id == "reviser":
                return ("SHOULD NOT RUN", [{"role": "assistant", "content": "reviser-said"}], None, None)
            if node_id == "writer":
                return ("done", [], None, None)
            raise AssertionError(f"unexpected node {node_id}")

        captured = []

        async def capturing_run_node(node_id, node_config, inputs, predecessor_messages):
            if node_id == "writer":
                captured.append(list(predecessor_messages or []))
            return await run_node(node_id, node_config, inputs, predecessor_messages)

        await run_graph(graph, {"input": "go"}, capturing_run_node, parallel=True)

        assert captured == [[{"role": "assistant", "content": "router-said"}]]

    @pytest.mark.asyncio
    async def test_loop_pass_gets_critics_latest_plus_persisted_forward_senders_history(self):
        """Planner -> Writer.plan, Writer -> Critic, Critic -(revise,
        loop)-> Writer.feedback. On Writer's SECOND run (after the loop
        back), its history must contain Planner's conversation (an
        out-of-loop forward sender, persisted across iterations) PLUS
        Critic's latest conversation (the loop sender) — not just whichever
        node happened to finish most recently."""
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

        writer_histories = []

        async def run_node(node_id, node_config, inputs, predecessor_messages):
            if node_id == "planner":
                return ("TopicPlan", [{"role": "assistant", "content": "planner-said"}], None, None)
            if node_id == "writer":
                writer_histories.append(list(predecessor_messages or []))
                idx = len(writer_histories) - 1
                return (f"draft{idx + 1}", [{"role": "assistant", "content": f"writer-said-{idx}"}], None, None)
            if node_id == "critic":
                if len(writer_histories) == 1:
                    return ("needs more work", [{"role": "assistant", "content": "critic-said-1"}], "revise", None)
                return ("approved!", [{"role": "assistant", "content": "critic-said-2"}], "approve", None)
            raise AssertionError(f"unexpected node {node_id}")

        await run_graph(graph, {"input": "go"}, run_node, parallel=True)

        assert len(writer_histories) == 2
        # Writer's FIRST run already starts after Planner has delivered
        # "plan" (readiness requires it), so Planner's conversation is
        # already part of Writer's very first history -- there is no
        # "critic" sender yet, since the loop hasn't fired back once.
        assert writer_histories[0] == [{"role": "assistant", "content": "planner-said"}]
        assert writer_histories[1] == [
            {"role": "assistant", "content": "planner-said"},
            {"role": "assistant", "content": "critic-said-1"},
        ]

    @pytest.mark.asyncio
    async def test_entry_node_with_no_senders_gets_empty_history(self):
        graph = {
            "nodes": {"only": {"name": "Only"}},
            "edges": [],
            "entry_point": "only",
            "exit_points": ["only"],
        }
        captured = []

        async def run_node(node_id, node_config, inputs, predecessor_messages):
            captured.append(predecessor_messages)
            return ("done", [], None, None)

        await run_graph(graph, {"input": "hi"}, run_node, parallel=True)
        assert captured == [[]]
