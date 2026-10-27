"""
Unit tests for cyclic graph support in AgentExecutor.

Tests cover:
- Message history accumulation across loop iterations
- Exit points allowing loop edges (fix for exit-point-blocks-loop-edges)
- Output-path routing via structured output unions
- max_loop_iterations safety limit
- Backward compatibility with single-node and simple DAG agents

Named-inputs contract (claude_plans/named-agent-inputs.md, plan step 4):
`AgentExecutor._execute_graph` / `execute_agent_node` take `inputs: dict`
instead of a single input string, and `_run_sub_agent` (mocked below)
receives that same per-node named-inputs dict — a legacy edge with no
target_input delivers under the "input" key.

Run with: pytest tests/test_agent_executor_loops.py -v
"""

import pytest
import asyncio
import sys
import os
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, MagicMock, patch, AsyncMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from executors.agent_executor import AgentExecutor
from runners.agent_runner import _build_output_path_types


# ============================================================================
# Helpers
# ============================================================================

class MockSession:
    """Mock database session."""
    def __init__(self):
        self.added = []
        self.committed = False

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def query(self, *args, **kwargs):
        return MockQuery()

    def refresh(self, obj):
        pass


class MockQuery:
    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return None

    def all(self):
        return []


def run_async(coro):
    """Run an async coroutine in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


def builts_for(graph_config):
    """built_agents map for mocked _run_sub_agent — values are never inspected."""
    return {node_id: Mock() for node_id in graph_config["nodes"]}


# ============================================================================
# Test _build_output_path_types
# ============================================================================

class TestBuildOutputPathTypes:
    def test_creates_union_from_two_paths(self):
        output_paths = {
            "revise": "Draft needs work",
            "approve": "Draft is ready",
        }
        union_type, class_to_path, models = _build_output_path_types(output_paths)

        assert "Revise" in class_to_path
        assert "Approve" in class_to_path
        assert class_to_path["Revise"] == "revise"
        assert class_to_path["Approve"] == "approve"
        assert "revise" in models
        assert "approve" in models

    def test_models_have_content_field(self):
        output_paths = {"revise": "Needs work"}
        _, _, models = _build_output_path_types(output_paths)

        ReviseModel = models["revise"]
        instance = ReviseModel(content="Fix the intro")
        assert instance.content == "Fix the intro"

    def test_single_path_returns_model_not_union(self):
        output_paths = {"done": "Task complete"}
        union_type, class_to_path, models = _build_output_path_types(output_paths)

        # Single path: union_type is the model itself, not a Union
        assert union_type is models["done"]

    def test_model_docstrings_from_descriptions(self):
        output_paths = {"revise": "Choose when draft needs improvement"}
        _, _, models = _build_output_path_types(output_paths)

        assert models["revise"].__doc__ == "Choose when draft needs improvement"


# ============================================================================
# Test _execute_graph — loop mechanics
# ============================================================================

class TestExecuteGraphLoops:
    """Test the BFS loop mechanics by mocking _run_sub_agent."""

    def _make_executor(self):
        session = MockSession()
        executor = AgentExecutor.__new__(AgentExecutor)
        executor.session = session
        executor.enable_retry = False
        executor.retry_config = None
        return executor

    @pytest.mark.asyncio
    async def test_simple_dag_backward_compatible(self):
        """Two-node DAG (no loops) should work identically to before, with
        named inputs: each node's input is now a dict (legacy edges land
        under the "input" key)."""
        executor = self._make_executor()

        call_log = []

        async def mock_run_sub_agent(node_id, node_config, inputs, built, predecessor_messages=None):
            call_log.append((node_id, inputs, predecessor_messages))
            if node_id == "writer":
                return ("Draft text", [], None)
            else:
                return ("Summary of draft", [], None)

        executor._run_sub_agent = mock_run_sub_agent

        graph_config = {
            "nodes": {
                "writer": {"agent_type": "pydanticai", "name": "Writer"},
                "summarizer": {"agent_type": "pydanticai", "name": "Summarizer"},
            },
            "edges": [{"from_node": "writer", "to_node": "summarizer", "is_loop": False}],
            "entry_point": "writer",
            "exit_points": ["summarizer"],
        }

        result = await executor._execute_graph(
            graph_config, {"input": "Write something"}, None, builts_for(graph_config))

        assert result["status"] == "completed"
        assert result["result"] == "Summary of draft"
        assert len(call_log) == 2
        # Writer gets the original external inputs dict, no history
        assert call_log[0] == ("writer", {"input": "Write something"}, None)
        # Summarizer gets Writer's output under the legacy "input" key, no history
        assert call_log[1] == ("summarizer", {"input": "Draft text"}, None)

    @pytest.mark.asyncio
    async def test_exit_point_follows_loop_edges(self):
        """Exit points should follow loop edges back, not just stop.
        With the old code, exit points skipped ALL successors including loops."""
        executor = self._make_executor()

        call_count = {"writer": 0, "reviewer": 0}

        async def mock_run_sub_agent(node_id, node_config, inputs, built, predecessor_messages=None):
            call_count[node_id] += 1
            return (f"Output from {node_id}", [], None)

        executor._run_sub_agent = mock_run_sub_agent

        graph_config = {
            "nodes": {
                "writer": {"agent_type": "pydanticai", "name": "Writer"},
                "reviewer": {"agent_type": "pydanticai", "name": "Reviewer"},
            },
            "edges": [
                {"from_node": "writer", "to_node": "reviewer", "is_loop": False},
                {"from_node": "reviewer", "to_node": "writer", "is_loop": True},
            ],
            "entry_point": "writer",
            "exit_points": ["reviewer"],  # Reviewer IS an exit point
            "max_loop_iterations": 1,
        }

        result = await executor._execute_graph(
            graph_config, {"input": "Write a poem"}, None, builts_for(graph_config))

        assert result["status"] == "completed"
        # With max_loop_iterations=1: writer→reviewer (loop back)→writer→reviewer (limit hit)
        # Reviewer called 2 times proves the loop edge fired from an exit point
        assert call_count["reviewer"] == 2
        assert call_count["writer"] == 2

    @pytest.mark.asyncio
    async def test_max_loop_iterations_stops_runaway(self):
        """Loops should stop after max_loop_iterations even if agent keeps looping."""
        executor = self._make_executor()

        call_count = {"total": 0}

        async def mock_run_sub_agent(node_id, node_config, inputs, built, predecessor_messages=None):
            call_count["total"] += 1
            return (f"Output from {node_id}", [], None)

        executor._run_sub_agent = mock_run_sub_agent

        graph_config = {
            "nodes": {
                "a": {"agent_type": "pydanticai", "name": "A"},
                "b": {"agent_type": "pydanticai", "name": "B"},
            },
            "edges": [
                {"from_node": "a", "to_node": "b", "is_loop": False},
                {"from_node": "b", "to_node": "a", "is_loop": True},
            ],
            "entry_point": "a",
            "exit_points": ["b"],
            "max_loop_iterations": 2,
        }

        result = await executor._execute_graph(
            graph_config, {"input": "Start"}, None, builts_for(graph_config))

        # a→b (loop back) → a→b (loop back) → a→b (limit reached, stop)
        # That's 3 iterations of a + 3 of b = 6 total, but b loops back only 2 times
        # a runs: initial + 2 loop-backs = 3 times
        # b runs: 3 times
        assert result["status"] == "completed"
        assert call_count["total"] <= 6  # Safety: shouldn't exceed this

    @pytest.mark.asyncio
    async def test_predecessor_messages_passed_on_reentry(self):
        """When a node re-enters via loop, it should receive its previous predecessor_messages."""
        executor = self._make_executor()

        histories_received = []

        async def mock_run_sub_agent(node_id, node_config, inputs, built, predecessor_messages=None):
            if node_id == "writer":
                histories_received.append(predecessor_messages)
                fake_messages = [{"role": "user", "content": inputs}]
                if predecessor_messages:
                    fake_messages = predecessor_messages + fake_messages
                return ("Draft", fake_messages, None)
            else:
                # A real run always returns non-empty all_messages()
                history = (predecessor_messages or []) + [{"role": "assistant", "content": "Feedback"}]
                return ("Feedback", history, None)

        executor._run_sub_agent = mock_run_sub_agent

        graph_config = {
            "nodes": {
                "writer": {"agent_type": "pydanticai", "name": "Writer"},
                "reviewer": {"agent_type": "pydanticai", "name": "Reviewer"},
            },
            "edges": [
                {"from_node": "writer", "to_node": "reviewer", "is_loop": False},
                {"from_node": "reviewer", "to_node": "writer", "is_loop": True},
            ],
            "entry_point": "writer",
            "exit_points": ["reviewer"],
            "max_loop_iterations": 1,
        }

        await executor._execute_graph(
            graph_config, {"input": "Write a poem"}, None, builts_for(graph_config))

        # First call: no history
        assert histories_received[0] is None
        # Second call (after loop): should have history from first run
        assert len(histories_received) >= 2
        assert histories_received[1] is not None
        assert len(histories_received[1]) > 0

    @pytest.mark.asyncio
    async def test_output_path_routing(self):
        """Edges with output_path should only fire when the node chooses that path."""
        executor = self._make_executor()

        visited = []

        async def mock_run_sub_agent(node_id, node_config, inputs, built, predecessor_messages=None):
            visited.append(node_id)
            if node_id == "reviewer":
                # Choose "approve" path
                return ("Looks good!", [], "approve")
            return (f"Output from {node_id}", [], None)

        executor._run_sub_agent = mock_run_sub_agent

        graph_config = {
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
            "max_loop_iterations": 3,
        }

        result = await executor._execute_graph(
            graph_config, {"input": "Write a blog"}, None, builts_for(graph_config))

        # Reviewer chose "approve" → should go to formatter, NOT loop back to writer
        assert result["status"] == "completed"
        assert visited == ["writer", "reviewer", "formatter"]

    @pytest.mark.asyncio
    async def test_output_path_revise_loops_back(self):
        """When the reviewer chooses 'revise', it should loop back to writer."""
        executor = self._make_executor()

        visited = []
        reviewer_calls = 0

        async def mock_run_sub_agent(node_id, node_config, inputs, built, predecessor_messages=None):
            nonlocal reviewer_calls
            visited.append(node_id)
            if node_id == "reviewer":
                reviewer_calls += 1
                if reviewer_calls >= 2:
                    return ("Final approval", [], "approve")
                return ("Needs work", [], "revise")
            return (f"Output from {node_id}", [], None)

        executor._run_sub_agent = mock_run_sub_agent

        graph_config = {
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

        result = await executor._execute_graph(
            graph_config, {"input": "Write a blog"}, None, builts_for(graph_config))

        # writer → reviewer (revise) → writer → reviewer (approve) → formatter
        assert result["status"] == "completed"
        assert visited == ["writer", "reviewer", "writer", "reviewer", "formatter"]

    @pytest.mark.asyncio
    async def test_single_node_backward_compatible(self):
        """Single-node agent should work identically to before."""
        executor = self._make_executor()

        async def mock_run_sub_agent(node_id, node_config, inputs, built, predecessor_messages=None):
            return ("Hello world", [], None)

        executor._run_sub_agent = mock_run_sub_agent

        graph_config = {
            "nodes": {"main": {"agent_type": "pydanticai", "name": "Main"}},
            "edges": [],
            "entry_point": "main",
            "exit_points": ["main"],
        }

        result = await executor._execute_graph(
            graph_config, {"input": "Hi"}, None, builts_for(graph_config))

        assert result["status"] == "completed"
        assert result["result"] == "Hello world"

    @pytest.mark.asyncio
    async def test_named_inputs_both_delivered_to_single_node(self):
        """Two edges into one node, each targeting a different named input,
        must both arrive — a dict, not the old single string."""
        executor = self._make_executor()

        received = []

        graph_config = {
            "nodes": {
                "planner": {"agent_type": "pydanticai", "name": "Planner"},
                "researcher": {"agent_type": "pydanticai", "name": "Researcher"},
                "writer": {"agent_type": "pydanticai", "name": "Writer"},
            },
            "edges": [
                {"from_node": "planner", "to_node": "writer", "target_input": "plan"},
                {"from_node": "planner", "to_node": "researcher"},
                {"from_node": "researcher", "to_node": "writer", "target_input": "research"},
            ],
            "entry_point": "planner",
            "exit_points": ["writer"],
        }

        async def run_with_outputs(node_id, node_config, inputs, built, predecessor_messages=None):
            received.append((node_id, dict(inputs)))
            if node_id == "planner":
                return ("PLAN", [], None)
            if node_id == "researcher":
                return (f"RESEARCH({inputs.get('input')})", [], None)
            return (f"combined:{inputs.get('plan')}+{inputs.get('research')}", [], None)

        executor._run_sub_agent = run_with_outputs

        result = await executor._execute_graph(
            graph_config, {"input": "go"}, None, builts_for(graph_config))

        assert result["status"] == "completed"
        writer_calls = [inp for nid, inp in received if nid == "writer"]
        assert len(writer_calls) == 1
        assert writer_calls[0] == {"plan": "PLAN", "research": "RESEARCH(PLAN)"}
        assert result["result"] == "combined:PLAN+RESEARCH(PLAN)"


# ============================================================================
# record_node session-per-write (parallel-agent-execution plan, step 2 /
# decision 6): record_node must open its OWN DatabaseManager session for each
# write, closed when done, rather than reusing AgentExecutor's shared
# self.session — the same rule flow_runner._execute_node_async follows
# because a shared Session isn't safe once sub-agents run under gather.
# ============================================================================

class _PoisonSession:
    """Stands in for AgentExecutor's shared self.session. Any use at all is a
    bug: record_node must never touch it."""

    def add(self, *a, **k):
        raise AssertionError("record_node used the shared executor session (add)")

    def commit(self, *a, **k):
        raise AssertionError("record_node used the shared executor session (commit)")

    def refresh(self, *a, **k):
        raise AssertionError("record_node used the shared executor session (refresh)")

    def query(self, *a, **k):
        raise AssertionError("record_node used the shared executor session (query)")

    def close(self):
        pass


class TestRecordNodeSessionPerWrite:
    @pytest.fixture
    def parent_row(self, tmp_path, monkeypatch):
        """A real per-test SQLite DB (same pattern as test_flow_runner_agents.py)
        with a parent Execution row already committed, so record_node's writes
        have a valid parent_id to attach to."""
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
        from database.database_setup import DatabaseManager, User, Execution

        manager = DatabaseManager()
        manager.create_tables()
        session = manager.get_session()
        user = User(username="tester", email="t@t.t", password_hash="x")
        session.add(user)
        session.commit()
        session.refresh(user)

        parent = Execution(user_id=user.id, execution_type="agent", name="Parent", status="running")
        session.add(parent)
        session.commit()
        session.refresh(parent)

        parent_id, user_id = parent.id, user.id
        session.close()
        yield parent_id, user_id

    @pytest.mark.asyncio
    async def test_record_node_opens_and_closes_its_own_session_per_write(self, parent_row, monkeypatch):
        parent_id, user_id = parent_row
        import executors.agent_executor as agent_executor_module
        from database.database_setup import DatabaseManager as RealDatabaseManager, Execution

        opened = []

        class TrackingDatabaseManager:
            """Fake DatabaseManager that tracks every session it hands out so
            the test can prove record_node opens one per write and closes it,
            instead of sharing a single long-lived Session (unsafe under
            concurrent sub-agent execution — see the plan's decision 6)."""

            def __init__(self):
                self._real = RealDatabaseManager()

            def get_session(self):
                session = self._real.get_session()
                record = {"session": session, "closed": False}
                real_close = session.close

                def close():
                    record["closed"] = True
                    real_close()

                session.close = close
                opened.append(record)
                return session

        # Patched with raising=False: agent_executor.py does not import
        # DatabaseManager at all today (it only ever uses the constructor's
        # shared self.session), so this name doesn't exist yet — that's the
        # point of this red test.
        monkeypatch.setattr(agent_executor_module, "DatabaseManager", TrackingDatabaseManager, raising=False)

        executor = AgentExecutor.__new__(AgentExecutor)
        executor.session = _PoisonSession()  # must never be touched by record_node
        executor.enable_retry = False
        executor.retry_config = None

        async def mock_run_sub_agent(node_id, node_config, inputs, built, predecessor_messages=None):
            if node_id == "writer":
                return ("Draft text", [], None)
            return ("Summary of draft", [], None)

        executor._run_sub_agent = mock_run_sub_agent

        graph_config = {
            "nodes": {
                "writer": {"agent_type": "pydanticai", "name": "Writer"},
                "summarizer": {"agent_type": "pydanticai", "name": "Summarizer"},
            },
            "edges": [{"from_node": "writer", "to_node": "summarizer", "is_loop": False}],
            "entry_point": "writer",
            "exit_points": ["summarizer"],
        }

        execution_stub = Mock(id=parent_id, user_id=user_id)

        result = await executor._execute_graph(
            graph_config, {"input": "Write something"}, execution_stub, builts_for(graph_config))

        assert result["status"] == "completed"

        # Two sub-nodes -> two writes -> two DISTINCT per-write sessions from
        # DatabaseManager (never the executor's shared self.session), each
        # closed once its write is done.
        assert len(opened) == 2, "record_node should open one DatabaseManager session per write"
        assert opened[0]["session"] is not opened[1]["session"]
        assert all(record["closed"] for record in opened), "each per-write session must be closed"

        # Independently verify the rows actually landed, via a fresh session.
        check = RealDatabaseManager().get_session()
        try:
            rows = (
                check.query(Execution)
                .filter(Execution.parent_id == parent_id, Execution.execution_type == "agent")
                .order_by(Execution.sequence)
                .all()
            )
            assert [r.node_id for r in rows] == ["writer", "summarizer"]
            assert rows[0].output_data == {"result": "Draft text"}
            assert rows[1].output_data == {"result": "Summary of draft"}
        finally:
            check.close()
