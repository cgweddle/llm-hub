"""
Unit tests for cyclic graph support in AgentExecutor.

Tests cover:
- Message history accumulation across loop iterations
- Exit points allowing loop edges (fix for exit-point-blocks-loop-edges)
- Output-path routing via structured output unions
- max_loop_iterations safety limit
- Backward compatibility with single-node and simple DAG agents

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

from executors.agent_executor import AgentExecutor, _build_output_path_types


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
        """Two-node DAG (no loops) should work identically to before."""
        executor = self._make_executor()

        call_log = []

        async def mock_run_sub_agent(node_id, node_config, node_input, message_history=None):
            call_log.append((node_id, node_input, message_history))
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

        result = await executor._execute_graph(graph_config, "Write something", None)

        assert result["status"] == "completed"
        assert result["result"] == "Summary of draft"
        assert len(call_log) == 2
        # Writer gets original input, no history
        assert call_log[0] == ("writer", "Write something", None)
        # Summarizer gets Writer's output, no history
        assert call_log[1] == ("summarizer", "Draft text", None)

    @pytest.mark.asyncio
    async def test_exit_point_follows_loop_edges(self):
        """Exit points should follow loop edges back, not just stop.
        With the old code, exit points skipped ALL successors including loops."""
        executor = self._make_executor()

        call_count = {"writer": 0, "reviewer": 0}

        async def mock_run_sub_agent(node_id, node_config, node_input, message_history=None):
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

        result = await executor._execute_graph(graph_config, "Write a poem", None)

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

        async def mock_run_sub_agent(node_id, node_config, node_input, message_history=None):
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

        result = await executor._execute_graph(graph_config, "Start", None)

        # a→b (loop back) → a→b (loop back) → a→b (limit reached, stop)
        # That's 3 iterations of a + 3 of b = 6 total, but b loops back only 2 times
        # a runs: initial + 2 loop-backs = 3 times
        # b runs: 3 times
        assert result["status"] == "completed"
        assert call_count["total"] <= 6  # Safety: shouldn't exceed this

    @pytest.mark.asyncio
    async def test_message_history_passed_on_reentry(self):
        """When a node re-enters via loop, it should receive its previous message_history."""
        executor = self._make_executor()

        histories_received = []

        async def mock_run_sub_agent(node_id, node_config, node_input, message_history=None):
            if node_id == "writer":
                histories_received.append(message_history)
                fake_messages = [{"role": "user", "content": node_input}]
                if message_history:
                    fake_messages = message_history + fake_messages
                return ("Draft", fake_messages, None)
            else:
                return ("Feedback", [], None)

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

        await executor._execute_graph(graph_config, "Write a poem", None)

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

        async def mock_run_sub_agent(node_id, node_config, node_input, message_history=None):
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

        result = await executor._execute_graph(graph_config, "Write a blog", None)

        # Reviewer chose "approve" → should go to formatter, NOT loop back to writer
        assert result["status"] == "completed"
        assert visited == ["writer", "reviewer", "formatter"]

    @pytest.mark.asyncio
    async def test_output_path_revise_loops_back(self):
        """When the reviewer chooses 'revise', it should loop back to writer."""
        executor = self._make_executor()

        visited = []
        reviewer_calls = 0

        async def mock_run_sub_agent(node_id, node_config, node_input, message_history=None):
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

        result = await executor._execute_graph(graph_config, "Write a blog", None)

        # writer → reviewer (revise) → writer → reviewer (approve) → formatter
        assert result["status"] == "completed"
        assert visited == ["writer", "reviewer", "writer", "reviewer", "formatter"]

    @pytest.mark.asyncio
    async def test_single_node_backward_compatible(self):
        """Single-node agent should work identically to before."""
        executor = self._make_executor()

        async def mock_run_sub_agent(node_id, node_config, node_input, message_history=None):
            return ("Hello world", [], None)

        executor._run_sub_agent = mock_run_sub_agent

        graph_config = {
            "nodes": {"main": {"agent_type": "pydanticai", "name": "Main"}},
            "edges": [],
            "entry_point": "main",
            "exit_points": ["main"],
        }

        result = await executor._execute_graph(graph_config, "Hi", None)

        assert result["status"] == "completed"
        assert result["result"] == "Hello world"
