"""
Unified Agent Executor
Executes all agents via graph traversal (BFS with cycle support).

Every agent — simple or complex — is represented as a graph_config with
nodes, edges, entry_point, and exit_points. A simple agent is just a
single-node graph.

Records execution trees in the database:
- Standalone agent runs create a top-level Execution(type='agent')
- Agent nodes within flows receive a parent_execution from FlowExecutor
- Internal tool call tracing is handled by LangFuse (automatic instrumentation)

Features:
- Unified graph-based execution (no agent_type routing)
- Execution record management via self-referencing execution tree
- LangFuse integration for internal agent telemetry
- Runs pre-built agents (runners.agent_runner.BuiltAgent) — compilation
  happens at flow prepare time, execution only runs them
- Automatic retry with exponential backoff for transient failures
- Error handling and logging
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from sqlalchemy.orm import Session

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.database import create_execution, update_execution
from database.database_setup import DatabaseManager, Execution
from utils.prompt_template import resolve_user_prompt_template
from runners.agent_runner import BuiltAgent
from executors.agent_graph_engine import run_graph

# Import retry utilities
try:
    from utils.retry import (
        RetryConfig,
        retry_async,
        DEFAULT_LLM_RETRY_CONFIG,
    )
    RETRY_AVAILABLE = True
except ImportError:
    RETRY_AVAILABLE = False
    RetryConfig = None
    DEFAULT_LLM_RETRY_CONFIG = None

from observability.langfuse_tracing import (
    LANGFUSE_AVAILABLE,
    langfuse_client,
    langfuse_observe,
)

logger = logging.getLogger(__name__)


@contextmanager
def _scoped_session():
    """A short-lived DB session for a single write. Sub-agent nodes under
    run_graph's parallel execution can't share AgentExecutor's long-lived
    self.session, so each write (record_node, trace-id-on-failure) gets its
    own session, closed immediately after."""
    session = DatabaseManager().get_session()
    try:
        yield session
    finally:
        session.close()

# Instrumentation is execution-side: this module runs agents, so it turns on
# span emission for all PydanticAI agent runs in this process.
if LANGFUSE_AVAILABLE:
    from pydantic_ai import Agent as _PydanticAIAgent
    _PydanticAIAgent.instrument_all()
    logger.info("LangFuse instrumentation enabled for PydanticAI agents")
else:
    logger.info("LangFuse not available — internal agent tracing disabled")


class AgentExecutor:
    """
    Unified agent execution service.

    All agents are executed via graph traversal. The graph_config on
    every Agent record defines nodes (sub-agent configs), edges
    (connections with optional loop markers), entry_point, and
    exit_points.

    For single-node agents, BFS naturally runs one node and returns.

    Internal tool call tracing is handled by LangFuse — every PydanticAI
    agent.run() call is automatically instrumented. The execution tree in
    the database records structural hierarchy (which agents/tools ran),
    while LangFuse records the detailed internal conversation.
    """

    def __init__(
        self,
        session: Session,
        retry_config: Optional["RetryConfig"] = None,
        enable_retry: bool = True,
    ):
        self.session = session

        # Configure retry behavior
        self.enable_retry = enable_retry and RETRY_AVAILABLE
        if self.enable_retry:
            self.retry_config = retry_config or DEFAULT_LLM_RETRY_CONFIG
            logger.debug(
                f"Retry enabled: max_retries={self.retry_config.max_retries}, "
                f"base_delay={self.retry_config.base_delay}s"
            )
        else:
            self.retry_config = None
            if enable_retry and not RETRY_AVAILABLE:
                logger.warning("Retry requested but retry utilities not available")

    async def execute_agent_node(
        self,
        graph_config: Dict[str, Any],
        inputs: Dict[str, Any],
        built_agents: Dict[str, BuiltAgent],
        parent_execution: Optional[Execution] = None
    ) -> str:
        """
        Execute an agent and return its text output.
        Entry point for FlowRunner — records structural node under parent_execution.
        Internal tracing is handled by LangFuse automatically.

        Args:
            graph_config: The agent's graph_config, captured at flow prepare time
            inputs: Named input values for the entry sub-agent
            built_agents: sub-node id → BuiltAgent, compiled at flow prepare time
            parent_execution: Parent Execution record from FlowRunner

        Returns:
            Agent text output as string
        """
        if not graph_config:
            raise ValueError("Agent has no graph_config")

        result = await self._execute_graph(
            graph_config, inputs, execution=parent_execution, built_agents=built_agents
        )
        return str(result.get("result", ""))

    async def _execute_graph(
        self,
        graph_config: Dict[str, Any],
        inputs: Dict[str, Any],
        execution: Optional[Execution],
        built_agents: Dict[str, BuiltAgent],
        max_loop_iterations: int = None
    ) -> Dict[str, Any]:
        """
        Execute an agent graph via BFS traversal with cycle support.

        Traversal lives in the pure agent_graph_engine.run_graph; this method
        supplies the effectful callables it injects. run_node wraps each run
        with LangFuse trace capture + retry (delegating to _run_sub_agent);
        record_node writes structural Execution rows for multi-node agents
        (single-node agents reuse the parent execution). Internal tool call
        details are captured by LangFuse.
        """
        is_multi_node = len(graph_config.get("nodes", {})) > 1
        step = {"seq": 0}

        async def run_node(node_id, node_config, node_inputs, predecessor_messages):
            built = built_agents.get(node_id)
            if built is None:
                raise ValueError(f"No compiled agent for sub-node '{node_id}'")

            predecessor_messages = predecessor_messages or None

            # Bound outside the LANGFUSE branch so the failure handler can read it.
            captured_trace_id = None
            try:
                if LANGFUSE_AVAILABLE and langfuse_observe:
                    # Capture the trace ID even if the agent fails
                    @langfuse_observe(name=node_config.get("name", node_id))
                    async def _observed_run():
                        nonlocal captured_trace_id
                        captured_trace_id = langfuse_client.get_current_trace_id()
                        return await self._run_sub_agent(
                            node_id, node_config, node_inputs, built,
                            predecessor_messages=predecessor_messages,
                        )

                    output, node_messages, chosen_path = await _observed_run()
                    langfuse_client.flush()
                else:
                    output, node_messages, chosen_path = await self._run_sub_agent(
                        node_id, node_config, node_inputs, built,
                        predecessor_messages=predecessor_messages,
                    )
            except Exception:
                # Store trace ID even on failure so the LangFuse trace is accessible
                if execution and captured_trace_id:
                    langfuse_client.flush()
                    with _scoped_session() as session:
                        update_execution(session, execution.id, langfuse_trace_id=captured_trace_id)
                raise

            return output, node_messages, chosen_path, captured_trace_id

        def record_node(node_id, node_config, node_inputs, output, chosen_path, trace_id):
            if execution and is_multi_node:
                with _scoped_session() as session:
                    create_execution(
                        session,
                        parent_id=execution.id,
                        user_id=execution.user_id,
                        execution_type='agent',
                        node_id=node_id,
                        name=node_config.get("name", node_id),
                        sequence=step["seq"],
                        input_data={name: str(value)[:2000] for name, value in node_inputs.items()},
                        output_data={"result": str(output)[:2000]},
                        status='completed',
                        started_at=datetime.now(),
                        completed_at=datetime.now(),
                        execution_metadata={
                            "agent_type": node_config.get("agent_type"),
                        },
                        langfuse_trace_id=trace_id,
                    )
                    step["seq"] += 1
            elif execution and trace_id:
                with _scoped_session() as session:
                    update_execution(session, execution.id, langfuse_trace_id=trace_id)

        result = await run_graph(
            graph_config,
            inputs,
            run_node,
            record_node=record_node,
            max_loop_iterations=max_loop_iterations,
            parallel=True,
        )
        result["execution_id"] = execution.id if execution else None
        return result

    async def _run_sub_agent(
        self, node_id: str, node_config: Dict, inputs: Dict[str, Any],
        built: BuiltAgent,
        predecessor_messages: Optional[List] = None,
    ) -> Tuple[str, Optional[List], Optional[str]]:
        """
        Run the pre-built sub-agent for a single graph node.
        Returns (output_text, messages, chosen_path) where:
        - messages is the PydanticAI message history (all_messages() from the run)
        - chosen_path is the output path name (None if no output_paths configured)
        Internal tracing is handled by LangFuse.
        """
        logger.debug(
            f"Running sub-agent: {node_config.get('name', node_id)} "
            f"(type: {node_config.get('agent_type')})"
        )

        return await self._run_pydanticai_node(node_config, inputs, built, predecessor_messages=predecessor_messages)

    @staticmethod
    def _apply_user_prompt(node_config: Dict, inputs: Dict[str, Any],
                           predecessor_messages: Optional[List] = None) -> str:
        """Resolve a node's user_prompt into the final user message from its
        named inputs."""
        user_prompt = node_config.get("user_prompt", "")
        return resolve_user_prompt_template(user_prompt, inputs, predecessor_messages)

    async def _run_pydanticai_node(
        self, node_config: Dict, inputs: Dict[str, Any],
        built: BuiltAgent,
        predecessor_messages: Optional[List] = None,
    ) -> Tuple[str, List, Optional[str]]:
        """Run a pre-built PydanticAI sub-agent (compiled by runners.agent_runner).
        Returns (output_text, all_messages, chosen_path).
        chosen_path is None for nodes without output_paths.
        LangFuse auto-captures internal tool calls and LLM I/O."""
        node_input = self._apply_user_prompt(node_config, inputs, predecessor_messages)

        # Run with retry if enabled
        async def run():
            return await built.agent.run(node_input)

        if self.enable_retry:
            def on_retry(attempt, exception, delay):
                logger.warning(f"Sub-agent retry {attempt + 1}: {type(exception).__name__}")
            result = await retry_async(run, config=self.retry_config, on_retry=on_retry)
        else:
            result = await run()

        # Extract output and determine chosen path
        result_data = result.output
        chosen_path = None

        if built.class_to_path and result_data is not None:
            # Determine which union member was chosen via isinstance
            class_name = type(result_data).__name__
            chosen_path = built.class_to_path.get(class_name)
            # Extract content from the structured output
            if hasattr(result_data, 'content'):
                output_str = str(result_data.content)
            elif hasattr(result_data, 'model_dump'):
                output_str = str(result_data.model_dump())
            else:
                output_str = str(result_data)
            logger.info(f"Output path chosen: {chosen_path} (class: {class_name})")
        else:
            if hasattr(result_data, 'model_dump'):
                result_data = result_data.model_dump()
            output_str = str(result_data) if result_data else ""

        return (output_str, result.all_messages(), chosen_path)

    def _extract_cost(self, result) -> Optional[Dict[str, Any]]:
        """Extract cost/token usage from PydanticAI result."""
        try:
            if hasattr(result, 'cost'):
                cost_info = result.cost()
                return {
                    "total_tokens": cost_info.total_tokens if hasattr(cost_info, 'total_tokens') else None,
                    "input_tokens": cost_info.request_tokens if hasattr(cost_info, 'request_tokens') else None,
                    "output_tokens": cost_info.response_tokens if hasattr(cost_info, 'response_tokens') else None,
                    "cost_usd": str(cost_info) if cost_info else None
                }
        except Exception as e:
            logger.warning(f"Could not extract cost information: {e}")
            return None
