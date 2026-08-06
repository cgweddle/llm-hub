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
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from sqlalchemy.orm import Session

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.database import create_execution, update_execution
from database.database_setup import Execution
from utils.prompt_template import resolve_user_prompt_template
from runners.agent_runner import BuiltAgent

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
        input_text: str,
        built_agents: Dict[str, BuiltAgent],
        parent_execution: Optional[Execution] = None
    ) -> str:
        """
        Execute an agent and return its text output.
        Entry point for FlowRunner — records structural node under parent_execution.
        Internal tracing is handled by LangFuse automatically.

        Args:
            graph_config: The agent's graph_config, captured at flow prepare time
            input_text: Input text
            built_agents: sub-node id → BuiltAgent, compiled at flow prepare time
            parent_execution: Parent Execution record from FlowRunner

        Returns:
            Agent text output as string
        """
        if not graph_config:
            raise ValueError("Agent has no graph_config")

        result = await self._execute_graph(
            graph_config, input_text, execution=parent_execution, built_agents=built_agents
        )
        return str(result.get("result", ""))

    async def _execute_graph(
        self,
        graph_config: Dict[str, Any],
        input_data: str,
        execution: Optional[Execution],
        built_agents: Dict[str, BuiltAgent],
        max_loop_iterations: int = None
    ) -> Dict[str, Any]:
        """
        Execute an agent graph via BFS traversal with cycle support.
        Records sub-agent nodes as child Execution rows for structural tracking.
        Internal tool call details are captured by LangFuse.
        """
        nodes_config = graph_config.get("nodes", {})
        edges_config = graph_config.get("edges", [])
        entry_point = graph_config.get("entry_point")
        exit_points = graph_config.get("exit_points", [])

        if max_loop_iterations is None:
            max_loop_iterations = graph_config.get("max_loop_iterations", 5)

        if not entry_point or not nodes_config:
            raise ValueError("Invalid graph_config: missing entry_point or nodes")

        logger.info(
            f"Graph execution: {len(nodes_config)} nodes, {len(edges_config)} edges, "
            f"entry: {entry_point}, exits: {exit_points}"
        )

        # Build adjacency list and edge metadata
        adjacency: Dict[str, list] = {node_id: [] for node_id in nodes_config}
        loop_edges: set = set()
        edge_output_paths: Dict[tuple, str] = {}  # (from, to) → output_path name
        for edge in edges_config:
            from_node = edge.get("from_node")
            to_node = edge.get("to_node")
            is_loop = edge.get("is_loop", False)
            output_path = edge.get("output_path")
            if from_node and to_node:
                adjacency[from_node].append(to_node)
                if is_loop:
                    loop_edges.add((from_node, to_node))
                if output_path:
                    edge_output_paths[(from_node, to_node)] = output_path

        # Track execution
        execution_trace = []
        node_outputs: Dict[str, Any] = {}
        messages: List = []  # Accumulating PydanticAI message history across all nodes
        loop_counts: Dict[tuple, int] = {}
        step_sequence = 0

        # BFS traversal
        current_nodes = [entry_point]
        current_input = input_data

        while current_nodes:
            next_nodes = []

            for node_id in current_nodes:
                if node_id not in nodes_config:
                    logger.warning(f"Node {node_id} not found in config, skipping")
                    continue

                # Determine input for this node
                node_input = node_outputs.get(node_id + "_input", current_input)

                built = built_agents.get(node_id)
                if built is None:
                    raise ValueError(f"No compiled agent for sub-node '{node_id}'")

                # Execute the sub-agent node
                chosen_path = None
                # Read by the except handler below even when LangFuse is off,
                # so it must be bound outside the LANGFUSE_AVAILABLE branch.
                _captured_trace_id = None
                try:
                    trace_id = None
                    predecessor_msgs = messages if messages else None

                    if LANGFUSE_AVAILABLE and langfuse_observe:
                        # Capture trace ID even if the agent fails
                        @langfuse_observe(name=nodes_config[node_id].get("name", node_id))
                        async def _observed_run():
                            nonlocal _captured_trace_id
                            _captured_trace_id = langfuse_client.get_current_trace_id()
                            result = await self._run_sub_agent(node_id, nodes_config[node_id], node_input, built, predecessor_messages=predecessor_msgs)
                            return result

                        output, node_result_messages, chosen_path = await _observed_run()
                        trace_id = _captured_trace_id
                        langfuse_client.flush()
                    else:
                        output, node_result_messages, chosen_path = await self._run_sub_agent(node_id, nodes_config[node_id], node_input, built, predecessor_messages=predecessor_msgs)
                    # Apply return behavior for the chosen path
                    if chosen_path:
                        node_output_paths = nodes_config[node_id].get("output_paths", {})
                        path_config = node_output_paths.get(chosen_path, {})
                        if isinstance(path_config, dict) and path_config.get("return_behavior") == "previous_output":
                            output = node_input

                    node_outputs[node_id] = output
                    if node_result_messages is not None:
                        messages = node_result_messages

                    execution_trace.append({
                        "node": node_id,
                        "name": nodes_config[node_id].get("name", node_id),
                        "agent_type": nodes_config[node_id].get("agent_type"),
                        "input": node_input[:200] + "..." if len(str(node_input)) > 200 else node_input,
                        "output": output[:200] + "..." if len(str(output)) > 200 else output,
                        "status": "completed"
                    })

                    # Record structural node execution for multi-node agents only.
                    # Single-node agents already have a parent execution representing them.
                    is_multi_node = len(nodes_config) > 1
                    if execution and is_multi_node:
                        create_execution(
                            self.session,
                            parent_id=execution.id,
                            user_id=execution.user_id,
                            execution_type='agent',
                            node_id=node_id,
                            name=nodes_config[node_id].get("name", node_id),
                            sequence=step_sequence,
                            input_data={"input": str(node_input)[:2000]},
                            output_data={"result": str(output)[:2000]},
                            status='completed',
                            started_at=datetime.now(),
                            completed_at=datetime.now(),
                            execution_metadata={
                                "agent_type": nodes_config[node_id].get("agent_type"),
                            },
                            langfuse_trace_id=trace_id
                        )
                        step_sequence += 1
                    elif execution and trace_id:
                        update_execution(self.session, execution.id, langfuse_trace_id=trace_id)

                    logger.debug(f"Sub-agent {node_id} completed")

                except Exception as e:
                    # Store trace ID even on failure so the LangFuse trace is accessible
                    if execution and _captured_trace_id:
                        langfuse_client.flush()
                        update_execution(self.session, execution.id, langfuse_trace_id=_captured_trace_id)

                    logger.error(f"Sub-agent {node_id} failed: {e}")
                    raise RuntimeError(
                        f"Agent node '{nodes_config[node_id].get('name', node_id)}' failed: {e}"
                    ) from e

                # Traverse successor nodes
                # Exit points skip forward edges but still follow loop edges
                is_exit = node_id in exit_points
                for successor in adjacency.get(node_id, []):
                    edge_key = (node_id, successor)
                    is_loop_edge = edge_key in loop_edges

                    # Exit points only follow loop edges back, not forward edges
                    if is_exit and not is_loop_edge:
                        continue

                    # Filter by output_path when the node chose a specific path
                    edge_path = edge_output_paths.get(edge_key)
                    if chosen_path is not None and edge_path is not None:
                        if edge_path != chosen_path:
                            continue

                    # Handle loop edges with iteration limit
                    if is_loop_edge:
                        loop_counts[edge_key] = loop_counts.get(edge_key, 0) + 1
                        if loop_counts[edge_key] > max_loop_iterations:
                            logger.warning(
                                f"Loop limit reached for edge {node_id} -> {successor}, skipping"
                            )
                            continue

                    # Pass output as input to successor
                    node_outputs[successor + "_input"] = output
                    if successor not in next_nodes:
                        next_nodes.append(successor)

            current_nodes = next_nodes

        # Collect final outputs from exit points
        final_outputs = {
            node_id: node_outputs.get(node_id, "")
            for node_id in exit_points
            if node_id in node_outputs
        }

        main_result = final_outputs.get(exit_points[0], "") if exit_points else ""

        return {
            "execution_id": execution.id if execution else None,
            "status": "completed",
            "result": main_result,
            "all_outputs": final_outputs,
            "execution_trace": execution_trace,
            "messages": execution_trace,
            "sub_agent_count": len(nodes_config)
        }

    async def _run_sub_agent(
        self, node_id: str, node_config: Dict, node_input: str,
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

        return await self._run_pydanticai_node(node_config, node_input, built, predecessor_messages=predecessor_messages)

    @staticmethod
    def _apply_user_prompt(node_config: Dict, node_input: str,
                           predecessor_messages: Optional[List] = None) -> str:
        """Resolve user_prompt templates into the final user message.

        If the user_prompt contains {input}, it replaces it with node_input
        and the resolved template IS the user message. If no user_prompt is
        set, node_input is used directly."""
        user_prompt = node_config.get("user_prompt", "").strip()
        if user_prompt:
            return resolve_user_prompt_template(
                user_prompt, node_input, predecessor_messages
            )
        return node_input

    async def _run_pydanticai_node(
        self, node_config: Dict, node_input: str,
        built: BuiltAgent,
        predecessor_messages: Optional[List] = None,
    ) -> Tuple[str, List, Optional[str]]:
        """Run a pre-built PydanticAI sub-agent (compiled by runners.agent_runner).
        Returns (output_text, all_messages, chosen_path).
        chosen_path is None for nodes without output_paths.
        LangFuse auto-captures internal tool calls and LLM I/O."""
        node_input = self._apply_user_prompt(node_config, node_input, predecessor_messages)

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
