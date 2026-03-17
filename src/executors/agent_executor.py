"""
Unified Agent Executor
Executes all agents via graph traversal (BFS with cycle support).

Every agent — simple or complex — is represented as a graph_config with
nodes, edges, entry_point, and exit_points. A simple agent is just a
single-node graph.

Features:
- Unified graph-based execution (no agent_type routing)
- Execution record management
- Message history storage
- Streaming support for PydanticAI
- Automatic retry with exponential backoff for transient failures
- Error handling and logging
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, Optional
from sqlalchemy.orm import Session

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.database import get_agent_by_id, get_tool_by_id
from database.database_setup import Execution, Message

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

logger = logging.getLogger(__name__)


class AgentExecutor:
    """
    Unified agent execution service.

    All agents are executed via graph traversal. The graph_config on
    every Agent record defines nodes (sub-agent configs), edges
    (connections with optional loop markers), entry_point, and
    exit_points.

    For single-node agents, BFS naturally runs one node and returns.
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

    async def execute_agent(
        self,
        agent_id: int,
        user_id: int,
        input_data: str,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Execute an agent and store results in the database.

        Args:
            agent_id: ID of the agent to execute
            user_id: ID of the user executing the agent
            input_data: User input/query for the agent
            stream: Whether to stream responses (only for single-node PydanticAI)

        Returns:
            Dict with execution results
        """
        agent_record = get_agent_by_id(self.session, agent_id)
        if not agent_record:
            raise ValueError(f"Agent with ID {agent_id} not found")

        graph_config = agent_record.graph_config
        if not graph_config:
            raise ValueError(f"Agent {agent_id} has no graph_config")

        logger.info(
            f"Executing agent: {agent_record.name} "
            f"(nodes: {len(graph_config.get('nodes', {}))}, user: {user_id}, stream: {stream})"
        )

        # Create execution record
        execution = self._create_execution(agent_id, user_id, input_data)

        try:
            # Handle streaming for single-node agents
            if stream:
                nodes = graph_config.get("nodes", {})
                if len(nodes) == 1:
                    return await self._execute_single_node_stream(
                        graph_config, input_data, execution
                    )
                # Multi-node streaming not supported — fall through to normal execution
                logger.warning("Streaming not supported for multi-node agents, using non-streaming")

            # Unified graph execution
            result = await self._execute_graph(graph_config, input_data, execution)

            self._complete_execution(execution, result)
            logger.info(f"Agent execution completed: {execution.id}")
            return result

        except Exception as e:
            self._fail_execution(execution, str(e))
            logger.error(f"Agent execution failed: {e}")
            raise RuntimeError(f"Agent execution failed: {e}")

    async def execute_agent_node(
        self,
        agent_id: int,
        input_text: str,
        session: Session
    ) -> str:
        """
        Execute an agent and return its text output.
        Lightweight entry point for flow executor — no Execution record created.

        Args:
            agent_id: Agent ID
            input_text: Input text
            session: Database session

        Returns:
            Agent text output as string
        """
        agent = get_agent_by_id(session, agent_id)
        if not agent:
            raise ValueError(f"Agent with ID {agent_id} not found")

        graph_config = agent.graph_config
        if not graph_config:
            raise ValueError(f"Agent {agent_id} has no graph_config")

        result = await self._execute_graph(graph_config, input_text, execution=None)
        return str(result.get("result", ""))

    async def _execute_graph(
        self,
        graph_config: Dict[str, Any],
        input_data: str,
        execution: Optional[Execution],
        max_loop_iterations: int = None
    ) -> Dict[str, Any]:
        """
        Execute an agent graph via BFS traversal with cycle support.

        Works identically for single-node and multi-node graphs.
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

        # Build adjacency list
        adjacency: Dict[str, list] = {node_id: [] for node_id in nodes_config}
        loop_edges: set = set()
        for edge in edges_config:
            from_node = edge.get("from_node")
            to_node = edge.get("to_node")
            is_loop = edge.get("is_loop", False)
            if from_node and to_node:
                adjacency[from_node].append(to_node)
                if is_loop:
                    loop_edges.add((from_node, to_node))

        # Track execution
        execution_trace = []
        node_outputs: Dict[str, Any] = {}
        loop_counts: Dict[tuple, int] = {}

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

                # Execute the sub-agent node
                try:
                    output = await self._run_sub_agent(node_id, nodes_config[node_id], node_input)
                    node_outputs[node_id] = output

                    execution_trace.append({
                        "node": node_id,
                        "name": nodes_config[node_id].get("name", node_id),
                        "agent_type": nodes_config[node_id].get("agent_type"),
                        "input": node_input[:200] + "..." if len(str(node_input)) > 200 else node_input,
                        "output": output[:200] + "..." if len(str(output)) > 200 else output,
                        "status": "completed"
                    })

                    # Store message if execution record exists
                    if execution:
                        self._store_node_message(execution.id, node_id, nodes_config[node_id], node_input, output)

                    logger.debug(f"Sub-agent {node_id} completed")

                except Exception as e:
                    logger.error(f"Sub-agent {node_id} failed: {e}")
                    execution_trace.append({
                        "node": node_id,
                        "name": nodes_config[node_id].get("name", node_id),
                        "input": str(node_input)[:200],
                        "error": str(e),
                        "status": "failed"
                    })
                    continue

                # Skip successor traversal if this is an exit point
                if node_id in exit_points:
                    continue

                # Add successor nodes
                for successor in adjacency.get(node_id, []):
                    edge_key = (node_id, successor)

                    # Handle loop edges with iteration limit
                    if edge_key in loop_edges:
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

    async def _run_sub_agent(self, node_id: str, node_config: Dict, node_input: str) -> str:
        """
        Create and run a sub-agent for a single graph node.
        """
        logger.debug(
            f"Running sub-agent: {node_config.get('name', node_id)} "
            f"(type: {node_config.get('agent_type')})"
        )

        agent_type = node_config.get("agent_type", "pydanticai")

        if agent_type == "react":
            return await self._run_react_node(node_config, node_input)
        else:
            return await self._run_pydanticai_node(node_config, node_input)

    @staticmethod
    def _apply_user_prompt(node_config: Dict, node_input: str) -> str:
        """Prepend user_prompt to node_input when present."""
        user_prompt = node_config.get("user_prompt", "").strip()
        if user_prompt:
            return f"{user_prompt}\n\n{node_input}"
        return node_input

    async def _run_pydanticai_node(self, node_config: Dict, node_input: str) -> str:
        """Run a PydanticAI sub-agent from node config."""
        from pydantic_ai import Agent

        node_input = self._apply_user_prompt(node_config, node_input)

        llm_provider = node_config.get("llm_provider", "")
        model_name = self._resolve_model_name(llm_provider)

        sub_agent = Agent(
            model=model_name,
            system_prompt=node_config.get("system_prompt", "You are a helpful assistant."),
        )

        # Load and register tools if tool_ids are specified
        tool_ids = node_config.get("tool_ids", [])
        if tool_ids:
            self._register_tools_on_agent(sub_agent, tool_ids)

        # Run with retry if enabled
        async def run():
            return await sub_agent.run(node_input)

        if self.enable_retry:
            def on_retry(attempt, exception, delay):
                logger.warning(f"Sub-agent retry {attempt + 1}: {type(exception).__name__}")
            result = await retry_async(run, config=self.retry_config, on_retry=on_retry)
        else:
            result = await run()

        result_data = result.data
        if hasattr(result_data, 'model_dump'):
            result_data = result_data.model_dump()

        return str(result_data) if result_data else ""

    async def _run_react_node(self, node_config: Dict, node_input: str) -> str:
        """Run a React (Google ADK) sub-agent from node config."""
        from factories.agent_factory import ReactAgentFactory, AgentConfig, setup_llm_environment, DatabaseToolLoader
        from utils import get_llm_config_by_name

        node_input = self._apply_user_prompt(node_config, node_input)

        llm_provider = node_config.get("llm_provider", "")
        llm_config = get_llm_config_by_name(llm_provider)
        if not llm_config:
            raise ValueError(f"LLM config '{llm_provider}' not found")

        llm_config['config_name'] = llm_provider
        model = setup_llm_environment(llm_config)

        tool_loader = DatabaseToolLoader(self.session)
        for tool_id in node_config.get("tool_ids", []):
            try:
                tool_loader.load_tool(tool_id)
            except Exception as e:
                logger.warning(f"Failed to load tool {tool_id}: {e}")

        from factories.agent_factory import ReActAgent
        agent = ReActAgent(
            name=node_config.get("name", "Agent"),
            model=model,
            tool_loader=tool_loader,
            agent_description=node_config.get("system_prompt", "You are a helpful assistant."),
        )

        result = await agent.run_async(node_input)
        return result.get("answer", "")

    def _register_tools_on_agent(self, agent, tool_ids):
        """Register database tools on a PydanticAI agent."""
        try:
            from converters.pydanticai_tool_converter import PydanticAIToolConverter
            converter = PydanticAIToolConverter(self.session)

            for tool_id in tool_ids:
                try:
                    tool_record = get_tool_by_id(self.session, tool_id)
                    if not tool_record:
                        logger.warning(f"Tool {tool_id} not found, skipping")
                        continue
                    tool_func, _, _ = converter.convert_tool(tool_record)
                    agent.tool(tool_func)
                    logger.debug(f"Registered tool: {tool_record.name}")
                except Exception as e:
                    logger.error(f"Failed to register tool {tool_id}: {e}")
        except ImportError:
            logger.warning("PydanticAI tool converter not available")

    async def _execute_single_node_stream(
        self,
        graph_config: Dict[str, Any],
        input_data: str,
        execution: Execution
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream execution for single-node PydanticAI agents.
        """
        nodes_config = graph_config.get("nodes", {})
        entry_point = graph_config.get("entry_point")
        node_config = nodes_config.get(entry_point, {})

        from pydantic_ai import Agent

        input_data = self._apply_user_prompt(node_config, input_data)

        llm_provider = node_config.get("llm_provider", "")
        model_name = self._resolve_model_name(llm_provider)

        agent = Agent(
            model=model_name,
            system_prompt=node_config.get("system_prompt", "You are a helpful assistant."),
        )

        tool_ids = node_config.get("tool_ids", [])
        if tool_ids:
            self._register_tools_on_agent(agent, tool_ids)

        async with agent.run_stream(input_data) as stream:
            async for message in stream:
                yield {
                    "type": "message",
                    "content": str(message.content) if hasattr(message, 'content') else str(message),
                    "timestamp": datetime.now().isoformat(),
                    "execution_id": execution.id
                }

            result = await stream.result()

            result_data = result.data
            if hasattr(result_data, 'model_dump'):
                result_data = result_data.model_dump()

            self._complete_execution(execution, {
                "result": result_data,
                "cost": self._extract_cost(result)
            })

            yield {
                "type": "complete",
                "result": result_data,
                "cost": self._extract_cost(result),
                "execution_id": execution.id
            }

    def _resolve_model_name(self, llm_provider: str) -> str:
        """Resolve an LLM provider name to a model string for PydanticAI."""
        try:
            import yaml
            config_path = os.path.expanduser("~/.llm_hub/config.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}

                models = config.get("models", [])
                for model_config in models:
                    if model_config.get("name") == llm_provider:
                        provider = model_config.get("provider", "openai")
                        model = model_config.get("model", "gpt-4")
                        return f"{provider}:{model}"
        except Exception as e:
            logger.warning(f"Could not load LLM config: {e}")

        return "openai:gpt-4"

    def _create_execution(self, agent_id: int, user_id: int, input_data: str) -> Execution:
        """Create an Execution record in the database."""
        execution = Execution(
            user_id=user_id,
            agent_id=agent_id,
            execution_type='agent',
            input_data={"input": input_data},
            status='running',
            started_at=datetime.now()
        )
        self.session.add(execution)
        self.session.commit()
        self.session.refresh(execution)
        logger.debug(f"Created execution record: {execution.id}")
        return execution

    def _complete_execution(self, execution: Execution, result: Dict[str, Any]):
        """Mark execution as completed and store results."""
        execution.status = 'completed'
        execution.completed_at = datetime.now()
        execution.output_data = {
            "result": result.get("result", ""),
            "model": result.get("model", ""),
            "cost": result.get("cost")
        }
        self.session.commit()
        logger.debug(f"Execution {execution.id} marked as completed")

    def _fail_execution(self, execution: Execution, error_message: str):
        """Mark execution as failed and store error."""
        execution.status = 'failed'
        execution.completed_at = datetime.now()
        execution.error_message = error_message
        self.session.commit()
        logger.debug(f"Execution {execution.id} marked as failed")

    def _store_node_message(self, execution_id: int, node_id: str, node_config: Dict, input_text: str, output_text: str):
        """Store a message for a graph node execution."""
        message = Message(
            execution_id=execution_id,
            role="assistant",
            content=str(output_text),
            sender=node_config.get("name", node_id),
            message_metadata={
                "node_id": node_id,
                "agent_type": node_config.get("agent_type"),
                "input_preview": str(input_text)[:200]
            }
        )
        self.session.add(message)
        self.session.commit()

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


# Convenience function for direct usage
async def execute_agent_by_id(
    agent_id: int,
    user_id: int,
    input_data: str,
    session: Session,
    stream: bool = False
) -> Dict[str, Any]:
    """Convenience function to execute an agent."""
    executor = AgentExecutor(session)
    return await executor.execute_agent(agent_id, user_id, input_data, stream)
