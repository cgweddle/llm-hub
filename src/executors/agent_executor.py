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
- Streaming support for PydanticAI
- Automatic retry with exponential backoff for transient failures
- Error handling and logging
"""

import logging
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, Optional, Tuple, List

from sqlalchemy.orm import Session

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.database import get_agent_by_id, get_tool_by_id, create_execution, update_execution
from database.database_setup import Execution
from utils.prompt_template import resolve_system_prompt_template, resolve_user_prompt_template

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

# Load .env before initializing LangFuse (needs LANGFUSE_* env vars)
from dotenv import load_dotenv
load_dotenv()

# Initialize LangFuse for automatic PydanticAI tracing
try:
    from langfuse import get_client as get_langfuse_client, observe as langfuse_observe
    langfuse_client = get_langfuse_client()
    # Instrument all PydanticAI agents — captures tool calls, LLM I/O, costs automatically
    from pydantic_ai import Agent as _PydanticAIAgent
    _PydanticAIAgent.instrument_all()
    LANGFUSE_AVAILABLE = True
    logger_init_msg = "LangFuse instrumentation enabled for PydanticAI agents"
except Exception:
    LANGFUSE_AVAILABLE = False
    langfuse_client = None
    langfuse_observe = None
    logger_init_msg = "LangFuse not available — internal agent tracing disabled"

logger = logging.getLogger(__name__)
logger.info(logger_init_msg)


def _get_path_description(path_config) -> str:
    """Extract description from an output path config (string or dict)."""
    if isinstance(path_config, str):
        return path_config
    return path_config.get("description", "")


def _build_output_path_types(output_paths: Dict[str, Any]):
    """Build dynamic Pydantic union types from output_paths config.

    Given {"revise": "Draft needs work", "approve": "Draft is ready"} or
    {"revise": {"description": "...", "return_behavior": "node_output"}, ...},
    creates model classes Revise and Approve each with a 'content' field,
    and returns (union_type, {ClassName: path_name} mapping).

    PydanticAI treats union members as separate output tools,
    so the LLM actively chooses which path to take.
    """
    from pydantic import BaseModel, create_model

    models = {}
    class_to_path = {}
    for path_name, path_config in output_paths.items():
        description = _get_path_description(path_config)
        # Create a model class with the path name capitalized
        class_name = path_name.capitalize()
        model = create_model(
            class_name,
            content=(str, ...),
            __doc__=description,
        )
        models[path_name] = model
        class_to_path[class_name] = path_name

    # Build the union type
    from typing import Union
    model_list = list(models.values())
    if len(model_list) == 1:
        union_type = model_list[0]
    else:
        union_type = Union[tuple(model_list)]

    return union_type, class_to_path, models


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

    async def execute_agent(
        self,
        agent_id: int,
        user_id: int,
        input_data: str,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a standalone agent and store results in the database.
        Creates a top-level Execution(type='agent'). Internal tool call
        tracing is handled automatically by LangFuse.
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

        # Create top-level execution record
        execution = create_execution(
            self.session,
            user_id=user_id,
            agent_id=agent_id,
            execution_type='agent',
            name=agent_record.name,
            input_data={"input": input_data},
            status='running',
            started_at=datetime.now()
        )

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

            update_execution(self.session, execution.id,
                status='completed',
                output_data={
                    "result": result.get("result", ""),
                    "model": result.get("model", ""),
                    "cost": result.get("cost")
                },
                completed_at=datetime.now()
            )
            logger.info(f"Agent execution completed: {execution.id}")
            return result

        except Exception as e:
            update_execution(self.session, execution.id,
                status='failed',
                error_message=str(e),
                completed_at=datetime.now()
            )
            logger.error(f"Agent execution failed: {e}")
            raise RuntimeError(f"Agent execution failed: {e}")

    async def execute_agent_node(
        self,
        agent_id: int,
        input_text: str,
        session: Session,
        parent_execution: Optional[Execution] = None
    ) -> str:
        """
        Execute an agent and return its text output.
        Entry point for FlowExecutor — records structural node under parent_execution.
        Internal tracing is handled by LangFuse automatically.

        Args:
            agent_id: Agent ID
            input_text: Input text
            session: Database session
            parent_execution: Parent Execution record from FlowExecutor

        Returns:
            Agent text output as string
        """
        agent = get_agent_by_id(session, agent_id)
        if not agent:
            raise ValueError(f"Agent with ID {agent_id} not found")

        graph_config = agent.graph_config
        if not graph_config:
            raise ValueError(f"Agent {agent_id} has no graph_config")

        result = await self._execute_graph(graph_config, input_text, execution=parent_execution)
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

                # Execute the sub-agent node
                chosen_path = None
                try:
                    trace_id = None
                    predecessor_msgs = messages if messages else None

                    if LANGFUSE_AVAILABLE and langfuse_observe:
                        # Capture trace ID even if the agent fails
                        _captured_trace_id = None

                        @langfuse_observe(name=nodes_config[node_id].get("name", node_id))
                        async def _observed_run():
                            nonlocal _captured_trace_id
                            _captured_trace_id = langfuse_client.get_current_trace_id()
                            result = await self._run_sub_agent(node_id, nodes_config[node_id], node_input, predecessor_messages=predecessor_msgs)
                            return result

                        output, node_result_messages, chosen_path = await _observed_run()
                        trace_id = _captured_trace_id
                        langfuse_client.flush()
                    else:
                        output, node_result_messages, chosen_path = await self._run_sub_agent(node_id, nodes_config[node_id], node_input, predecessor_messages=predecessor_msgs)
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
        predecessor_messages: Optional[List] = None,
    ) -> Tuple[str, Optional[List], Optional[str]]:
        """
        Create and run a sub-agent for a single graph node.
        Returns (output_text, messages, chosen_path) where:
        - messages is the PydanticAI message history (all_messages() from the run)
        - chosen_path is the output path name (None if no output_paths configured)
        Internal tracing is handled by LangFuse.
        """
        logger.debug(
            f"Running sub-agent: {node_config.get('name', node_id)} "
            f"(type: {node_config.get('agent_type')})"
        )

        return await self._run_pydanticai_node(node_config, node_input, predecessor_messages=predecessor_messages)

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
        predecessor_messages: Optional[List] = None,
    ) -> Tuple[str, List, Optional[str]]:
        """Run a PydanticAI sub-agent. Returns (output_text, all_messages, chosen_path).
        chosen_path is None for nodes without output_paths.
        LangFuse auto-captures internal tool calls and LLM I/O."""
        from pydantic_ai import Agent

        node_input = self._apply_user_prompt(node_config, node_input, predecessor_messages)

        llm_provider = node_config.get("llm_provider", "")
        model_name = self._resolve_model_name(llm_provider)

        # Build output type and routing from output_paths if configured
        output_paths = node_config.get("output_paths")
        output_type = None
        class_to_path = None
        if output_paths and len(output_paths) > 0:
            output_type, class_to_path, _ = _build_output_path_types(output_paths)

        # Fetch tool records for template resolution and registration
        tool_ids = node_config.get("tool_ids", [])
        tool_records = [get_tool_by_id(self.session, tid) for tid in tool_ids]
        tool_records = [t for t in tool_records if t is not None]

        system_prompt = node_config.get("system_prompt", "You are a helpful assistant.")
        system_prompt = resolve_system_prompt_template(system_prompt, node_config, tool_records)

        # Append routing instructions when output paths are configured
        if output_paths:
            routing_lines = ["\n\nYou must choose one of the following output paths:"]
            for path_name, path_config in output_paths.items():
                description = _get_path_description(path_config)
                routing_lines.append(f'- "{path_name.capitalize()}": {description}')
            system_prompt += "\n".join(routing_lines)

        agent_kwargs = dict(
            model=model_name,
            system_prompt=system_prompt,
        )
        if output_type is not None:
            agent_kwargs["output_type"] = output_type

        sub_agent = Agent(**agent_kwargs)

        # Register tools on the agent
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

        # Extract output and determine chosen path
        result_data = result.output
        chosen_path = None

        if class_to_path and result_data is not None:
            # Determine which union member was chosen via isinstance
            class_name = type(result_data).__name__
            chosen_path = class_to_path.get(class_name)
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
                    agent.tool_plain(tool_func)
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
        LangFuse captures the full trace automatically.
        """
        nodes_config = graph_config.get("nodes", {})
        entry_point = graph_config.get("entry_point")
        node_config = nodes_config.get(entry_point, {})

        from pydantic_ai import Agent

        input_data = self._apply_user_prompt(node_config, input_data)

        llm_provider = node_config.get("llm_provider", "")
        model_name = self._resolve_model_name(llm_provider)

        # Fetch tool records for template resolution and registration
        tool_ids = node_config.get("tool_ids", [])
        tool_records = [get_tool_by_id(self.session, tid) for tid in tool_ids]
        tool_records = [t for t in tool_records if t is not None]

        system_prompt = node_config.get("system_prompt", "You are a helpful assistant.")
        system_prompt = resolve_system_prompt_template(system_prompt, node_config, tool_records)

        agent = Agent(
            model=model_name,
            system_prompt=system_prompt,
        )

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

            result_data = result.output
            if hasattr(result_data, 'model_dump'):
                result_data = result_data.model_dump()

            cost = self._extract_cost(result)

            update_execution(self.session, execution.id,
                status='completed',
                output_data={
                    "result": result_data,
                    "cost": cost
                },
                completed_at=datetime.now()
            )

            yield {
                "type": "complete",
                "result": result_data,
                "cost": cost,
                "execution_id": execution.id
            }

    def _resolve_model_name(self, llm_provider: str) -> str:
        """Resolve an LLM provider name to a model string for PydanticAI.
        Also sets api_key/base_url as env vars so PydanticAI can pick them up."""
        try:
            import yaml
            config_path = os.path.expanduser("~/.llm_hub/config.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}

                models = config.get("models", [])
                for model_config in models:
                    if model_config.get("name") == llm_provider:
                        provider = model_config.get("provider")
                        model = model_config.get("model")
                        api_key = model_config.get("api_key")
                        base_url = model_config.get("base_url")

                        if provider == "lmstudio":
                            api_key = api_key or "lm-studio"
                            base_url = base_url or "http://localhost:1234/v1"
                            provider = "openai"

                        if api_key:
                            if provider == "anthropic":
                                os.environ["ANTHROPIC_API_KEY"] = api_key
                            else:
                                os.environ["OPENAI_API_KEY"] = api_key
                        if base_url:
                            os.environ["OPENAI_BASE_URL"] = base_url

                        return f"{provider}:{model}"
        except Exception as e:
            logger.warning(f"Could not load LLM config: {e}")

        raise ValueError(f"LLM provider '{llm_provider}' not found in ~/.llm_hub/config.yaml")

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
