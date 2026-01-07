"""
Custom ReAct Agent Factory using Google ADK
Creates ReAct-style agents from database configurations using Google's Agent Development Kit.
LLM configuration is loaded by model_name from ~/.llm_hub/config.yaml
"""

import json
import logging
import os
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.runners import Runner
from google.genai import types

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.database import (
    get_session,
    get_tool_by_id,
    get_agent_by_id,
    create_agent as db_create_agent
)
from utils import get_llm_config_by_name

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Enum representing the current state of the agent"""
    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class AgentConfig:
    """
    Configuration for creating a ReAct agent.

    The model_name field references an LLM configuration by name from ~/.llm_hub/config.yaml,
    following the same pattern used by flow tool nodes.
    """
    name: str
    description: str
    model_name: Optional[str] = None  # Reference to LLM config in ~/.llm_hub/config.yaml
    system_prompt: Optional[str] = None
    tool_ids: Optional[List[int]] = None
    user_id: int = 1
    is_public: bool = False
    max_iterations: int = 10
    agent_metadata: Dict[str, Any] = field(default_factory=dict)


def setup_llm_environment(llm_config: Dict) -> str:
    """
    Setup environment variables for the LLM provider and return the model string.
    Follows the same pattern as FlowExecutor._setup_llm_environment()

    Args:
        llm_config: Dict with keys: provider, model, api_key, base_url, config_name

    Returns:
        Model string to use with Google ADK
    """
    provider = llm_config["provider"]
    model = llm_config["model"]
    api_key = llm_config.get("api_key")
    base_url = llm_config.get("base_url")
    config_name = llm_config.get("config_name")

    # Pass the config name for reference
    if config_name:
        os.environ["LLMHUB_CONFIG_NAME"] = config_name
    os.environ["LLMHUB_MODEL_NAME"] = model

    # Set up provider-specific environment variables
    if provider == "gemini" or provider == "google":
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
    elif provider == "anthropic":
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        if base_url:
            os.environ["ANTHROPIC_BASE_URL"] = base_url
    elif provider == "openai":
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url
    elif provider == "lmstudio":
        os.environ["OPENAI_API_KEY"] = api_key or "lm-studio"
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url
    elif provider == "azure":
        if api_key:
            os.environ["AZURE_API_KEY"] = api_key
        if base_url:
            os.environ["AZURE_API_BASE"] = base_url

    return model


class DatabaseToolLoader:
    """
    Loads tools from the database and converts them to Google ADK FunctionTools.
    """

    def __init__(self, session=None):
        self.session = session or get_session()
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._loaded_functions: Dict[str, Callable] = {}
        self._adk_tools: List[FunctionTool] = []

    def load_tool(self, tool_id: int) -> Dict[str, Any]:
        """Load a tool from the database and prepare it for execution"""
        tool = get_tool_by_id(self.session, tool_id)
        if not tool:
            raise ValueError(f"Tool with ID {tool_id} not found")

        tool_info = {
            "id": tool.id,
            "name": tool.name,
            "description": tool.description,
            "tool_type": tool.tool_type,
            "input_schema": tool.input_schema or {},
            "output_schema": tool.output_schema or {},
            "function_code": tool.function_code,
            "main_function": tool.main_function,
            "helper_functions": tool.helper_functions or {}
        }

        self.tools[tool.name] = tool_info

        # Compile the function if it's a function-based tool
        if tool.tool_type == "function" and tool.function_code:
            self._compile_and_convert_tool(tool.name, tool_info)

        return tool_info

    def _compile_and_convert_tool(self, tool_name: str, tool_info: Dict[str, Any]) -> None:
        """Compile a tool's function code and convert to ADK FunctionTool"""
        try:
            # Create a namespace for the function
            namespace = {}

            # Execute helper functions first
            for helper_name, helper_code in tool_info.get("helper_functions", {}).items():
                exec(helper_code, namespace)

            # Execute the main function code
            exec(tool_info["function_code"], namespace)

            # Get the main function from the namespace
            main_func_name = tool_info.get("main_function", tool_name)
            if main_func_name in namespace:
                func = namespace[main_func_name]
                self._loaded_functions[tool_name] = func

                # Create ADK FunctionTool
                adk_tool = FunctionTool(func)
                self._adk_tools.append(adk_tool)

                logger.info(f"Loaded and converted tool '{tool_name}' to ADK FunctionTool")
            else:
                logger.warning(f"Main function '{main_func_name}' not found for tool '{tool_name}'")

        except Exception as e:
            logger.error(f"Failed to compile tool '{tool_name}': {e}")
            raise

    def get_adk_tools(self) -> List[FunctionTool]:
        """Get list of ADK FunctionTools"""
        return self._adk_tools

    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all loaded tools"""
        descriptions = []
        for name, info in self.tools.items():
            desc = f"- {name}: {info['description']}"
            if info.get("input_schema"):
                params = info["input_schema"].get("properties", {})
                if params:
                    param_str = ", ".join([f"{k}: {v.get('type', 'any')}" for k, v in params.items()])
                    desc += f"\n  Parameters: {param_str}"
            descriptions.append(desc)
        return "\n".join(descriptions)

    def get_tool_names(self) -> List[str]:
        """Get list of all loaded tool names"""
        return list(self.tools.keys())


class ReActAgent:
    """
    ReAct-style agent implementation using Google ADK.

    Google ADK's LlmAgent has built-in tool-calling capabilities that implement
    a reasoning + acting pattern. The agent will:
    1. Analyze the user's request
    2. Decide which tools to call
    3. Execute tools and observe results
    4. Continue reasoning until the task is complete
    """

    REACT_INSTRUCTION_TEMPLATE = """You are a ReAct agent that thinks step-by-step and uses tools to accomplish tasks.

{agent_description}

APPROACH:
1. Think carefully about what the user is asking
2. Decide which tool(s) would help accomplish the task
3. Call tools as needed to gather information or perform actions
4. Synthesize the results into a clear, helpful response

IMPORTANT RULES:
- Always explain your reasoning before using a tool
- Use tools to get accurate information rather than guessing
- If a tool returns an error, explain what went wrong
- Provide clear, actionable responses based on tool results
- If you cannot complete a task, explain why

Begin!"""

    def __init__(
        self,
        name: str,
        model: str,
        tool_loader: DatabaseToolLoader,
        agent_description: str = "You are a helpful AI assistant.",
        max_iterations: int = 10,
        verbose: bool = True
    ):
        self.name = name
        self.model = model
        self.tool_loader = tool_loader
        self.agent_description = agent_description
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.state = AgentState.IDLE
        self.history: List[Dict[str, Any]] = []

        # Build the ADK agent
        self._agent = self._build_agent()
        self._runner = None

    def _build_instruction(self) -> str:
        """Build the instruction prompt for the agent"""
        return self.REACT_INSTRUCTION_TEMPLATE.format(
            agent_description=self.agent_description
        )

    def _build_agent(self) -> LlmAgent:
        """Build the Google ADK LlmAgent"""
        instruction = self._build_instruction()
        adk_tools = self.tool_loader.get_adk_tools()

        agent = LlmAgent(
            name=self.name,
            model=self.model,
            instruction=instruction,
            tools=adk_tools if adk_tools else None
        )

        logger.info(f"Built ADK agent '{self.name}' with model '{self.model}' and {len(adk_tools)} tools")
        return agent

    async def run_async(self, user_input: str) -> Dict[str, Any]:
        """
        Run the agent asynchronously.

        Args:
            user_input: The user's query or task

        Returns:
            Dictionary with response and execution details
        """
        self.state = AgentState.RUNNING
        self.history = []

        try:
            # Create a runner for this execution
            runner = Runner(
                agent=self._agent,
                app_name=f"{self.name}_app"
            )

            # Create a new session for this run
            session_id = f"session_{self.name}_{id(user_input)}"
            user_id = "default_user"

            # Run the agent
            final_response = None

            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=user_input)]
                )
            ):
                # Log events for debugging
                if self.verbose:
                    logger.debug(f"Event: {event}")

                # Track tool calls in history
                if hasattr(event, 'tool_calls') and event.tool_calls:
                    for tool_call in event.tool_calls:
                        self.history.append({
                            "type": "tool_call",
                            "tool": tool_call.name if hasattr(tool_call, 'name') else str(tool_call),
                            "input": tool_call.args if hasattr(tool_call, 'args') else None
                        })

                # Capture the final response
                if hasattr(event, 'content') and event.content:
                    final_response = event.content

            self.state = AgentState.FINISHED

            # Extract text from response
            response_text = ""
            if final_response:
                if hasattr(final_response, 'parts'):
                    for part in final_response.parts:
                        if hasattr(part, 'text'):
                            response_text += part.text
                elif isinstance(final_response, str):
                    response_text = final_response

            if self.verbose:
                print(f"\n✅ Agent Response:\n{response_text}")

            return {
                "success": True,
                "answer": response_text,
                "history": self.history,
                "model": self.model
            }

        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Agent execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "history": self.history
            }

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Run the agent synchronously.

        Args:
            user_input: The user's query or task

        Returns:
            Dictionary with response and execution details
        """
        try:
            # Handle event loop properly
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an event loop, use a thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self.run_async(user_input))
                        return future.result()
                else:
                    return loop.run_until_complete(self.run_async(user_input))
            except RuntimeError:
                # No event loop exists, create one
                return asyncio.run(self.run_async(user_input))

        except Exception as e:
            logger.error(f"Sync agent execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "history": self.history
            }


class ReactAgentFactory:
    """
    Factory for creating ReAct agents from database configurations using Google ADK.
    Handles loading agent configs, tools, and constructing the agent.
    """

    def __init__(self, session=None):
        self.session = session or get_session()

    def create_from_config(self, config: AgentConfig) -> ReActAgent:
        """
        Create a ReAct agent from an AgentConfig object.

        Args:
            config: AgentConfig with agent settings

        Returns:
            Configured ReActAgent instance
        """
        if not config.model_name:
            raise ValueError(
                "model_name is required. Configure an LLM in the LLM Providers panel "
                "and provide its name."
            )

        # Load LLM config by name and setup environment
        llm_config = get_llm_config_by_name(config.model_name)
        if not llm_config:
            raise ValueError(
                f"LLM config '{config.model_name}' not found in ~/.llm_hub/config.yaml. "
                f"Please configure it in the LLM Providers panel."
            )

        llm_config['config_name'] = config.model_name
        model = setup_llm_environment(llm_config)

        # Create tool loader and load tools
        tool_loader = DatabaseToolLoader(self.session)

        if config.tool_ids:
            for tool_id in config.tool_ids:
                try:
                    tool_loader.load_tool(tool_id)
                except Exception as e:
                    logger.warning(f"Failed to load tool {tool_id}: {e}")

        # Build agent description
        agent_description = config.system_prompt or config.description

        # Create and return the agent
        return ReActAgent(
            name=config.name,
            model=model,
            tool_loader=tool_loader,
            agent_description=agent_description,
            max_iterations=config.max_iterations,
            verbose=True
        )

    def create_from_database(self, agent_id: int) -> ReActAgent:
        """
        Create a ReAct agent from a database record.

        Args:
            agent_id: ID of the agent in the database

        Returns:
            Configured ReActAgent instance
        """
        agent_record = get_agent_by_id(self.session, agent_id)
        if not agent_record:
            raise ValueError(f"Agent with ID {agent_id} not found")

        if agent_record.agent_type != "react":
            raise ValueError(f"Agent {agent_id} is not a ReAct agent (type: {agent_record.agent_type})")

        # Extract tool IDs from the agent's tools relationship
        tool_ids = [tool.id for tool in agent_record.tools]

        # Also check tools_config for additional tool IDs
        tools_config = agent_record.tools_config or {}
        if "tool_ids" in tools_config:
            tool_ids.extend(tools_config["tool_ids"])
            tool_ids = list(set(tool_ids))  # Remove duplicates

        # Get model_name from llm_config
        llm_config = agent_record.llm_config or {}
        model_name = llm_config.get("model_name")

        if not model_name:
            raise ValueError(
                f"Agent {agent_id} does not have a model_name configured. "
                "Please set an LLM configuration for this agent."
            )

        config = AgentConfig(
            name=agent_record.name,
            description=agent_record.description or "",
            model_name=model_name,
            system_prompt=agent_record.system_prompt,
            tool_ids=tool_ids,
            user_id=agent_record.user_id,
            is_public=agent_record.is_public,
            max_iterations=agent_record.agent_metadata.get("max_iterations", 10) if agent_record.agent_metadata else 10,
            agent_metadata=agent_record.agent_metadata or {}
        )

        return self.create_from_config(config)

    def save_agent_to_database(self, config: AgentConfig) -> int:
        """
        Save an agent configuration to the database.

        Args:
            config: AgentConfig to save

        Returns:
            The ID of the created agent
        """
        # Store model_name in llm_config (following the pattern)
        llm_config = {"model_name": config.model_name}

        # Prepare tools_config with tool IDs
        tools_config = {"tool_ids": config.tool_ids or []}

        # Prepare metadata
        metadata = config.agent_metadata.copy()
        metadata["max_iterations"] = config.max_iterations

        agent = db_create_agent(
            session=self.session,
            user_id=config.user_id,
            name=config.name,
            description=config.description,
            agent_type="react",
            system_prompt=config.system_prompt or config.description,
            llm_config=llm_config,
            tools_config=tools_config,
            metadata=metadata
        )

        return agent.id


# Convenience function for quick agent creation
def create_react_agent(
    name: str,
    description: str,
    model_name: str,
    tool_ids: Optional[List[int]] = None,
    max_iterations: int = 10,
    **kwargs
) -> ReActAgent:
    """
    Convenience function to quickly create a ReAct agent using Google ADK.

    Args:
        name: Agent name
        description: Agent description/system prompt
        model_name: Name of LLM config from ~/.llm_hub/config.yaml (e.g., "My Gemini Config")
        tool_ids: List of database tool IDs to load
        max_iterations: Maximum iterations for reasoning
        **kwargs: Additional agent metadata options

    Returns:
        Configured ReActAgent instance

    Example:
        >>> agent = create_react_agent(
        ...     name="Research Assistant",
        ...     description="You help users research topics",
        ...     model_name="My Gemini Config",  # Must exist in ~/.llm_hub/config.yaml
        ...     tool_ids=[1, 2, 3]
        ... )
        >>> result = agent.run("What is the capital of France?")
    """
    config = AgentConfig(
        name=name,
        description=description,
        model_name=model_name,
        tool_ids=tool_ids,
        max_iterations=max_iterations,
        agent_metadata=kwargs
    )

    factory = ReactAgentFactory()
    return factory.create_from_config(config)
