"""
PydanticAI Agent Factory
Creates PydanticAI agents from database configurations
"""

import logging
import os
import sys
from typing import Dict, Any, List, Optional, Type, Union
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.database import get_session, get_tool_by_id, get_agent_by_id
from utils import get_llm_config_by_name
from converters.pydanticai_tool_converter import PydanticAIToolConverter

logger = logging.getLogger(__name__)

# Maps provider names to their PydanticAI model class
PROVIDER_MAP = {
    "anthropic": AnthropicModel,
    "openai": OpenAIModel,
    "lmstudio": OpenAIModel,
}


class PydanticAIAgentFactory:
    """
    Factory for creating PydanticAI agents from database configurations.

    This factory:
    1. Loads agent configuration from database
    2. Creates appropriate LLM model instance
    3. Converts database tools to PydanticAI tools
    4. Registers tools with the agent
    5. Handles structured output configuration

    Pattern mirrors ReactAgentFactory for consistency.
    """

    def __init__(self, session=None):
        """
        Initialize the PydanticAI agent factory.

        Args:
            session: Optional database session. If not provided, creates a new one.
        """
        self.session = session or get_session()
        self.tool_converter = PydanticAIToolConverter(self.session)

    def create_from_database(self, agent_id: int) -> Agent:
        """
        Create a PydanticAI agent from a database Agent record.

        Args:
            agent_id: ID of the agent in the database

        Returns:
            Configured PydanticAI Agent instance

        Raises:
            ValueError: If agent not found, wrong type, or configuration invalid

        Example:
            >>> factory = PydanticAIAgentFactory()
            >>> agent = factory.create_from_database(agent_id=5)
            >>> result = await agent.run("What is 2+2?")
        """
        # Load agent record from database
        agent_record = get_agent_by_id(self.session, agent_id)
        if not agent_record:
            raise ValueError(f"Agent with ID {agent_id} not found in database")

        # Validate agent type
        if agent_record.agent_type != "pydanticai":
            raise ValueError(
                f"Agent {agent_id} is not a PydanticAI agent (type: {agent_record.agent_type}). "
                f"Use ReactAgentFactory for 'react' agent types."
            )

        logger.info(f"Creating PydanticAI agent: {agent_record.name} (ID: {agent_id})")

        # Get LLM configuration and create model
        llm_config = self._get_llm_config(agent_record)
        model = self._create_model(llm_config)

        # Check for structured output configuration
        result_type = self._get_result_type(agent_record)

        # Get system prompt
        system_prompt = agent_record.system_prompt

        # Create PydanticAI agent
        if result_type:
            logger.info(f"Creating agent with structured output: {result_type.__name__}")
            agent = Agent(
                model=model,
                result_type=result_type,
                system_prompt=system_prompt
            )
        else:
            agent = Agent(
                model=model,
                system_prompt=system_prompt
            )

        # Load and register tools
        self._register_tools(agent, agent_record)

        logger.info(f"✓ Successfully created PydanticAI agent: {agent_record.name}")
        return agent

    def _get_llm_config(self, agent_record) -> Dict[str, Any]:
        """
        Get LLM configuration for the agent.

        Args:
            agent_record: Database Agent object

        Returns:
            Dict with LLM configuration (provider, model, api_key, etc.)

        Raises:
            ValueError: If LLM configuration is invalid or not found
        """
        llm_config = agent_record.llm_config or {}
        model_name = llm_config.get("model_name")

        if not model_name:
            raise ValueError(
                f"Agent {agent_record.id} does not have a model_name configured. "
                f"Please set an LLM configuration for this agent via the LLM Providers panel."
            )

        # Load full config from ~/.llm_hub/config.yaml
        full_config = get_llm_config_by_name(model_name)
        if not full_config:
            raise ValueError(
                f"LLM config '{model_name}' not found in ~/.llm_hub/config.yaml. "
                f"Please configure it in the LLM Providers panel."
            )

        # Add config name for tracking
        full_config['config_name'] = model_name

        logger.debug(f"Loaded LLM config: {model_name} (provider: {full_config.get('provider')})")
        return full_config

    def _create_model(self, llm_config: Dict[str, Any]) -> Union[AnthropicModel, OpenAIModel]:
        """
        Create a PydanticAI model instance from LLM configuration.

        Args:
            llm_config: Dict with keys:
                - provider: str ('anthropic', 'openai', 'lmstudio')
                - model: str (e.g., 'claude-3-5-sonnet-20241022', 'gpt-4')
                - api_key: Optional[str]
                - base_url: Optional[str]
                - config_name: Optional[str] (for logging/tracking)

        Returns:
            PydanticAI model instance (AnthropicModel or OpenAIModel)

        Raises:
            ValueError: If provider is unsupported or required fields are missing
        """
        provider = llm_config.get("provider")
        model_name = llm_config.get("model")
        api_key = llm_config.get("api_key")
        base_url = llm_config.get("base_url")
        config_name = llm_config.get("config_name", "unknown")

        if not provider:
            raise ValueError("Provider is required in llm_config")

        if not model_name:
            raise ValueError("Model name is required in llm_config")

        model_class = PROVIDER_MAP.get(provider)
        if not model_class:
            supported = ", ".join(PROVIDER_MAP.keys())
            raise ValueError(f"Unsupported provider: {provider}. Supported: {supported}")

        logger.info(f"Creating PydanticAI model: provider={provider}, model={model_name}, config={config_name}")

        # LM Studio defaults
        if provider == "lmstudio":
            api_key = "lm-studio"
            if not base_url:
                base_url = "http://localhost:1234/v1"
                logger.warning(f"No base_url provided for LM Studio, using default: {base_url}")

        # Build kwargs, only including values that are set
        kwargs = {"model_name": model_name}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        return model_class(**kwargs)

    def _get_result_type(self, agent_record) -> Optional[Type[BaseModel]]:
        """
        Get structured output result type from agent configuration.

        Checks for output_schema in the following order:
        1. agent_record.output_schema (dedicated column, preferred)
        2. agent_record.agent_metadata["result_schema"] (legacy fallback)

        Args:
            agent_record: Database Agent object

        Returns:
            Pydantic model class for structured output, or None if not configured
        """
        # Prefer the dedicated output_schema column
        output_schema = getattr(agent_record, 'output_schema', None)

        # Fallback to agent_metadata for backward compatibility
        if not output_schema:
            agent_metadata = agent_record.agent_metadata or {}
            output_schema = agent_metadata.get("result_schema")

        if not output_schema:
            return None

        try:
            # Convert JSON schema to Pydantic model
            result_model = self.tool_converter.json_schema_to_pydantic(
                schema=output_schema,
                model_name=f"{agent_record.name.replace(' ', '_')}_Result"
            )
            return result_model

        except Exception as e:
            logger.error(f"Failed to create result type for agent {agent_record.id}: {e}")
            logger.warning("Agent will run without structured output validation")
            return None

    def _register_tools(self, agent: Agent, agent_record):
        """
        Register database tools with the PydanticAI agent.

        Args:
            agent: PydanticAI Agent instance
            agent_record: Database Agent object

        Note:
            Tools are loaded from:
            1. agent_record.tools (many-to-many relationship)
            2. agent_record.tools_config["tool_ids"] (additional tool IDs)
        """
        # Get tool IDs from relationship
        tool_ids = [tool.id for tool in agent_record.tools]

        # Also check tools_config for additional tool IDs
        tools_config = agent_record.tools_config or {}
        if "tool_ids" in tools_config:
            additional_tool_ids = tools_config["tool_ids"]
            if additional_tool_ids:
                tool_ids.extend(additional_tool_ids)
                tool_ids = list(set(tool_ids))  # Remove duplicates

        if not tool_ids:
            logger.warning(f"Agent {agent_record.name} has no tools configured")
            return

        logger.info(f"Registering {len(tool_ids)} tools with agent {agent_record.name}")

        # Convert and register each tool
        for tool_id in tool_ids:
            try:
                tool_record = get_tool_by_id(self.session, tool_id)
                if not tool_record:
                    logger.warning(f"Tool with ID {tool_id} not found, skipping")
                    continue

                # Convert tool to PydanticAI format
                tool_func, input_model, output_model = self.tool_converter.convert_tool(tool_record)

                # Register tool with agent
                agent.tool(tool_func)

                logger.debug(f"Registered tool: {tool_record.name}")

            except Exception as e:
                logger.error(f"Failed to register tool {tool_id}: {e}")
                # Continue with other tools even if one fails
                continue

        logger.info(f"✓ Successfully registered {len(tool_ids)} tools")

    def validate_agent_config(self, agent_id: int) -> Dict[str, Any]:
        """
        Validate agent configuration without creating the agent.

        Useful for checking configuration before execution.

        Args:
            agent_id: ID of the agent to validate

        Returns:
            Dict with validation results:
            {
                "valid": bool,
                "errors": List[str],
                "warnings": List[str],
                "config": Dict[str, Any]
            }

        Example:
            >>> factory = PydanticAIAgentFactory()
            >>> validation = factory.validate_agent_config(5)
            >>> if validation["valid"]:
            ...     agent = factory.create_from_database(5)
        """
        errors = []
        warnings = []
        config = {}

        try:
            agent_record = get_agent_by_id(self.session, agent_id)
            if not agent_record:
                errors.append(f"Agent {agent_id} not found")
                return {"valid": False, "errors": errors, "warnings": warnings, "config": config}

            # Check agent type
            if agent_record.agent_type != "pydanticai":
                errors.append(f"Agent type is '{agent_record.agent_type}', expected 'pydanticai'")

            # Check LLM config
            try:
                llm_config = self._get_llm_config(agent_record)
                config["llm_config"] = llm_config
            except ValueError as e:
                errors.append(f"LLM config error: {e}")

            # Check tools
            tool_ids = [tool.id for tool in agent_record.tools]
            tools_config = agent_record.tools_config or {}
            if "tool_ids" in tools_config:
                tool_ids.extend(tools_config["tool_ids"])
            tool_ids = list(set(tool_ids))

            if not tool_ids:
                warnings.append("Agent has no tools configured")

            config["tool_count"] = len(tool_ids)
            config["tools"] = [
                {"id": tid, "exists": get_tool_by_id(self.session, tid) is not None}
                for tid in tool_ids
            ]

            # Check for structured output schema (new column or legacy metadata)
            has_output_schema = bool(getattr(agent_record, 'output_schema', None))
            agent_metadata = agent_record.agent_metadata or {}
            has_legacy_schema = "result_schema" in agent_metadata

            config["has_structured_output"] = has_output_schema or has_legacy_schema
            config["output_schema_source"] = (
                "output_schema" if has_output_schema else
                "agent_metadata" if has_legacy_schema else
                None
            )

            valid = len(errors) == 0
            return {
                "valid": valid,
                "errors": errors,
                "warnings": warnings,
                "config": config
            }

        except Exception as e:
            errors.append(f"Validation error: {e}")
            return {"valid": False, "errors": errors, "warnings": warnings, "config": config}


# Convenience function for direct usage
def create_pydanticai_agent_from_database(agent_id: int, session=None) -> Agent:
    """
    Convenience function to quickly create a PydanticAI agent from database.

    Args:
        agent_id: Database agent ID
        session: Optional database session

    Returns:
        Configured PydanticAI Agent instance

    Example:
        >>> agent = create_pydanticai_agent_from_database(agent_id=5)
        >>> result = await agent.run("What is the weather?")
    """
    factory = PydanticAIAgentFactory(session=session)
    return factory.create_from_database(agent_id)
