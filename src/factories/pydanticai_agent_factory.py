"""
PydanticAI Agent Factory
Creates PydanticAI agents from database configurations or node config dicts.
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
from utils.prompt_template import resolve_system_prompt_template
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
    Factory for creating PydanticAI agents from database configurations
    or from node config dicts (used by the graph executor).
    """

    def __init__(self, session=None):
        self.session = session or get_session()
        self.tool_converter = PydanticAIToolConverter(self.session)

    def create_from_node_config(
        self,
        node_config: Dict[str, Any],
        output_schema: Optional[Dict] = None
    ) -> Agent:
        """
        Create a PydanticAI agent from a graph_config node dict.

        Args:
            node_config: Node config with keys: system_prompt, llm_provider, tool_ids
            output_schema: Optional JSON schema for structured output

        Returns:
            Configured PydanticAI Agent instance
        """
        llm_provider = node_config.get("llm_provider", "")
        llm_config = self._get_llm_config_by_name(llm_provider)
        model = self._create_model(llm_config)

        result_type = self._build_result_type(output_schema) if output_schema else None

        # Fetch tool records for template resolution and registration
        tool_ids = node_config.get("tool_ids", [])
        tool_records = [get_tool_by_id(self.session, tid) for tid in tool_ids]
        tool_records = [t for t in tool_records if t is not None]

        system_prompt = node_config.get("system_prompt", "You are a helpful assistant.")
        system_prompt = resolve_system_prompt_template(system_prompt, node_config, tool_records)

        if result_type:
            agent = Agent(model=model, result_type=result_type, system_prompt=system_prompt)
        else:
            agent = Agent(model=model, system_prompt=system_prompt)

        if tool_ids:
            self._register_tools_by_ids(agent, tool_ids)

        return agent

    def create_from_database(self, agent_id: int) -> Agent:
        """
        Create a PydanticAI agent from a database Agent record.
        Reads config from graph_config.nodes[entry_point].

        Args:
            agent_id: ID of the agent in the database

        Returns:
            Configured PydanticAI Agent instance
        """
        agent_record = get_agent_by_id(self.session, agent_id)
        if not agent_record:
            raise ValueError(f"Agent with ID {agent_id} not found in database")

        graph_config = agent_record.graph_config
        if not graph_config:
            raise ValueError(f"Agent {agent_id} has no graph_config")

        entry_point = graph_config.get("entry_point")
        nodes = graph_config.get("nodes", {})
        node_config = nodes.get(entry_point) if entry_point else None

        if not node_config:
            raise ValueError(
                f"Agent {agent_id} graph_config has no valid entry_point node. "
                f"entry_point='{entry_point}', available nodes: {list(nodes.keys())}"
            )

        logger.info(f"Creating PydanticAI agent: {agent_record.name} (ID: {agent_id})")
        return self.create_from_node_config(node_config, agent_record.output_schema)

    def _get_llm_config_by_name(self, llm_provider: str) -> Dict[str, Any]:
        """
        Get LLM configuration by provider name from ~/.llm_hub/config.yaml.
        """
        if not llm_provider:
            raise ValueError(
                "llm_provider is not set. "
                "Please configure an LLM in the LLM Providers panel."
            )

        full_config = get_llm_config_by_name(llm_provider)
        if not full_config:
            raise ValueError(
                f"LLM config '{llm_provider}' not found in ~/.llm_hub/config.yaml. "
                f"Please configure it in the LLM Providers panel."
            )

        full_config['config_name'] = llm_provider
        logger.debug(f"Loaded LLM config: {llm_provider} (provider: {full_config.get('provider')})")
        return full_config

    def _create_model(self, llm_config: Dict[str, Any]) -> Union[AnthropicModel, OpenAIModel]:
        """
        Create a PydanticAI model instance from LLM configuration.
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

        if provider == "lmstudio":
            api_key = "lm-studio"
            if not base_url:
                base_url = "http://localhost:1234/v1"

        kwargs = {"model_name": model_name}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        return model_class(**kwargs)

    def _build_result_type(self, output_schema: Dict) -> Optional[Type[BaseModel]]:
        """
        Convert a JSON schema to a Pydantic model class for structured output.
        """
        if not output_schema:
            return None
        try:
            result_model = self.tool_converter.json_schema_to_pydantic(
                schema=output_schema,
                model_name="AgentResult"
            )
            return result_model
        except Exception as e:
            logger.error(f"Failed to create result type from output_schema: {e}")
            return None

    def _register_tools_by_ids(self, agent: Agent, tool_ids: List[int]):
        """
        Register database tools with the PydanticAI agent by tool IDs.
        """
        if not tool_ids:
            return

        logger.info(f"Registering {len(tool_ids)} tools")

        for tool_id in tool_ids:
            try:
                tool_record = get_tool_by_id(self.session, tool_id)
                if not tool_record:
                    logger.warning(f"Tool with ID {tool_id} not found, skipping")
                    continue
                tool_func, _, _ = self.tool_converter.convert_tool(tool_record)
                agent.tool_plain(tool_func)
                logger.debug(f"Registered tool: {tool_record.name}")
            except Exception as e:
                logger.error(f"Failed to register tool {tool_id}: {e}")

    def validate_agent_config(self, agent_id: int) -> Dict[str, Any]:
        """
        Validate agent configuration without creating the agent.
        """
        errors = []
        warnings = []
        config = {}

        try:
            agent_record = get_agent_by_id(self.session, agent_id)
            if not agent_record:
                errors.append(f"Agent {agent_id} not found")
                return {"valid": False, "errors": errors, "warnings": warnings, "config": config}

            graph_config = agent_record.graph_config
            if not graph_config:
                errors.append("Agent has no graph_config")
                return {"valid": False, "errors": errors, "warnings": warnings, "config": config}

            entry_point = graph_config.get("entry_point")
            nodes = graph_config.get("nodes", {})
            if not entry_point or entry_point not in nodes:
                errors.append(f"Invalid entry_point '{entry_point}'")

            # Validate the entry node's LLM config
            node_config = nodes.get(entry_point, {})
            llm_provider = node_config.get("llm_provider")
            if llm_provider:
                try:
                    llm_config = self._get_llm_config_by_name(llm_provider)
                    config["llm_config"] = llm_config
                except ValueError as e:
                    errors.append(f"LLM config error: {e}")
            else:
                errors.append("Entry node has no llm_provider configured")

            # Check tools
            tool_ids = node_config.get("tool_ids", [])
            if not tool_ids:
                warnings.append("Entry node has no tools configured")
            config["tool_count"] = len(tool_ids)
            config["tools"] = [
                {"id": tid, "exists": get_tool_by_id(self.session, tid) is not None}
                for tid in tool_ids
            ]

            config["has_structured_output"] = bool(agent_record.output_schema)
            config["node_count"] = len(nodes)

            valid = len(errors) == 0
            return {"valid": valid, "errors": errors, "warnings": warnings, "config": config}

        except Exception as e:
            errors.append(f"Validation error: {e}")
            return {"valid": False, "errors": errors, "warnings": warnings, "config": config}


# Convenience function for direct usage
def create_pydanticai_agent_from_database(agent_id: int, session=None) -> Agent:
    """
    Convenience function to quickly create a PydanticAI agent from database.
    """
    factory = PydanticAIAgentFactory(session=session)
    return factory.create_from_database(agent_id)
