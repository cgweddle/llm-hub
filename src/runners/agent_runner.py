"""
Agent Runner

Prepare-time agent compilation for flow execution. `compile_agent` builds a
ready-to-run PydanticAI Agent from one graph sub-node config: model object
with credentials bound at construction (no env-var side effects), resolved
system prompt with output-path routing, structured output type, and
registered tools. Creation only — no `.run()`, no DB writes, no langfuse.

Mirrors tool_runner.compile_tool: flow_runner compiles agents once at prepare
time, caches the BuiltAgents on the run context, execution runs the cached
agents (reflection loops reuse the same BuiltAgent across iterations), and
resume recompiles on a fresh session.

Tool registration execs the tool's code, so this module is only ever imported
by flow-running processes (the local flow child, the hosted flow-runner
container, pytest) — never by the backend API process.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from converters.pydanticai_tool_converter import PydanticAIToolConverter
from utils.prompt_template import resolve_system_prompt_template

logger = logging.getLogger(__name__)


@dataclass
class BuiltAgent:
    """A compiled, ready-to-run agent for one graph sub-node.

    class_to_path maps structured-output class names back to output path
    names (None when the node has no output_paths).
    """
    agent: Agent
    class_to_path: Optional[Dict[str, str]] = None


def resolve_provider_config(llm_config: Dict[str, Any], llm_provider: str) -> Dict[str, Any]:
    """Find the named provider entry in llm_config["models"].

    Pure lookup — credential handling happens in _create_model, which binds
    the api_key into the model object instead of mutating os.environ.
    """
    for model_config in llm_config.get("models", []):
        if model_config.get("name") == llm_provider:
            return model_config
    raise ValueError(f"LLM provider '{llm_provider}' not found in config")


def _create_model(provider: str, model: str,
                  api_key: Optional[str] = None, base_url: Optional[str] = None):
    """Build a PydanticAI model object with credentials bound at construction.

    api_key stays a separate parameter from the provider/model/base_url recipe
    so callers control where the key value comes from (runtime config lookup
    vs an emitted os.environ read in exported code).
    """
    if not provider:
        raise ValueError("Provider is required in provider config")
    if not model:
        raise ValueError("Model name is required in provider config")

    if provider == "lmstudio":
        api_key = api_key or "lm-studio"
        base_url = base_url or "http://localhost:1234/v1"
        provider = "openai"

    if provider == "anthropic":
        return AnthropicModel(model, provider=AnthropicProvider(api_key=api_key, base_url=base_url))
    if provider == "openai":
        return OpenAIChatModel(model, provider=OpenAIProvider(api_key=api_key, base_url=base_url))
    raise ValueError(f"Unsupported provider: {provider}. Supported: anthropic, openai, lmstudio")


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
    and returns (union_type, {ClassName: path_name} mapping, models).

    PydanticAI treats union members as separate output tools,
    so the LLM actively chooses which path to take.
    """
    from pydantic import create_model

    models = {}
    class_to_path = {}
    for path_name, path_config in output_paths.items():
        description = _get_path_description(path_config)
        class_name = path_name.capitalize()
        model = create_model(
            class_name,
            content=(str, ...),
            __doc__=description,
        )
        models[path_name] = model
        class_to_path[class_name] = path_name

    from typing import Union
    model_list = list(models.values())
    if len(model_list) == 1:
        union_type = model_list[0]
    else:
        union_type = Union[tuple(model_list)]

    return union_type, class_to_path, models


def compile_agent(sub_node_config: Dict[str, Any], tool_records: List[Any],
                  provider_config: Dict[str, Any]) -> BuiltAgent:
    """Compile one graph sub-node into a BuiltAgent, without running it.

    tool_records are pre-fetched Tool ORM objects (attributes loaded — the
    converter reads them without a session). provider_config is the resolved
    entry from resolve_provider_config, one per flow agent node.
    """
    model = _create_model(
        provider=provider_config.get("provider"),
        model=provider_config.get("model"),
        api_key=provider_config.get("api_key"),
        base_url=provider_config.get("base_url"),
    )

    output_paths = sub_node_config.get("output_paths")
    output_type = None
    class_to_path = None
    if output_paths:
        output_type, class_to_path, _ = _build_output_path_types(output_paths)

    system_prompt = sub_node_config.get("system_prompt", "You are a helpful assistant.")
    system_prompt = resolve_system_prompt_template(system_prompt, sub_node_config, tool_records)

    if output_paths:
        routing_lines = ["\n\nYou must choose one of the following output paths:"]
        for path_name, path_config in output_paths.items():
            description = _get_path_description(path_config)
            routing_lines.append(f'- "{path_name.capitalize()}": {description}')
        system_prompt += "\n".join(routing_lines)

    agent_kwargs = dict(
        model=model,
        system_prompt=system_prompt,
    )
    if output_type is not None:
        agent_kwargs["output_type"] = output_type

    agent = Agent(**agent_kwargs)

    converter = PydanticAIToolConverter()
    for tool_record in tool_records:
        try:
            tool_func, _, _ = converter.convert_tool(tool_record)
            agent.tool_plain(tool_func)
            logger.debug(f"Registered tool: {tool_record.name}")
        except Exception as e:
            logger.error(f"Failed to register tool {getattr(tool_record, 'id', '?')}: {e}")

    return BuiltAgent(agent=agent, class_to_path=class_to_path)
