"""
AI-powered system prompt generation module for agents.

This module provides functionality to generate agent system prompts and
user prompts (task instructions) using LLMs, based on agent name,
description, and available tools.

Prompts are loaded from the database (prompts table, name: 'agent_prompt_gen').
Run 'python src/prompts/upload_prompts.py' to populate from markdown files.
"""

import json
import os
import sys
from typing import Any, AsyncIterator, Dict, List, Optional

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pydantic_ai import Agent

from src.database.database import get_prompt_by_name


PROMPT_NAME = "agent_prompt_gen"
USER_PROMPT_GEN_NAME = "agent_user_prompt_gen"


def _build_agent(llm_config: Dict, system_prompt: str) -> Agent:
    provider = llm_config["provider"]
    model = llm_config["model"]
    api_key = llm_config.get("api_key")
    base_url = llm_config.get("base_url")

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

    return Agent(model=f"{provider}:{model}", system_prompt=system_prompt)


async def generate_system_prompt_stream(
    session: Any,
    agent_name: str,
    agent_description: str,
    tool_names: List[str],
    llm_config: Dict,
    additional_instructions: Optional[str] = None,
) -> AsyncIterator[str]:
    prompt_record = get_prompt_by_name(session, PROMPT_NAME)

    if not prompt_record:
        yield json.dumps({
            "error": f"Prompt '{PROMPT_NAME}' not found in database. "
                     "Run 'python src/prompts/upload_prompts.py' to upload prompts."
        }) + "\n"
        return

    system_prompt = prompt_record.system_prompt
    user_prompt_template = prompt_record.user_prompt

    if not system_prompt or not user_prompt_template:
        yield json.dumps({"error": "System prompt or user prompt is empty in database"}) + "\n"
        return

    if tool_names:
        tools_list = "\n".join(f"- {name}" for name in tool_names)
        tools_section = f"Available Tools:\n{tools_list}\n\nThe agent should know how to use these tools effectively."
    else:
        tools_section = "This agent has no specific tools assigned."

    if additional_instructions and additional_instructions.strip():
        additional_section = f"Additional Requirements:\n{additional_instructions.strip()}"
    else:
        additional_section = ""

    user_prompt_filled = user_prompt_template.replace("{AGENT_NAME}", agent_name)
    user_prompt_filled = user_prompt_filled.replace("{AGENT_DESCRIPTION}", agent_description)
    user_prompt_filled = user_prompt_filled.replace("{TOOLS_SECTION}", tools_section)
    user_prompt_filled = user_prompt_filled.replace("{ADDITIONAL_SECTION}", additional_section)

    try:
        agent = _build_agent(llm_config, system_prompt)
    except ValueError as e:
        yield json.dumps({"error": f"Failed to create LLM: {str(e)}"}) + "\n"
        return

    accumulated_text = ""
    try:
        async with agent.run_stream(user_prompt_filled) as stream:
            async for chunk in stream.stream_text(delta=True):
                accumulated_text += chunk
                yield json.dumps({"chunk": chunk}) + "\n"
    except Exception as e:
        yield json.dumps({"error": f"LLM streaming failed: {str(e)}"}) + "\n"
        return

    yield json.dumps({
        "done": True,
        "system_prompt": accumulated_text.strip()
    }) + "\n"


async def generate_user_prompt_stream(
    session: Any,
    agent_name: str,
    agent_description: str,
    tool_names: List[str],
    llm_config: Dict,
    generated_system_prompt: str,
    additional_instructions: Optional[str] = None,
) -> AsyncIterator[str]:
    prompt_record = get_prompt_by_name(session, USER_PROMPT_GEN_NAME)

    if not prompt_record:
        yield json.dumps({
            "error": f"Prompt '{USER_PROMPT_GEN_NAME}' not found in database. "
                     "Run 'python src/prompts/upload_prompts.py' to upload prompts."
        }) + "\n"
        return

    system_prompt = prompt_record.system_prompt
    user_prompt_template = prompt_record.user_prompt

    if not system_prompt or not user_prompt_template:
        yield json.dumps({"error": "System prompt or user prompt is empty in database"}) + "\n"
        return

    if tool_names:
        tools_list = "\n".join(f"- {name}" for name in tool_names)
        tools_section = f"Available Tools:\n{tools_list}\n\nThe task instruction may reference using these tools."
    else:
        tools_section = "This agent has no specific tools assigned."

    if additional_instructions and additional_instructions.strip():
        additional_section = f"Additional Requirements:\n{additional_instructions.strip()}"
    else:
        additional_section = ""

    user_prompt_filled = user_prompt_template.replace("{AGENT_NAME}", agent_name)
    user_prompt_filled = user_prompt_filled.replace("{AGENT_DESCRIPTION}", agent_description)
    user_prompt_filled = user_prompt_filled.replace("{TOOLS_SECTION}", tools_section)
    user_prompt_filled = user_prompt_filled.replace("{ADDITIONAL_SECTION}", additional_section)
    user_prompt_filled = user_prompt_filled.replace("{SYSTEM_PROMPT}", generated_system_prompt)

    try:
        agent = _build_agent(llm_config, system_prompt)
    except ValueError as e:
        yield json.dumps({"error": f"Failed to create LLM: {str(e)}"}) + "\n"
        return

    accumulated_text = ""
    try:
        async with agent.run_stream(user_prompt_filled) as stream:
            async for chunk in stream.stream_text(delta=True):
                accumulated_text += chunk
                yield json.dumps({"chunk": chunk}) + "\n"
    except Exception as e:
        yield json.dumps({"error": f"LLM streaming failed: {str(e)}"}) + "\n"
        return

    yield json.dumps({
        "done": True,
        "user_prompt": accumulated_text.strip()
    }) + "\n"
