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
from typing import Any, Iterator, List, Optional

# Add project root to path
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.llm_setup import create_llm
from src.database.database import get_prompt_by_name
from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage


PROMPT_NAME = "agent_prompt_gen"
USER_PROMPT_GEN_NAME = "agent_user_prompt_gen"


def generate_system_prompt_stream(
    session: Any,
    agent_name: str,
    agent_description: str,
    tool_names: List[str],
    llm_config: Dict,
    additional_instructions: Optional[str] = None,
) -> Iterator[str]:
    """
    Generate an agent system prompt using LLM with streaming.

    Args:
        session: Database session
        agent_name: Name of the agent
        agent_description: Description of what the agent does
        tool_names: List of tool names available to the agent
        llm_model: Config name from ~/.llm_hub/config.yaml
        additional_instructions: Optional extra instructions for prompt generation

    Yields:
        JSON strings with streaming updates
    """
    # 1. Query prompts from database
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

    # 2. Extract LLM config fields
    provider = llm_config["provider"]
    model = llm_config["model"]
    api_key = llm_config.get("api_key")
    base_url = llm_config.get("base_url")

    # 3. Build tools section
    if tool_names:
        tools_list = "\n".join(f"- {name}" for name in tool_names)
        tools_section = f"Available Tools:\n{tools_list}\n\nThe agent should know how to use these tools effectively."
    else:
        tools_section = "This agent has no specific tools assigned."

    # Build additional instructions section
    if additional_instructions and additional_instructions.strip():
        additional_section = f"Additional Requirements:\n{additional_instructions.strip()}"
    else:
        additional_section = ""

    # 4. Fill placeholders in user prompt
    user_prompt_filled = user_prompt_template.replace("{AGENT_NAME}", agent_name)
    user_prompt_filled = user_prompt_filled.replace("{AGENT_DESCRIPTION}", agent_description)
    user_prompt_filled = user_prompt_filled.replace("{TOOLS_SECTION}", tools_section)
    user_prompt_filled = user_prompt_filled.replace("{ADDITIONAL_SECTION}", additional_section)

    # 5. Prepare LLM credentials
    llm_api_key = api_key
    llm_model = model

    if provider == "lmstudio":
        llm_api_key = "dummy-key-for-local-llm"
        if not llm_model.startswith("openai/"):
            llm_model = f"openai/{llm_model}"

    # 6. Create LLM instance
    try:
        llm = create_llm(
            provider=provider,
            model=llm_model,
            temperature=0.7,
            api_key=llm_api_key,
            base_url=base_url
        )
    except ValueError as e:
        yield json.dumps({"error": f"Failed to create LLM: {str(e)}"}) + "\n"
        return

    # 7. Stream LLM response
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt_filled)
    ]

    accumulated_text = ""
    try:
        for chunk in llm.stream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                accumulated_text += chunk.content
                yield json.dumps({"chunk": chunk.content}) + "\n"
    except Exception as e:
        yield json.dumps({"error": f"LLM streaming failed: {str(e)}"}) + "\n"
        return

    # Send final result
    yield json.dumps({
        "done": True,
        "system_prompt": accumulated_text.strip()
    }) + "\n"


def generate_user_prompt_stream(
    session: Any,
    agent_name: str,
    agent_description: str,
    tool_names: List[str],
    llm_config: Dict,
    generated_system_prompt: str,
    additional_instructions: Optional[str] = None,
) -> Iterator[str]:
    """
    Generate a task-specific user prompt for an agent using LLM with streaming.

    Uses a separate prompt template (agent_user_prompt_gen) that is aware of
    the agent's system prompt, so the user prompt complements it without overlap.

    Args:
        session: Database session
        agent_name: Name of the agent
        agent_description: Description of what the agent does
        tool_names: List of tool names available to the agent
        llm_model: Config name from ~/.llm_hub/config.yaml
        generated_system_prompt: The already-generated system prompt for this agent
        additional_instructions: Optional extra instructions

    Yields:
        JSON strings with streaming updates
    """
    # 1. Query user prompt generation template from database
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

    # 2. Extract LLM config fields
    provider = llm_config["provider"]
    model = llm_config["model"]
    api_key = llm_config.get("api_key")
    base_url = llm_config.get("base_url")

    # 3. Build tools section
    if tool_names:
        tools_list = "\n".join(f"- {name}" for name in tool_names)
        tools_section = f"Available Tools:\n{tools_list}\n\nThe task instruction may reference using these tools."
    else:
        tools_section = "This agent has no specific tools assigned."

    if additional_instructions and additional_instructions.strip():
        additional_section = f"Additional Requirements:\n{additional_instructions.strip()}"
    else:
        additional_section = ""

    # 4. Fill placeholders in user prompt template
    user_prompt_filled = user_prompt_template.replace("{AGENT_NAME}", agent_name)
    user_prompt_filled = user_prompt_filled.replace("{AGENT_DESCRIPTION}", agent_description)
    user_prompt_filled = user_prompt_filled.replace("{TOOLS_SECTION}", tools_section)
    user_prompt_filled = user_prompt_filled.replace("{ADDITIONAL_SECTION}", additional_section)
    user_prompt_filled = user_prompt_filled.replace("{SYSTEM_PROMPT}", generated_system_prompt)

    # 5. Prepare LLM credentials
    llm_api_key = api_key
    llm_model = model

    if provider == "lmstudio":
        llm_api_key = "dummy-key-for-local-llm"
        if not llm_model.startswith("openai/"):
            llm_model = f"openai/{llm_model}"

    # 6. Create LLM instance
    try:
        llm = create_llm(
            provider=provider,
            model=llm_model,
            temperature=0.7,
            api_key=llm_api_key,
            base_url=base_url
        )
    except ValueError as e:
        yield json.dumps({"error": f"Failed to create LLM: {str(e)}"}) + "\n"
        return

    # 7. Stream LLM response
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt_filled)
    ]

    accumulated_text = ""
    try:
        for chunk in llm.stream(messages):
            if hasattr(chunk, 'content') and chunk.content:
                accumulated_text += chunk.content
                yield json.dumps({"chunk": chunk.content}) + "\n"
    except Exception as e:
        yield json.dumps({"error": f"LLM streaming failed: {str(e)}"}) + "\n"
        return

    # Send final result
    yield json.dumps({
        "done": True,
        "user_prompt": accumulated_text.strip()
    }) + "\n"
