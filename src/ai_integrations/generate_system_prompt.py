"""
AI-powered system prompt generation module for agents.

This module provides functionality to generate agent system prompts using LLMs,
based on agent name, description, and available tools.
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
from langchain_core.messages import SystemMessage, HumanMessage


META_SYSTEM_PROMPT = """You are an expert AI prompt engineer. Your task is to write a clear, effective system prompt for an AI agent.

The system prompt you write should:
1. Define the agent's role and personality
2. Specify its capabilities and limitations
3. Include instructions for how to use any available tools
4. Set appropriate behavioral guidelines
5. Be well-structured and comprehensive

Output ONLY the system prompt text. Do not include any meta-commentary, markdown formatting, or explanations."""

USER_PROMPT_TEMPLATE = """Write a system prompt for an AI agent with the following details:

Agent Name: {agent_name}
Agent Description: {agent_description}

{tools_section}

{additional_section}

Write the system prompt now."""


def generate_system_prompt_stream(
    agent_name: str,
    agent_description: str,
    tool_names: List[str],
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    additional_instructions: Optional[str] = None
) -> Iterator[str]:
    """
    Generate an agent system prompt using LLM with streaming.

    Args:
        agent_name: Name of the agent
        agent_description: Description of what the agent does
        tool_names: List of tool names available to the agent
        provider: LLM provider (e.g., 'anthropic', 'openai', 'gemini', 'lmstudio')
        model: Model name
        api_key: Optional API key for the provider
        base_url: Optional base URL
        additional_instructions: Optional extra instructions for prompt generation

    Yields:
        JSON strings with streaming updates
    """
    # Build tools section
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

    # Fill user prompt template
    user_prompt = USER_PROMPT_TEMPLATE.format(
        agent_name=agent_name,
        agent_description=agent_description,
        tools_section=tools_section,
        additional_section=additional_section
    )

    # Prepare LLM credentials
    llm_api_key = api_key
    llm_base_url = base_url
    llm_model = model

    if provider == "lmstudio":
        llm_api_key = "dummy-key-for-local-llm"
        if not llm_model.startswith("openai/"):
            llm_model = f"openai/{llm_model}"

    # Create LLM instance
    try:
        llm = create_llm(
            provider=provider,
            model=llm_model,
            temperature=0.7,
            api_key=llm_api_key,
            base_url=llm_base_url
        )
    except ValueError as e:
        yield json.dumps({"error": f"Failed to create LLM: {str(e)}"}) + "\n"
        return

    # Stream LLM response
    messages = [
        SystemMessage(content=META_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
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
