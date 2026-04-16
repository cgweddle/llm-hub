"""
AI-powered Python tool code generation module.

This module provides functionality to generate Python tool code using LLMs,
based on tool name and description provided by the user.
"""

import re
import ast
import os
import sys
import json
from typing import Any, AsyncIterator, Dict, Optional

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from pydantic_ai import Agent

from src.database.database import get_prompt_by_name


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


def _extract_code_from_markdown(text: str) -> str:
    code_block_pattern = r"```(?:python)?\n(.*?)\n```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()


def _extract_main_function_name(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Generated code has syntax errors: {str(e)}")

    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    if not functions:
        raise ValueError("No functions found in generated code")
    return functions[-1].name


async def generate_tool_code_stream(
    session: Any,
    tool_name: str,
    tool_description: str,
    llm_config: Dict,
    additional_instructions: Optional[str] = None,
) -> AsyncIterator[str]:
    prompt_record = get_prompt_by_name(session, "python_code_gen")

    if not prompt_record:
        yield json.dumps({"error": "Code generation prompts not found in database"}) + "\n"
        return

    system_prompt = prompt_record.system_prompt
    user_prompt = prompt_record.user_prompt

    if not system_prompt or not user_prompt:
        yield json.dumps({"error": "System prompt or user prompt is empty in database"}) + "\n"
        return

    user_prompt_filled = user_prompt.replace("{TOOL_NAME}", tool_name)
    user_prompt_filled = user_prompt_filled.replace("{TOOL_DESCRIPTION}", tool_description)

    if additional_instructions and additional_instructions.strip():
        user_prompt_filled += f"\n\nAdditional instructions about the script:\n{additional_instructions.strip()}"

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

    generated_code = _extract_code_from_markdown(accumulated_text)

    try:
        main_function = _extract_main_function_name(generated_code)
        yield json.dumps({
            "done": True,
            "script_code": generated_code,
            "main_function": main_function
        }) + "\n"
    except Exception as e:
        yield json.dumps({
            "done": True,
            "script_code": generated_code,
            "error": f"Failed to extract main function: {str(e)}"
        }) + "\n"


async def edit_tool_code_stream(
    session: Any,
    existing_code: str,
    editing_instructions: str,
    tool_name: str,
    tool_description: str,
    llm_config: Dict,
) -> AsyncIterator[str]:
    prompt_record = get_prompt_by_name(session, "python_code_gen")

    if not prompt_record:
        yield json.dumps({"error": "Code generation prompts not found in database"}) + "\n"
        return

    system_prompt = prompt_record.system_prompt
    user_prompt_template = prompt_record.user_prompt

    if not system_prompt or not user_prompt_template:
        yield json.dumps({"error": "System prompt or user prompt is empty in database"}) + "\n"
        return

    user_prompt = f"""I need you to edit the following Python script.

Tool name: {tool_name}
Tool description: {tool_description}

Current code:
```python
{existing_code}
```

Editing instructions:
{editing_instructions}

Please provide the complete edited Python script following all the rules in the system prompt."""

    try:
        agent = _build_agent(llm_config, system_prompt)
    except ValueError as e:
        yield json.dumps({"error": f"Failed to create LLM: {str(e)}"}) + "\n"
        return

    accumulated_text = ""
    try:
        async with agent.run_stream(user_prompt) as stream:
            async for chunk in stream.stream_text(delta=True):
                accumulated_text += chunk
                yield json.dumps({"chunk": chunk}) + "\n"
    except Exception as e:
        yield json.dumps({"error": f"LLM streaming failed: {str(e)}"}) + "\n"
        return

    generated_code = _extract_code_from_markdown(accumulated_text)

    try:
        main_function = _extract_main_function_name(generated_code)
        yield json.dumps({
            "done": True,
            "script_code": generated_code,
            "main_function": main_function
        }) + "\n"
    except Exception as e:
        yield json.dumps({
            "done": True,
            "script_code": generated_code,
            "error": f"Failed to extract main function: {str(e)}"
        }) + "\n"
