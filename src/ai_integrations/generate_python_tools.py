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
from typing import Any, Dict, Optional, Iterator

# Add project root to path
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.database.database import get_prompt_by_name
from src.llm_setup import create_llm
from langchain_core.messages import SystemMessage, HumanMessage


def generate_tool_code(
    session: Any,
    tool_name: str,
    tool_description: str,
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate Python tool code using LLM based on tool name and description.

    Args:
        session: Database session
        tool_name: Name of the tool to generate
        tool_description: Description of what the tool should do
        provider: LLM provider (e.g., 'anthropic', 'openai', 'gemini', 'lmstudio')
        model: Model name (e.g., 'claude-3-5-sonnet-20241022')
        api_key: Optional API key for the provider
        base_url: Optional base URL (for LM Studio or custom endpoints)

    Returns:
        Dict with 'script_code' and 'main_function' keys

    Raises:
        ValueError: If prompts not found or code generation fails
        Exception: If LLM call fails
    """
    # 1. Query prompts from database
    prompt_record = get_prompt_by_name(session, "python_code_gen")

    if not prompt_record:
        raise ValueError(
            "Code generation prompts not found in database. "
            "Please run 'python src/prompts/upload_prompts.py' to upload prompts."
        )

    system_prompt = prompt_record.system_prompt
    user_prompt = prompt_record.user_prompt

    if not system_prompt or not user_prompt:
        raise ValueError("System prompt or user prompt is empty in database")

    # 2. Replace placeholders in user_prompt
    user_prompt_filled = user_prompt.replace("{TOOL_NAME}", tool_name)
    user_prompt_filled = user_prompt_filled.replace("{TOOL_DESCRIPTION}", tool_description)

    # 3. Prepare API credentials and base URL
    llm_api_key = api_key
    llm_base_url = base_url
    llm_model = model

    # Special handling for LM Studio (OpenAI-compatible local server)
    if provider == "lmstudio":
        # Use a dummy API key for LM Studio (required by LiteLLM but not validated by local server)
        llm_api_key = "dummy-key-for-local-llm"
        # Prefix model with "openai/" to tell LiteLLM to use OpenAI format
        if not llm_model.startswith("openai/"):
            llm_model = f"openai/{llm_model}"

    # 4. Create LLM instance
    try:
        llm = create_llm(
            provider=provider,
            model=llm_model,
            temperature=0.3,
            api_key=llm_api_key,
            base_url=llm_base_url
        )
    except ValueError as e:
        raise ValueError(f"Failed to create LLM: {str(e)}")

    # 5. Call LLM to generate code
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt_filled)
    ]

    try:
        response = llm.invoke(messages)
        generated_text = response.content
    except Exception as e:
        raise Exception(f"LLM invocation failed: {str(e)}")

    # 6. Extract Python code from markdown if present
    generated_code = _extract_code_from_markdown(generated_text)

    # 7. Extract main function name from generated code
    try:
        main_function = _extract_main_function_name(generated_code)
    except Exception as e:
        # If extraction fails, use a default or raise error
        raise ValueError(f"Failed to extract main function name from generated code: {str(e)}")

    return {
        "script_code": generated_code,
        "main_function": main_function
    }


def _extract_code_from_markdown(text: str) -> str:
    """
    Extract Python code from markdown code blocks.

    If the text contains ```python or ``` code blocks, extract the code.
    Otherwise, return the text as-is.

    Args:
        text: Text that may contain markdown code blocks

    Returns:
        Extracted Python code
    """
    # Pattern to match markdown code blocks with optional 'python' language identifier
    code_block_pattern = r"```(?:python)?\n(.*?)\n```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)

    if matches:
        # Return the first code block found
        return matches[0].strip()

    # No code block found, return the text as-is
    return text.strip()


def generate_tool_code_stream(
    session: Any,
    tool_name: str,
    tool_description: str,
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    additional_instructions: Optional[str] = None
) -> Iterator[str]:
    """
    Generate Python tool code using LLM with streaming.

    Args:
        session: Database session
        tool_name: Name of the tool to generate
        tool_description: Description of what the tool should do
        provider: LLM provider (e.g., 'anthropic', 'openai', 'gemini', 'lmstudio')
        model: Model name (e.g., 'claude-3-5-sonnet-20241022')
        api_key: Optional API key for the provider
        base_url: Optional base URL (for LM Studio or custom endpoints)
        additional_instructions: Optional additional instructions to append to user prompt

    Yields:
        JSON strings with streaming updates

    Raises:
        ValueError: If prompts not found or code generation fails
        Exception: If LLM call fails
    """
    # 1. Query prompts from database
    prompt_record = get_prompt_by_name(session, "python_code_gen")

    if not prompt_record:
        yield json.dumps({"error": "Code generation prompts not found in database"}) + "\n"
        return

    system_prompt = prompt_record.system_prompt
    user_prompt = prompt_record.user_prompt

    if not system_prompt or not user_prompt:
        yield json.dumps({"error": "System prompt or user prompt is empty in database"}) + "\n"
        return

    # 2. Replace placeholders in user_prompt
    user_prompt_filled = user_prompt.replace("{TOOL_NAME}", tool_name)
    user_prompt_filled = user_prompt_filled.replace("{TOOL_DESCRIPTION}", tool_description)

    # Append additional instructions if provided
    if additional_instructions and additional_instructions.strip():
        user_prompt_filled += f"\n\nAdditional instructions about the script:\n{additional_instructions.strip()}"

    # 3. Prepare API credentials and base URL
    llm_api_key = api_key
    llm_base_url = base_url
    llm_model = model

    # Special handling for LM Studio (OpenAI-compatible local server)
    if provider == "lmstudio":
        llm_api_key = "dummy-key-for-local-llm"
        if not llm_model.startswith("openai/"):
            llm_model = f"openai/{llm_model}"

    # 4. Create LLM instance
    try:
        llm = create_llm(
            provider=provider,
            model=llm_model,
            temperature=0.3,
            api_key=llm_api_key,
            base_url=llm_base_url
        )
    except ValueError as e:
        yield json.dumps({"error": f"Failed to create LLM: {str(e)}"}) + "\n"
        return

    # 5. Stream LLM response
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

    # 6. Extract Python code from markdown if present
    generated_code = _extract_code_from_markdown(accumulated_text)

    # 7. Extract main function name from generated code
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


def edit_tool_code_stream(
    session: Any,
    existing_code: str,
    editing_instructions: str,
    tool_name: str,
    tool_description: str,
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> Iterator[str]:
    """
    Edit existing Python tool code using LLM with streaming.

    Args:
        session: Database session
        existing_code: The current Python code to edit
        editing_instructions: Instructions for how to modify the code
        tool_name: Name of the tool
        tool_description: Description of what the tool does
        provider: LLM provider (e.g., 'anthropic', 'openai', 'gemini', 'lmstudio')
        model: Model name (e.g., 'claude-3-5-sonnet-20241022')
        api_key: Optional API key for the provider
        base_url: Optional base URL (for LM Studio or custom endpoints)

    Yields:
        JSON strings with streaming updates

    Raises:
        ValueError: If prompts not found or code editing fails
        Exception: If LLM call fails
    """
    # 1. Query prompts from database
    prompt_record = get_prompt_by_name(session, "python_code_gen")

    if not prompt_record:
        yield json.dumps({"error": "Code generation prompts not found in database"}) + "\n"
        return

    system_prompt = prompt_record.system_prompt
    user_prompt_template = prompt_record.user_prompt

    if not system_prompt or not user_prompt_template:
        yield json.dumps({"error": "System prompt or user prompt is empty in database"}) + "\n"
        return

    # 2. Build editing prompt
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

    # 3. Prepare API credentials and base URL
    llm_api_key = api_key
    llm_base_url = base_url
    llm_model = model

    # Special handling for LM Studio (OpenAI-compatible local server)
    if provider == "lmstudio":
        llm_api_key = "dummy-key-for-local-llm"
        if not llm_model.startswith("openai/"):
            llm_model = f"openai/{llm_model}"

    # 4. Create LLM instance
    try:
        llm = create_llm(
            provider=provider,
            model=llm_model,
            temperature=0.3,
            api_key=llm_api_key,
            base_url=llm_base_url
        )
    except ValueError as e:
        yield json.dumps({"error": f"Failed to create LLM: {str(e)}"}) + "\n"
        return

    # 5. Stream LLM response
    messages = [
        SystemMessage(content=system_prompt),
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

    # 6. Extract Python code from markdown if present
    generated_code = _extract_code_from_markdown(accumulated_text)

    # 7. Extract main function name from generated code
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


def _extract_main_function_name(code: str) -> str:
    """
    Extract the main function name from Python code using AST parsing.

    The main function is identified as:
    1. The last function defined in the module (convention for entry point)
    2. If no functions found, raise an error

    Args:
        code: Python code as string

    Returns:
        Name of the main function

    Raises:
        ValueError: If no functions found or code cannot be parsed
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Generated code has syntax errors: {str(e)}")

    # Find all function definitions
    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

    if not functions:
        raise ValueError("No functions found in generated code")

    # Get the last defined function (typically the entry point)
    main_func = functions[-1]

    return main_func.name
