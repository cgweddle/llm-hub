"""
AI-powered judge prompt generation for evaluations.

Generates a system prompt for an LLM-as-a-judge evaluator based on
the evaluation's configured parameters (name, score type, categories,
input variables, return fields).

Prompts are loaded from the database (prompts table, name: 'eval_prompt_gen').
Run 'python src/prompts/upload_prompts.py' to populate from markdown files.
"""

import json
import os
import sys
from typing import Any, Dict, Iterator, List, Optional

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.llm_setup import create_llm
from src.database.database import get_prompt_by_name
from src.utils import get_llm_config_by_name
from langchain_core.messages import SystemMessage, HumanMessage


PROMPT_NAME = "eval_prompt_gen"


def _build_score_type_section(
    score_type: str,
    score_categories: Optional[List[Dict[str, str]]],
) -> str:
    """Build the score type description for the user prompt template."""
    parts = [f"Score type: {score_type}"]

    if score_type == "NUMERIC":
        parts.append("The judge must return a float score between 0.0 and 1.0.")
        if score_categories:
            parts.append("\nScore ranges:")
            for cat in score_categories:
                name = cat.get("name", "")
                desc = cat.get("description", "")
                parts.append(f"  - {name}" + (f": {desc}" if desc else ""))
    elif score_type == "CATEGORICAL":
        parts.append("The judge must pick exactly one category.")
        if score_categories:
            parts.append("\nAllowed categories:")
            for cat in score_categories:
                name = cat.get("name", "")
                desc = cat.get("description", "")
                parts.append(f'  - "{name}"' + (f" — {desc}" if desc else ""))
    elif score_type == "BOOLEAN":
        parts.append("The judge must return true or false.")

    return "\n".join(parts)


def _build_input_variables_section(input_variables: List[str]) -> str:
    """Build the input variables description for the user prompt template."""
    input_labels = {
        "input": "the user's input to the agent",
        "output": "the agent's output",
        "tool_output": "output from tool calls made by the agent",
    }
    input_desc = ", ".join(input_labels.get(v, v) for v in input_variables)
    return f"The judge will receive: {input_desc}"


def _build_return_format_section(
    score_type: str,
    score_categories: Optional[List[Dict[str, str]]],
    return_fields: Optional[List[str]],
) -> str:
    """Build the return format specification for the user prompt template."""
    return_format = {}
    if score_type == "NUMERIC":
        return_format["score"] = "float (0.0-1.0)"
    elif score_type == "BOOLEAN":
        return_format["score"] = "boolean"
    elif score_type == "CATEGORICAL":
        cat_names = [c.get("name", "") for c in (score_categories or [])]
        return_format["score"] = f"one of {cat_names}" if cat_names else "string"

    if return_fields:
        for field in return_fields:
            return_format[field] = "string"
    else:
        return_format["reasoning"] = "string"

    return f"The judge must return JSON in this format: {json.dumps(return_format)}"


def generate_eval_prompt_stream(
    session: Any,
    eval_name: str,
    eval_description: str,
    score_type: str,
    score_categories: Optional[List[Dict[str, str]]],
    return_fields: Optional[List[str]],
    input_variables: List[str],
    llm_model: str,
    additional_instructions: Optional[str] = None,
) -> Iterator[str]:
    """
    Generate a judge system prompt for an evaluation using LLM with streaming.

    Yields JSON strings with streaming updates.
    """
    # 1. Load prompts from database
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

    # 2. Load LLM config
    llm_config = get_llm_config_by_name(llm_model)
    if not llm_config:
        yield json.dumps({"error": f"LLM config '{llm_model}' not found in ~/.llm_hub/config.yaml"}) + "\n"
        return

    provider = llm_config["provider"]
    model = llm_config["model"]
    api_key = llm_config.get("api_key")
    base_url = llm_config.get("base_url")

    # 3. Build template sections
    eval_description_section = f"\nDescription: {eval_description}" if eval_description else ""
    score_type_section = _build_score_type_section(score_type, score_categories)
    input_variables_section = _build_input_variables_section(input_variables)
    return_format_section = _build_return_format_section(score_type, score_categories, return_fields)

    if additional_instructions and additional_instructions.strip():
        additional_section = f"\nAdditional Requirements:\n{additional_instructions.strip()}"
    else:
        additional_section = ""

    # 4. Fill placeholders in user prompt template
    user_message = user_prompt_template.replace("{EVAL_NAME}", eval_name)
    user_message = user_message.replace("{EVAL_DESCRIPTION_SECTION}", eval_description_section)
    user_message = user_message.replace("{SCORE_TYPE_SECTION}", score_type_section)
    user_message = user_message.replace("{INPUT_VARIABLES_SECTION}", input_variables_section)
    user_message = user_message.replace("{RETURN_FORMAT_SECTION}", return_format_section)
    user_message = user_message.replace("{ADDITIONAL_SECTION}", additional_section)

    # 5. Create LLM
    llm_api_key = api_key
    llm_model_name = model

    if provider == "lmstudio":
        llm_api_key = "dummy-key-for-local-llm"
        if not llm_model_name.startswith("openai/"):
            llm_model_name = f"openai/{llm_model_name}"

    try:
        llm = create_llm(
            provider=provider,
            model=llm_model_name,
            temperature=0.7,
            api_key=llm_api_key,
            base_url=base_url,
        )
    except ValueError as e:
        yield json.dumps({"error": f"Failed to create LLM: {str(e)}"}) + "\n"
        return

    # 6. Stream LLM response
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    accumulated_text = ""
    try:
        for chunk in llm.stream(messages):
            if hasattr(chunk, "content") and chunk.content:
                accumulated_text += chunk.content
                yield json.dumps({"chunk": chunk.content}) + "\n"
    except Exception as e:
        yield json.dumps({"error": f"LLM streaming failed: {str(e)}"}) + "\n"
        return

    yield json.dumps({"done": True, "system_prompt": accumulated_text}) + "\n"
