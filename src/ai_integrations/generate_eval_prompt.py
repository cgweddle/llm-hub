"""
AI-powered judge prompt generation for evaluations.

Generates a system prompt for an LLM-as-a-judge evaluator based on
the evaluation's configured parameters (name, score type, categories,
input variables, return fields).
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
from src.utils import get_llm_config_by_name
from langchain_core.messages import SystemMessage, HumanMessage


META_SYSTEM_PROMPT = """You are an expert at writing LLM-as-a-judge evaluation prompts.

Your task is to generate a clear, specific system prompt that will be used by a judge LLM to evaluate outputs from AI agents.

The generated prompt should:
- Clearly explain what aspect of the output is being evaluated
- Provide specific, actionable scoring criteria
- Specify the exact return format as JSON
- Be concise but thorough
- Not include example inputs — those will be provided at runtime

Output ONLY the system prompt text. Do not wrap it in quotes or markdown."""


def generate_eval_prompt_stream(
    eval_name: str,
    eval_description: str,
    score_type: str,
    score_categories: Optional[List[Dict[str, str]]],
    return_fields: Optional[List[str]],
    input_variables: List[str],
    llm_model: str,
) -> Iterator[str]:
    """
    Generate a judge system prompt for an evaluation using LLM with streaming.

    Yields JSON strings with streaming updates.
    """
    llm_config = get_llm_config_by_name(llm_model)
    if not llm_config:
        yield json.dumps({"error": f"LLM config '{llm_model}' not found in ~/.llm_hub/config.yaml"}) + "\n"
        return

    provider = llm_config["provider"]
    model = llm_config["model"]
    api_key = llm_config.get("api_key")
    base_url = llm_config.get("base_url")

    # Build the user message describing what prompt to generate
    parts = [f"Generate a judge system prompt for an evaluation called \"{eval_name}\"."]

    if eval_description:
        parts.append(f"\nDescription: {eval_description}")

    # Score type section
    parts.append(f"\nScore type: {score_type}")
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

    # Input variables
    input_labels = {
        "input": "the user's input to the agent",
        "output": "the agent's output",
        "tool_output": "output from tool calls made by the agent",
    }
    input_desc = ", ".join(input_labels.get(v, v) for v in input_variables)
    parts.append(f"\nThe judge will receive: {input_desc}")

    # Return format
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

    parts.append(f"\nThe judge must return JSON in this format: {json.dumps(return_format)}")

    user_message = "\n".join(parts)

    # Create LLM
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

    messages = [
        SystemMessage(content=META_SYSTEM_PROMPT),
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
