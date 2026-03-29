"""
Prompt template resolvers.

System prompt placeholders (resolved at agent creation):
  {AGENT_NAME}        → node name
  {AGENT_DESCRIPTION} → node description
  {TOOLS_SECTION}     → formatted list of tool names

User prompt placeholders (resolved at execution time):
  {input}             → node_input text (preceding node's output or user's runtime input)
  {message_history}   → serialized PydanticAI message history from previous nodes

Backward compatible: prompts without placeholders pass through unchanged.
"""

from typing import Any, Dict, List, Optional

from utils.message_serializer import serialize_messages


def resolve_system_prompt_template(
    system_prompt: str,
    node_config: Dict[str, Any],
    tool_records: Optional[List[Any]] = None,
) -> str:
    """
    Replace template placeholders in a system prompt with runtime values.

    Args:
        system_prompt: Raw system prompt, possibly containing placeholders.
        node_config: Graph node dict with 'name', 'description', etc.
        tool_records: List of Tool ORM objects (must have a .name attribute).

    Returns:
        System prompt with placeholders resolved.
    """
    result = system_prompt

    result = result.replace("{AGENT_NAME}", node_config.get("name", "Agent"))
    result = result.replace("{AGENT_DESCRIPTION}", node_config.get("description", ""))

    if tool_records:
        tools_list = "\n".join(f"- {t.name}" for t in tool_records)
        tools_section = (
            f"Available Tools:\n{tools_list}\n\n"
            "The agent should know how to use these tools effectively."
        )
    else:
        tools_section = "This agent has no specific tools assigned."

    result = result.replace("{TOOLS_SECTION}", tools_section)

    return result


def resolve_user_prompt_template(
    user_prompt: str,
    node_input: str,
    predecessor_messages: Optional[List] = None,
) -> str:
    """
    Replace template placeholders in a user prompt with runtime values.

    Args:
        user_prompt: Raw user prompt, possibly containing {input} and {message_history}.
        node_input: The node's input text (preceding node's output or user's runtime input).
        predecessor_messages: Accumulated PydanticAI messages from previous nodes.

    Returns:
        User prompt with placeholders resolved.
    """
    result = user_prompt

    result = result.replace("{input}", node_input)

    if predecessor_messages:
        result = result.replace("{message_history}", serialize_messages(predecessor_messages))
    else:
        result = result.replace("{message_history}", "")

    return result
