"""
System prompt template resolver.

Resolves placeholders in agent system prompts at runtime:
  {AGENT_NAME}        → node name
  {AGENT_DESCRIPTION} → node description
  {TOOLS_SECTION}     → formatted list of tool names

Backward compatible: prompts without placeholders pass through unchanged.
"""

from typing import Any, Dict, List, Optional


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
