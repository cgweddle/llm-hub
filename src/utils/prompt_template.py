"""
Prompt template resolvers.

System prompt placeholders (resolved at agent creation):
  {AGENT_NAME}        → node name
  {AGENT_DESCRIPTION} → node description
  {TOOLS_SECTION}     → formatted list of tool names

User prompt placeholders (named inputs, resolved at execution time, see
claude_plans/named-agent-inputs.md): every {name} placeholder in a
sub-agent's user_prompt is a named input (decision 2); {{name}} is an escape
for the literal text "{name}" and creates no input (decision 3);
{message_history} is reserved and is not an input — it is replaced with the
serialized PydanticAI message history from previous nodes; an empty
user_prompt is treated as "{input}" (decision 2).

Backward compatible: prompts without placeholders pass through unchanged.
"""

import re
from typing import Any, Dict, List, Optional

from utils.message_serializer import serialize_messages

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


def extract_prompt_inputs(user_prompt: str) -> List[str]:
    """
    List a user prompt's named inputs: ordered, de-duplicated, skipping
    {{escaped}} braces and the reserved {message_history} placeholder.

    An empty (or whitespace-only) prompt implicitly means "{input}" (decision
    2), so it returns ["input"]. A non-empty prompt with zero {name}
    placeholders returns [].
    """
    if not user_prompt or not user_prompt.strip():
        return ["input"]

    names: List[str] = []
    seen = set()
    i = 0
    n = len(user_prompt)
    while i < n:
        if user_prompt[i] == "{":
            if i + 1 < n and user_prompt[i + 1] == "{":
                i += 2
                continue
            close = user_prompt.find("}", i + 1)
            if close == -1:
                i += 1
                continue
            name = user_prompt[i + 1:close]
            if name != "message_history" and _NAME_RE.match(name) and name not in seen:
                seen.add(name)
                names.append(name)
            i = close + 1
        else:
            i += 1
    return names


def resolve_user_prompt_template(
    user_prompt: str,
    inputs: Dict[str, str],
    predecessor_messages: Optional[List] = None,
) -> str:
    """
    Replace a user prompt's named-input placeholders with runtime values.

    Args:
        user_prompt: Raw user prompt — {name} placeholders, {{name}} escapes,
            and the reserved {message_history} placeholder.
        inputs: Named input values for this node. A missing input resolves
            to "" (decision 6).
        predecessor_messages: Accumulated PydanticAI messages from previous nodes.

    Returns:
        User prompt with placeholders resolved.
    """
    if not user_prompt or not user_prompt.strip():
        return inputs.get("input", "")

    result: List[str] = []
    i = 0
    n = len(user_prompt)
    while i < n:
        c = user_prompt[i]
        if c == "{":
            if i + 1 < n and user_prompt[i + 1] == "{":
                result.append("{")
                i += 2
                continue
            close = user_prompt.find("}", i + 1)
            if close == -1:
                result.append(c)
                i += 1
                continue
            name = user_prompt[i + 1:close]
            if name == "message_history":
                result.append(serialize_messages(predecessor_messages) if predecessor_messages else "")
            elif _NAME_RE.match(name):
                result.append(str(inputs.get(name, "")))
            else:
                result.append(user_prompt[i:close + 1])
            i = close + 1
        elif c == "}" and i + 1 < n and user_prompt[i + 1] == "}":
            result.append("}")
            i += 2
        else:
            result.append(c)
            i += 1
    return "".join(result)


def agent_external_inputs(graph_config: Dict[str, Any]) -> List[str]:
    """
    The agent's flow-level inputs (decision 8): the entry sub-agent's inputs
    that no internal edge feeds. These are the handles shown on the agent
    node on the flow canvas.
    """
    nodes_config = graph_config.get("nodes", {})
    entry_point = graph_config.get("entry_point")
    entry_config = nodes_config.get(entry_point, {})
    entry_inputs = extract_prompt_inputs(entry_config.get("user_prompt", ""))

    fed = {
        edge.get("target_input") or "input"
        for edge in graph_config.get("edges", [])
        if edge.get("to_node") == entry_point
    }
    return [name for name in entry_inputs if name not in fed]
