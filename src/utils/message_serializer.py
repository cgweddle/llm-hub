"""
Serialize PydanticAI messages into human-readable text.

Used by user prompt template resolution to render {message_history}.
Messages are stored in native PydanticAI format throughout the executor;
serialization only happens at template resolution time.
"""

from typing import List


def serialize_messages(messages: List) -> str:
    """Serialize PydanticAI messages into human-readable text.

    Iterates over ModelRequest/ModelResponse objects and their parts,
    producing a labelled transcript of the conversation.
    """
    if not messages:
        return ""

    lines: list[str] = []

    for message in messages:
        parts = getattr(message, "parts", [])
        for part in parts:
            kind = getattr(part, "part_kind", "")

            if kind == "system-prompt":
                content = getattr(part, "content", "")
                lines.append(f"[System Prompt]\n{content}")

            elif kind == "user-prompt":
                content = getattr(part, "content", "")
                lines.append(f"[User]\n{content}")

            elif kind == "text":
                content = getattr(part, "content", "")
                lines.append(f"[Assistant]\n{content}")

            elif kind == "tool-call":
                name = getattr(part, "tool_name", "unknown")
                args = getattr(part, "args", "")
                if hasattr(args, "args_json"):
                    args = args.args_json
                lines.append(f"[Tool Call: {name}]\n{args}")

            elif kind == "tool-return":
                name = getattr(part, "tool_name", "unknown")
                content = getattr(part, "content", "")
                lines.append(f"[Tool Result: {name}]\n{content}")

            # Skip ThinkingPart, RetryPromptPart, and unknown types

    return "\n\n".join(lines)
