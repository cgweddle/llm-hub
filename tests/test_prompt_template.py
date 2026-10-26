"""Tests for system prompt template resolution."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pydantic_ai.messages import ModelResponse, TextPart

from utils.prompt_template import resolve_system_prompt_template


class FakeTool:
    """Minimal stand-in for a Tool ORM record."""
    def __init__(self, name: str):
        self.name = name


class TestResolveSystemPromptTemplate:

    def test_all_placeholders_resolved(self):
        template = "You are {AGENT_NAME}. {AGENT_DESCRIPTION}\n\n{TOOLS_SECTION}"
        node_config = {"name": "Email Bot", "description": "Classifies emails by priority"}
        tools = [FakeTool("send_email"), FakeTool("read_inbox")]

        result = resolve_system_prompt_template(template, node_config, tools)

        assert "Email Bot" in result
        assert "Classifies emails by priority" in result
        assert "- send_email" in result
        assert "- read_inbox" in result
        assert "{AGENT_NAME}" not in result
        assert "{AGENT_DESCRIPTION}" not in result
        assert "{TOOLS_SECTION}" not in result

    def test_no_placeholders_passthrough(self):
        """Old baked prompts without placeholders should pass through unchanged."""
        baked = "You are an email classifier. You handle incoming messages."
        node_config = {"name": "Email Bot", "description": "Classifies emails"}

        result = resolve_system_prompt_template(baked, node_config)

        assert result == baked

    def test_partial_placeholders(self):
        template = "You are {AGENT_NAME}. You help with tasks."
        node_config = {"name": "Helper", "description": "General assistant"}

        result = resolve_system_prompt_template(template, node_config)

        assert "Helper" in result
        assert "{AGENT_NAME}" not in result
        # Description placeholder was never in the template, so description value won't appear
        assert "General assistant" not in result

    def test_empty_tools(self):
        template = "You are {AGENT_NAME}.\n\n{TOOLS_SECTION}"
        node_config = {"name": "Bot"}

        result = resolve_system_prompt_template(template, node_config, tool_records=[])

        assert "no specific tools assigned" in result

    def test_none_tools(self):
        template = "{TOOLS_SECTION}"
        node_config = {"name": "Bot"}

        result = resolve_system_prompt_template(template, node_config, tool_records=None)

        assert "no specific tools assigned" in result

    def test_missing_name_uses_default(self):
        template = "You are {AGENT_NAME}."
        node_config = {}

        result = resolve_system_prompt_template(template, node_config)

        assert result == "You are Agent."

    def test_missing_description_uses_empty(self):
        template = "Role: {AGENT_DESCRIPTION}"
        node_config = {}

        result = resolve_system_prompt_template(template, node_config)

        assert result == "Role: "

    def test_tools_section_format(self):
        template = "{TOOLS_SECTION}"
        node_config = {"name": "Bot"}
        tools = [FakeTool("search"), FakeTool("calculate")]

        result = resolve_system_prompt_template(template, node_config, tools)

        assert result.startswith("Available Tools:")
        assert "- search" in result
        assert "- calculate" in result
        assert "use these tools effectively" in result

    def test_multiple_occurrences(self):
        template = "{AGENT_NAME} is great. Call me {AGENT_NAME}."
        node_config = {"name": "Aria"}

        result = resolve_system_prompt_template(template, node_config)

        assert result == "Aria is great. Call me Aria."
        assert "{AGENT_NAME}" not in result


# ─── Named inputs (claude_plans/named-agent-inputs.md) ───────────────────────
#
# New surface under test, not yet implemented:
#   extract_prompt_inputs(user_prompt: str) -> list[str]
#   resolve_user_prompt_template(user_prompt: str, inputs: dict[str, str],
#                                 predecessor_messages=None) -> str
#   agent_external_inputs(graph_config: dict) -> list[str]
#
# These imports are expected to fail (ImportError) until the implementer adds
# them to src/utils/prompt_template.py — that is the "red" we want.

from utils.prompt_template import (
    extract_prompt_inputs,
    resolve_user_prompt_template,
    agent_external_inputs,
)


class TestExtractPromptInputs:

    def test_single_input(self):
        assert extract_prompt_inputs("Summarize: {input}") == ["input"]

    def test_order_preserved(self):
        assert extract_prompt_inputs("{plan} then {draft} then {plan}") == ["plan", "draft"]

    def test_duplicates_removed_keeps_first_position(self):
        result = extract_prompt_inputs("{a} {b} {a} {c} {b}")
        assert result == ["a", "b", "c"]

    def test_message_history_excluded(self):
        assert extract_prompt_inputs("{input}\n\n{message_history}") == ["input"]

    def test_only_message_history_yields_no_inputs(self):
        assert extract_prompt_inputs("{message_history}") == []

    def test_double_brace_is_escape_not_input(self):
        # {{template}} is literal text "{template}" sent to the LLM — no input.
        result = extract_prompt_inputs("Use {{template}} but fill in {name}.")
        assert result == ["name"]

    def test_escape_only_yields_no_inputs(self):
        assert extract_prompt_inputs("Just {{literal}} braces, nothing else.") == []

    def test_json_like_braces_ignored(self):
        # Not a valid [A-Za-z_][A-Za-z0-9_]* identifier, so it's not an input.
        result = extract_prompt_inputs('Context: {"a": 1} and also {real_name}')
        assert result == ["real_name"]

    def test_empty_prompt_returns_input(self):
        assert extract_prompt_inputs("") == ["input"]

    def test_whitespace_only_prompt_returns_input(self):
        assert extract_prompt_inputs("   \n\t  ") == ["input"]

    def test_underscore_and_digits_in_name(self):
        assert extract_prompt_inputs("{draft_v2} review") == ["draft_v2"]

    def test_name_cannot_start_with_digit(self):
        # "2bad" is not a valid identifier start; the whole {2bad} is ignored.
        assert extract_prompt_inputs("{2bad} {ok_name}") == ["ok_name"]

    def test_no_placeholders_at_all_is_empty_list(self):
        # Non-empty prompt with zero {name} placeholders — not the "empty
        # prompt" special case, so no implicit ["input"].
        assert extract_prompt_inputs("Just a plain instruction, no braces.") == []


class TestResolveUserPromptTemplate:

    def test_substitutes_named_input(self):
        result = resolve_user_prompt_template("Hello {name}!", {"name": "World"})
        assert result == "Hello World!"

    def test_multiple_named_inputs(self):
        result = resolve_user_prompt_template(
            "{plan} / {feedback}", {"plan": "P", "feedback": "F"}
        )
        assert result == "P / F"

    def test_missing_input_resolves_to_empty_string(self):
        result = resolve_user_prompt_template("Notes: {feedback}", {})
        assert result == "Notes: "

    def test_missing_input_among_others(self):
        result = resolve_user_prompt_template(
            "{plan} {feedback}", {"plan": "P"}
        )
        assert result == "P "

    def test_double_brace_escape_renders_literal(self):
        result = resolve_user_prompt_template("{{name}}", {"name": "ignored"})
        assert result == "{name}"

    def test_escape_alongside_real_input(self):
        result = resolve_user_prompt_template(
            "Use {{template}} for {name}.", {"name": "World"}
        )
        assert result == "Use {template} for World."

    def test_empty_prompt_uses_input_key(self):
        result = resolve_user_prompt_template("", {"input": "fallback text"})
        assert result == "fallback text"

    def test_empty_prompt_with_no_input_key_is_empty(self):
        result = resolve_user_prompt_template("", {"other": "value"})
        assert result == ""

    def test_message_history_blank_when_no_predecessor(self):
        result = resolve_user_prompt_template(
            "{input} -- {message_history}", {"input": "x"}, predecessor_messages=None
        )
        assert result == "x -- "

    def test_message_history_filled_when_predecessor_present(self):
        # predecessor_messages are always real PydanticAI message objects at
        # runtime (never plain dicts) — use the same objects the executor
        # and exporter deal with.
        predecessor_messages = [ModelResponse(parts=[TextPart(content="hi")])]
        result = resolve_user_prompt_template(
            "{input} -- {message_history}",
            {"input": "x"},
            predecessor_messages=predecessor_messages,
        )
        assert "{message_history}" not in result
        assert result.startswith("x -- ")
        assert len(result) > len("x -- ")
        assert "hi" in result

    def test_input_is_an_ordinary_dict_key(self):
        # {input} is not special — it's just whatever key "input" maps to.
        result = resolve_user_prompt_template("{input}", {"input": "seed value"})
        assert result == "seed value"


class TestAgentExternalInputs:

    def test_single_node_no_edges(self):
        graph_config = {
            "nodes": {"main": {"user_prompt": "{input}"}},
            "edges": [],
            "entry_point": "main",
            "exit_points": ["main"],
        }
        assert agent_external_inputs(graph_config) == ["input"]

    def test_entry_inputs_not_fed_internally_are_external(self):
        graph_config = {
            "nodes": {
                "writer": {"user_prompt": "{plan} {feedback}"},
                "critic": {"user_prompt": "{draft}"},
            },
            "edges": [
                {"from_node": "writer", "to_node": "critic", "is_loop": False},
                {"from_node": "critic", "to_node": "writer", "is_loop": True,
                 "target_input": "feedback"},
            ],
            "entry_point": "writer",
            "exit_points": ["critic"],
        }
        # "feedback" is fed by critic's internal loop edge; "plan" has nothing
        # feeding it, so it's the agent's external flow-level input.
        assert agent_external_inputs(graph_config) == ["plan"]

    def test_legacy_edge_without_target_input_feeds_input(self):
        graph_config = {
            "nodes": {
                "writer": {"user_prompt": "{input}"},
                "other": {"user_prompt": "{input}"},
            },
            "edges": [
                # No target_input — legacy edges target "input" (decision 5).
                {"from_node": "other", "to_node": "writer", "is_loop": True},
            ],
            "entry_point": "writer",
            "exit_points": ["writer"],
        }
        assert agent_external_inputs(graph_config) == []

    def test_order_preserved_for_external_inputs(self):
        graph_config = {
            "nodes": {"writer": {"user_prompt": "{a} {b} {c}"}},
            "edges": [],
            "entry_point": "writer",
            "exit_points": ["writer"],
        }
        assert agent_external_inputs(graph_config) == ["a", "b", "c"]
