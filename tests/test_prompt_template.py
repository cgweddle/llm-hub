"""Tests for system prompt template resolution."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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
