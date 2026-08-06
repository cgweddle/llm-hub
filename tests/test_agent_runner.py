"""
Unit tests for agent_runner.compile_agent — prepare-time agent compilation.

Verifies the creation recipe without running any agent: bound-credential
model objects (no os.environ side effects), system prompt assembly with
routing lines, output-path union types, and tool registration.

A few assertions read pydantic-ai internals (_system_prompts,
_function_toolset) — the executor never does, but tests need to observe
what was compiled without running it.

Run with: pytest tests/test_agent_runner.py -v
"""
import os

import pytest

from src.runners.agent_runner import (
    BuiltAgent,
    compile_agent,
    resolve_provider_config,
    _create_model,
    _build_output_path_types,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel


LLM_CONFIG = {"models": [
    {"name": "claude-dev", "provider": "anthropic", "model": "claude-x", "api_key": "test-key"},
    {"name": "local-lm", "provider": "lmstudio", "model": "qwen"},
]}

PROVIDER_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL")


class FakeTool:
    id = 1
    name = "adder"
    description = "adds two numbers"
    tool_type = "function"
    main_function = "add"
    script_code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    function_code = script_code
    input_schema = {"a": {"type": "int", "optional": False},
                    "b": {"type": "int", "optional": False}}
    output_schema = {"type": "int"}
    helper_functions = None


def reviewer_node(**overrides):
    node = {
        "name": "Reviewer",
        "description": "Reviews drafts",
        "system_prompt": "You are {AGENT_NAME}.",
        "tool_ids": [],
    }
    node.update(overrides)
    return node


class TestResolveProviderConfig:
    def test_returns_named_entry(self):
        entry = resolve_provider_config(LLM_CONFIG, "claude-dev")
        assert entry["api_key"] == "test-key"
        assert entry["provider"] == "anthropic"

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="not found in config"):
            resolve_provider_config(LLM_CONFIG, "nope")


class TestCreateModel:
    def test_anthropic_model(self):
        model = _create_model("anthropic", "claude-x", api_key="k")
        assert isinstance(model, AnthropicModel)
        assert model.model_name == "claude-x"

    def test_lmstudio_maps_to_openai_with_defaults(self):
        model = _create_model("lmstudio", "qwen")
        assert isinstance(model, OpenAIChatModel)
        assert "localhost:1234" in str(model.client.base_url)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            _create_model("gemini", "g")

    def test_missing_model_raises(self):
        with pytest.raises(ValueError, match="Model name is required"):
            _create_model("anthropic", "")


class TestCompileAgent:
    def test_env_untouched(self):
        before = {k: os.environ.get(k) for k in PROVIDER_ENV_VARS}
        compile_agent(reviewer_node(), [], resolve_provider_config(LLM_CONFIG, "claude-dev"))
        after = {k: os.environ.get(k) for k in PROVIDER_ENV_VARS}
        assert after == before

    def test_plain_node_has_no_path_routing(self):
        built = compile_agent(reviewer_node(), [],
                              resolve_provider_config(LLM_CONFIG, "claude-dev"))
        assert isinstance(built, BuiltAgent)
        assert built.class_to_path is None
        assert isinstance(built.agent.model, AnthropicModel)

    def test_system_prompt_resolves_template_and_routing(self):
        node = reviewer_node(output_paths={
            "approve": {"description": "Output meets quality bar"},
            "revise": "Needs improvement",
        })
        built = compile_agent(node, [], resolve_provider_config(LLM_CONFIG, "claude-dev"))
        prompt = built.agent._system_prompts[0]
        assert "Reviewer" in prompt
        assert "You must choose one of the following output paths" in prompt
        assert '"Approve": Output meets quality bar' in prompt
        assert '"Revise": Needs improvement' in prompt
        assert built.class_to_path == {"Approve": "approve", "Revise": "revise"}

    def test_tools_registered_from_records(self):
        built = compile_agent(reviewer_node(tool_ids=[1]), [FakeTool()],
                              resolve_provider_config(LLM_CONFIG, "claude-dev"))
        assert "adder" in built.agent._function_toolset.tools

    def test_output_path_types_shape(self):
        union_type, class_to_path, models = _build_output_path_types(
            {"approve": "good", "revise": "bad"})
        assert set(class_to_path) == {"Approve", "Revise"}
        assert set(models) == {"approve", "revise"}
        assert models["approve"].__doc__ == "good"
