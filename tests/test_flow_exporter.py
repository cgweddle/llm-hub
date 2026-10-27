"""
Tests for the flow exporter (src/exporters/flow_exporter.py).

Every emitted .py file must compile; wiring assertions check that edge
mappings, typed-in values, and parallelism are inlined as readable source;
guard tests cover Phase-1 limits (single-node agents only), credential-leak
prevention, and the no-llmhub-imports contract.

Uses a real per-test SQLite DB like test_flow_runner_agents.py.
"""
import asyncio
import ast
import importlib
import inspect
import io
import json
import os
import re
import sys
import typing
import zipfile
from pathlib import Path

import pytest

import pydantic_ai
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import FunctionModel

from src.database.database_setup import (
    DatabaseManager, User, Tool, Flow, Agent as AgentModel,
)
from src.exporters.flow_exporter import (
    FlowExportError, _docstring_text, export_flow, export_flow_zip,
)


LLM_CONFIG = {"models": [
    {"name": "anth", "provider": "anthropic", "model": "claude-x",
     "api_key": "sk-SECRET123"},
    {"name": "lm", "provider": "lmstudio", "model": "qwen"},
]}


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    manager = DatabaseManager()
    manager.create_tables()
    session = manager.get_session()
    user = User(username="tester", email="t@t.t", password_hash="x")
    session.add(user)
    session.commit()
    session.refresh(user)
    yield session, user
    session.close()


def _purge_export_modules():
    """Drop any previously imported export package so tests never leak state."""
    for name in list(sys.modules):
        if (name == "agents" or name.startswith("agents.")
                or name == "tools" or name.startswith("tools.")):
            del sys.modules[name]


@pytest.fixture
def import_exported(tmp_path, monkeypatch):
    """Write an exported file set under tmp_path and import one of its modules.

    Puts tmp_path first on sys.path (some environments have an unrelated
    installed "agents" package), purges "agents"/"tools" modules before
    importing and at teardown, and sets a dummy ANTHROPIC_API_KEY so the
    real (never-called) model object can be constructed at import time.
    """
    _purge_export_modules()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    def _do(files, module_name):
        for rel_path, content in files.items():
            full = tmp_path / rel_path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        _purge_export_modules()
        return importlib.import_module(module_name)

    yield _do
    _purge_export_modules()


def make_tool(session, user, name, script, main_function,
              input_schema=None, output_schema=None, required_packages=None):
    tool = Tool(
        user_id=user.id, name=name, tool_type="function",
        main_function=main_function, script_code=script,
        input_schema=input_schema or {"properties": {"x": {"type": "str"}}},
        output_schema=output_schema,
        required_packages=required_packages,
    )
    session.add(tool)
    session.commit()
    session.refresh(tool)
    return tool


def make_agent(session, user, graph_config, name="Reviewer"):
    agent = AgentModel(user_id=user.id, name=name, graph_config=graph_config)
    session.add(agent)
    session.commit()
    session.refresh(agent)
    return agent


def make_flow(session, user, nodes, edges, entry, exits, name="test-flow"):
    flow = Flow(
        user_id=user.id, name=name,
        graph_config={"nodes": nodes, "edges": edges,
                      "entry_point": entry, "exit_points": exits},
        entry_point=entry, exit_points=exits,
    )
    session.add(flow)
    session.commit()
    session.refresh(flow)
    return flow


def make_fetch_tool(session, user):
    return make_tool(
        session, user, "Fetch Page",
        'def fetch_page(url):\n    return {"text": url + "!", "status": 200}\n',
        "fetch_page",
        input_schema={"properties": {"url": {"type": "str"}}},
        output_schema={"type": "object",
                       "properties": {"text": {"type": "str"},
                                      "status": {"type": "int"}}},
        required_packages=["requests"],
    )


def make_summarize_tool(session, user):
    return make_tool(
        session, user, "Summarize",
        "def summarize_text(text, max_words):\n    return text[:max_words]\n",
        "summarize_text",
        input_schema={"properties": {"text": {"type": "str"},
                                     "max_words": {"type": "int"}}},
        output_schema={"type": "str"},
        required_packages=["pandas", "sqlalchemy"],
    )


def single_node_agent_config(**overrides):
    node = {"agent_type": "pydanticai", "name": "Main",
            "system_prompt": "You are {AGENT_NAME}."}
    node.update(overrides)
    return {"nodes": {"main": node}, "edges": [],
            "entry_point": "main", "exit_points": ["main"]}


def writer_reviewer_agent_config(writer_overrides=None, reviewer_overrides=None,
                                  edges=None, max_loop_iterations=2):
    """The canonical multi-node fixture: Writer -> Reviewer, reviewer routes
    revise/approve, the revise path loops back to Writer."""
    writer = {"agent_type": "pydanticai", "name": "writer", "system_prompt": "w"}
    writer.update(writer_overrides or {})
    reviewer = {
        "agent_type": "reflection", "name": "reviewer", "system_prompt": "r",
        "output_paths": {
            "revise": {"description": "Needs another pass"},
            "approve": {"description": "Good to go"},
        },
    }
    reviewer.update(reviewer_overrides or {})
    return {
        "nodes": {"writer": writer, "reviewer": reviewer},
        "edges": edges if edges is not None else [
            {"from_node": "writer", "to_node": "reviewer", "is_loop": False},
            {"from_node": "reviewer", "to_node": "writer", "is_loop": True,
             "output_path": "revise"},
        ],
        "entry_point": "writer", "exit_points": ["reviewer"],
        "max_loop_iterations": max_loop_iterations,
    }


def assert_all_compile(files):
    for path, content in files.items():
        if path.endswith(".py"):
            compile(content, path, "exec")


def assert_standalone(files):
    for path, content in files.items():
        assert not re.search(r"(from|import)\s+src\.", content), path
        assert "llm_hub" not in content and "llmhub" not in content, path


def assert_no_secret_leak(files):
    assert all("sk-SECRET123" not in content for content in files.values())


# ─── Tool-only flows ─────────────────────────────────────────────────────────

def test_parallel_flow_wiring(db):
    """Trigger fans out to two tools (parallel level), which join into a third."""
    session, user = db
    fetch = make_fetch_tool(session, user)
    summarize = make_summarize_tool(session, user)
    join = make_tool(
        session, user, "Join",
        "def join_parts(page_text, summary):\n    return page_text + summary\n",
        "join_parts",
        input_schema={"properties": {"page_text": {"type": "str"},
                                     "summary": {"type": "str"}}},
        output_schema={"type": "str"},
    )
    flow = make_flow(
        session, user,
        nodes={
            "T": {"node_type": "trigger", "id": 0, "name": "Trigger",
                  "input_value": "https://example.com"},
            "F": {"node_type": "tool", "id": fetch.id, "name": "Fetch Page"},
            "S": {"node_type": "tool", "id": summarize.id, "name": "Summarize",
                  "input_values": {"max_words": 100}},
            "J": {"node_type": "tool", "id": join.id, "name": "Join"},
        },
        edges=[
            {"from_node": "T", "to_node": "F", "mapping": {"": "url"}},
            {"from_node": "T", "to_node": "S"},
            {"from_node": "F", "to_node": "J", "mapping": {"text": "page_text"}},
            {"from_node": "S", "to_node": "J", "mapping": {"": "summary"}},
        ],
        entry="T", exits=["J"],
    )

    files = export_flow(flow, session, LLM_CONFIG)
    assert_all_compile(files)
    assert_standalone(files)
    assert "agents/__init__.py" not in files

    flow_py = files["flow.py"]
    # entry trigger seeds from initial_input with the typed-in default
    assert "initial_input if initial_input else 'https://example.com'" in flow_py
    # "" mapping → whole output; field mapping → static subscript
    assert "url=trigger_out" in flow_py
    assert "page_text=fetch_page_out['text']" in flow_py
    assert "summary=summarize_out" in flow_py
    # trigger text auto-wraps into Summarize's unfilled param, literals kept
    assert "max_words=100" in flow_py
    assert "text=trigger_out" in flow_py
    # parallel level runs sync tools in threads; the join is a direct call
    assert "await asyncio.gather(" in flow_py
    assert "asyncio.to_thread(fetch_page, url=trigger_out)" in flow_py
    assert "join_parts(" in flow_py and "await asyncio.to_thread(join_parts" not in flow_py
    assert "return join_out" in flow_py


def test_parallel_false_emits_sequential_flow(db):
    """parallel=False: same wiring, but branches run one after another."""
    session, user = db
    fetch = make_fetch_tool(session, user)
    summarize = make_summarize_tool(session, user)
    flow = make_flow(
        session, user,
        nodes={
            "T": {"node_type": "trigger", "id": 0, "name": "Trigger",
                  "input_value": "hi"},
            "F": {"node_type": "tool", "id": fetch.id, "name": "Fetch Page"},
            "S": {"node_type": "tool", "id": summarize.id, "name": "Summarize",
                  "input_values": {"max_words": 100}},
        },
        edges=[
            {"from_node": "T", "to_node": "F", "mapping": {"": "url"}},
            {"from_node": "T", "to_node": "S"},
        ],
        entry="T", exits=["F", "S"],
    )
    files = export_flow(flow, session, LLM_CONFIG, parallel=False)
    assert_all_compile(files)
    flow_py = files["flow.py"]
    assert "await asyncio.gather(" not in flow_py and "to_thread(" not in flow_py
    assert "fetch_page_out = fetch_page(url=trigger_out)" in flow_py
    assert "summarize_out = summarize_text(" in flow_py
    assert "sequentially in dependency order" in flow_py


def test_sequential_flow_direct_calls(db):
    session, user = db
    fetch = make_fetch_tool(session, user)
    summarize = make_summarize_tool(session, user)
    flow = make_flow(
        session, user,
        nodes={
            "F": {"node_type": "tool", "id": fetch.id, "name": "Fetch Page"},
            "S": {"node_type": "tool", "id": summarize.id, "name": "Summarize",
                  "input_values": {"max_words": 10}},
        },
        edges=[{"from_node": "F", "to_node": "S", "mapping": {"text": "text"}}],
        entry="F", exits=["S"],
    )
    files = export_flow(flow, session, LLM_CONFIG)
    assert_all_compile(files)
    flow_py = files["flow.py"]
    # entry tool takes initial_input as kwargs; no gather, no to_thread
    assert "fetch_page(**initial_input)" in flow_py
    assert "await asyncio.gather(" not in flow_py and "to_thread(" not in flow_py
    assert "text=fetch_page_out['text']" in flow_py


def test_dict_passthrough_merge(db):
    session, user = db
    fetch = make_fetch_tool(session, user)
    consume = make_tool(
        session, user, "Consume",
        "def consume(text, status):\n    return text * status\n",
        "consume",
        input_schema={"properties": {"text": {"type": "str"},
                                     "status": {"type": "int"}}},
    )
    flow = make_flow(
        session, user,
        nodes={
            "F": {"node_type": "tool", "id": fetch.id, "name": "Fetch"},
            "C": {"node_type": "tool", "id": consume.id, "name": "Consume"},
        },
        edges=[{"from_node": "F", "to_node": "C"}],
        entry="F", exits=["C"],
    )
    files = export_flow(flow, session, LLM_CONFIG)
    assert_all_compile(files)
    assert "consume(**fetch_out)" in files["flow.py"]


def test_unknown_upstream_uses_merge_auto_helper(db):
    session, user = db
    mystery = make_tool(
        session, user, "Mystery",
        "def mystery(x):\n    return x\n",
        "mystery",
        input_schema={"properties": {"x": {"type": "str"}}},
        output_schema={"type": "Any"},
    )
    consume = make_tool(
        session, user, "Consume",
        "def consume(a, b):\n    return (a, b)\n",
        "consume",
        input_schema={"properties": {"a": {"type": "str"}, "b": {"type": "str"}}},
    )
    flow = make_flow(
        session, user,
        nodes={
            "M": {"node_type": "tool", "id": mystery.id, "name": "Mystery"},
            "C": {"node_type": "tool", "id": consume.id, "name": "Consume",
                  "input_values": {"b": "fixed"}},
        },
        edges=[{"from_node": "M", "to_node": "C"}],
        entry="M", exits=["C"],
    )
    files = export_flow(flow, session, LLM_CONFIG)
    assert_all_compile(files)
    flow_py = files["flow.py"]
    assert "def _merge_auto(" in flow_py
    assert "_merge_auto({'b': 'fixed'}, mystery_out, params=['a', 'b'])" in flow_py


def test_tool_files_and_requirements(db):
    session, user = db
    fetch = make_fetch_tool(session, user)
    summarize = make_summarize_tool(session, user)
    flow = make_flow(
        session, user,
        nodes={
            "F": {"node_type": "tool", "id": fetch.id, "name": "Fetch"},
            "S": {"node_type": "tool", "id": summarize.id, "name": "Summarize",
                  "input_values": {"max_words": 5}},
        },
        edges=[{"from_node": "F", "to_node": "S", "mapping": {"text": "text"}}],
        entry="F", exits=["S"],
    )
    files = export_flow(flow, session, LLM_CONFIG)
    # verbatim tool scripts under slugged module names
    assert 'return {"text": url + "!", "status": 200}' in files["tools/fetch_page.py"]
    assert "def summarize_text(text, max_words):" in files["tools/summarize.py"]
    assert files["tools/__init__.py"] == ""
    # union of required_packages across tools; no agent pins
    requirements = files["requirements.txt"].splitlines()
    assert "requests" in requirements and "pandas" in requirements
    assert "sqlalchemy" in requirements
    assert not any(r.startswith("pydantic-ai") for r in requirements)


def test_requirements_keep_every_tool_package(db):
    """Every package a tool's script actually imports must survive into the
    export's requirements.txt, including packages that used to be filtered
    out by the (now-removed) infra denylist."""
    session, user = db
    tool = make_tool(
        session, user, "Net Tool",
        "def net_call(x):\n    return x\n",
        "net_call",
        input_schema={"properties": {"x": {"type": "str"}}},
        required_packages=["httpx", "python-dotenv", "redis", "langfuse"],
    )
    flow = make_flow(
        session, user,
        nodes={"N": {"node_type": "tool", "id": tool.id, "name": "Net Tool"}},
        edges=[],
        entry="N", exits=["N"],
    )
    files = export_flow(flow, session, LLM_CONFIG)
    assert_all_compile(files)
    requirements = files["requirements.txt"].splitlines()
    for pkg in ["httpx", "python-dotenv", "redis", "langfuse"]:
        assert pkg in requirements
    assert requirements == sorted(set(requirements))
    assert not any(r.startswith("pydantic-ai") for r in requirements)


def test_requirements_agent_pin_and_tool_package_both_present(db):
    """A tool-required package that happens to share a name with an SDK pin
    (e.g. "openai") must still appear verbatim alongside the pinned, versioned
    line the agent needs -- the two are distinct lines, not duplicates."""
    session, user = db
    tool = make_tool(
        session, user, "OpenAI Tool",
        "def call_openai(x):\n    return x\n",
        "call_openai",
        input_schema={"properties": {"x": {"type": "str"}}},
        required_packages=["openai"],
    )
    agent = make_agent(session, user, single_node_agent_config(name="Reviewer"))
    flow = agent_flow(session, user, agent, tool)
    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "lm"})
    assert_all_compile(files)
    requirements = files["requirements.txt"].splitlines()
    assert "openai" in requirements
    assert "openai==2.12.0" in requirements
    assert requirements == list(dict.fromkeys(requirements))


def test_multi_exit_returns_tuple(db):
    session, user = db
    fetch = make_fetch_tool(session, user)
    summarize = make_summarize_tool(session, user)
    flow = make_flow(
        session, user,
        nodes={
            "T": {"node_type": "trigger", "id": 0, "name": "Trigger",
                  "input_value": "hi"},
            "F": {"node_type": "tool", "id": fetch.id, "name": "Fetch"},
            "S": {"node_type": "tool", "id": summarize.id, "name": "Summarize",
                  "input_values": {"max_words": 5}},
        },
        edges=[
            {"from_node": "T", "to_node": "F", "mapping": {"": "url"}},
            {"from_node": "T", "to_node": "S"},
        ],
        entry="T", exits=["F", "S"],
    )
    files = export_flow(flow, session, LLM_CONFIG)
    assert_all_compile(files)
    assert "return (fetch_out, summarize_out)" in files["flow.py"]


# ─── Agent flows ─────────────────────────────────────────────────────────────

def agent_flow(session, user, agent, tool):
    return make_flow(
        session, user,
        nodes={
            "F": {"node_type": "tool", "id": tool.id, "name": "Fetch"},
            "AG": {"node_type": "agent", "id": agent.id, "name": "Reviewer"},
        },
        edges=[{"from_node": "F", "to_node": "AG"}],
        entry="F", exits=["AG"],
    )


def test_agent_module_anthropic(db):
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, single_node_agent_config(
        name="Reviewer",
        user_prompt="Review this: {input}",
        tool_ids=[fetch.id],
        output_paths={
            "approve": {"description": "Looks good",
                        "return_behavior": "previous_output"},
            "revise": "Needs changes",
        },
    ))
    flow = agent_flow(session, user, agent, fetch)

    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)
    assert_standalone(files)

    agent_py = files["agents/reviewer.py"]
    # credentials from env, never embedded; provider/model baked in
    assert 'os.environ["ANTHROPIC_AP' + 'I_KEY"]' in agent_py
    assert all("sk-SECRET123" not in content for content in files.values())
    assert "'claude-x'" in agent_py
    # output paths as plain classes with routing appended to the prompt
    assert "class Approve(BaseModel):" in agent_py
    assert "class Revise(BaseModel):" in agent_py
    assert "You must choose one of the following output paths:" in agent_py
    assert "PREVIOUS_OUTPUT_PATHS = ['approve']" in agent_py
    # system prompt template resolved at export time
    assert "You are Reviewer." in agent_py
    # user prompt template resolved at run time, from a NAMED-inputs dict
    assert "async def run_reviewer(inputs: dict) -> str:" in agent_py
    # agent tool registered as the raw imported function
    assert "from tools.fetch_page import fetch_page" in agent_py
    assert "agent.tool_plain(fetch_page)" in agent_py

    flow_py = files["flow.py"]
    assert "from agents.reviewer import run_reviewer" in flow_py
    assert 'await run_reviewer({"input": _as_text(fetch_out)})' in flow_py

    requirements = files["requirements.txt"].splitlines()
    assert "pydantic-ai==1.31.0" in requirements
    assert "anthropic==0.77.0" in requirements
    assert not any(r.startswith("openai==") for r in requirements)
    assert "ANTHROPIC_AP" + "I_KEY" in files["README.md"]


def test_agent_module_lmstudio_defaults(db):
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, single_node_agent_config())
    flow = agent_flow(session, user, agent, fetch)

    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "lm"})
    assert_all_compile(files)
    agent_py = files["agents/reviewer.py"]
    assert "base_url='http://localhost:1234/v1'" in agent_py
    assert 'os.environ.get("LMSTUDIO_AP' + 'I_KEY", "lm-studio")' in agent_py
    assert "OpenAIChatModel" in agent_py
    requirements = files["requirements.txt"].splitlines()
    assert "openai==2.12.0" in requirements


def test_agent_without_llm_selection_fails(db):
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, single_node_agent_config())
    flow = agent_flow(session, user, agent, fetch)
    with pytest.raises(FlowExportError, match="No LLM selected"):
        export_flow(flow, session, LLM_CONFIG)
    with pytest.raises(FlowExportError, match="not found in config"):
        export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "nope"})


# ─── Complex (multi-node / looping) agent exports ────────────────────────────
# See claude_plans/complex-agent-export.md §7. These replace the old
# Phase-1 rejection test; they describe the target behavior once multi-node
# / looping agents can be exported, so most of them currently fail because
# _check_phase1() still rejects any agent with more than one node.

def test_reflection_loop_agent_wiring(db):
    """Writer -> Reviewer, reviewer routes revise/approve, revise loops back."""
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, writer_reviewer_agent_config())
    flow = agent_flow(session, user, agent, fetch)

    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)
    assert_standalone(files)
    assert_no_secret_leak(files)

    pkg = "agents/reviewer"
    expected_agent_files = {
        "agents/__init__.py",
        f"{pkg}/__init__.py",
        f"{pkg}/agent.py",
        f"{pkg}/model.py",
        f"{pkg}/graph.json",
        f"{pkg}/agents/__init__.py",
        f"{pkg}/agents/writer.py",
        f"{pkg}/agents/reviewer.py",
        "agents/_runtime/__init__.py",
        "agents/_runtime/graph_engine.py",
        "agents/_runtime/message_serializer.py",
        "agents/_runtime/prompts.py",
    }
    actual_agent_files = {p for p in files if p.startswith("agents/")}
    assert actual_agent_files == expected_agent_files
    assert "agents/reviewer.py" not in files

    model_py = files[f"{pkg}/model.py"]
    assert model_py.count("model = ") == 1

    writer_py = files[f"{pkg}/agents/writer.py"]
    reviewer_py = files[f"{pkg}/agents/reviewer.py"]
    for sub_py in (writer_py, reviewer_py):
        assert "from agents.reviewer.model import model" in sub_py
        assert "Agent(model=model" in sub_py

    assert "output_type=_output_types()" in reviewer_py
    writer_agent_line = next(l for l in writer_py.splitlines() if l.startswith("agent = Agent("))
    assert "output_type=" not in writer_agent_line

    graph = json.loads(files[f"{pkg}/graph.json"])
    loop_edge = next(e for e in graph["edges"]
                      if e.get("from_node") == "reviewer" and e.get("to_node") == "writer")
    assert loop_edge["is_loop"] is True
    assert loop_edge["output_path"] == "revise"
    assert graph["max_loop_iterations"] == 2
    assert graph["entry_point"] == "writer"
    assert graph["exit_points"] == ["reviewer"]
    for node_id in ("writer", "reviewer"):
        node = graph["nodes"][node_id]
        assert set(node) <= {"name", "agent_type", "output_paths"}
        assert "system_prompt" not in node
        assert "user_prompt" not in node
        assert "tool_ids" not in node
        assert "llm_provider" not in node
    graph_json_text = files[f"{pkg}/graph.json"]
    assert graph_json_text.endswith("\n")
    assert "\n" in graph_json_text.rstrip("\n")  # pretty-printed, multi-line

    sub_init = files[f"{pkg}/agents/__init__.py"]
    assert '"writer": writer.run' in sub_init
    assert '"reviewer": reviewer.run' in sub_init

    agent_py = files[f"{pkg}/agent.py"]
    assert "from agents._runtime.graph_engine import run_graph" in agent_py
    assert "from agents.reviewer.agents import RUNNERS" in agent_py
    assert "async def run_reviewer(inputs: dict) -> str:" in agent_py
    assert "run_graph(GRAPH_CONFIG, inputs" in agent_py

    init_py = files[f"{pkg}/__init__.py"]
    assert "run_reviewer" in init_py

    flow_py = files["flow.py"]
    assert "from agents.reviewer import run_reviewer" in flow_py
    assert "await run_reviewer(" in flow_py


def test_vendored_files_byte_identical(db):
    """agents/_runtime vendors the engine + serializer verbatim; single-node
    exports get none of it."""
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, writer_reviewer_agent_config())
    flow = agent_flow(session, user, agent, fetch)
    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})

    src_dir = Path(__file__).resolve().parent.parent / "src"
    engine_source = (src_dir / "executors" / "agent_graph_engine.py").read_text(encoding="utf-8")
    serializer_source = (src_dir / "utils" / "message_serializer.py").read_text(encoding="utf-8")

    assert files["agents/_runtime/graph_engine.py"] == engine_source
    assert files["agents/_runtime/message_serializer.py"] == serializer_source

    single_agent = make_agent(session, user, single_node_agent_config(), name="Simple")
    single_flow = agent_flow(session, user, single_agent, fetch)
    single_files = export_flow(single_flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert not any(p.startswith("agents/_runtime/") for p in single_files)


def test_multinode_user_prompts(db, import_exported):
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, writer_reviewer_agent_config(
        reviewer_overrides={"user_prompt": "Review: {input}\n\n{message_history}"},
    ))
    flow = agent_flow(session, user, agent, fetch)
    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)
    assert_standalone(files)

    prompts_py = files["agents/_runtime/prompts.py"]
    assert prompts_py.count("def apply_prompt(") == 1

    writer_py = files["agents/reviewer/agents/writer.py"]
    reviewer_py = files["agents/reviewer/agents/reviewer.py"]
    assert "def apply_prompt(" not in writer_py
    assert "def apply_prompt(" not in reviewer_py
    assert 'USER_PROMPT = ""' in writer_py
    assert "USER_PROMPT = " in reviewer_py
    assert "Review: {input}" in reviewer_py
    assert "apply_prompt(USER_PROMPT, node_input, messages)" in writer_py
    assert "apply_prompt(USER_PROMPT, node_input, messages)" in reviewer_py

    # the no-prompt fixture must also produce USER_PROMPT = "" in both files
    agent2 = make_agent(session, user, writer_reviewer_agent_config(), name="NoPrompts")
    flow2 = agent_flow(session, user, agent2, fetch)
    files2 = export_flow(flow2, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert 'USER_PROMPT = ""' in files2["agents/noprompts/agents/writer.py"]
    assert 'USER_PROMPT = ""' in files2["agents/noprompts/agents/reviewer.py"]

    # whitespace-only user_prompt is also written as ""
    agent3 = make_agent(session, user, {
        "nodes": {
            "a": {"agent_type": "pydanticai", "name": "a", "system_prompt": "s",
                  "user_prompt": "   "},
            "b": {"agent_type": "pydanticai", "name": "b", "system_prompt": "s"},
        },
        "edges": [{"from_node": "a", "to_node": "b", "is_loop": False}],
        "entry_point": "a", "exit_points": ["b"],
    }, name="Whitespace")
    flow3 = agent_flow(session, user, agent3, fetch)
    files3 = export_flow(flow3, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert 'USER_PROMPT = ""' in files3["agents/whitespace/agents/a.py"]

    module = import_exported(files, "agents._runtime.prompts")
    assert module.apply_prompt("", {"input": "hi"}, None) == "hi"
    assert module.apply_prompt("Q: {input}", {"input": "hi"}, None) == "Q: hi"
    assert module.apply_prompt("{input} / {message_history}", {"input": "hi"}, None) == "hi / "


def test_multinode_previous_output(db):
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, writer_reviewer_agent_config(
        reviewer_overrides={"output_paths": {
            "revise": {"description": "Needs another pass"},
            "approve": {"description": "Good to go",
                        "return_behavior": "previous_output"},
        }},
    ))
    flow = agent_flow(session, user, agent, fetch)
    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    graph = json.loads(files["agents/reviewer/graph.json"])
    assert graph["nodes"]["reviewer"]["output_paths"]["approve"]["return_behavior"] == "previous_output"


def test_multinode_requirements(db):
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, writer_reviewer_agent_config())
    flow = agent_flow(session, user, agent, fetch)
    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    requirements = files["requirements.txt"].splitlines()
    assert "pydantic-ai==1.31.0" in requirements
    assert "anthropic==0.77.0" in requirements
    assert not any(r.startswith("openai==") for r in requirements)


def test_output_path_classes_scoped_per_node(db, import_exported):
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, {
        "nodes": {
            "node_x": {"agent_type": "pydanticai", "name": "Node X", "system_prompt": "s",
                       "output_paths": {
                           "approve": {"description": "ok"},
                           "revise": {"description": "no"},
                       }},
            "node_y": {"agent_type": "pydanticai", "name": "Node Y", "system_prompt": "s",
                       "output_paths": {
                           "approve": {"description": "ok2"},
                           "agent": {"description": "weird"},
                       }},
        },
        "edges": [{"from_node": "node_x", "to_node": "node_y", "is_loop": False}],
        "entry_point": "node_x", "exit_points": ["node_y"],
    }, name="Router")
    flow = agent_flow(session, user, agent, fetch)
    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)
    assert_standalone(files)

    node_x_py = files["agents/router/agents/node_x.py"]
    node_y_py = files["agents/router/agents/node_y.py"]

    for sub_py in (node_x_py, node_y_py):
        assert re.search(r"^class ", sub_py, re.MULTILINE) is None
        assert "output_type=_output_types()" in sub_py
        assert "CLASS_TO_PATH = " in sub_py

    assert "class Approve" in node_x_py
    assert "class Revise" in node_x_py
    assert "class Approve" in node_y_py
    assert "class Agent" in node_y_py

    mod_x = import_exported(files, "agents.router.agents.node_x")
    mod_y = importlib.import_module("agents.router.agents.node_y")

    assert mod_x.Agent is pydantic_ai.Agent
    assert mod_y.Agent is pydantic_ai.Agent
    assert {c.__name__ for c in typing.get_args(mod_x._output_types())} == {"Approve", "Revise"}
    assert {c.__name__ for c in typing.get_args(mod_y._output_types())} == {"Approve", "Agent"}


def test_multinode_canvas_node_ids(db):
    """Sub-agent file names come from display names, never raw canvas ids."""
    session, user = db
    fetch = make_fetch_tool(session, user)
    node_ids = [
        "1729543212345", "1729543212346", "1729543212347",
        "1729543212348", "1729543212349",
    ]
    names = ["Writer", "Writer", "", "Node", "Run Writer"]
    nodes = {
        nid: {"agent_type": "pydanticai", "name": name, "system_prompt": "s"}
        for nid, name in zip(node_ids, names)
    }
    edges = [
        {"from_node": node_ids[i], "to_node": node_ids[i + 1], "is_loop": False}
        for i in range(len(node_ids) - 1)
    ]
    agent = make_agent(session, user, {
        "nodes": nodes, "edges": edges,
        "entry_point": node_ids[0], "exit_points": [node_ids[-1]],
    }, name="Writer")
    flow = agent_flow(session, user, agent, fetch)
    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)
    assert_standalone(files)

    for path in files:
        for nid in node_ids:
            assert nid not in path, path

    pkg = "agents/writer"
    expected_subagents = {
        f"{pkg}/agents/writer.py",
        f"{pkg}/agents/writer_2.py",
        f"{pkg}/agents/node.py",
        f"{pkg}/agents/node_2.py",
        f"{pkg}/agents/run_writer.py",
    }
    assert expected_subagents <= set(files)

    sub_init = files[f"{pkg}/agents/__init__.py"]
    assert f'"{node_ids[0]}": writer.run' in sub_init
    assert f'"{node_ids[1]}": writer_2.run' in sub_init
    assert f'"{node_ids[2]}": node.run' in sub_init
    assert f'"{node_ids[3]}": node_2.run' in sub_init
    assert f'"{node_ids[4]}": run_writer.run' in sub_init

    agent_py = files[f"{pkg}/agent.py"]
    assert "async def run_writer(" in agent_py
    assert agent_py.count("from agents.writer.agents import") == 1
    assert "from agents.writer.agents import RUNNERS" in agent_py

    graph = json.loads(files[f"{pkg}/graph.json"])
    assert set(graph["nodes"]) == set(node_ids)
    assert graph["entry_point"] == node_ids[0]
    assert graph["exit_points"] == [node_ids[-1]]
    for edge in graph["edges"]:
        assert edge["from_node"] in node_ids
        assert edge["to_node"] in node_ids


def test_reflection_loop_runs(db, import_exported):
    """Behavioral test: drive the exported writer/reviewer loop with fake
    models and check it matches the platform's routing/loop/previous_output
    semantics. Runs against the dev env's installed pydantic-ai, not the
    export's pin."""
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, writer_reviewer_agent_config(
        reviewer_overrides={
            "user_prompt": "Review: {input}\n\n{message_history}",
            "output_paths": {
                "revise": {"description": "Needs another pass"},
                "approve": {"description": "Good to go",
                            "return_behavior": "previous_output"},
            },
        },
    ), name="Reflection")
    flow = agent_flow(session, user, agent, fetch)
    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})

    pkg_module = import_exported(files, "agents.reflection")
    writer_mod = importlib.import_module("agents.reflection.agents.writer")
    reviewer_mod = importlib.import_module("agents.reflection.agents.reviewer")

    def make_writer_fn(calls):
        def writer_fn(messages, info):
            calls.append(messages[-1].parts[-1].content)
            return ModelResponse(parts=[TextPart(f"draft {len(calls)}")])
        return writer_fn

    def make_reviewer_fn(script, prompts):
        def reviewer_fn(messages, info):
            choice = script.pop(0)
            prompts.append(messages[-1].parts[-1].content)
            return ModelResponse(parts=[ToolCallPart(
                f"final_result_{choice.capitalize()}", {"content": f"{choice} note"})])
        return reviewer_fn

    def run_scenario(script):
        writer_calls, reviewer_prompts = [], []
        with writer_mod.agent.override(model=FunctionModel(make_writer_fn(writer_calls))), \
             reviewer_mod.agent.override(model=FunctionModel(make_reviewer_fn(list(script), reviewer_prompts))):
            result = asyncio.run(pkg_module.run_reflection({"input": "otters"}))
        return result, writer_calls, reviewer_prompts

    # approve after revising: previous_output returns the writer's draft
    result, writer_calls, reviewer_prompts = run_scenario(["revise", "approve"])
    assert len(writer_calls) == 2
    assert len(reviewer_prompts) == 2
    assert result == "draft 2"
    assert reviewer_prompts[0].startswith("Review: draft 1")
    assert "[Assistant]\ndraft 1" in reviewer_prompts[0]
    assert writer_calls[1] == "revise note"

    # never satisfied: capped at max_loop_iterations=2, no error
    result, writer_calls, reviewer_prompts = run_scenario(["revise", "revise", "revise"])
    assert len(writer_calls) == 3
    assert len(reviewer_prompts) == 3
    assert result == "revise note"

    # approve straight away: approve does not follow the revise-tagged loop edge
    result, writer_calls, reviewer_prompts = run_scenario(["approve"])
    assert len(writer_calls) == 1
    assert len(reviewer_prompts) == 1
    assert result == "draft 1"


def test_multinode_invalid_edges_rejected(db):
    session, user = db
    fetch = make_fetch_tool(session, user)

    # from_node not a sub-node
    agent_a = make_agent(session, user, writer_reviewer_agent_config(
        edges=[
            {"from_node": "writer", "to_node": "reviewer", "is_loop": False},
            {"from_node": "reviewer", "to_node": "writer", "is_loop": True, "output_path": "revise"},
            {"from_node": "ghost", "to_node": "reviewer", "is_loop": False},
        ],
    ), name="BadFrom")
    flow_a = agent_flow(session, user, agent_a, fetch)
    with pytest.raises(FlowExportError) as exc_a:
        export_flow(flow_a, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert "ghost" in str(exc_a.value)
    assert "BadFrom" in str(exc_a.value)

    # to_node not a sub-node
    agent_b = make_agent(session, user, writer_reviewer_agent_config(
        edges=[
            {"from_node": "writer", "to_node": "reviewer", "is_loop": False},
            {"from_node": "reviewer", "to_node": "writer", "is_loop": True, "output_path": "revise"},
            {"from_node": "reviewer", "to_node": "ghost2", "is_loop": False},
        ],
    ), name="BadTo")
    flow_b = agent_flow(session, user, agent_b, fetch)
    with pytest.raises(FlowExportError) as exc_b:
        export_flow(flow_b, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert "ghost2" in str(exc_b.value)
    assert "BadTo" in str(exc_b.value)

    # from_node/to_node missing entirely
    agent_c = make_agent(session, user, writer_reviewer_agent_config(
        edges=[
            {"from_node": "writer", "to_node": "reviewer", "is_loop": False},
            {"from_node": "reviewer", "to_node": "writer", "is_loop": True, "output_path": "revise"},
            {"from_node": "writer"},
        ],
    ), name="BadMissing")
    flow_c = agent_flow(session, user, agent_c, fetch)
    with pytest.raises(FlowExportError) as exc_c:
        export_flow(flow_c, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert "BadMissing" in str(exc_c.value)


def test_subagent_tool_name_clash(db, import_exported):
    session, user = db
    fetch = make_fetch_tool(session, user)
    tool_model = make_tool(
        session, user, "Model Lookup",
        "def model(x):\n    return x\n", "model",
        input_schema={"properties": {"x": {"type": "str"}}},
    )
    tool_run = make_tool(
        session, user, "Runner",
        "def run(x):\n    return x\n", "run",
        input_schema={"properties": {"x": {"type": "str"}}},
    )
    agent = make_agent(session, user, writer_reviewer_agent_config(
        writer_overrides={"tool_ids": [tool_model.id, tool_run.id]},
    ))
    flow = agent_flow(session, user, agent, fetch)
    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)
    assert_standalone(files)

    writer_py = files["agents/reviewer/agents/writer.py"]
    assert "from tools.model_lookup import model as model_lookup_model" in writer_py
    assert "from tools.runner import run as runner_run" in writer_py
    assert "agent.tool_plain(model_lookup_model)" in writer_py
    assert "agent.tool_plain(runner_run)" in writer_py
    assert "from agents.reviewer.model import model" in writer_py

    module = import_exported(files, "agents.reviewer.agents.writer")
    assert isinstance(module.model, Model)
    assert inspect.iscoroutinefunction(module.run)

    recorded = {}

    def fn(messages, info):
        recorded["names"] = [t.name for t in info.function_tools]
        return ModelResponse(parts=[TextPart("ok")])

    with module.agent.override(model=FunctionModel(fn)):
        asyncio.run(module.agent.run("hi"))

    assert "model" in recorded["names"]
    assert "run" in recorded["names"]


def test_single_node_output_path_named_agent(db, import_exported):
    """An output path literally named "agent" would define a top-level
    `class Agent(BaseModel)` that replaces the imported pydantic_ai.Agent.
    The exporter must detect the clash and emit the output-path classes
    inside a `_output_types()` function instead (same shape multi-node
    sub-agent files already use for clashing path names)."""
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, single_node_agent_config(
        output_paths={
            "human": {"description": "For a person"},
            "agent": {"description": "For another agent"},
        },
    ))
    flow = agent_flow(session, user, agent, fetch)

    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)
    assert_standalone(files)

    agent_py = files["agents/reviewer.py"]
    assert "def _output_types():" in agent_py
    assert "output_type=_output_types()" in agent_py
    agent_call_line = next(
        l for l in agent_py.splitlines() if l.startswith("agent = Agent("))
    assert "output_type=_output_types()" in agent_call_line

    module = import_exported(files, "agents.reviewer")
    assert module.Agent is pydantic_ai.Agent
    assert {c.__name__ for c in typing.get_args(module._output_types())} == {
        "Human", "Agent"}

    def fn(messages, info):
        return ModelResponse(parts=[ToolCallPart(
            "final_result_Agent", {"content": "to agent"})])

    with module.agent.override(model=FunctionModel(fn)):
        result = asyncio.run(module.run_reviewer({"input": "x"}))
    assert result == "to agent"


def test_single_node_tool_name_clash(db, import_exported):
    """A tool whose main function is named model/agent/os would be imported
    under that bare name and silently overwritten by the LLM model object
    (or shadow the real os module), then get registered as an LLM tool
    instead of a callable. The exporter must alias the import to
    <module>_<func> and use that alias both for the import and for
    agent.tool_plain(...); pydantic-ai still names the tool after the
    function's __name__, so the LLM-visible tool name is unaffected."""
    session, user = db
    fetch = make_fetch_tool(session, user)
    tool_model = make_tool(
        session, user, "Model Lookup",
        "def model(x):\n    return x\n", "model",
        input_schema={"properties": {"x": {"type": "str"}}},
    )
    tool_agent = make_tool(
        session, user, "Agent Lookup",
        "def agent(x):\n    return x\n", "agent",
        input_schema={"properties": {"x": {"type": "str"}}},
    )
    tool_os = make_tool(
        session, user, "Os Lookup",
        "def os(x):\n    return x\n", "os",
        input_schema={"properties": {"x": {"type": "str"}}},
    )
    agent = make_agent(session, user, single_node_agent_config(
        tool_ids=[tool_model.id, tool_agent.id, tool_os.id],
    ))
    flow = agent_flow(session, user, agent, fetch)

    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)
    assert_standalone(files)

    agent_py = files["agents/reviewer.py"]
    assert "from tools.model_lookup import model as model_lookup_model" in agent_py
    assert "from tools.agent_lookup import agent as agent_lookup_agent" in agent_py
    assert "from tools.os_lookup import os as os_lookup_os" in agent_py
    assert "agent.tool_plain(model_lookup_model)" in agent_py
    assert "agent.tool_plain(agent_lookup_agent)" in agent_py
    assert "agent.tool_plain(os_lookup_os)" in agent_py

    module = import_exported(files, "agents.reviewer")
    assert isinstance(module.model, Model)
    assert module.os is os

    recorded = {}

    def fn(messages, info):
        recorded["names"] = [t.name for t in info.function_tools]
        return ModelResponse(parts=[TextPart("ok")])

    with module.agent.override(model=FunctionModel(fn)):
        asyncio.run(module.agent.run("hi"))

    assert "model" in recorded["names"]
    assert "agent" in recorded["names"]
    assert "os" in recorded["names"]


def test_single_node_ordinary_names_unchanged(db):
    """Sanity/regression guard: a normal single-node agent (no clashing
    output-path or tool names) must keep emitting plain top-level output
    classes and a bare import — i.e. every agent that exports correctly
    today must stay byte-identical once the clash fixes land."""
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, single_node_agent_config(
        tool_ids=[fetch.id],
        output_paths={
            "approve": {"description": "Looks good"},
            "revise": {"description": "Needs changes"},
        },
    ))
    flow = agent_flow(session, user, agent, fetch)

    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)

    agent_py = files["agents/reviewer.py"]
    assert re.search(r"^class Approve\(BaseModel\):", agent_py, re.MULTILINE)
    assert re.search(r"^class Revise\(BaseModel\):", agent_py, re.MULTILINE)
    assert "output_type=Approve | Revise" in agent_py
    assert "_output_types" not in agent_py
    assert "from tools.fetch_page import fetch_page" in agent_py
    assert " as fetch_page" not in agent_py


def test_agent_gets_selected_field_from_expanded_output(db):
    """{field: ""} mapping (expanded dict output → agent) feeds just that field."""
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, single_node_agent_config())
    flow = make_flow(
        session, user,
        nodes={
            "F": {"node_type": "tool", "id": fetch.id, "name": "Fetch"},
            "AG": {"node_type": "agent", "id": agent.id, "name": "Reviewer"},
        },
        edges=[{"from_node": "F", "to_node": "AG", "mapping": {"text": ""}}],
        entry="F", exits=["AG"],
    )
    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)
    flow_py = files["flow.py"]
    # "text" is a str field in the schema — direct subscript, no _as_text;
    # {field: ""} targets the agent's legacy "input" input (decision 5).
    assert 'await run_reviewer({"input": fetch_out[\'text\']})' in flow_py


def test_agent_selected_nonstr_field_is_stringified(db):
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, single_node_agent_config())
    flow = make_flow(
        session, user,
        nodes={
            "F": {"node_type": "tool", "id": fetch.id, "name": "Fetch"},
            "AG": {"node_type": "agent", "id": agent.id, "name": "Reviewer"},
        },
        edges=[{"from_node": "F", "to_node": "AG", "mapping": {"status": ""}}],
        entry="F", exits=["AG"],
    )
    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert 'await run_reviewer({"input": _as_text(fetch_out[\'status\'])})' in files["flow.py"]


def test_tool_generic_input_autowraps_selected_field(db):
    """{field: ""} into a tool merges the field like an unmapped scalar."""
    session, user = db
    fetch = make_fetch_tool(session, user)
    summarize = make_summarize_tool(session, user)
    flow = make_flow(
        session, user,
        nodes={
            "F": {"node_type": "tool", "id": fetch.id, "name": "Fetch"},
            "S": {"node_type": "tool", "id": summarize.id, "name": "Summarize",
                  "input_values": {"max_words": 10}},
        },
        edges=[{"from_node": "F", "to_node": "S", "mapping": {"text": ""}}],
        entry="F", exits=["S"],
    )
    files = export_flow(flow, session, LLM_CONFIG)
    assert_all_compile(files)
    assert "text=fetch_out['text']" in files["flow.py"]


def test_agent_selected_field_from_unknown_output_uses_pick(db):
    session, user = db
    mystery = make_tool(
        session, user, "Mystery", "def mystery(x):\n    return x\n", "mystery",
        input_schema={"properties": {"x": {"type": "str"}}},
        output_schema={"type": "Any"},
    )
    agent = make_agent(session, user, single_node_agent_config())
    flow = make_flow(
        session, user,
        nodes={
            "M": {"node_type": "tool", "id": mystery.id, "name": "Mystery"},
            "AG": {"node_type": "agent", "id": agent.id, "name": "Reviewer"},
        },
        edges=[{"from_node": "M", "to_node": "AG", "mapping": {"text": ""}}],
        entry="M", exits=["AG"],
    )
    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)
    flow_py = files["flow.py"]
    assert "_as_text(_pick(mystery_out, 'text'))" in flow_py
    assert "def _pick(" in flow_py



# ─── Escaping of user-controlled text in generated source ───────────────────
# The exporter writes node ids, agent/sub-agent/flow/tool/node names, and
# output-path descriptions into generated .py files using raw f-strings.
# Text containing quotes, backslashes, triple-quotes or newlines must not
# break the generated syntax or let injected code run on import.

def test_runner_ids_escaped(db, import_exported, monkeypatch):
    """Sub-node ids containing quotes/backslashes/injection text must survive
    as plain RUNNERS keys, not get spliced into the generated source as code."""
    session, user = db
    fetch = make_fetch_tool(session, user)
    monkeypatch.delenv("EXPORT_PWNED", raising=False)

    node_ids = [
        'a"b',
        "c'd",
        'e\\f',
        'x": __import__("os").environ.__setitem__("EXPORT_PWNED", "1"), "y',
    ]
    nodes = {
        nid: {"agent_type": "pydanticai", "name": f"Node{i}", "system_prompt": "s"}
        for i, nid in enumerate(node_ids)
    }
    edges = [
        {"from_node": node_ids[i], "to_node": node_ids[i + 1], "is_loop": False}
        for i in range(len(node_ids) - 1)
    ]
    agent = make_agent(session, user, {
        "nodes": nodes, "edges": edges,
        "entry_point": node_ids[0], "exit_points": [node_ids[-1]],
    }, name="Escaped")
    flow = agent_flow(session, user, agent, fetch)

    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)
    assert_standalone(files)
    assert_no_secret_leak(files)

    pkg_folder = next(
        p.rsplit("/", 2)[0] for p in files
        if p.endswith("/agents/__init__.py") and not p.startswith("agents/_runtime/")
    )
    agents_pkg_module = pkg_folder.replace("/", ".") + ".agents"

    monkeypatch.delenv("EXPORT_PWNED", raising=False)
    mod = import_exported(files, agents_pkg_module)
    assert "EXPORT_PWNED" not in os.environ

    assert set(mod.RUNNERS) == set(node_ids)

    graph = json.loads(files[f"{pkg_folder}/graph.json"])
    assert set(graph["nodes"]) == set(node_ids)
    assert graph["entry_point"] == node_ids[0]
    assert graph["exit_points"] == [node_ids[-1]]


def test_docstring_text_escaped(db, import_exported, monkeypatch):
    """Names/descriptions containing quotes, a literal triple-quote, path-like
    backslashes, or a trailing quote must not break or escape out of any
    generated docstring (single-node agent, multi-node model/agent/sub-agent
    files, flow.py, __main__.py)."""
    session, user = db
    monkeypatch.delenv("EXPORT_PWNED", raising=False)

    triple_quote_payload = 'Evil"""\nimport os; os.environ["EXPORT_PWNED"] = "1"\n"""'
    win_path = 'C:\\new\\Names'
    trailing_quote = 'Say "yes"'

    fetch = make_fetch_tool(session, user)

    single_agent = make_agent(session, user, single_node_agent_config(
        output_paths={"go": {"description": trailing_quote}},
    ), name=win_path)

    multi_agent = make_agent(session, user, {
        "nodes": {
            "writer": {"agent_type": "pydanticai", "name": triple_quote_payload,
                       "system_prompt": "s"},
            "reviewer": {"agent_type": "reflection", "name": win_path,
                         "system_prompt": "s",
                         "output_paths": {
                             "approve": {"description": trailing_quote},
                             "revise": {"description": "fine"},
                         }},
        },
        "edges": [
            {"from_node": "writer", "to_node": "reviewer", "is_loop": False},
            {"from_node": "reviewer", "to_node": "writer", "is_loop": True,
             "output_path": "revise"},
        ],
        "entry_point": "writer", "exit_points": ["reviewer"],
        "max_loop_iterations": 2,
    }, name=triple_quote_payload)

    flow = make_flow(
        session, user,
        nodes={
            "F": {"node_type": "tool", "id": fetch.id, "name": "Fetch"},
            "AG1": {"node_type": "agent", "id": single_agent.id, "name": "Simple"},
            "AG2": {"node_type": "agent", "id": multi_agent.id, "name": "Complex"},
        },
        edges=[
            {"from_node": "F", "to_node": "AG1"},
            {"from_node": "AG1", "to_node": "AG2"},
        ],
        entry="F", exits=["AG2"], name=triple_quote_payload,
    )

    files = export_flow(
        flow, session, LLM_CONFIG, agent_llms={"AG1": "anth", "AG2": "anth"},
    )
    assert_all_compile(files)
    assert_standalone(files)
    assert_no_secret_leak(files)

    single_path = next(
        p for p in files
        if re.match(r"^agents/[^/]+\.py$", p) and p != "agents/__init__.py"
    )
    single_module = single_path[:-3].replace("/", ".")

    multi_folder = next(
        p.rsplit("/", 1)[0] for p in files
        if p.endswith("/agent.py") and not p.startswith("agents/_runtime/")
    )
    multi_module = multi_folder.replace("/", ".")
    sub_paths = [
        p for p in files
        if p.startswith(multi_folder + "/agents/") and p.endswith(".py")
        and not p.endswith("__init__.py")
    ]
    assert len(sub_paths) == 2

    monkeypatch.delenv("EXPORT_PWNED", raising=False)

    single_mod = import_exported(files, single_module)
    assert "EXPORT_PWNED" not in os.environ
    assert win_path in (single_mod.__doc__ or "")
    assert trailing_quote in (single_mod.Go.__doc__ or "")

    model_mod = importlib.import_module(f"{multi_module}.model")
    assert "EXPORT_PWNED" not in os.environ
    assert triple_quote_payload in (model_mod.__doc__ or "")

    agent_mod = importlib.import_module(f"{multi_module}.agent")
    assert "EXPORT_PWNED" not in os.environ
    assert triple_quote_payload in (agent_mod.__doc__ or "")

    found_trailing_in_class = False
    for sub_path in sub_paths:
        sub_module = sub_path[:-3].replace("/", ".")
        sub_mod = importlib.import_module(sub_module)
        assert "EXPORT_PWNED" not in os.environ
        doc = sub_mod.__doc__ or ""
        assert (triple_quote_payload in doc) or (win_path in doc)
        if hasattr(sub_mod, "_output_types"):
            classes = typing.get_args(sub_mod._output_types())
            docs = [c.__doc__ or "" for c in classes]
            if any(trailing_quote in d for d in docs):
                found_trailing_in_class = True
    assert found_trailing_in_class

    main_py = files["__main__.py"]
    main_tree = ast.parse(main_py)
    main_doc = ast.get_docstring(main_tree) or ""
    assert triple_quote_payload in main_doc

    flow_tree = ast.parse(files["flow.py"])
    flow_doc = ast.get_docstring(flow_tree) or ""
    assert triple_quote_payload in flow_doc

    assert "EXPORT_PWNED" not in os.environ


def test_comment_text_single_line(db, import_exported, monkeypatch):
    """A newline embedded in a node/tool name must not turn into a literal
    newline inside a generated '#' comment line — the text after it would
    then become real, uncommented code instead of a comment."""
    session, user = db
    monkeypatch.delenv("EXPORT_PWNED", raising=False)

    node_name = 'Fetch\nimport os; os.environ["EXPORT_PWNED"] = "1"'
    tool_name = 'Grab\nimport os; os.environ["EXPORT_PWNED"] = "1"'

    fetch = make_tool(
        session, user, tool_name,
        'def fetch_page(url):\n    return {"text": url + "!", "status": 200}\n',
        "fetch_page",
        input_schema={"properties": {"url": {"type": "str"}}},
        output_schema={"type": "object",
                       "properties": {"text": {"type": "str"},
                                      "status": {"type": "int"}}},
    )
    summarize = make_summarize_tool(session, user)
    flow = make_flow(
        session, user,
        nodes={
            "T": {"node_type": "trigger", "id": 0, "name": "Trigger",
                  "input_value": "https://example.com"},
            "F": {"node_type": "tool", "id": fetch.id, "name": node_name},
            "S": {"node_type": "tool", "id": summarize.id, "name": "Summarize",
                  "input_values": {"max_words": 100}},
        },
        edges=[
            {"from_node": "T", "to_node": "F", "mapping": {"": "url"}},
            {"from_node": "T", "to_node": "S"},
        ],
        entry="T", exits=["F", "S"],
    )

    files = export_flow(flow, session, LLM_CONFIG)
    assert_all_compile(files)
    assert_standalone(files)

    flow_py = files["flow.py"]
    assert node_name not in flow_py
    assert "Fetch import os; os.environ" in flow_py

    tool_paths = [
        p for p in files
        if p.startswith("tools/") and p.endswith(".py") and p != "tools/__init__.py"
    ]
    fetch_tool_path = next(p for p in tool_paths if "def fetch_page(" in files[p])
    tool_py = files[fetch_tool_path]
    assert tool_name not in tool_py
    assert "Grab import os; os.environ" in tool_py

    monkeypatch.delenv("EXPORT_PWNED", raising=False)
    sys.modules.pop("flow", None)
    import_exported(files, "flow")
    assert "EXPORT_PWNED" not in os.environ

    tool_module = fetch_tool_path[:-3].replace("/", ".")
    sys.modules.pop(tool_module, None)
    importlib.import_module(tool_module)
    assert "EXPORT_PWNED" not in os.environ



# ─── _docstring_text fuzz follow-up: leading-quote injection ───────────────
# A fuzz check found _docstring_text only escapes a `"` when the text
# contains `"""` or ends with `"`. Every docstring template puts a `"`
# immediately BEFORE the user text (e.g. `"""Agent "{name}" — ..."""`), so
# text starting with `""` combines with that `"` into a closing `"""`,
# ending the docstring early and turning the rest of the name into real
# source. These tests reproduce that for every template position.

PWNED_PAYLOAD = '""\nimport os; os.environ["EXPORT_PWNED"] = "1"\n#'


def test_leading_quote_injection_single_node_agent_name(db, import_exported, monkeypatch):
    """A single-node agent name starting with two double-quotes closes
    the triple-quoted docstring template early (_emit_single_node_agent's
    Agent-name line), letting the injected line run as real code on
    import."""
    session, user = db
    monkeypatch.delenv("EXPORT_PWNED", raising=False)
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, single_node_agent_config(), name=PWNED_PAYLOAD)
    flow = agent_flow(session, user, agent, fetch)

    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)

    single_path = next(
        p for p in files
        if re.match(r"^agents/[^/]+\.py$", p) and p != "agents/__init__.py"
    )
    single_module = single_path[:-3].replace("/", ".")

    monkeypatch.delenv("EXPORT_PWNED", raising=False)
    mod = import_exported(files, single_module)
    assert "EXPORT_PWNED" not in os.environ
    assert PWNED_PAYLOAD in (mod.__doc__ or "")


def test_leading_quote_injection_multi_node_agent_name(db, import_exported, monkeypatch):
    """Same bug for a multi-node agent's name, which is spliced into both
    model.py's and agent.py's docstring templates (_emit_multi_model,
    _emit_agent_orchestrator)."""
    session, user = db
    monkeypatch.delenv("EXPORT_PWNED", raising=False)
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, writer_reviewer_agent_config(), name=PWNED_PAYLOAD)
    flow = agent_flow(session, user, agent, fetch)

    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)

    multi_folder = next(
        p.rsplit("/", 1)[0] for p in files
        if p.endswith("/agent.py") and not p.startswith("agents/_runtime/")
    )
    multi_module = multi_folder.replace("/", ".")

    monkeypatch.delenv("EXPORT_PWNED", raising=False)
    model_mod = import_exported(files, f"{multi_module}.model")
    assert "EXPORT_PWNED" not in os.environ
    assert PWNED_PAYLOAD in (model_mod.__doc__ or "")

    monkeypatch.delenv("EXPORT_PWNED", raising=False)
    agent_mod = importlib.import_module(f"{multi_module}.agent")
    assert "EXPORT_PWNED" not in os.environ
    assert PWNED_PAYLOAD in (agent_mod.__doc__ or "")


def test_leading_quote_injection_sub_agent_name(db, import_exported, monkeypatch):
    """Same bug for a sub-agent's display name, spliced into the
    sub-agent docstring template (_emit_sub_agent)."""
    session, user = db
    monkeypatch.delenv("EXPORT_PWNED", raising=False)
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, writer_reviewer_agent_config(
        writer_overrides={"name": PWNED_PAYLOAD},
    ), name="Reflection2")
    flow = agent_flow(session, user, agent, fetch)

    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)

    multi_folder = next(
        p.rsplit("/", 1)[0] for p in files
        if p.endswith("/agent.py") and not p.startswith("agents/_runtime/")
    )
    sub_path = next(
        p for p in files
        if p.startswith(multi_folder + "/agents/") and p.endswith(".py")
        and not p.endswith("__init__.py") and not p.endswith("/reviewer.py")
    )
    sub_module = sub_path[:-3].replace("/", ".")

    monkeypatch.delenv("EXPORT_PWNED", raising=False)
    sub_mod = import_exported(files, sub_module)
    assert "EXPORT_PWNED" not in os.environ
    assert PWNED_PAYLOAD in (sub_mod.__doc__ or "")


def test_leading_quote_injection_flow_name(db):
    """Same bug for the flow name, spliced into both flow.py's and
    __main__.py's docstring templates. In flow.py the early close leaves a
    stray closing triple-quote dangling, which then opens a second,
    unterminated triple-quoted string that swallows the rest of the file —
    flow.py fails to even compile. __main__.py still parses, but gains the
    injected assignment as a real top-level statement and its docstring
    loses the rest of the flow name."""
    session, user = db
    fetch = make_fetch_tool(session, user)
    flow = make_flow(
        session, user,
        nodes={"F": {"node_type": "tool", "id": fetch.id, "name": "Fetch"}},
        edges=[], entry="F", exits=["F"], name=PWNED_PAYLOAD,
    )
    baseline_flow = make_flow(
        session, user,
        nodes={"F": {"node_type": "tool", "id": fetch.id, "name": "Fetch"}},
        edges=[], entry="F", exits=["F"], name="Ordinary Flow",
    )

    files = export_flow(flow, session, LLM_CONFIG)
    baseline_files = export_flow(baseline_flow, session, LLM_CONFIG)

    # flow.py must still be syntactically valid Python.
    compile(files["flow.py"], "flow.py", "exec")

    # __main__.py: still parses, but must not gain extra top-level
    # statements, and its docstring must retain the exact flow name.
    main_tree = ast.parse(files["__main__.py"])
    baseline_tree = ast.parse(baseline_files["__main__.py"])
    assert len(main_tree.body) == len(baseline_tree.body)
    main_doc = ast.get_docstring(main_tree) or ""
    assert PWNED_PAYLOAD in main_doc


# ─── _docstring_text unit tests ──────────────────────────────────────────────

TRICKY_DOCSTRING_STRINGS = [
    '""x', 'x"', 'a""', '"', '""', '"""', '\\', 'a\\"',
    'C:\\new\\Names', 'say "hi" now', 'cr\rhere', 'nul\x00here',
    'tab\there', 'line1\nline2', 'é😀',
]
TRICKY_DOCSTRING_IDS = [
    "leading-double-quote", "trailing-quote", "double-trailing-quote",
    "single-quote", "double-quote", "triple-quote", "backslash",
    "char-then-backslash-quote", "windows-path", "quoted-word",
    "carriage-return", "nul-byte", "tab", "two-lines", "unicode",
]


@pytest.mark.parametrize("s", TRICKY_DOCSTRING_STRINGS, ids=TRICKY_DOCSTRING_IDS)
def test_docstring_text_safe_in_agent_name_context(s):
    """_docstring_text(s) dropped into the agent-name docstring shape
    (an opening triple-quote, the word Agent, a double-quote, the escaped
    text, a closing double-quote, more text, a closing triple-quote) must
    always parse as a single module-level string-constant statement whose
    value still contains the original text — i.e. it must never close the
    docstring early or break the syntax."""
    esc = _docstring_text(s)
    source = '"""Agent "' + esc + '" \u2014 x."""'
    tree = ast.parse(source)
    assert len(tree.body) == 1
    stmt = tree.body[0]
    assert isinstance(stmt, ast.Expr)
    assert isinstance(stmt.value, ast.Constant)
    assert isinstance(stmt.value.value, str)
    assert s in stmt.value.value


@pytest.mark.parametrize("s", TRICKY_DOCSTRING_STRINGS, ids=TRICKY_DOCSTRING_IDS)
def test_docstring_text_safe_in_bare_context(s):
    """Same as above but for the bare triple-quoted shape used for
    output-path descriptions — the escaped text must round-trip to exactly
    s."""
    esc = _docstring_text(s)
    source = '"""' + esc + '"""'
    tree = ast.parse(source)
    assert len(tree.body) == 1
    stmt = tree.body[0]
    assert isinstance(stmt, ast.Expr)
    assert isinstance(stmt.value, ast.Constant)
    assert isinstance(stmt.value.value, str)
    assert stmt.value.value == s


def test_docstring_text_unchanged_for_ordinary_text():
    """Plain names/descriptions (no quotes, backslashes, or odd control
    characters) must pass through _docstring_text untouched."""
    for s in ["writer", "Reflection", "Needs another pass", "My Agent (v2)",
              "é😀", "line1\nline2"]:
        assert _docstring_text(s) == s


# ─── Zip wrapper ─────────────────────────────────────────────────────────────

def test_zip_round_trip(db):
    session, user = db
    fetch = make_fetch_tool(session, user)
    flow = make_flow(
        session, user,
        nodes={"F": {"node_type": "tool", "id": fetch.id, "name": "Fetch"}},
        edges=[], entry="F", exits=["F"], name="My Flow",
    )
    zip_bytes, filename = export_flow_zip(flow, session, LLM_CONFIG)
    assert filename == "my_flow.zip"
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        names = archive.namelist()
        assert all(n.startswith("my_flow/") for n in names)
        assert "my_flow/flow.py" in names
        assert "my_flow/tools/fetch_page.py" in names
        assert "my_flow/README.md" in names
        assert "my_flow/__main__.py" in names


# ─── Named inputs (claude_plans/named-agent-inputs.md, plan step 6) ──────────
#
# Flow edges into an agent emit a dict literal over every incoming edge
# (not just incoming[0]); single-node agents emit `run(inputs: dict)` and
# resolve every {name} placeholder (plus the {{name}} escape) from that
# dict; multi-node agents thread the same dict through run_graph end to end.

def test_single_node_agent_dict_literal_input_in_flow_py(db, import_exported):
    """Two flow edges into one agent node, each selecting a different field
    of the same upstream dict output, must both appear as entries of a
    single dict literal passed to the agent's run() — not just
    incoming[0]."""
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, single_node_agent_config(
        user_prompt="Draft: {draft}\nNotes: {notes}\nTemplate: {{literal}}",
    ))
    flow = make_flow(
        session, user,
        nodes={
            "F": {"node_type": "tool", "id": fetch.id, "name": "Fetch"},
            "AG": {"node_type": "agent", "id": agent.id, "name": "Reviewer"},
        },
        edges=[
            {"from_node": "F", "to_node": "AG", "mapping": {"text": "draft"}},
            {"from_node": "F", "to_node": "AG", "mapping": {"status": "notes"}},
        ],
        entry="F", exits=["AG"],
    )
    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)
    assert_standalone(files)

    agent_py = files["agents/reviewer.py"]
    assert "async def run_reviewer(inputs: dict) -> str:" in agent_py

    flow_py = files["flow.py"]
    assert "run_reviewer({" in flow_py
    assert '"draft":' in flow_py
    assert '"notes":' in flow_py

    module = import_exported(files, "agents.reviewer")
    captured = []

    def fn(messages, info):
        captured.append(messages[-1].parts[-1].content)
        return ModelResponse(parts=[TextPart("ok")])

    with module.agent.override(model=FunctionModel(fn)):
        asyncio.run(module.run_reviewer({"draft": "D1", "notes": "N1"}))

    prompt = captured[0]
    assert "Draft: D1" in prompt
    assert "Notes: N1" in prompt
    # {{literal}} is an escape: literal "{literal}" text, not substituted.
    assert "Template: {literal}" in prompt


def test_single_node_agent_missing_input_resolves_empty(db, import_exported):
    """An input with no connected edge resolves to "" (decision 6) — the
    run() call must not KeyError or leave the placeholder un-substituted."""
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, single_node_agent_config(
        user_prompt="Draft: {draft}\nNotes: {notes}",
    ))
    flow = make_flow(
        session, user,
        nodes={
            "F": {"node_type": "tool", "id": fetch.id, "name": "Fetch"},
            "AG": {"node_type": "agent", "id": agent.id, "name": "Reviewer"},
        },
        edges=[{"from_node": "F", "to_node": "AG", "mapping": {"text": "draft"}}],
        entry="F", exits=["AG"],
    )
    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)

    module = import_exported(files, "agents.reviewer")
    captured = []

    def fn(messages, info):
        captured.append(messages[-1].parts[-1].content)
        return ModelResponse(parts=[TextPart("ok")])

    with module.agent.override(model=FunctionModel(fn)):
        asyncio.run(module.run_reviewer({"draft": "D1"}))

    assert captured[0] == "Draft: D1\nNotes: "


def reflection_named_inputs_config():
    return {
        "nodes": {
            "planner": {"agent_type": "pydanticai", "name": "planner", "system_prompt": "p",
                        "user_prompt": "{input}"},
            "writer": {"agent_type": "pydanticai", "name": "writer", "system_prompt": "w",
                       "user_prompt": "{plan} / {feedback}"},
            "reviewer": {"agent_type": "reflection", "name": "reviewer", "system_prompt": "r",
                         "user_prompt": "{input}",
                         "output_paths": {
                             "revise": {"description": "Needs another pass"},
                             "approve": {"description": "Good to go",
                                         "return_behavior": "previous_output",
                                         "return_input": "input"},
                         }},
        },
        "edges": [
            {"from_node": "planner", "to_node": "writer", "target_input": "plan"},
            {"from_node": "writer", "to_node": "reviewer"},
            {"from_node": "reviewer", "to_node": "writer", "is_loop": True,
             "output_path": "revise", "target_input": "feedback"},
        ],
        "entry_point": "planner",
        "exit_points": ["reviewer"],
        "max_loop_iterations": 2,
    }


def test_multinode_reflection_loop_named_inputs_runs(db, import_exported):
    """Behavioral, end-to-end test of the new engine through an export:
    Planner -> Writer.plan, Writer -> Reviewer, Reviewer -(revise, loop)->
    Writer.feedback. Writer must keep `plan` across iterations (planner is
    outside the loop) while `feedback` updates every iteration — driven
    with FunctionModel fakes, no network."""
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, reflection_named_inputs_config(),
                        name="ReflectionNamed")
    flow = agent_flow(session, user, agent, fetch)
    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)

    pkg_module = import_exported(files, "agents.reflectionnamed")
    planner_mod = importlib.import_module("agents.reflectionnamed.agents.planner")
    writer_mod = importlib.import_module("agents.reflectionnamed.agents.writer")
    reviewer_mod = importlib.import_module("agents.reflectionnamed.agents.reviewer")

    def planner_fn(messages, info):
        return ModelResponse(parts=[TextPart("PLAN")])

    def make_writer_fn(prompts):
        def writer_fn(messages, info):
            prompts.append(messages[-1].parts[-1].content)
            return ModelResponse(parts=[TextPart(f"draft{len(prompts)}")])
        return writer_fn

    def make_reviewer_fn(script):
        def reviewer_fn(messages, info):
            choice = script.pop(0)
            return ModelResponse(parts=[ToolCallPart(
                f"final_result_{choice.capitalize()}", {"content": f"{choice} note"})])
        return reviewer_fn

    writer_prompts = []
    with planner_mod.agent.override(model=FunctionModel(planner_fn)), \
         writer_mod.agent.override(model=FunctionModel(make_writer_fn(writer_prompts))), \
         reviewer_mod.agent.override(model=FunctionModel(make_reviewer_fn(["revise", "approve"]))):
        result = asyncio.run(pkg_module.run_reflectionnamed({"input": "go"}))

    assert len(writer_prompts) == 2
    assert writer_prompts[0] == "PLAN / "
    assert writer_prompts[1] == "PLAN / revise note"
    # approve's previous_output returns the reviewer's own "input" (the
    # last thing it reviewed, i.e. writer's second draft).
    assert result == "draft2"


# ─── parallel-agent-execution: exported run_graph(..., parallel=...) ────────
#
# claude_plans/parallel-agent-execution.md, plan step 3: the emitted
# multi-node orchestrator (agents/<slug>/agent.py) must forward the export's
# `parallel` flag into its run_graph(...) call — `parallel=True` (live
# default) when the export dialog's Parallel toggle is on, `parallel=False`
# when it's off. Single-node agents are unaffected (no run_graph call at
# all), so these tests only touch the writer/reviewer multi-node fixture.

def test_multinode_agent_orchestrator_run_graph_parallel_true_by_default(db):
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, writer_reviewer_agent_config())
    flow = agent_flow(session, user, agent, fetch)

    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})
    assert_all_compile(files)

    agent_py = files["agents/reviewer/agent.py"]
    run_graph_line = next(
        l for l in agent_py.splitlines() if "run_graph(GRAPH_CONFIG, inputs" in l)
    assert "parallel=True" in run_graph_line, run_graph_line


def test_multinode_agent_orchestrator_run_graph_parallel_false_when_exported_sequential(db):
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, writer_reviewer_agent_config())
    flow = agent_flow(session, user, agent, fetch)

    files = export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"}, parallel=False)
    assert_all_compile(files)

    agent_py = files["agents/reviewer/agent.py"]
    run_graph_line = next(
        l for l in agent_py.splitlines() if "run_graph(GRAPH_CONFIG, inputs" in l)
    assert "parallel=False" in run_graph_line, run_graph_line
