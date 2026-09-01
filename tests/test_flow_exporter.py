"""
Tests for the flow exporter (src/exporters/flow_exporter.py).

Every emitted .py file must compile; wiring assertions check that edge
mappings, typed-in values, and parallelism are inlined as readable source;
guard tests cover Phase-1 limits (single-node agents only), credential-leak
prevention, and the no-llmhub-imports contract.

Uses a real per-test SQLite DB like test_flow_runner_agents.py.
"""
import io
import re
import zipfile

import pytest

from src.database.database_setup import (
    DatabaseManager, User, Tool, Flow, Agent as AgentModel,
)
from src.exporters.flow_exporter import (
    FlowExportError, export_flow, export_flow_zip,
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


def assert_all_compile(files):
    for path, content in files.items():
        if path.endswith(".py"):
            compile(content, path, "exec")


def assert_standalone(files):
    for path, content in files.items():
        assert not re.search(r"(from|import)\s+src\.", content), path
        assert "llm_hub" not in content and "llmhub" not in content, path


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
    # union of required_packages minus infra denylist; no agent pins
    requirements = files["requirements.txt"].splitlines()
    assert "requests" in requirements and "pandas" in requirements
    assert "sqlalchemy" not in requirements
    assert not any(r.startswith("pydantic-ai") for r in requirements)


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
    # user prompt template resolved at run time
    assert 'USER_PROMPT.replace("{input}", node_input)' in agent_py
    # agent tool registered as the raw imported function
    assert "from tools.fetch_page import fetch_page" in agent_py
    assert "agent.tool_plain(fetch_page)" in agent_py

    flow_py = files["flow.py"]
    assert "from agents.reviewer import run_reviewer" in flow_py
    assert "await run_reviewer(_as_text(fetch_out))" in flow_py

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


def test_multi_node_agent_rejected(db):
    session, user = db
    fetch = make_fetch_tool(session, user)
    agent = make_agent(session, user, {
        "nodes": {
            "writer": {"agent_type": "pydanticai", "name": "writer",
                       "system_prompt": "w"},
            "reviewer": {"agent_type": "reflection", "name": "reviewer",
                         "system_prompt": "r"},
        },
        "edges": [
            {"from_node": "writer", "to_node": "reviewer", "is_loop": False},
            {"from_node": "reviewer", "to_node": "writer", "is_loop": True},
        ],
        "entry_point": "writer", "exit_points": ["reviewer"],
        "max_loop_iterations": 2,
    })
    flow = agent_flow(session, user, agent, fetch)
    with pytest.raises(FlowExportError, match="single-node agents only"):
        export_flow(flow, session, LLM_CONFIG, agent_llms={"AG": "anth"})


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
    # "text" is a str field in the schema — direct subscript, no _as_text
    assert "await run_reviewer(fetch_out['text'])" in flow_py


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
    assert "await run_reviewer(_as_text(fetch_out['status']))" in files["flow.py"]


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
