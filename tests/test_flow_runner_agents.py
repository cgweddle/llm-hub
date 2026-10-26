"""
Flow-level tests for prepare-time agent compilation (FlowRunner._prepare_agents):
compile-once per run (reflection loops reuse BuiltAgents), per-node provider
resolution (multi-provider flows), and recompilation on resume after an
agent edit.

compile_agent is patched at the flow_runner seam with a counting fake that
returns TestModel-backed BuiltAgents — no network, no API keys. Each compiled
agent's output encodes its compile generation ("out-<n>") so tests can tell
WHICH compilation produced a node's output.

Uses a real per-test SQLite DB and the real FlowRunner/AgentExecutor code
path, like test_flow_runner_resume.py.

Named-inputs additions (claude_plans/named-agent-inputs.md, plan step 5):
flow edges into an agent reuse the tool-edge `mapping` shape, and the
agent's named inputs arrive as a dict (`{name: value}`), with per-input
previews recorded on the agent's execution row.
"""
from datetime import datetime

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from src.database.database_setup import (
    DatabaseManager, User, Tool, Flow, Execution, Agent as AgentModel,
)
import src.runners.flow_runner as fr
from src.runners.flow_runner import FlowRunner
from src.runners.agent_runner import BuiltAgent


LLM_CONFIG = {"models": [
    {"name": "anth", "provider": "anthropic", "model": "claude-x", "api_key": "ka"},
    {"name": "lm", "provider": "lmstudio", "model": "qwen"},
]}

OK = "def {f}(x):\n    return x + '{f}'\n"
BOOM = "def {f}(x):\n    raise ValueError('boom')\n"


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


@pytest.fixture
def patched_compile(monkeypatch):
    """Replace compile_agent with a counting fake; returns the call log."""
    calls = []

    def fake_compile_agent(sub_node_config, tool_records, provider_config):
        calls.append({"sub": sub_node_config.get("name"),
                      "provider": provider_config["name"]})
        return BuiltAgent(agent=Agent(
            TestModel(custom_output_text=f"out-{len(calls)}"),
            system_prompt="t"))

    monkeypatch.setattr(fr, "compile_agent", fake_compile_agent)
    return calls


@pytest.fixture
def capturing_compile(monkeypatch):
    """Replace compile_agent with a fake whose model records the exact
    resolved user message it receives for every run — lets a test see
    whether a named input actually reached the sub-agent's prompt."""
    captured = []

    def fn(messages, info):
        captured.append(messages[-1].parts[-1].content)
        return ModelResponse(parts=[TextPart("ok")])

    def fake_compile_agent(sub_node_config, tool_records, provider_config):
        return BuiltAgent(agent=Agent(FunctionModel(fn), system_prompt="t"))

    monkeypatch.setattr(fr, "compile_agent", fake_compile_agent)
    return captured


def make_agent(session, user, graph_config, name="agent"):
    agent = AgentModel(user_id=user.id, name=name, graph_config=graph_config)
    session.add(agent)
    session.commit()
    session.refresh(agent)
    return agent


def make_tool(session, user, name, script, main_function):
    tool = Tool(
        user_id=user.id, name=name, tool_type="function",
        main_function=main_function, script_code=script,
        input_schema={"properties": {"x": {"type": "str"}}},
    )
    session.add(tool)
    session.commit()
    session.refresh(tool)
    return tool


def make_flow(session, user, nodes, edges, entry, exits):
    flow = Flow(
        user_id=user.id, name="test-flow",
        graph_config={"nodes": nodes, "edges": edges,
                      "entry_point": entry, "exit_points": exits},
        entry_point=entry, exit_points=exits,
    )
    session.add(flow)
    session.commit()
    session.refresh(flow)
    return flow


def single_node_agent_config(name="Main"):
    return {
        "nodes": {"main": {"agent_type": "pydanticai", "name": name,
                            "system_prompt": "You help."}},
        "edges": [],
        "entry_point": "main",
        "exit_points": ["main"],
    }


def two_input_agent_config(name="Main"):
    return {
        "nodes": {"main": {"agent_type": "pydanticai", "name": name,
                            "system_prompt": "You help.",
                            "user_prompt": "{draft} / {notes}"}},
        "edges": [],
        "entry_point": "main",
        "exit_points": ["main"],
    }


def single_named_input_agent_config(input_name="draft", name="Main"):
    return {
        "nodes": {"main": {"agent_type": "pydanticai", "name": name,
                            "system_prompt": "You help.",
                            "user_prompt": f"{{{input_name}}}"}},
        "edges": [],
        "entry_point": "main",
        "exit_points": ["main"],
    }


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_reflection_loop_compiles_once_per_sub_node(db, patched_compile):
    """A looping two-node agent runs its sub-nodes repeatedly but compiles
    each sub-node exactly once per run."""
    session, user = db
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
        "entry_point": "writer",
        "exit_points": ["reviewer"],
        "max_loop_iterations": 2,
    })
    flow = make_flow(
        session, user,
        nodes={"AG": {"node_type": "agent", "id": agent.id, "name": "AG"}},
        edges=[], entry="AG", exits=["AG"],
    )

    runner = FlowRunner(session, flow.id, user.id,
                        llm_config=LLM_CONFIG, agent_llms={"AG": "anth"})
    result = runner.run("go")

    assert result["status"] == "completed"
    assert [c["sub"] for c in patched_compile] == ["writer", "reviewer"]

    # The loop actually iterated: the multi-node agent recorded more
    # sub-node execution rows than it has sub-nodes.
    agent_child = (session.query(Execution)
                   .filter(Execution.parent_id == result["execution_id"])
                   .one())
    sub_runs = (session.query(Execution)
                .filter(Execution.parent_id == agent_child.id)
                .count())
    assert sub_runs > 2, f"expected loop iterations, got {sub_runs} sub-runs"


def test_multi_provider_flow_resolves_per_node(db, patched_compile):
    """Two agent nodes at the same topological level get their own provider
    configs — the regression case for the old shared-env-var resolution."""
    session, user = db
    a1 = make_agent(session, user, single_node_agent_config("One"), name="one")
    a2 = make_agent(session, user, single_node_agent_config("Two"), name="two")
    entry = make_tool(session, user, "T", OK.format(f="t"), "t")
    flow = make_flow(
        session, user,
        nodes={
            "T": {"node_type": "tool", "id": entry.id, "name": "T", "input_values": {}},
            "A1": {"node_type": "agent", "id": a1.id, "name": "A1"},
            "A2": {"node_type": "agent", "id": a2.id, "name": "A2"},
        },
        edges=[{"from_node": "T", "to_node": "A1"},
               {"from_node": "T", "to_node": "A2"}],
        entry="T", exits=["A1", "A2"],
    )

    runner = FlowRunner(session, flow.id, user.id, llm_config=LLM_CONFIG,
                        agent_llms={"A1": "anth", "A2": "lm"})
    result = runner.run({"x": "hello"})

    assert result["status"] == "completed"
    providers = {c["sub"]: c["provider"] for c in patched_compile}
    assert providers == {"One": "anth", "Two": "lm"}


def test_resume_recompiles_edited_agent(db, patched_compile):
    """After a failure, editing the agent and resuming recompiles it on the
    fresh session AND re-runs it (fingerprint invalidation), so the resumed
    flow uses the new agent's output."""
    session, user = db
    agent = make_agent(session, user, single_node_agent_config())
    boom = make_tool(session, user, "C", BOOM.format(f="c"), "c")
    flow = make_flow(
        session, user,
        nodes={
            "AG": {"node_type": "agent", "id": agent.id, "name": "AG"},
            "C": {"node_type": "tool", "id": boom.id, "name": "C", "input_values": {}},
        },
        edges=[{"from_node": "AG", "to_node": "C"}],
        entry="AG", exits=["C"],
    )

    runner = FlowRunner(session, flow.id, user.id,
                        llm_config=LLM_CONFIG, agent_llms={"AG": "anth"})
    result = runner.run("hello")
    assert result["status"] == "failed"
    assert len(patched_compile) == 1  # compiled once for the first run

    # Fix the tool and edit the agent (explicit updated_at: SQLite's
    # func.now() is second-granular, same-second edits would be invisible
    # to fingerprinting)
    boom.script_code = OK.format(f="c")
    boom.updated_at = datetime.now()
    agent.graph_config = single_node_agent_config("Edited")
    agent.updated_at = datetime.now()
    session.commit()

    result2 = runner.resume()

    assert result2["status"] == "completed"
    assert len(patched_compile) == 2  # recompiled on resume
    # The RE-COMPILED agent's output (out-2) flowed into C, not the cached out-1
    assert "out-2c" in str(result2["final_output"])


# ─── Named inputs (claude_plans/named-agent-inputs.md, plan step 5) ──────────

def test_two_flow_edges_deliver_both_named_inputs(db, capturing_compile):
    """Two flow edges into the same agent node, each mapping a different
    upstream field to a different named input, must BOTH reach the entry
    sub-agent — as a dict, not last-writer-wins."""
    session, user = db
    agent = make_agent(session, user, two_input_agent_config())
    tool = make_tool(
        session, user, "T",
        "def t(x):\n    return {'draft_part': 'D-' + x, 'notes_part': 'N-' + x}\n",
        "t",
    )
    flow = make_flow(
        session, user,
        nodes={
            "T": {"node_type": "tool", "id": tool.id, "name": "T", "input_values": {}},
            "AG": {"node_type": "agent", "id": agent.id, "name": "AG"},
        },
        edges=[
            {"from_node": "T", "to_node": "AG", "mapping": {"draft_part": "draft"}},
            {"from_node": "T", "to_node": "AG", "mapping": {"notes_part": "notes"}},
        ],
        entry="T", exits=["AG"],
    )

    runner = FlowRunner(session, flow.id, user.id,
                        llm_config=LLM_CONFIG, agent_llms={"AG": "anth"})
    result = runner.run({"x": "seed"})

    assert result["status"] == "completed"
    assert len(capturing_compile) == 1
    resolved_prompt = capturing_compile[0]
    assert "D-seed" in resolved_prompt
    assert "N-seed" in resolved_prompt


def test_legacy_single_edge_flow_lands_in_input_key(db, patched_compile):
    """A flow edge with no `mapping` at all (today's shape) still delivers
    to the agent's "input" key — the one-time legacy read (decision 5)."""
    session, user = db
    agent = make_agent(session, user, single_node_agent_config())
    tool = make_tool(session, user, "T", OK.format(f="t"), "t")
    flow = make_flow(
        session, user,
        nodes={
            "T": {"node_type": "tool", "id": tool.id, "name": "T", "input_values": {}},
            "AG": {"node_type": "agent", "id": agent.id, "name": "AG"},
        },
        edges=[{"from_node": "T", "to_node": "AG"}],
        entry="T", exits=["AG"],
    )

    runner = FlowRunner(session, flow.id, user.id,
                        llm_config=LLM_CONFIG, agent_llms={"AG": "anth"})
    result = runner.run({"x": "seed"})

    assert result["status"] == "completed"
    root_id = result["execution_id"]
    ag_row = (session.query(Execution)
              .filter(Execution.parent_id == root_id, Execution.node_id == "AG")
              .one())
    assert isinstance(ag_row.input_data, dict)
    assert set(ag_row.input_data.keys()) == {"input"}
    assert "seedt" in ag_row.input_data["input"]


def test_agent_row_input_data_is_dict_of_bounded_previews(db, patched_compile):
    """Decision 17: the flow-level agent row's input_data is {name:
    preview<=1000}, one entry per named input, each independently capped."""
    session, user = db
    agent = make_agent(session, user, two_input_agent_config())
    tool = make_tool(
        session, user, "T",
        "def t(x):\n    return {'draft_part': x * 1200, 'notes_part': 'short'}\n",
        "t",
    )
    flow = make_flow(
        session, user,
        nodes={
            "T": {"node_type": "tool", "id": tool.id, "name": "T", "input_values": {}},
            "AG": {"node_type": "agent", "id": agent.id, "name": "AG"},
        },
        edges=[
            {"from_node": "T", "to_node": "AG", "mapping": {"draft_part": "draft"}},
            {"from_node": "T", "to_node": "AG", "mapping": {"notes_part": "notes"}},
        ],
        entry="T", exits=["AG"],
    )

    runner = FlowRunner(session, flow.id, user.id,
                        llm_config=LLM_CONFIG, agent_llms={"AG": "anth"})
    result = runner.run({"x": "z"})

    assert result["status"] == "completed"
    root_id = result["execution_id"]
    ag_row = (session.query(Execution)
              .filter(Execution.parent_id == root_id, Execution.node_id == "AG")
              .one())
    assert isinstance(ag_row.input_data, dict)
    assert set(ag_row.input_data.keys()) == {"draft", "notes"}
    assert len(ag_row.input_data["draft"]) <= 1000
    assert ag_row.input_data["notes"] == "short"


def test_flow_edge_into_renamed_agent_input_fails_before_any_node_runs(db, patched_compile):
    """Decision 16: FlowRunner._prepare_agents rejects a flow edge into an
    agent input that doesn't exist (e.g. the agent's user_prompt was edited
    after the flow was built), before any node — tool or agent — runs."""
    session, user = db
    agent = make_agent(session, user, single_named_input_agent_config("draft"))
    tool = make_tool(session, user, "T", OK.format(f="t"), "t")
    flow = make_flow(
        session, user,
        nodes={
            "T": {"node_type": "tool", "id": tool.id, "name": "T", "input_values": {}},
            "AG": {"node_type": "agent", "id": agent.id, "name": "AG"},
        },
        # "notes" doesn't exist on the agent (it only has "draft").
        edges=[{"from_node": "T", "to_node": "AG", "mapping": {"": "notes"}}],
        entry="T", exits=["AG"],
    )

    runner = FlowRunner(session, flow.id, user.id,
                        llm_config=LLM_CONFIG, agent_llms={"AG": "anth"})
    result = runner.run({"x": "hello"})

    assert result["status"] == "failed"
    assert "notes" in (result.get("error") or "")

    root_id = result["execution_id"]
    children = session.query(Execution).filter(Execution.parent_id == root_id).count()
    assert children == 0, "no node (tool or agent) should have run"
