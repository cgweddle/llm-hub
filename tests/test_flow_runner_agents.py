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
"""
from datetime import datetime

import pytest
from pydantic_ai import Agent
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
