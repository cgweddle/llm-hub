"""
Tests for src/runners/tool_runner.py (in-process tool compilation) and the
engine's global tool timeout (TOOL_TIMEOUT_SECONDS in flow_runner).

Flow-level tests run FlowRunner inside pytest against a per-test SQLite DB —
the same engine code the local flow child and hosted container execute.
"""
import pytest

from src.database.database_setup import DatabaseManager, User, Tool, Flow, Execution
from src.runners import flow_runner
from src.runners.flow_runner import FlowRunner
from src.runners.tool_runner import ToolCompileError, compile_tool


class FakeTool:
    def __init__(self, script_code, main_function):
        self.script_code = script_code
        self.main_function = main_function


# ─── compile_tool unit tests ─────────────────────────────────────────────────

def test_compile_tool_returns_native_objects():
    func = compile_tool(FakeTool("def make(x):\n    return {x, 'native'}\n", "make"))
    result = func(x="a")
    assert isinstance(result, set)
    assert result == {"a", "native"}


def test_compile_tool_syntax_error():
    with pytest.raises(ToolCompileError, match="failed to load"):
        compile_tool(FakeTool("def broken(:\n", "broken"))


def test_compile_tool_missing_main():
    with pytest.raises(ToolCompileError, match="not found or not callable"):
        compile_tool(FakeTool("def other():\n    return 1\n", "main"))


def test_compile_tool_uncallable_main():
    with pytest.raises(ToolCompileError, match="not found or not callable"):
        compile_tool(FakeTool("main = 42\n", "main"))


def test_compile_tool_main_block_inert(tmp_path):
    marker = tmp_path / "ran.txt"
    script = (
        "def f(x):\n"
        "    return x\n"
        "if __name__ == '__main__':\n"
        f"    open({str(marker)!r}, 'w').write('ran')\n"
    )
    func = compile_tool(FakeTool(script, "f"))
    assert func(x=1) == 1
    assert not marker.exists()


def test_compile_tool_fresh_globals_per_compile():
    script = "state = []\ndef f(x):\n    state.append(x)\n    return len(state)\n"
    first = compile_tool(FakeTool(script, "f"))
    assert first(x=1) == 1
    assert first(x=2) == 2
    second = compile_tool(FakeTool(script, "f"))
    assert second(x=3) == 1  # recompile does not share module state


# ─── flow-level tests (engine integration) ───────────────────────────────────

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


def make_tool(session, user, name, script, main_function, params):
    tool = Tool(
        user_id=user.id,
        name=name,
        tool_type="function",
        main_function=main_function,
        script_code=script,
        input_schema={"properties": {p: {"type": "str"} for p in params}},
    )
    session.add(tool)
    session.commit()
    session.refresh(tool)
    return tool


def make_flow(session, user, nodes, edges, entry, exits):
    flow = Flow(
        user_id=user.id,
        name="test-flow",
        graph_config={"nodes": nodes, "edges": edges,
                      "entry_point": entry, "exit_points": exits},
        entry_point=entry,
        exit_points=exits,
    )
    session.add(flow)
    session.commit()
    session.refresh(flow)
    return flow


def tool_node(tool, name):
    return {"node_type": "tool", "id": tool.id, "name": name, "input_values": {}}


def test_global_timeout_marks_node_failed(db, monkeypatch):
    session, user = db
    monkeypatch.setattr(flow_runner, "TOOL_TIMEOUT_SECONDS", 1)
    slow = make_tool(
        session, user, "SLOW",
        "import time\ndef slow(x):\n    time.sleep(5)\n    return x\n",
        "slow", ["x"],
    )
    flow = make_flow(session, user, nodes={"SLOW": tool_node(slow, "SLOW")},
                     edges=[], entry="SLOW", exits=["SLOW"])

    runner = FlowRunner(session, flow.id, user.id)
    result = runner.run({"x": "1"})
    assert result["status"] == "failed"
    assert "timed out after 1s" in result["error"]

    check = DatabaseManager().get_session()
    try:
        row = (
            check.query(Execution)
            .filter(Execution.parent_id == result["execution_id"])
            .one()
        )
        assert row.status == "failed"  # not stuck at 'running'
        assert "timed out" in row.error_message
        assert row.completed_at is not None
    finally:
        check.close()


def test_native_objects_pass_between_nodes(db):
    session, user = db
    a = make_tool(
        session, user, "A",
        "def a(x):\n    return {x, 'from_a'}\n",
        "a", ["x"],
    )
    b = make_tool(
        session, user, "B",
        "def b(data):\n"
        "    assert isinstance(data, set), type(data).__name__\n"
        "    return ','.join(sorted(data))\n",
        "b", ["data"],
    )
    flow = make_flow(
        session, user,
        nodes={"A": tool_node(a, "A"), "B": tool_node(b, "B")},
        edges=[{"from_node": "A", "to_node": "B"}],
        entry="A", exits=["B"],
    )

    runner = FlowRunner(session, flow.id, user.id)
    result = runner.run({"x": "1"})
    assert result["status"] == "completed"
    assert result["final_output"] == "1,from_a"
    assert isinstance(runner.ctx.state["A"], set)  # raw object, never serialized


def test_tool_error_message_includes_type(db):
    session, user = db
    boom = make_tool(
        session, user, "BOOM",
        "def boom(x):\n    raise KeyError('missing_column')\n",
        "boom", ["x"],
    )
    flow = make_flow(session, user, nodes={"BOOM": tool_node(boom, "BOOM")},
                     edges=[], entry="BOOM", exits=["BOOM"])

    runner = FlowRunner(session, flow.id, user.id)
    result = runner.run({"x": "1"})
    assert result["status"] == "failed"

    check = DatabaseManager().get_session()
    try:
        row = (
            check.query(Execution)
            .filter(Execution.parent_id == result["execution_id"])
            .one()
        )
        assert row.error_message.startswith("KeyError:")
    finally:
        check.close()
