"""
E2E tests for the local flow child (src/runners/local_flow_child.py +
src/tasks/run_flow.py stdin transport): real child processes spawned against a
per-test file-backed SQLite DB, exactly as the backend does in LOCAL mode.

The DB is the only child→test channel (same as child→backend in production);
stdin carries resume/test messages in.
"""
import json
import subprocess
import sys
import time
from datetime import datetime

import pytest

from src.database.database_setup import DatabaseManager, User, Tool, Flow, Execution
from src.database.database import create_execution
from src.runners.local_flow_child import _resolve_interpreter, spawn_local_flow_child


# ─── _resolve_interpreter unit tests ─────────────────────────────────────────

def test_resolve_interpreter_unset():
    assert _resolve_interpreter(None) == (sys.executable, None)
    assert _resolve_interpreter("") == (sys.executable, None)


def test_resolve_interpreter_env_dir(tmp_path):
    env_dir = tmp_path / "env"
    (env_dir / "bin").mkdir(parents=True)
    python = env_dir / "bin" / "python"
    python.touch()
    assert _resolve_interpreter(str(env_dir)) == (str(python), str(env_dir / "bin"))


def test_resolve_interpreter_dir_without_python(tmp_path):
    env_dir = tmp_path / "env"
    (env_dir / "bin").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="python not found"):
        _resolve_interpreter(str(env_dir))


def test_resolve_interpreter_file(tmp_path):
    python = tmp_path / "python3"
    python.touch()
    assert _resolve_interpreter(str(python)) == (str(python), None)


def test_resolve_interpreter_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="python not found"):
        _resolve_interpreter(str(tmp_path / "nope"))


# ─── E2E fixtures/helpers ────────────────────────────────────────────────────

OK = "def {f}(x):\n    return x + '{f}'\n"
BOOM = "def {f}(x):\n    raise ValueError('boom')\n"


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
def reaper():
    children = []
    yield children
    for child in children:
        child.shutdown()


def make_tool(session, user, name, script, main_function):
    tool = Tool(
        user_id=user.id,
        name=name,
        tool_type="function",
        main_function=main_function,
        script_code=script,
        input_schema={"properties": {"x": {"type": "str"}}},
    )
    session.add(tool)
    session.commit()
    session.refresh(tool)
    return tool


def edit_tool(session, tool, script):
    tool.script_code = script
    tool.updated_at = datetime.now()
    session.commit()


def make_chain_flow(session, user, with_downstream=False):
    """A(ok) → C(boom) [→ D(ok)]. D is downstream of the failure → untestable."""
    a = make_tool(session, user, "A", OK.format(f="a"), "a")
    c = make_tool(session, user, "C", BOOM.format(f="c"), "c")
    nodes = {
        "A": {"node_type": "tool", "id": a.id, "name": "A", "input_values": {}},
        "C": {"node_type": "tool", "id": c.id, "name": "C", "input_values": {}},
    }
    edges = [{"from_node": "A", "to_node": "C"}]
    exits = ["C"]
    if with_downstream:
        d = make_tool(session, user, "D", OK.format(f="d"), "d")
        nodes["D"] = {"node_type": "tool", "id": d.id, "name": "D", "input_values": {}}
        edges.append({"from_node": "C", "to_node": "D"})
        exits = ["D"]
    flow = Flow(
        user_id=user.id,
        name="child-test-flow",
        graph_config={"nodes": nodes, "edges": edges,
                      "entry_point": "A", "exit_points": exits},
        entry_point="A",
        exit_points=exits,
    )
    session.add(flow)
    session.commit()
    session.refresh(flow)
    return flow, {"a": a, "c": c}


def start_child(session, user, flow, reaper, initial_input=None):
    initial_input = initial_input if initial_input is not None else {"x": "1"}
    execution = create_execution(
        session,
        user_id=user.id,
        flow_id=flow.id,
        execution_type="flow",
        name=flow.name,
        input_data=initial_input,
        status="pending",
        started_at=datetime.now(),
    )
    child = spawn_local_flow_child(flow.id, user.id, execution.id, initial_input, None, {})
    reaper.append(child)
    return child, execution.id


def get_root(execution_id):
    session = DatabaseManager().get_session()
    try:
        return session.get(Execution, execution_id)
    finally:
        session.close()


def wait_for_status(execution_id, statuses, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root = get_root(execution_id)
        if root is not None and root.status in statuses:
            return root
        time.sleep(0.3)
    raise AssertionError(f"Execution {execution_id} never reached {statuses}")


def child_row_count(execution_id):
    session = DatabaseManager().get_session()
    try:
        return (
            session.query(Execution)
            .filter(Execution.parent_id == execution_id)
            .count()
        )
    finally:
        session.close()


def wait_for_test_result(execution_id, request_id, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        root = get_root(execution_id)
        result = (root.execution_metadata or {}).get("last_test_result") if root else None
        if result and result.get("request_id") == request_id:
            return result
        time.sleep(0.3)
    raise AssertionError(f"No test result for request {request_id}")


# ─── E2E tests ───────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_fail_resident_resume_completes(db, reaper):
    session, user = db
    flow, tools = make_chain_flow(session, user)

    child, execution_id = start_child(session, user, flow, reaper)
    wait_for_status(execution_id, {"failed"})
    assert child.is_alive()  # resident, awaiting resume

    edit_tool(session, tools["c"], OK.format(f="c"))
    assert child.signal({"action": "resume"})
    assert child.popen.wait(timeout=60) == 0

    root = wait_for_status(execution_id, {"completed"})
    assert root.execution_metadata["resume_count"] == 1


@pytest.mark.slow
def test_resume_window_expiry_exits_nonzero(db, reaper, monkeypatch):
    session, user = db
    monkeypatch.setenv("FLOW_RUNNER_RESUME_TIMEOUT_SECONDS", "2")
    flow, _ = make_chain_flow(session, user)

    child, execution_id = start_child(session, user, flow, reaper)
    wait_for_status(execution_id, {"failed"})
    assert child.popen.wait(timeout=30) == 1


@pytest.mark.slow
def test_stdin_eof_exits(db, reaper):
    session, user = db
    flow, _ = make_chain_flow(session, user)

    child, execution_id = start_child(session, user, flow, reaper)
    wait_for_status(execution_id, {"failed"})
    assert child.is_alive()

    child.popen.stdin.close()  # backend death from the child's perspective
    assert child.popen.wait(timeout=30) == 1


@pytest.mark.slow
def test_in_run_tool_test_writes_no_rows(db, reaper):
    session, user = db
    flow, tools = make_chain_flow(session, user, with_downstream=True)

    child, execution_id = start_child(session, user, flow, reaper)
    wait_for_status(execution_id, {"failed"})
    rows_before = child_row_count(execution_id)

    # Testable node: A already ran; its (entry) input comes from initial_input.
    assert child.signal({"action": "test", "node_id": "A", "request_id": "req-a"})
    result = wait_for_test_result(execution_id, "req-a")
    assert result["status"] == "success"
    assert result["result"] == "1a"

    # The failing node is testable too (upstream A is in ctx.state) — and an
    # edit made since the failure is what gets tested.
    edit_tool(session, tools["c"], OK.format(f="c"))
    assert child.signal({"action": "test", "node_id": "C", "request_id": "req-c"})
    result = wait_for_test_result(execution_id, "req-c")
    assert result["status"] == "success"
    assert result["result"] == "1ac"

    # Downstream of the failure: inputs unavailable.
    assert child.signal({"action": "test", "node_id": "D", "request_id": "req-d"})
    result = wait_for_test_result(execution_id, "req-d")
    assert result["status"] == "error"
    assert "inputs unavailable" in result["error"]

    # Transient: no execution rows were created, the run is still failed,
    # and the child is still resident.
    assert child_row_count(execution_id) == rows_before
    assert get_root(execution_id).status == "failed"
    assert child.is_alive()
