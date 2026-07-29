"""
Tests for FlowRunner in-memory resume: short-circuit replay, bounded previews,
graph-aware invalidation, row reuse, and the cancelled-sibling sweep.

Uses a real per-test SQLite DB (DATABASE_URL is read at every
DatabaseManager() construction, including the per-node sessions inside
_execute_node_async) and real in-process tool execution — the same code path
production uses.
"""
import time
from datetime import datetime

import pytest

from src.database.database_setup import DatabaseManager, User, Tool, Flow, Execution
from src.runners.flow_runner import FlowRunner
from src.runners.live_run_store import LiveRunStore


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


def edit_tool(session, tool, script):
    """Edit a tool's code the way the API would, with a guaranteed-fresh
    updated_at (SQLite's func.now() is second-granular; same-second edits
    would otherwise be invisible to fingerprinting)."""
    tool.script_code = script
    tool.updated_at = datetime.now()
    session.commit()


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


def counter_script(func, path, body="return x + '{}'".format):
    """Tool script that appends a line to `path` on every invocation."""
    return (
        f"def {func}(x):\n"
        f"    with open({str(path)!r}, 'a') as f:\n"
        f"        f.write('ran\\n')\n"
        f"    return x + '{func}'\n"
    )


def count_runs(path):
    try:
        with open(path) as f:
            return len(f.readlines())
    except FileNotFoundError:
        return 0


BOOM = "def {f}(x):\n    raise ValueError('boom')\n"
OK = "def {f}(x):\n    return x + '{f}'\n"
JOIN = "def {f}(x, y):\n    return x + y\n"


def diamond(session, user, tmp_path, c_script=None):
    """A → (B, C) → D. B counts its invocations in a side-effect file."""
    b_marker = tmp_path / "b_runs.txt"
    a = make_tool(session, user, "A", OK.format(f="a"), "a", ["x"])
    b = make_tool(session, user, "B", counter_script("b", b_marker), "b", ["x"])
    c = make_tool(session, user, "C", c_script or BOOM.format(f="c"), "c", ["x"])
    d = make_tool(session, user, "D", JOIN.format(f="d"), "d", ["x", "y"])
    flow = make_flow(
        session, user,
        nodes={"A": tool_node(a, "A"), "B": tool_node(b, "B"),
               "C": tool_node(c, "C"), "D": tool_node(d, "D")},
        edges=[{"from_node": "A", "to_node": "B"},
               {"from_node": "A", "to_node": "C"},
               {"from_node": "B", "to_node": "D"},
               {"from_node": "C", "to_node": "D"}],
        entry="A", exits=["D"],
    )
    return flow, {"a": a, "b": b, "c": c, "d": d}, b_marker


def children_by_node(session, root_id):
    rows = session.query(Execution).filter(Execution.parent_id == root_id).all()
    result = {}
    for row in rows:
        result.setdefault(row.node_id, []).append(row)
    return result


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_diamond_resume_short_circuits_and_reuses_rows(db, tmp_path):
    session, user = db
    flow, tools, b_marker = diamond(session, user, tmp_path)

    runner = FlowRunner(session, flow.id, user.id)
    result = runner.run({"x": "1"})
    assert result["status"] == "failed"
    assert result["execution_id"] is not None
    assert "execution_trace" not in result
    assert set(runner.ctx.state) == {"A", "B"}
    assert count_runs(b_marker) == 1

    root_before = session.get(Execution, result["execution_id"])
    original_started_at = root_before.started_at

    edit_tool(session, tools["c"], OK.format(f="c"))

    resumed = runner.resume()
    assert resumed["status"] == "completed"
    assert resumed["execution_id"] == result["execution_id"]
    assert resumed["final_output"] == "1ab1ac"
    assert "execution_trace" not in resumed
    assert count_runs(b_marker) == 1  # B short-circuited

    check = DatabaseManager().get_session()
    try:
        by_node = children_by_node(check, result["execution_id"])
        assert set(by_node) == {"A", "B", "C", "D"}
        assert all(len(rows) == 1 for rows in by_node.values())
        c_row = by_node["C"][0]
        assert c_row.status == "completed"
        assert c_row.error_message is None
        root = check.get(Execution, result["execution_id"])
        assert root.status == "completed"
        assert root.started_at == original_started_at
        assert root.execution_metadata["resumed_from_nodes"] == ["C"]
        assert root.execution_metadata["resume_count"] == 1
    finally:
        check.close()


def test_previews_bounded(db, tmp_path):
    session, user = db
    big = make_tool(session, user, "BIG", "def big(x):\n    return 'z' * 5000\n",
                    "big", ["x"])
    flow = make_flow(session, user,
                     nodes={"BIG": tool_node(big, "BIG")},
                     edges=[], entry="BIG", exits=["BIG"])

    runner = FlowRunner(session, flow.id, user.id)
    result = runner.run({"x": "1"})
    assert result["status"] == "completed"
    assert isinstance(result["final_output"], str) and len(result["final_output"]) <= 1000

    check = DatabaseManager().get_session()
    try:
        root = check.get(Execution, result["execution_id"])
        assert root.input_data == {"x": "1"}  # root keeps the full initial input
        for rows in children_by_node(check, root.id).values():
            for row in rows:
                for value in (row.input_data, row.output_data):
                    assert value is None or (isinstance(value, str) and len(value) <= 1000)
    finally:
        check.close()


def test_trigger_entry_records_preview(db):
    session, user = db
    echo = make_tool(session, user, "E", OK.format(f="e"), "e", ["x"])
    flow = make_flow(
        session, user,
        nodes={"T": {"node_type": "trigger", "name": "T", "input_value": "hello"},
               "E": tool_node(echo, "E")},
        edges=[{"from_node": "T", "to_node": "E"}],
        entry="T", exits=["E"],
    )
    runner = FlowRunner(session, flow.id, user.id)
    result = runner.run(None)
    assert result["status"] == "completed"

    check = DatabaseManager().get_session()
    try:
        t_row = children_by_node(check, result["execution_id"])["T"][0]
        assert t_row.input_data == "hello"
        assert t_row.output_data == "hello"
    finally:
        check.close()


def test_double_failure_re_retains(db, tmp_path):
    session, user = db
    flow, tools, b_marker = diamond(session, user, tmp_path)

    runner = FlowRunner(session, flow.id, user.id)
    result = runner.run({"x": "1"})
    assert result["status"] == "failed"

    still_failed = runner.resume()  # C still broken
    assert still_failed["status"] == "failed"
    assert set(runner.ctx.state) == {"A", "B"}

    edit_tool(session, tools["c"], OK.format(f="c"))
    resumed = runner.resume()
    assert resumed["status"] == "completed"
    assert count_runs(b_marker) == 1

    check = DatabaseManager().get_session()
    try:
        root = check.get(Execution, result["execution_id"])
        assert root.execution_metadata["resume_count"] == 2
    finally:
        check.close()


def test_resume_without_run_raises(db):
    session, user = db
    tool = make_tool(session, user, "A", OK.format(f="a"), "a", ["x"])
    flow = make_flow(session, user, nodes={"A": tool_node(tool, "A")},
                     edges=[], entry="A", exits=["A"])
    runner = FlowRunner(session, flow.id, user.id)
    with pytest.raises(ValueError, match="No live run to resume"):
        runner.resume()


def test_cancelled_sibling_sweep(db, tmp_path):
    session, user = db
    slow = (
        "def c(x):\n"
        "    import time\n"
        "    time.sleep(2)\n"
        "    return x + 'c'\n"
    )
    flow, tools, b_marker = diamond(session, user, tmp_path, c_script=slow)
    edit_tool(session, tools["b"], BOOM.format(f="b"))  # B fails fast instead

    runner = FlowRunner(session, flow.id, user.id)
    result = runner.run({"x": "1"})
    assert result["status"] == "failed"

    check = DatabaseManager().get_session()
    try:
        by_node = children_by_node(check, result["execution_id"])
        c_row = by_node["C"][0]
        assert c_row.status == "failed"  # swept, not stuck at 'running'
        assert "sibling" in (c_row.error_message or "")
    finally:
        check.close()

    edit_tool(session, tools["b"], OK.format(f="b"))
    resumed = runner.resume()
    assert resumed["status"] == "completed"
    assert resumed["final_output"] == "1ab1ac"

    check = DatabaseManager().get_session()
    try:
        by_node = children_by_node(check, result["execution_id"])
        assert all(len(rows) == 1 for rows in by_node.values())
        assert all(rows[0].status == "completed" for rows in by_node.values())
    finally:
        check.close()


def test_graph_aware_edited_tool_invalidates(db, tmp_path):
    session, user = db
    flow, tools, b_marker = diamond(session, user, tmp_path)

    runner = FlowRunner(session, flow.id, user.id)
    result = runner.run({"x": "1"})
    assert result["status"] == "failed"
    assert count_runs(b_marker) == 1

    edit_tool(session, tools["c"], OK.format(f="c"))
    edit_tool(session, tools["b"], counter_script("b", tmp_path / "b_runs.txt"))

    resumed = runner.resume()
    assert resumed["status"] == "completed"
    assert count_runs(b_marker) == 2  # edited B re-ran despite prior success

    check = DatabaseManager().get_session()
    try:
        root = check.get(Execution, result["execution_id"])
        assert root.execution_metadata["resumed_from_nodes"] == ["B", "C"]
        by_node = children_by_node(check, root.id)
        assert all(len(rows) == 1 for rows in by_node.values())
    finally:
        check.close()


def test_graph_aware_new_node_and_edge(db, tmp_path):
    session, user = db
    flow, tools, b_marker = diamond(session, user, tmp_path)

    runner = FlowRunner(session, flow.id, user.id)
    result = runner.run({"x": "1"})
    assert result["status"] == "failed"

    edit_tool(session, tools["c"], OK.format(f="c"))
    e = make_tool(session, user, "E", OK.format(f="e"), "e", ["x"])
    config = dict(flow.graph_config)
    config["nodes"] = dict(config["nodes"], E=tool_node(e, "E"))
    config["edges"] = [
        {"from_node": "A", "to_node": "B"},
        {"from_node": "A", "to_node": "C"},
        {"from_node": "B", "to_node": "D"},
        {"from_node": "C", "to_node": "E"},
        {"from_node": "E", "to_node": "D"},
    ]
    flow.graph_config = config
    session.commit()

    resumed = runner.resume()
    assert resumed["status"] == "completed"
    assert resumed["final_output"] == "1ab1ace"
    assert count_runs(b_marker) == 1  # untouched branch stays cached

    check = DatabaseManager().get_session()
    try:
        by_node = children_by_node(check, result["execution_id"])
        assert set(by_node) == {"A", "B", "C", "D", "E"}
        assert all(len(rows) == 1 for rows in by_node.values())
    finally:
        check.close()


def test_transitive_invalidation(db, tmp_path):
    session, user = db
    a_marker = tmp_path / "a_runs.txt"
    b_marker = tmp_path / "b_runs.txt"
    a = make_tool(session, user, "A", counter_script("a", a_marker), "a", ["x"])
    b = make_tool(session, user, "B", counter_script("b", b_marker), "b", ["x"])
    c = make_tool(session, user, "C", BOOM.format(f="c"), "c", ["x"])
    flow = make_flow(
        session, user,
        nodes={"A": tool_node(a, "A"), "B": tool_node(b, "B"), "C": tool_node(c, "C")},
        edges=[{"from_node": "A", "to_node": "B"},
               {"from_node": "B", "to_node": "C"}],
        entry="A", exits=["C"],
    )

    runner = FlowRunner(session, flow.id, user.id)
    result = runner.run({"x": "1"})
    assert result["status"] == "failed"
    assert count_runs(a_marker) == 1 and count_runs(b_marker) == 1

    edit_tool(session, c, OK.format(f="c"))
    edit_tool(session, a, counter_script("a", a_marker))  # upstream edit

    resumed = runner.resume()
    assert resumed["status"] == "completed"
    # A changed → B invalidated transitively despite being unchanged itself.
    assert count_runs(a_marker) == 2
    assert count_runs(b_marker) == 2

    check = DatabaseManager().get_session()
    try:
        by_node = children_by_node(check, result["execution_id"])
        assert all(len(rows) == 1 for rows in by_node.values())
    finally:
        check.close()


def test_live_run_store():
    store = LiveRunStore()

    class StubChild:
        def __init__(self, execution_id, alive=True):
            self.execution_id = execution_id
            self.alive = alive
            self.shutdowns = 0

        def is_alive(self):
            return self.alive

        def shutdown(self):
            self.shutdowns += 1
            self.alive = False

    c1, c2 = StubChild(1), StubChild(2)
    store.retain(c1)
    assert store.pop(2) is None          # wrong id → no claim
    assert store.get(1) is c1            # peek is non-consuming
    assert store.get(1) is c1
    assert store.pop(1) is c1            # atomic claim
    assert store.pop(1) is None          # second claim loses

    store.retain(c1)
    store.retain(c2)                     # single slot: any new failure supersedes
    assert c1.shutdowns == 1             # superseded child is shut down
    assert store.pop(1) is None
    assert store.pop(2) is c2

    dead = StubChild(3, alive=False)
    store.retain(dead)
    assert store.get(3) is None          # dead children are reaped on access
    assert store.pop(3) is None
