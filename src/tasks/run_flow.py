"""
Flow execution entrypoint — runs as the per-run flow child process.

HOSTED: invoked by deploy/flow-runner/entrypoint.sh inside the flow-runner
Podman container; resume/test messages arrive over redis pub/sub.
LOCAL: spawned directly by the backend (src/runners/local_flow_child.py) using
the flow's own python environment; resume/test messages arrive as
newline-delimited JSON on stdin.

Reads task parameters from FLOW_RUNNER_* environment variables, opens its own
DB session, and calls FlowRunner. On failure the process stays resident for
FLOW_RUNNER_RESUME_TIMEOUT_SECONDS holding the runner's ctx.state so the run
can be resumed (or individual tool nodes tested) in place. All results flow
back to the backend through the database — the inbound message channel is the
only direct link.

Exits 0 on eventual success, non-zero on failure/window expiry.
"""
import json
import logging
import os
import sys
from datetime import datetime


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("flow_runner")


def _install_enabled() -> bool:
    """Local children skip package installs (the flow's env is user-managed);
    hosted containers default to installing tools' required_packages."""
    return os.environ.get("FLOW_RUNNER_INSTALL_PACKAGES", "").lower() != "false"


def _attempt_resume(runner, flow_id: int) -> dict:
    """Re-install packages if enabled (a tool edit may add some), then resume."""
    if _install_enabled():
        from src.database.database_setup import DatabaseManager
        from src.tasks.install_required_packages import install_required_packages_for_flow

        session = DatabaseManager().get_session()
        try:
            install_required_packages_for_flow(session, flow_id)
        finally:
            session.close()
    return runner.resume()


def _handle_test_request(runner, node_id, request_id) -> None:
    """Run one tool node against the resident ctx.state and write the outcome
    to the root execution's execution_metadata["last_test_result"].

    Transient by design: never writes execution rows, never touches ctx.state.
    The backend polls the metadata field for a matching request_id. The tool is
    re-fetched so an edit made since the failure is what gets tested.
    """
    import concurrent.futures
    import contextlib
    import io
    import traceback

    from src.database.database_setup import DatabaseManager
    from src.database.database import get_execution_by_id, get_tool_by_id, update_execution
    from src.runners.flow_runner import (
        TOOL_TIMEOUT_SECONDS,
        _build_node_input,
        _incoming_edges,
        _preview,
    )
    from src.runners.tool_runner import ToolCompileError, compile_tool

    outcome = {"request_id": request_id, "node_id": node_id, "status": "error"}
    ctx = getattr(runner, "ctx", None)
    root_id = getattr(runner, "root_execution_id", None)
    if ctx is None or root_id is None:
        logger.warning("Test request ignored: no resident run state")
        return

    try:
        node_config = (ctx.graph_config.get("nodes") or {}).get(node_id)
        if node_config is None:
            outcome["error"] = f"node '{node_id}' not found in flow"
            outcome["error_type"] = "ValueError"
        elif (node_config.get("node_type") or "tool") != "tool":
            outcome["error"] = f"node '{node_id}' is not a tool node"
            outcome["error_type"] = "ValueError"
        else:
            missing = [
                e["from_node"]
                for e in _incoming_edges(ctx.graph_config, node_id)
                if e["from_node"] not in ctx.state
            ]
            if missing:
                outcome["error"] = (
                    "node not testable — inputs unavailable "
                    f"(upstream nodes not run: {', '.join(missing)})"
                )
                outcome["error_type"] = "ValueError"
            else:
                session = DatabaseManager().get_session()
                try:
                    tool_id = node_config.get("id") or node_config.get("tool_id")
                    tool = get_tool_by_id(session, tool_id) if tool_id else None
                    if tool is None:
                        raise ValueError(f"Tool {tool_id} not found for node '{node_id}'")
                    func = compile_tool(tool)
                finally:
                    session.close()

                node_input = _build_node_input(ctx, node_id, runner._initial_input)
                if not isinstance(node_input, dict):
                    raise TypeError(
                        f"Tool node '{node_id}' expected a dict of kwargs, "
                        f"got {type(node_input).__name__}"
                    )

                def _call():
                    out_buf, err_buf = io.StringIO(), io.StringIO()
                    try:
                        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                            value = func(**node_input)
                        return {
                            "status": "success",
                            "result": _preview(value),
                            "stdout": _preview(out_buf.getvalue()),
                            "stderr": _preview(err_buf.getvalue()),
                        }
                    except Exception as e:
                        return {
                            "status": "error",
                            "error": f"{type(e).__name__}: {e}",
                            "error_type": type(e).__name__,
                            "traceback": traceback.format_exc()[-1000:],
                            "stdout": _preview(out_buf.getvalue()),
                            "stderr": _preview(err_buf.getvalue()),
                        }

                pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                try:
                    outcome.update(pool.submit(_call).result(timeout=TOOL_TIMEOUT_SECONDS))
                except concurrent.futures.TimeoutError:
                    outcome["error"] = (
                        f"Tool node '{node_id}' timed out after {TOOL_TIMEOUT_SECONDS}s"
                    )
                    outcome["error_type"] = "TimeoutError"
                finally:
                    pool.shutdown(wait=False)
    except ToolCompileError as e:
        outcome["error"] = str(e)
        outcome["error_type"] = "ToolCompileError"
    except Exception as e:
        import traceback as tb
        outcome["error"] = f"{type(e).__name__}: {e}"
        outcome["error_type"] = type(e).__name__
        outcome["traceback"] = tb.format_exc()[-1000:]

    session = DatabaseManager().get_session()
    try:
        existing = get_execution_by_id(session, root_id)
        meta = dict(existing.execution_metadata or {}) if existing else {}
        meta["last_test_result"] = outcome
        update_execution(session, root_id, execution_metadata=meta)
        logger.info(
            "Test of node '%s' finished: %s (request %s)",
            node_id, outcome.get("status"), request_id,
        )
    finally:
        session.close()


def _await_resume_redis(runner, flow_id: int, execution_id: int) -> int:
    """Hold the failed run resident; act on redis 'resume:'/'test:' messages.

    The runner's ctx.state (raw upstream outputs) lives in this process's
    memory — exiting would make the failure unresumable, so the container
    stays alive for FLOW_RUNNER_RESUME_TIMEOUT_SECONDS after each failure.
    Returns 0 once a resume completes, 1 on window expiry (or when the resume
    plumbing isn't configured). Stale queued messages are drained after every
    resume attempt so duplicate publishes are no-ops. Test messages run a
    single tool node transiently and do NOT consume the resume deadline.
    """
    import time

    redis_url = os.environ.get("FLOW_RUNNER_REDIS_URL")
    if not redis_url:
        return 1
    timeout = int(os.environ.get("FLOW_RUNNER_RESUME_TIMEOUT_SECONDS", "1800"))

    try:
        import redis
        client = redis.Redis.from_url(redis_url)
        pubsub = client.pubsub()
        pubsub.subscribe(f"resume:{execution_id}", f"test:{execution_id}")
    except Exception:
        logger.exception("Cannot subscribe for resume; exiting")
        return 1

    try:
        deadline = time.monotonic() + timeout
        logger.info("Awaiting resume for execution %s (%ss window)", execution_id, timeout)
        while time.monotonic() < deadline:
            msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=5.0)
            if not msg:
                continue

            channel = msg.get("channel")
            channel = channel.decode() if isinstance(channel, bytes) else channel
            if channel == f"test:{execution_id}":
                try:
                    payload = json.loads(msg.get("data") or "{}")
                except (TypeError, ValueError):
                    logger.warning("Ignoring malformed test message")
                    continue
                _handle_test_request(runner, payload.get("node_id"), payload.get("request_id"))
                continue

            result = _attempt_resume(runner, flow_id)
            if result.get("status") == "completed":
                logger.info("Resume completed for execution %s", execution_id)
                return 0

            while pubsub.get_message(ignore_subscribe_messages=True, timeout=0.0):
                pass
            deadline = time.monotonic() + timeout
            logger.info("Resume attempt failed; awaiting another resume")
        logger.info("Resume window expired for execution %s", execution_id)
        return 1
    finally:
        try:
            pubsub.unsubscribe()
            pubsub.close()
            client.close()
        except Exception:
            pass


def _await_resume_stdin(runner, flow_id: int, execution_id: int) -> int:
    """Local-mode counterpart of _await_resume_redis: newline-JSON on stdin.

    Reads the RAW stdin fd via selectors + os.read — a buffered readline would
    block past the deadline and hide EOF. b"" from os.read means the parent
    (backend) died; the orphaned child exits instead of lingering. Messages:
    {"action": "resume"} and {"action": "test", "node_id": ..., "request_id": ...}.
    """
    import selectors
    import time

    timeout = int(os.environ.get("FLOW_RUNNER_RESUME_TIMEOUT_SECONDS", "1800"))
    fd = sys.stdin.fileno()
    sel = selectors.DefaultSelector()
    sel.register(fd, selectors.EVENT_READ)
    buffer = b""

    try:
        deadline = time.monotonic() + timeout
        logger.info("Awaiting resume on stdin for execution %s (%ss window)", execution_id, timeout)
        while time.monotonic() < deadline:
            if not sel.select(timeout=5.0):
                continue
            chunk = os.read(fd, 65536)
            if chunk == b"":
                logger.info("stdin EOF (backend gone); exiting")
                return 1
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except ValueError:
                    logger.warning("Ignoring malformed stdin message")
                    continue

                action = msg.get("action")
                if action == "test":
                    _handle_test_request(runner, msg.get("node_id"), msg.get("request_id"))
                elif action == "resume":
                    result = _attempt_resume(runner, flow_id)
                    if result.get("status") == "completed":
                        logger.info("Resume completed for execution %s", execution_id)
                        return 0
                    deadline = time.monotonic() + timeout
                    logger.info("Resume attempt failed; awaiting another resume")
                else:
                    logger.warning("Ignoring unknown stdin action: %r", action)
        logger.info("Resume window expired for execution %s", execution_id)
        return 1
    finally:
        sel.unregister(fd)
        sel.close()


def _await_resume(runner, flow_id: int, execution_id: int) -> int:
    transport = os.environ.get("FLOW_RUNNER_RESUME_TRANSPORT", "")
    if transport == "stdin":
        return _await_resume_stdin(runner, flow_id, execution_id)
    if os.environ.get("FLOW_RUNNER_REDIS_URL"):
        return _await_resume_redis(runner, flow_id, execution_id)
    return 1


def main() -> int:
    try:
        flow_id = int(os.environ["FLOW_RUNNER_FLOW_ID"])
        user_id = int(os.environ["FLOW_RUNNER_USER_ID"])
        execution_id = int(os.environ["FLOW_RUNNER_EXECUTION_ID"])
        initial_input = json.loads(os.environ["FLOW_RUNNER_INITIAL_INPUT"])
        conda_env = os.environ.get("FLOW_RUNNER_CONDA_ENV") or None
        agent_llms = json.loads(os.environ.get("FLOW_RUNNER_AGENT_LLMS", "{}"))
    except (KeyError, ValueError) as e:
        logger.error("Missing or invalid flow-runner env vars: %s", e)
        return 2

    from src.database.database_setup import DatabaseManager
    from src.database.database import update_execution
    from src.runners.flow_runner import FlowRunner
    from src.tasks.install_required_packages import install_required_packages_for_flow
    from src.utils.llm_config import load_llm_provider_config

    llm_config = load_llm_provider_config()

    session = DatabaseManager().get_session()
    try:
        if _install_enabled():
            install_required_packages_for_flow(session, flow_id)
        runner = FlowRunner(session, flow_id, user_id, llm_config=llm_config, agent_llms=agent_llms)
        result = runner.run(
            initial_input, conda_env, execution_id=execution_id
        )
        status = result.get("status")
        logger.info("Flow %s finished with status=%s", flow_id, status)
        if status == "completed":
            return 0
        return _await_resume(runner, flow_id, execution_id)
    except Exception as e:
        logger.exception("Flow execution crashed")
        update_execution(
            session,
            execution_id,
            status="failed",
            error_message=str(e),
            completed_at=datetime.now(),
        )
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
