"""
Celery task for flow execution.

Spawns a Podman container (llmhub-flow-runner) that runs FlowExecutor in
isolation. The worker itself only orchestrates the container — it never
executes user-authored tool scripts in its own process.

Falls back to in-worker execution when FLOW_RUNNER_USE_PODMAN=false, which
is useful for local testing of the Celery layer without Podman.
"""
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

from src.celery_app import celery_app
from src.database.database_setup import DatabaseManager
from src.database.database import update_execution, get_execution_by_id

logger = logging.getLogger(__name__)


FLOW_RUNNER_IMAGE = os.getenv("FLOW_RUNNER_IMAGE", "llmhub-flow-runner")
FLOW_RUNNER_NETWORK = os.getenv("FLOW_RUNNER_NETWORK", "llmhub-net")
FLOW_RUNNER_MEMORY = os.getenv("FLOW_RUNNER_MEMORY", "1g")
FLOW_RUNNER_CPUS = float(os.getenv("FLOW_RUNNER_CPUS", "2"))
FLOW_RUNNER_TIMEOUT = int(os.getenv("FLOW_RUNNER_TIMEOUT_SECONDS", "3600"))
FLOW_RUNNER_RESUME_TIMEOUT = int(os.getenv("FLOW_RUNNER_RESUME_TIMEOUT_SECONDS", "1800"))
PODMAN_SOCKET_URI = os.getenv("CONTAINER_HOST", "unix:///run/podman/podman.sock")

# Env vars forwarded from worker into the flow-runner container.
# LLM credentials are pre-resolved by the worker and passed via
# FLOW_RUNNER_LLM_CONFIG, so neither the encryption key nor direct
# access to llm_provider_configs is needed in the container.
_FORWARDED_ENV_VARS = [
    "SQL_DEBUG", "ENVIRONMENT",
    "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST",
]


def _build_container_env(flow_id, user_id, initial_input, conda_env, execution_id, agent_llms, llm_config):
    env = {
        "FLOW_RUNNER_FLOW_ID": str(flow_id),
        "FLOW_RUNNER_USER_ID": str(user_id),
        "FLOW_RUNNER_EXECUTION_ID": str(execution_id),
        "FLOW_RUNNER_INITIAL_INPUT": json.dumps(initial_input),
        "FLOW_RUNNER_CONDA_ENV": conda_env or "",
        "FLOW_RUNNER_AGENT_LLMS": json.dumps(agent_llms or {}),
        "FLOW_RUNNER_LLM_CONFIG": json.dumps(llm_config),
    }
    for var in _FORWARDED_ENV_VARS:
        if var in os.environ:
            env[var] = os.environ[var]

    # Resume plumbing: the container subscribes to resume:<execution_id> on
    # this redis after a failure and stays resident until the window expires.
    if os.environ.get("CELERY_BROKER_URL"):
        env["FLOW_RUNNER_REDIS_URL"] = os.environ["CELERY_BROKER_URL"]
    env["FLOW_RUNNER_RESUME_TIMEOUT_SECONDS"] = str(FLOW_RUNNER_RESUME_TIMEOUT)

    # Use the restricted flowrunner DB user if configured, otherwise
    # fall back to the main DATABASE_URL (e.g., local dev without Postgres roles).
    flowrunner_db_url = os.environ.get("FLOWRUNNER_DATABASE_URL")
    env["DATABASE_URL"] = flowrunner_db_url or os.environ.get("DATABASE_URL", "")

    return env


def _get_execution_status(execution_id: int) -> Optional[str]:
    session = DatabaseManager().get_session()
    try:
        ex = get_execution_by_id(session, execution_id)
        return ex.status if ex else None
    finally:
        session.close()


def _run_in_podman(flow_id, user_id, initial_input, conda_env, execution_id, agent_llms, llm_config) -> str:
    """
    Spawn a flow-runner container and watch the execution row until it reaches
    a terminal status, then detach and return that status.

    The worker deliberately does NOT wait for container exit: on failure the
    container stays resident awaiting a resume message (run_flow._await_resume)
    and blocking here would pin a worker slot for the whole resume window.
    The container owns its own lifecycle from the first terminal status —
    it self-terminates on resume-window expiry and auto-removes on exit.
    """
    # Import lazily so a worker running with FLOW_RUNNER_USE_PODMAN=false
    # (e.g., local testing) doesn't require the podman python bindings.
    from podman import PodmanClient

    env = _build_container_env(flow_id, user_id, initial_input, conda_env, execution_id, agent_llms, llm_config)

    logger.info(
        "Spawning flow-runner container: image=%s network=%s execution_id=%s",
        FLOW_RUNNER_IMAGE, FLOW_RUNNER_NETWORK, execution_id,
    )

    with PodmanClient(base_url=PODMAN_SOCKET_URI) as client:
        container = client.containers.create(
            image=FLOW_RUNNER_IMAGE,
            command=["/usr/local/bin/flow-runner-entrypoint.sh"],
            environment=env,
            network_mode="bridge",
            mem_limit=FLOW_RUNNER_MEMORY,
            cpu_quota=int(FLOW_RUNNER_CPUS * 100_000),
            cpu_period=100_000,
            remove=True,
        )
        watch_failed = False
        try:
            network = client.networks.get(FLOW_RUNNER_NETWORK)
            network.connect(container)
            container.start()

            def _relay_logs():
                try:
                    for line in container.logs(stream=True, follow=True, stdout=True, stderr=True):
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", errors="replace")
                        logger.info("[flow-runner] %s", line.rstrip())
                except Exception:
                    logger.info("[flow-runner] log stream ended")

            threading.Thread(target=_relay_logs, daemon=True).start()

            deadline = time.monotonic() + FLOW_RUNNER_TIMEOUT
            while time.monotonic() < deadline:
                status = _get_execution_status(execution_id)
                if status in ("completed", "failed"):
                    logger.info(
                        "Execution %s reached status=%s; detaching from container",
                        execution_id, status,
                    )
                    return status
                time.sleep(5)

            # Never reached a terminal status: the container is stuck or died
            # before writing to the DB. Reap it.
            watch_failed = True
            _mark_failed_if_still_running(
                execution_id,
                f"flow-runner did not reach a terminal status within {FLOW_RUNNER_TIMEOUT}s",
            )
            return "failed"
        except Exception:
            watch_failed = True
            raise
        finally:
            # Kill only on the timeout/error path. A clean detach after
            # status='failed' must leave the resident container (and its
            # in-memory checkpoint) alive.
            if watch_failed:
                try:
                    container.reload()
                    if container.status == "running":
                        container.kill()
                except Exception:
                    pass


def _run_inline(flow_id, user_id, initial_input, conda_env, execution_id, agent_llms, llm_config):
    """
    Fallback: run FlowRunner directly in the worker process (Phase 1 behavior).
    Selected via FLOW_RUNNER_USE_PODMAN=false.
    """
    from src.runners.flow_runner import FlowRunner

    session = DatabaseManager().get_session()
    try:
        runner = FlowRunner(session, flow_id, user_id, llm_config=llm_config, agent_llms=agent_llms)
        result = runner.run(initial_input, conda_env, execution_id=execution_id)
        return {"execution_id": execution_id, "status": result.get("status")}
    finally:
        session.close()


def _collect_needed_providers(flow_id: int, agent_llms: Dict[str, str]) -> set:
    """Return the set of llm_provider names this flow can reach at runtime.

    Union of per-node overrides (agent_llms) and the llm_provider baked into
    each agent sub-node's graph_config. Scoping the config this way keeps the
    YAML written into the flow-runner container minimal.
    """
    from src.database.database_setup import Flow, Agent

    providers = set(agent_llms.values()) if agent_llms else set()

    session = DatabaseManager().get_session()
    try:
        flow = session.query(Flow).filter(Flow.id == flow_id).first()
        if not flow or not flow.graph_config:
            return providers

        agent_ids = set()
        for node_info in flow.graph_config.get("nodes", {}).values():
            if node_info.get("node_type") != "agent":
                continue
            agent_id = node_info.get("agent_id") or node_info.get("id")
            if agent_id:
                agent_ids.add(agent_id)

        if not agent_ids:
            return providers

        agents = session.query(Agent).filter(Agent.id.in_(agent_ids)).all()
        for agent in agents:
            if not agent.graph_config:
                continue
            for sub_node in agent.graph_config.get("nodes", {}).values():
                provider = sub_node.get("llm_provider")
                if provider:
                    providers.add(provider)
    finally:
        session.close()

    return providers


def _mark_failed_if_still_running(execution_id: int, reason: str):
    """
    Safety net: if the flow-runner container died without updating the DB
    (e.g., OOM kill, timeout, crash before connecting to Postgres), mark
    the execution as failed so the frontend doesn't see it stuck in 'running'.
    """
    session = DatabaseManager().get_session()
    try:
        ex = get_execution_by_id(session, execution_id)
        if ex and ex.status in ("running", "pending"):
            update_execution(
                session, execution_id,
                status="failed",
                error_message=reason,
                completed_at=datetime.now(),
            )
    except Exception:
        logger.exception("Failed to mark execution %s as failed", execution_id)
    finally:
        session.close()


@celery_app.task(bind=True, name="src.tasks.tasks.execute_flow_task")
def execute_flow_task(
    self,
    flow_id: int,
    user_id: int,
    initial_input: Any,
    conda_env: Optional[str],
    execution_id: int,
    agent_llms: Optional[Dict[str, str]] = None,
):
    """
    Execute a flow, either inside a Podman container (production default) or
    inline in the worker (set FLOW_RUNNER_USE_PODMAN=false to opt out).
    """
    from src.database.database import get_user_llm_provider_configs

    agent_llms = agent_llms or {}
    use_podman = os.getenv("FLOW_RUNNER_USE_PODMAN", "true").lower() == "true"

    # Resolve LLM credentials in the worker so the flow-runner never needs
    # the encryption key or access to llm_provider_configs. The worker only
    # runs in HOSTED deployments, so we read the DB directly.
    session = DatabaseManager().get_session()
    try:
        models = get_user_llm_provider_configs(session, user_id)
    finally:
        session.close()
    needed_providers = _collect_needed_providers(flow_id, agent_llms)
    llm_config = {
        "models": [m for m in models if m.get("name") in needed_providers]
    }

    if not use_podman:
        try:
            return _run_inline(flow_id, user_id, initial_input, conda_env, execution_id, agent_llms, llm_config)
        except Exception as e:
            logger.exception("Inline flow execution failed for execution_id=%s", execution_id)
            _mark_failed_if_still_running(execution_id, f"inline execution error: {e}")
            raise

    try:
        final_status = _run_in_podman(flow_id, user_id, initial_input, conda_env, execution_id, agent_llms, llm_config)
    except Exception as e:
        logger.exception("flow-runner container launch failed for execution_id=%s", execution_id)
        _mark_failed_if_still_running(execution_id, f"podman spawn error: {e}")
        raise

    return {"execution_id": execution_id, "status": final_status}
