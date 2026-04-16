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

    # Use the restricted flowrunner DB user if configured, otherwise
    # fall back to the main DATABASE_URL (e.g., local dev without Postgres roles).
    flowrunner_db_url = os.environ.get("FLOWRUNNER_DATABASE_URL")
    env["DATABASE_URL"] = flowrunner_db_url or os.environ.get("DATABASE_URL", "")

    return env


def _run_in_podman(flow_id, user_id, initial_input, conda_env, execution_id, agent_llms, llm_config) -> int:
    """
    Spawn a flow-runner Podman container, stream logs, and return its exit code.
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
            command=["python", "-m", "src.tasks.run_flow"],
            environment=env,
            mem_limit=FLOW_RUNNER_MEMORY,
            cpu_quota=int(FLOW_RUNNER_CPUS * 100_000),
            cpu_period=100_000,
            remove=True,
        )
        try:
            network = client.networks.get(FLOW_RUNNER_NETWORK)
            network.connect(container)
            container.start()
            # Stream logs in the worker's log stream for observability
            for line in container.logs(stream=True, follow=True, stdout=True, stderr=True):
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                logger.info("[flow-runner] %s", line.rstrip())

            result = container.wait(condition="exited", timeout=FLOW_RUNNER_TIMEOUT)
            # podman-py returns an int or dict depending on version
            if isinstance(result, dict):
                return int(result.get("StatusCode", 1))
            return int(result)
        finally:
            # Container was created with remove=True, but in case of early failure:
            try:
                container.reload()
                if container.status == "running":
                    container.kill()
            except Exception:
                pass


def _run_inline(flow_id, user_id, initial_input, conda_env, execution_id, agent_llms, llm_config):
    """
    Fallback: run FlowExecutor directly in the worker process (Phase 1 behavior).
    Selected via FLOW_RUNNER_USE_PODMAN=false.
    """
    from src.executors.flow_executor import FlowExecutor

    session = DatabaseManager().get_session()
    try:
        executor = FlowExecutor(session, flow_id, user_id, llm_config=llm_config, agent_llms=agent_llms)
        result = executor.execute_flow(initial_input, conda_env, execution_id=execution_id)
        return {"execution_id": execution_id, "status": result.get("status")}
    finally:
        session.close()


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
    from src.utils import load_llm_provider_config

    agent_llms = agent_llms or {}
    use_podman = os.getenv("FLOW_RUNNER_USE_PODMAN", "true").lower() == "true"

    # Resolve LLM credentials in the worker so the flow-runner never needs
    # the encryption key or access to llm_provider_configs.
    llm_config = load_llm_provider_config(user_id=user_id)
    needed_providers = set(agent_llms.values()) if agent_llms else set()
    if needed_providers:
        llm_config["models"] = [
            m for m in llm_config.get("models", [])
            if m.get("name") in needed_providers
        ]
    else:
        llm_config = {"models": []}

    if not use_podman:
        try:
            return _run_inline(flow_id, user_id, initial_input, conda_env, execution_id, agent_llms, llm_config)
        except Exception as e:
            logger.exception("Inline flow execution failed for execution_id=%s", execution_id)
            _mark_failed_if_still_running(execution_id, f"inline execution error: {e}")
            raise

    try:
        returncode = _run_in_podman(flow_id, user_id, initial_input, conda_env, execution_id, agent_llms, llm_config)
    except Exception as e:
        logger.exception("flow-runner container launch failed for execution_id=%s", execution_id)
        _mark_failed_if_still_running(execution_id, f"podman spawn error: {e}")
        raise

    if returncode != 0:
        _mark_failed_if_still_running(
            execution_id,
            f"flow-runner container exited with code {returncode}",
        )

    return {"execution_id": execution_id, "returncode": returncode}
