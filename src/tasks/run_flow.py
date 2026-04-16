"""
Flow execution entrypoint that runs inside the flow-runner Podman container.

Invoked by: `podman run llmhub-flow-runner python -m src.tasks.run_flow`

Reads task parameters from environment variables (set by the Celery worker via
`podman run -e`), opens its own DB session, and calls FlowExecutor.

Exits 0 on successful flow completion, non-zero on failure.
"""
import json
import logging
import os
import sys
from datetime import datetime


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("flow_runner")


def main() -> int:
    try:
        flow_id = int(os.environ["FLOW_RUNNER_FLOW_ID"])
        user_id = int(os.environ["FLOW_RUNNER_USER_ID"])
        execution_id = int(os.environ["FLOW_RUNNER_EXECUTION_ID"])
        initial_input = json.loads(os.environ["FLOW_RUNNER_INITIAL_INPUT"])
        conda_env = os.environ.get("FLOW_RUNNER_CONDA_ENV") or None
        agent_llms = json.loads(os.environ.get("FLOW_RUNNER_AGENT_LLMS", "{}"))
        llm_config = json.loads(os.environ.get("FLOW_RUNNER_LLM_CONFIG", '{"models": []}'))
    except (KeyError, ValueError) as e:
        logger.error("Missing or invalid flow-runner env vars: %s", e)
        return 2

    from src.database.database_setup import DatabaseManager
    from src.database.database import update_execution
    from src.executors.flow_executor import FlowExecutor
    from src.tasks.install_required_packages import install_required_packages_for_flow

    session = DatabaseManager().get_session()
    try:
        install_required_packages_for_flow(session, flow_id)
        executor = FlowExecutor(session, flow_id, user_id, llm_config=llm_config, agent_llms=agent_llms)
        result = executor.execute_flow(
            initial_input, conda_env, execution_id=execution_id
        )
        status = result.get("status")
        logger.info("Flow %s finished with status=%s", flow_id, status)
        return 0 if status == "completed" else 1
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
