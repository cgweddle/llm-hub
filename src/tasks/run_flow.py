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
    except (KeyError, ValueError) as e:
        logger.error("Missing or invalid flow-runner env vars: %s", e)
        return 2

    from src.database.database_setup import DatabaseManager
    from src.database.database import update_execution
    from src.executors.flow_executor import FlowExecutor
    from src.utils import load_llm_provider_config

    session = DatabaseManager().get_session()
    try:
        llm_config = load_llm_provider_config(user_id=user_id)
        executor = FlowExecutor(session, flow_id, user_id, llm_config=llm_config)
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
