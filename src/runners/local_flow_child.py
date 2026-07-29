"""
Local flow child process management (LOCAL mode).

Local flows never execute inside the backend API process: each run is spawned
as a child process running `python -m src.tasks.run_flow` — a bare-subprocess
mirror of the hosted flow-runner container — under the flow's own python
environment (Flow.conda_env). The child writes all results to the database;
the backend polls. Its stdin pipe carries the only direct backend→child
messages (resume / test), as newline-delimited JSON.

The environment must contain the runner's own dependencies (see
deploy/flow-runner/requirements.txt, minus psycopg2/redis) — envs are
user-managed and nothing is auto-installed here.
"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_interpreter(conda_env: Optional[str]) -> Tuple[str, Optional[str]]:
    """Map Flow.conda_env to (python executable, bin dir to prepend to PATH).

    A directory is treated as a conda env or venv root: use <dir>/bin/python
    directly (not `conda run`, whose wrapper would swallow terminate()) and
    prepend <dir>/bin so tools shelling out find env binaries. A file is used
    as the interpreter itself. Unset falls back to the backend's interpreter.
    """
    if not conda_env:
        return sys.executable, None
    path = Path(conda_env)
    if path.is_dir():
        python = path / "bin" / "python"
        if not python.exists():
            raise FileNotFoundError(f"flow environment python not found: {python}")
        return str(python), str(path / "bin")
    if path.is_file():
        return str(path), None
    raise FileNotFoundError(f"flow environment python not found: {conda_env}")


class LocalFlowChild:
    """Handle on one spawned flow-run process."""

    def __init__(self, popen: subprocess.Popen, execution_id: int):
        self.popen = popen
        self.execution_id = execution_id

    def is_alive(self) -> bool:
        return self.popen.poll() is None

    def signal(self, msg: Dict[str, Any]) -> bool:
        """Send one JSON message on the child's stdin. False if the pipe is gone."""
        try:
            self.popen.stdin.write(json.dumps(msg) + "\n")
            self.popen.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    def shutdown(self) -> None:
        """Terminate (then kill) and reap the child. Safe on a dead child."""
        if self.popen.poll() is None:
            self.popen.terminate()
            try:
                self.popen.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.popen.kill()
                try:
                    self.popen.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Flow child for execution %s did not die after kill", self.execution_id
                    )


def spawn_local_flow_child(
    flow_id: int,
    user_id: int,
    execution_id: int,
    initial_input: Any,
    conda_env: Optional[str],
    agent_llms: Optional[Dict[str, str]],
) -> LocalFlowChild:
    """Spawn the per-run child. Raises FileNotFoundError/OSError on spawn failure.

    stdout/stderr are inherited so tool prints and runner logs land in the
    backend console. There is no reply pipe — results come back via the DB.
    """
    interpreter, bin_dir = _resolve_interpreter(conda_env)

    env = dict(os.environ)
    env.update({
        "FLOW_RUNNER_FLOW_ID": str(flow_id),
        "FLOW_RUNNER_USER_ID": str(user_id),
        "FLOW_RUNNER_EXECUTION_ID": str(execution_id),
        "FLOW_RUNNER_INITIAL_INPUT": json.dumps(initial_input),
        "FLOW_RUNNER_AGENT_LLMS": json.dumps(agent_llms or {}),
        "FLOW_RUNNER_RESUME_TRANSPORT": "stdin",
        "FLOW_RUNNER_INSTALL_PACKAGES": "false",
    })
    if conda_env:
        env["FLOW_RUNNER_CONDA_ENV"] = conda_env
    else:
        env.pop("FLOW_RUNNER_CONDA_ENV", None)
    if bin_dir:
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")

    popen = subprocess.Popen(
        [interpreter, "-m", "src.tasks.run_flow"],
        cwd=str(REPO_ROOT),
        env=env,
        stdin=subprocess.PIPE,
        text=True,
    )
    logger.info(
        "Spawned flow child pid=%s for execution %s (interpreter=%s)",
        popen.pid, execution_id, interpreter,
    )
    return LocalFlowChild(popen, execution_id)
