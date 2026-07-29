"""
In-process registry of the resident failed flow child awaiting resume (LOCAL mode).

Strict single slot: at most one live checkpoint exists per API process; any
new failure supersedes it (and shuts the superseded child down). Local dev
runs uvicorn with workers=1 (start_backend.py), so one process sees all
requests. HOSTED mode never uses this — each run's checkpoint lives in its
own resident container.

The slot holds a LocalFlowChild (process handle), not a runner: the failed
run's ctx.state lives in the child's memory. Dead children are reaped on
access — no reaper thread.
"""
import threading
from typing import Optional

from src.runners.local_flow_child import LocalFlowChild


class LiveRunStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._child: Optional[LocalFlowChild] = None

    def retain(self, child: LocalFlowChild) -> None:
        """Register a resident failed child, shutting down any superseded one."""
        with self._lock:
            superseded = self._child
            self._child = child
        if superseded is not None and superseded is not child:
            superseded.shutdown()

    def get(self, execution_id: int) -> Optional[LocalFlowChild]:
        """Non-consuming peek at the child for `execution_id` (in-run tests)."""
        with self._lock:
            child = self._reap_locked()
            if child is None or child.execution_id != execution_id:
                return None
            return child

    def pop(self, execution_id: int) -> Optional[LocalFlowChild]:
        """Atomically claim the child for `execution_id` (resume).

        Returns None if the slot is empty, holds a different run, or the child
        has died — a second concurrent resume, a superseded checkpoint, and a
        restarted server all land here.
        """
        with self._lock:
            child = self._reap_locked()
            if child is None or child.execution_id != execution_id:
                return None
            self._child = None
            return child

    def clear(self, execution_id: int) -> None:
        self.pop(execution_id)

    def _reap_locked(self) -> Optional[LocalFlowChild]:
        """Drop (and implicitly reap via poll) a child that has exited."""
        if self._child is not None and not self._child.is_alive():
            self._child = None
        return self._child


live_run_store = LiveRunStore()
