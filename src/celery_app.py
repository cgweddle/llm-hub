"""
Celery application instance.

Shared between the API process (which dispatches tasks via `.delay()`)
and the worker process (which executes them). Kept in its own module to
avoid circular imports from backend.py.
"""
import os
from celery import Celery

broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", broker_url)

celery_app = Celery("llmhub", broker=broker_url, backend=result_backend)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

celery_app.autodiscover_tasks(["src.tasks"])
