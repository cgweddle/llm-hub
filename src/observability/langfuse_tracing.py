"""
LangFuse Tracing

Single home for the shared LangFuse client. Imported by every process that
talks to LangFuse: the backend's trace/score proxy endpoints, the evaluation
executor's score posting, and the agent executor's per-node trace capture.

This module only initializes the client (which registers LangFuse as the
OpenTelemetry exporter). It does NOT call pydantic_ai's Agent.instrument_all()
— that globally enables span emission for every agent run and belongs in the
modules that actually run agents (agent_executor, evaluation_executor), so
processes that merely read trace data never instrument.

Degrades gracefully: LANGFUSE_AVAILABLE is False when langfuse isn't
installed or the client fails to initialize (missing LANGFUSE_* env vars).
"""

import logging

logger = logging.getLogger(__name__)

# Load .env before initializing LangFuse (needs LANGFUSE_* env vars)
from dotenv import load_dotenv
load_dotenv()

try:
    from langfuse import get_client as _get_langfuse_client, observe as langfuse_observe
    langfuse_client = _get_langfuse_client()
    LANGFUSE_AVAILABLE = True
    logger.info("LangFuse client initialized")
except Exception:
    LANGFUSE_AVAILABLE = False
    langfuse_client = None
    langfuse_observe = None
    logger.info("LangFuse not available — tracing disabled")
