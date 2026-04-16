"""Install the Python packages a flow's tools need into the flow-runner container.

Runs inside the flow-runner container at startup, before FlowExecutor is
constructed. Uses `uv pip install --user` so the non-root `llmhub` user
can write into `~/.local/...` without touching system site-packages.

Best-effort: any failure is logged as a warning and the flow runs anyway.
A genuinely missing package will still surface as ImportError during tool
execution, which is a clearer signal than a pip abort for a transient
network hiccup.
"""
from __future__ import annotations

import importlib
import logging
import subprocess
from typing import Any

from src.database.database import get_flow_by_id

logger = logging.getLogger(__name__)

_UV_TIMEOUT_SECONDS = 300


def install_required_packages_for_flow(session: Any, flow_id: int) -> None:
    flow = get_flow_by_id(session, flow_id)
    if flow is None:
        logger.warning("install_required_packages: flow %s not found", flow_id)
        return

    packages: set[str] = set()
    for tool in flow.tools:
        if tool.required_packages:
            packages.update(tool.required_packages)
    for agent in flow.agents:
        for tool in agent.tools:
            if tool.required_packages:
                packages.update(tool.required_packages)

    if not packages:
        logger.info("install_required_packages: flow %s has no extra packages required", flow_id)
        return

    sorted_packages = sorted(packages)
    logger.info("install_required_packages: installing %s", sorted_packages)

    try:
        result = subprocess.run(
            ["uv", "pip", "install", "--user", "--no-cache", *sorted_packages],
            capture_output=True,
            text=True,
            timeout=_UV_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        logger.warning("install_required_packages: `uv` binary not on PATH; skipping install")
        return
    except subprocess.TimeoutExpired:
        logger.warning(
            "install_required_packages: uv timed out after %ss; continuing without packages",
            _UV_TIMEOUT_SECONDS,
        )
        return
    except Exception:
        logger.exception("install_required_packages: unexpected error invoking uv")
        return

    if result.returncode != 0:
        logger.warning(
            "install_required_packages: uv pip install failed (code=%s) stderr=%s",
            result.returncode,
            (result.stderr or "")[:2000],
        )
        return

    importlib.invalidate_caches()
    logger.info("install_required_packages: installed %d package(s)", len(sorted_packages))
