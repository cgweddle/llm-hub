"""Tests for install_required_packages_for_flow.

We don't spin up a real DB. Instead, `get_flow_by_id` is monkey-patched to
return a hand-built object with `.tools` and `.agents` attributes — which
matches the SQLAlchemy relationship surface the helper actually uses.
`subprocess.run` is mocked so tests don't require `uv` to be installed.
"""

import os
import subprocess
import sys
import types
from unittest.mock import patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.tasks import install_required_packages as module
from src.tasks.install_required_packages import install_required_packages_for_flow


def _tool(required_packages=None):
    t = types.SimpleNamespace()
    t.required_packages = required_packages
    return t


def _agent(tools):
    a = types.SimpleNamespace()
    a.tools = tools
    return a


def _flow(tools=(), agents=()):
    f = types.SimpleNamespace()
    f.tools = list(tools)
    f.agents = list(agents)
    return f


def _ok_result():
    r = types.SimpleNamespace()
    r.returncode = 0
    r.stdout = ""
    r.stderr = ""
    return r


def _fail_result(stderr="boom"):
    r = types.SimpleNamespace()
    r.returncode = 1
    r.stdout = ""
    r.stderr = stderr
    return r


class TestCollection:
    def test_no_tools_skips_install(self):
        with patch.object(module, "get_flow_by_id", return_value=_flow()), \
             patch("subprocess.run") as run:
            install_required_packages_for_flow(session=None, flow_id=1)
        run.assert_not_called()

    def test_tool_packages_are_collected_sorted_and_deduped(self):
        flow = _flow(tools=[
            _tool(required_packages=["requests", "rich"]),
            _tool(required_packages=["requests", "httpx"]),
        ])
        with patch.object(module, "get_flow_by_id", return_value=flow), \
             patch("subprocess.run", return_value=_ok_result()) as run:
            install_required_packages_for_flow(session=None, flow_id=1)

        run.assert_called_once()
        cmd = run.call_args.args[0]
        # Prefix is fixed; the packages follow in sorted order
        assert cmd[:5] == ["uv", "pip", "install", "--user", "--no-cache"]
        assert cmd[5:] == ["httpx", "requests", "rich"]

    def test_agent_tool_packages_are_included(self):
        flow = _flow(
            tools=[_tool(required_packages=["numpy"])],
            agents=[_agent(tools=[_tool(required_packages=["httpx"])])],
        )
        with patch.object(module, "get_flow_by_id", return_value=flow), \
             patch("subprocess.run", return_value=_ok_result()) as run:
            install_required_packages_for_flow(session=None, flow_id=1)

        cmd = run.call_args.args[0]
        assert cmd[5:] == ["httpx", "numpy"]

    def test_null_required_packages_are_ignored(self):
        flow = _flow(tools=[
            _tool(required_packages=None),
            _tool(required_packages=[]),
            _tool(required_packages=["pandas"]),
        ])
        with patch.object(module, "get_flow_by_id", return_value=flow), \
             patch("subprocess.run", return_value=_ok_result()) as run:
            install_required_packages_for_flow(session=None, flow_id=1)

        cmd = run.call_args.args[0]
        assert cmd[5:] == ["pandas"]

    def test_missing_flow_is_a_noop(self):
        with patch.object(module, "get_flow_by_id", return_value=None), \
             patch("subprocess.run") as run:
            install_required_packages_for_flow(session=None, flow_id=999)
        run.assert_not_called()


class TestFailureModes:
    def test_nonzero_exit_logs_and_returns(self):
        flow = _flow(tools=[_tool(required_packages=["pandas"])])
        with patch.object(module, "get_flow_by_id", return_value=flow), \
             patch("subprocess.run", return_value=_fail_result("some uv error")):
            # Must not raise
            install_required_packages_for_flow(session=None, flow_id=1)

    def test_uv_binary_missing_does_not_raise(self):
        flow = _flow(tools=[_tool(required_packages=["pandas"])])
        with patch.object(module, "get_flow_by_id", return_value=flow), \
             patch("subprocess.run", side_effect=FileNotFoundError("uv")):
            install_required_packages_for_flow(session=None, flow_id=1)

    def test_timeout_does_not_raise(self):
        flow = _flow(tools=[_tool(required_packages=["pandas"])])

        def _timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=300)

        with patch.object(module, "get_flow_by_id", return_value=flow), \
             patch("subprocess.run", side_effect=_timeout):
            install_required_packages_for_flow(session=None, flow_id=1)

    def test_unexpected_exception_does_not_raise(self):
        flow = _flow(tools=[_tool(required_packages=["pandas"])])
        with patch.object(module, "get_flow_by_id", return_value=flow), \
             patch("subprocess.run", side_effect=RuntimeError("kaboom")):
            install_required_packages_for_flow(session=None, flow_id=1)


class TestCacheInvalidation:
    def test_importlib_caches_invalidated_on_success(self):
        flow = _flow(tools=[_tool(required_packages=["pandas"])])
        with patch.object(module, "get_flow_by_id", return_value=flow), \
             patch("subprocess.run", return_value=_ok_result()), \
             patch("importlib.invalidate_caches") as inv:
            install_required_packages_for_flow(session=None, flow_id=1)
        inv.assert_called_once()

    def test_importlib_caches_not_invalidated_on_failure(self):
        flow = _flow(tools=[_tool(required_packages=["pandas"])])
        with patch.object(module, "get_flow_by_id", return_value=flow), \
             patch("subprocess.run", return_value=_fail_result()), \
             patch("importlib.invalidate_caches") as inv:
            install_required_packages_for_flow(session=None, flow_id=1)
        inv.assert_not_called()
