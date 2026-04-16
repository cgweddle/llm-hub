"""Detect third-party PyPI package dependencies of a Python script via pigar.

The pigar CLI scans source for imports and maps module names to their PyPI
distribution names (e.g. ``cv2`` → ``opencv-python``). We only keep the names;
versions are discarded because the host environment may not have the packages
installed, so any version pigar emits would be unreliable or missing.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_PIGAR_TIMEOUT_SECONDS = 15
_VERSION_SEPARATOR_RE = re.compile(r"[=<>~!;\s]")


def detect_required_packages(script_code: str) -> list[str]:
    """Return a sorted list of PyPI package names imported by ``script_code``.

    Stdlib modules are excluded by pigar itself. Best-effort: returns an empty
    list (and logs a warning) on any failure, so tool creation never breaks
    when pigar is missing, slow, or unhappy about the script.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="pigar_") as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "tool_script.py").write_text(script_code)
            reqs_path = tmp_path / "requirements.txt"

            result = subprocess.run(
                [
                    "pigar", "generate",
                    "-f", str(reqs_path),
                    "-c", "-",
                    "--auto-select",
                    "--question-answer", "yes",
                    str(tmp_path),
                ],
                capture_output=True,
                text=True,
                timeout=_PIGAR_TIMEOUT_SECONDS,
                stdin=subprocess.DEVNULL,
            )

            if not reqs_path.exists():
                logger.warning(
                    "pigar did not produce requirements.txt (exit=%s, stderr=%s)",
                    result.returncode,
                    (result.stderr or "")[:300],
                )
                return []

            return _parse_requirements(reqs_path.read_text())
    except FileNotFoundError:
        logger.warning("pigar binary not on PATH; skipping import detection")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("pigar timed out after %ss; skipping import detection", _PIGAR_TIMEOUT_SECONDS)
        return []
    except Exception as e:
        logger.warning("pigar detection failed: %s", e)
        return []


def _parse_requirements(content: str) -> list[str]:
    names: set[str] = set()
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = _VERSION_SEPARATOR_RE.split(line, 1)[0].strip()
        if name:
            names.add(name.lower())
    return sorted(names)
