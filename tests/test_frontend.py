"""Runs the jsdom frontend suite as part of the normal test run.

Skipped where node or jsdom isn't available, so the Python suite stays
runnable on its own; the assertions live in tests/frontend/test_workbench.mjs.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SUITE = Path(__file__).parent / "frontend" / "test_workbench.mjs"


def _has_jsdom() -> bool:
    if shutil.which("node") is None:
        return False
    probe = subprocess.run(["node", "-e", "require('jsdom')"],
                           cwd=SUITE.parents[2], capture_output=True)
    return probe.returncode == 0


@pytest.mark.skipif(not _has_jsdom(), reason="node with jsdom not available")
def test_workbench_survives_missing_and_failing_data():
    result = subprocess.run(["node", str(SUITE)], cwd=SUITE.parents[2],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "checks passed" in result.stdout
