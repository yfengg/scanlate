"""Detector-layer tests: the CTD subprocess adapter's JSON contract and error
handling, the provisional reading-order heuristic, and auto/ctd/opencv/none
resolution. Real inference is exercised separately (a disposable, gitignored
spike, not part of this suite) -- these tests fake the subprocess boundary.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scanlate.imaging.detection import (CtdComicTextDetector, DetectedRegion, DetectorError,
                                        DetectorUnavailable, NullDetector, OpenCvComicTextDetector,
                                        _simple_japanese_reading_order, build_detector)
from scanlate.imaging.layout import BoundingBox, Polygon


def _configured(tmp_path) -> CtdComicTextDetector:
    python_path = tmp_path / "python.exe"
    script_path = tmp_path / "bridge.py"
    python_path.write_text("")
    script_path.write_text("")
    return CtdComicTextDetector(str(python_path), str(script_path))


def _fake_run_writing(payload: dict):
    def fake_run(cmd, capture_output, text, timeout):
        out_path = Path(cmd[cmd.index("--output") + 1])
        out_path.write_text(json.dumps(payload), encoding="utf8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return fake_run


# --- CtdComicTextDetector: availability and the subprocess boundary -------
def test_ctd_unavailable_without_configured_paths():
    detector = CtdComicTextDetector(None, None)
    assert detector.available() is False
    with pytest.raises(DetectorUnavailable):
        detector.detect(b"not-an-image")


def test_ctd_unavailable_when_configured_paths_dont_exist(tmp_path):
    detector = CtdComicTextDetector(str(tmp_path / "missing-python"), str(tmp_path / "missing-script"))
    assert detector.available() is False


def test_ctd_detect_parses_the_json_contract(tmp_path, monkeypatch):
    detector = _configured(tmp_path)
    payload = {"page_size": [600, 850], "regions": [
        {"box": {"x": 10, "y": 20, "width": 30, "height": 40}, "confidence": None,
         "vertical": True, "mask_polygon": [[10, 20], [40, 20], [40, 60], [10, 60]]},
    ]}
    monkeypatch.setattr(subprocess, "run", _fake_run_writing(payload))

    regions = detector.detect(b"fake-image-bytes")
    assert len(regions) == 1
    region = regions[0]
    assert region.box == BoundingBox(10, 20, 30, 40)
    # v1 never fabricates a confidence the reference wrapper doesn't provide.
    assert region.confidence is None
    assert region.mask == Polygon([(10, 20), (40, 20), (40, 60), (10, 60)])


def test_ctd_detect_handles_a_region_with_no_mask(tmp_path, monkeypatch):
    detector = _configured(tmp_path)
    payload = {"page_size": [100, 100], "regions": [
        {"box": {"x": 0, "y": 0, "width": 10, "height": 10}, "confidence": 0.9,
         "vertical": False, "mask_polygon": None},
    ]}
    monkeypatch.setattr(subprocess, "run", _fake_run_writing(payload))

    regions = detector.detect(b"fake-image-bytes")
    assert regions[0].mask is None
    assert regions[0].confidence == 0.9


def test_ctd_detect_raises_on_nonzero_exit(tmp_path, monkeypatch):
    detector = _configured(tmp_path)

    def fake_run(cmd, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(DetectorError, match="boom"):
        detector.detect(b"fake-image-bytes")


def test_ctd_detect_raises_on_timeout(tmp_path, monkeypatch):
    detector = _configured(tmp_path)

    def fake_run(cmd, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(DetectorError, match="timed out"):
        detector.detect(b"fake-image-bytes")


def test_ctd_detect_raises_on_malformed_json(tmp_path, monkeypatch):
    detector = _configured(tmp_path)

    def fake_run(cmd, capture_output, text, timeout):
        Path(cmd[cmd.index("--output") + 1]).write_text("not json", encoding="utf8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(DetectorError):
        detector.detect(b"fake-image-bytes")


# --- provisional Japanese reading order ------------------------------------
def test_simple_japanese_reading_order_groups_rows_right_to_left():
    top_left = DetectedRegion(BoundingBox(10, 10, 50, 100))
    top_right = DetectedRegion(BoundingBox(200, 15, 50, 100))
    bottom = DetectedRegion(BoundingBox(50, 300, 50, 100))

    ordered = _simple_japanese_reading_order([bottom, top_left, top_right])
    assert ordered == [top_right, top_left, bottom]


def test_simple_japanese_reading_order_handles_empty_and_single():
    assert _simple_japanese_reading_order([]) == []
    one = DetectedRegion(BoundingBox(0, 0, 10, 10))
    assert _simple_japanese_reading_order([one]) == [one]


# --- build_detector: auto prefers ctd, falls back to opencv, then none ----
def test_build_detector_auto_prefers_ctd_when_configured(tmp_path, monkeypatch):
    python_path, script_path = tmp_path / "python.exe", tmp_path / "bridge.py"
    python_path.write_text("")
    script_path.write_text("")
    monkeypatch.setenv("SCANLATE_CTD_PYTHON", str(python_path))
    monkeypatch.setenv("SCANLATE_CTD_SCRIPT", str(script_path))

    assert isinstance(build_detector("auto"), CtdComicTextDetector)
    assert isinstance(build_detector(None), CtdComicTextDetector)  # unset env defaults to auto


def test_build_detector_auto_falls_back_to_opencv_without_ctd_configured(monkeypatch):
    monkeypatch.delenv("SCANLATE_CTD_PYTHON", raising=False)
    monkeypatch.delenv("SCANLATE_CTD_SCRIPT", raising=False)

    detector = build_detector("auto")
    assert not isinstance(detector, CtdComicTextDetector)
    assert isinstance(detector, (OpenCvComicTextDetector, NullDetector))


def test_build_detector_ctd_falls_back_to_null_when_unconfigured(monkeypatch):
    monkeypatch.delenv("SCANLATE_CTD_PYTHON", raising=False)
    monkeypatch.delenv("SCANLATE_CTD_SCRIPT", raising=False)
    assert isinstance(build_detector("ctd"), NullDetector)


def test_build_detector_none_is_unaffected():
    assert isinstance(build_detector("none"), NullDetector)
