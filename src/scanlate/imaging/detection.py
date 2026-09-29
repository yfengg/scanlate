"""Text-region detection.

``TextRegionDetector`` is the seam: the API and the UI know only this
interface, so swapping in a trained model later touches one module.

The shipped implementation is classical OpenCV — MSER for text-like blobs,
morphological grouping into lines and blocks, then geometric filtering. Chosen
because it runs locally in milliseconds with no model download, no GPU, and no
network, which keeps the workflow usable offline and keeps tests deterministic.
It is a starting point, not a finished detector: accuracy on screentone and
hand-lettered SFX is modest, which is exactly why manual region editing is a
required part of the workflow rather than a fallback.

No general-purpose LLM is involved in detection.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ..core.types import SegmentKind
from .layout import BoundingBox, Polygon


class DetectorError(Exception):
    """Detection failed. The caller keeps the manual workflow available."""


class DetectorUnavailable(DetectorError):
    """No detector is installed or configured.

    Distinct from a detector that ran and found nothing: one means "automatic
    detection isn't available here", the other means "there is no text to find".
    The UI says different things about them; manual regions work in both.
    """


@dataclass
class DetectedRegion:
    box: BoundingBox
    confidence: float | None = 0.5
    suggested_kind: SegmentKind = SegmentKind.DIALOGUE
    # A text-segmentation outline in page coordinates, where the detector can
    # produce one (see CtdComicTextDetector). None for box-only detectors.
    mask: Polygon | None = None
    data: dict = field(default_factory=dict)


class TextRegionDetector(ABC):
    name: str = "detector"

    @abstractmethod
    def detect(self, image_bytes: bytes) -> list[DetectedRegion]:
        ...

    def available(self) -> bool:
        return True


class NullDetector(TextRegionDetector):
    """Used when no detector is installed. Says so rather than reporting zero."""
    name = "none"

    def available(self) -> bool:
        return False

    def detect(self, image_bytes: bytes) -> list[DetectedRegion]:
        raise DetectorUnavailable(
            "Automatic detection isn't available. Draw regions by hand to continue.")


class OpenCvComicTextDetector(TextRegionDetector):
    name = "opencv-mser"

    def __init__(self, min_area: int = 220, max_area_ratio: float = 0.28,
                 line_gap: int = 14, min_letters: int = 2):
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.line_gap = line_gap
        self.min_letters = min_letters

    def available(self) -> bool:
        try:
            import cv2  # noqa: F401
            return True
        except Exception:
            return False

    def detect(self, image_bytes: bytes) -> list[DetectedRegion]:
        try:
            import cv2
            import numpy as np
        except Exception as error:
            raise DetectorUnavailable(f"OpenCV isn't installed: {error}")

        try:
            buffer = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is None:
                raise DetectorError("The image couldn't be decoded for detection.")
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = grey.shape

            # MSER finds stable blobs; comic lettering is high-contrast, so both
            # dark-on-light and light-on-dark are picked up by running it twice.
            letters = self._letter_boxes(cv2, grey) + self._letter_boxes(cv2, 255 - grey)
            if not letters:
                return []

            mask = np.zeros((height, width), dtype=np.uint8)
            for box in letters:
                mask[box.y:box.y + box.height, box.x:box.x + box.width] = 255

            # Grow horizontally then vertically so glyphs merge into lines and
            # lines into a bubble-sized block.
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (self.line_gap, 3)))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, self.line_gap)))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            regions: list[DetectedRegion] = []
            page_area = width * height
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w * h < self.min_area or w * h > page_area * self.max_area_ratio:
                    continue
                inside = sum(1 for b in letters
                             if x <= b.x and b.x + b.width <= x + w
                             and y <= b.y and b.y + b.height <= y + h)
                if inside < self.min_letters:
                    continue
                pad = 3
                box = BoundingBox(max(0, x - pad), max(0, y - pad),
                                  min(width - max(0, x - pad), w + pad * 2),
                                  min(height - max(0, y - pad), h + pad * 2))
                regions.append(DetectedRegion(
                    box=box,
                    confidence=round(min(0.9, 0.4 + inside / 40), 2),
                    suggested_kind=self._suggest_kind(cv2, grey, box),
                    data={"letters": inside}))
        except DetectorError:
            raise
        except Exception as error:
            raise DetectorError(f"Detection failed: {error}")

        # Reading order: right-to-left by default would be a Japanese
        # assumption, so order top-to-bottom and let the user reorder.
        regions.sort(key=lambda r: (r.box.y, r.box.x))
        return self._drop_contained(regions)

    def _letter_boxes(self, cv2, grey) -> list[BoundingBox]:
        mser = cv2.MSER_create()
        mser.setMinArea(18)
        mser.setMaxArea(4000)
        boxes = []
        for points in mser.detectRegions(grey)[0]:
            x, y, w, h = cv2.boundingRect(points.reshape(-1, 1, 2))
            ratio = w / h if h else 0
            if 0.12 < ratio < 6 and 4 < h < 140:
                boxes.append(BoundingBox(int(x), int(y), int(w), int(h)))
        return boxes

    def _suggest_kind(self, cv2, grey, box: BoundingBox) -> SegmentKind:
        """A weak hint the user can override, never a decision.

        Text sitting on a bright, flat background is usually a speech bubble;
        anything else is left as 'other' rather than guessed at.
        """
        patch = grey[box.y:box.y + box.height, box.x:box.x + box.width]
        if patch.size == 0:
            return SegmentKind.OTHER
        bright = float((patch > 200).mean())
        return SegmentKind.DIALOGUE if bright > 0.45 else SegmentKind.OTHER

    def _drop_contained(self, regions: list[DetectedRegion]) -> list[DetectedRegion]:
        kept: list[DetectedRegion] = []
        for region in sorted(regions, key=lambda r: -(r.box.width * r.box.height)):
            if any(_contains(other.box, region.box) for other in kept):
                continue
            kept.append(region)
        return sorted(kept, key=lambda r: (r.box.y, r.box.x))


def _contains(outer: BoundingBox, inner: BoundingBox) -> bool:
    return (outer.x <= inner.x and outer.y <= inner.y
            and outer.x + outer.width >= inner.x + inner.width
            and outer.y + outer.height >= inner.y + inner.height)


class CtdComicTextDetector(TextRegionDetector):
    """Adapter for comic-text-detector (dmMaze/comic-text-detector, GPL-3.0),
    a manga/comic-specific detector trained on real manga/comic data — unlike
    ``OpenCvComicTextDetector``, it also produces a real per-region text
    segmentation mask, not just a box.

    **Out-of-process by design.** comic-text-detector's Python code is never
    imported into, or vendored in, this process or repository. This class
    only knows how to run an external command and parse a small JSON contract
    it writes back — the detector itself, its weights, and its dependencies
    live in a separate environment the operator prepares (see
    ``tools/ctd_bridge.py`` for the bridge script and setup notes). This keeps
    Scanlate's own process and repository free of GPL code; it does not by
    itself resolve every licensing question a real deployment might raise, so
    treat it as a cleaner separation for now, not a legal conclusion.

    Configuration is two environment variables, not a generic pluggable
    external-command system — this adapter is specific to comic-text-detector,
    not a contract other detectors are expected to implement the same way:

    * ``SCANLATE_CTD_PYTHON`` — the interpreter in the separate environment
      that has comic-text-detector's dependencies installed.
    * ``SCANLATE_CTD_SCRIPT`` — the bridge script to run with it.
    """
    name = "ctd"

    def __init__(self, python_path: str | None = None, script_path: str | None = None,
                 timeout: float = 60.0):
        self.python_path = python_path
        self.script_path = script_path
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.python_path and self.script_path
                    and Path(self.python_path).exists() and Path(self.script_path).exists())

    def detect(self, image_bytes: bytes) -> list[DetectedRegion]:
        if not self.available():
            raise DetectorUnavailable(
                "comic-text-detector isn't configured (SCANLATE_CTD_PYTHON/SCANLATE_CTD_SCRIPT).")
        with tempfile.TemporaryDirectory(prefix="scanlate-ctd-") as tmp:
            in_path = Path(tmp) / "page.png"
            out_path = Path(tmp) / "result.json"
            in_path.write_bytes(image_bytes)
            try:
                completed = subprocess.run(
                    [self.python_path, self.script_path,
                     "--input", str(in_path), "--output", str(out_path)],
                    capture_output=True, text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired as error:
                raise DetectorError(f"comic-text-detector timed out: {error}")
            except OSError as error:
                raise DetectorUnavailable(f"couldn't run comic-text-detector: {error}")
            if completed.returncode != 0:
                raise DetectorError(
                    f"comic-text-detector failed (exit {completed.returncode}): "
                    f"{completed.stderr.strip()[-500:]}")
            try:
                payload = json.loads(out_path.read_text(encoding="utf8"))
            except (OSError, json.JSONDecodeError) as error:
                raise DetectorError(f"comic-text-detector produced no readable result: {error}")

        regions = []
        for entry in payload.get("regions", []):
            box = entry["box"]
            mask_points = entry.get("mask_polygon")
            regions.append(DetectedRegion(
                box=BoundingBox(int(box["x"]), int(box["y"]), int(box["width"]), int(box["height"])),
                confidence=(float(entry["confidence"]) if entry.get("confidence") is not None else None),
                suggested_kind=SegmentKind.DIALOGUE,
                mask=Polygon([(int(x), int(y)) for x, y in mask_points]) if mask_points else None,
                data={"vertical": bool(entry.get("vertical", False))}))
        return _simple_japanese_reading_order(regions)


def _simple_japanese_reading_order(regions: list[DetectedRegion]) -> list[DetectedRegion]:
    """Group into rows by vertical proximity, then order each row right to
    left, rows top to bottom. A provisional heuristic, not panel analysis:
    it has no notion of panel boundaries and will misorder regions that
    straddle a row band it didn't expect. Good enough to get most ordinary,
    roughly-aligned dialogue on a page into a sane default order; reordering
    by hand remains available exactly as it is for any other region.
    """
    if not regions:
        return regions
    row_gap = max(1, sorted(r.box.height for r in regions)[len(regions) // 2])
    ordered = sorted(regions, key=lambda r: r.box.y)
    rows: list[list[DetectedRegion]] = []
    for region in ordered:
        if rows and region.box.y - rows[-1][-1].box.y <= row_gap:
            rows[-1].append(region)
        else:
            rows.append([region])
    result = []
    for row in rows:
        result.extend(sorted(row, key=lambda r: -r.box.x))
    return result


def build_detector(name: str | None = None) -> TextRegionDetector:
    import os
    name = (name or "auto").lower()
    if name in {"none", "null"}:
        return NullDetector()
    if name in {"opencv", "opencv-mser", "mser"}:
        detector = OpenCvComicTextDetector()
        return detector if detector.available() else NullDetector()
    if name == "ctd":
        detector = CtdComicTextDetector(os.environ.get("SCANLATE_CTD_PYTHON"),
                                        os.environ.get("SCANLATE_CTD_SCRIPT"))
        return detector if detector.available() else NullDetector()
    if name == "auto":
        ctd = CtdComicTextDetector(os.environ.get("SCANLATE_CTD_PYTHON"),
                                   os.environ.get("SCANLATE_CTD_SCRIPT"))
        if ctd.available():
            return ctd
        opencv = OpenCvComicTextDetector()
        return opencv if opencv.available() else NullDetector()
    raise ValueError(f"Unknown detector {name!r}")
