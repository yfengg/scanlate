"""Conservative bubble-fill cleanup for the export renderer.

Deliberately minimal: this never infers a bubble's true outline from the
detected region, and it never touches art or borders. It only looks at a
small inset within the stored text-area rectangle and, when those pixels are
obviously a flat, light fill, blanks *that inset* — nothing outside it. If the
sampled area is anything else (art, a gradient, a dark caption box), the
region is left completely untouched and reported so a human can handle it.

No mask is persisted: the decision and the fill colour are cheap to recompute
from the source image every time, so there is nothing to keep in sync.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw

from .layout import BoundingBox


@dataclass
class CleanupResult:
    ok: bool
    reason: str | None = None
    # Only set when ok: the inset rectangle that was judged safe and (once
    # ``BubbleCleaner.apply`` runs) filled. Always inside the original box,
    # never equal to it — the region's own edges are never touched.
    cleaned_box: BoundingBox | None = None
    fill_color: tuple[int, int, int] | None = None


class BubbleCleaner:
    """Flat-fill cleanup for ordinary white/flat-colour bubbles only.

    ``inset_ratio``/``min_inset`` shrink the stored region into the area that
    is actually sampled and cleaned, so a bubble's border ink and any
    neighbouring artwork right at the box edge are never at risk.
    ``uniformity_threshold`` caps how much the sampled pixels may vary
    (per-channel standard deviation) before the background is judged "not
    obviously flat". ``lightness_threshold`` additionally requires that flat
    area to be light — a uniform dark or saturated patch is exactly as likely
    to be art as a caption box, and blanking it would be a guess, not cleanup.
    """

    def __init__(self, inset_ratio: float = 0.14, min_inset: int = 3,
                 uniformity_threshold: float = 18.0, lightness_threshold: float = 175.0):
        self.inset_ratio = inset_ratio
        self.min_inset = min_inset
        self.uniformity_threshold = uniformity_threshold
        self.lightness_threshold = lightness_threshold

    def _inset(self, box: BoundingBox) -> BoundingBox | None:
        mx = max(self.min_inset, round(box.width * self.inset_ratio))
        my = max(self.min_inset, round(box.height * self.inset_ratio))
        width, height = box.width - 2 * mx, box.height - 2 * my
        if width <= 0 or height <= 0:
            return None
        return BoundingBox(box.x + mx, box.y + my, width, height)

    def inspect(self, image: Image.Image, box: BoundingBox) -> CleanupResult:
        """Decide whether ``box``'s inset is safe to blank. Never mutates ``image``."""
        inset = self._inset(box)
        if inset is None:
            return CleanupResult(ok=False, reason="region too small to inset safely")

        patch = image.crop((inset.x, inset.y, inset.x + inset.width,
                            inset.y + inset.height)).convert("RGB")
        pixels = list(patch.getdata())
        if not pixels:
            return CleanupResult(ok=False, reason="region has no sampled pixels")

        n = len(pixels)
        mean = tuple(sum(p[c] for p in pixels) / n for c in range(3))
        stdev = tuple((sum((p[c] - mean[c]) ** 2 for p in pixels) / n) ** 0.5 for c in range(3))
        brightness = sum(mean) / 3

        if max(stdev) > self.uniformity_threshold:
            return CleanupResult(ok=False,
                reason=f"background isn't flat (max channel std-dev {max(stdev):.1f})")
        if brightness < self.lightness_threshold:
            return CleanupResult(ok=False,
                reason=f"background isn't light enough (mean brightness {brightness:.1f})")

        return CleanupResult(ok=True, cleaned_box=inset,
                              fill_color=tuple(round(c) for c in mean))

    def apply(self, image: Image.Image, result: CleanupResult) -> None:
        """Blank exactly ``result.cleaned_box`` with ``result.fill_color``. No-op if not ok."""
        if not result.ok or result.cleaned_box is None or result.fill_color is None:
            return
        box = result.cleaned_box
        ImageDraw.Draw(image).rectangle(
            [box.x, box.y, box.x + box.width - 1, box.y + box.height - 1],
            fill=result.fill_color)
