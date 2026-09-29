"""Conservative bubble-fill cleanup for the export renderer.

Three evaluation strategies, in this order of preference:

* A real text-segmentation mask (``mask`` given, from a detector like
  comic-text-detector) — ground truth for the foreground/background split,
  not an estimate. This is strictly better than guessing, so it always wins
  when available, regardless of ``apply_inset``.
* ``apply_inset=False`` (no mask, but an explicit production area) — the box
  is expected to contain real glyph ink; requiring the whole rectangle to be
  uniform would reject exactly the case this exists for. Instead: estimate
  the background colour robustly (the per-channel *median*, not the mean, so
  a minority of dark ink pixels cannot drag the estimate toward them), cap
  how much of the box may be classified as foreground/ink, and apply the
  flatness/lightness checks only to the pixels classified as background.
* ``apply_inset=True`` (legacy, no mask, no explicit production area) —
  shrink the stored text box inward and require the *entire* shrunk area to
  be flat and light. Unchanged from before this module gained the other two
  strategies: this path only ever samples a small, already-trimmed sliver
  that was never expected to contain ink, so "everything here must be
  uniform" is the right (and simplest) test for it.

None of the three paths ever infers a bubble's shape from the region box, and
none touches art or borders outside the box it was given. If a box fails its
check it is left completely untouched and reported so a human can handle it.

No fill mask is persisted: all three decisions are cheap to recompute from
the source image (and, for the first, the region's own stored polygon) every
time, so there is nothing to keep in sync.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from PIL import Image, ImageDraw

from .layout import BoundingBox, Polygon


@dataclass
class CleanupResult:
    ok: bool
    reason: str | None = None
    # Only set when ok: the rectangle that was judged safe and (once
    # ``BubbleCleaner.apply`` runs) filled. In the legacy path this is the
    # inset, always inside and smaller than the box given to ``inspect``. In
    # the explicit-production-area path this is exactly the box given —
    # nothing is inferred or shrunk on top of a manual selection.
    cleaned_box: BoundingBox | None = None
    fill_color: tuple[int, int, int] | None = None


class BubbleCleaner:
    """Flat-fill cleanup for ordinary white/flat-colour bubbles.

    ``inset_ratio``/``min_inset`` shrink the stored region into the area
    sampled and cleaned in the legacy (no explicit production area) path, so
    a bubble's border ink and any neighbouring artwork right at the box edge
    are never at risk there.

    ``uniformity_threshold`` caps how much sampled pixels may vary
    (per-channel standard deviation) before an area is judged "not obviously
    flat" — applied to every pixel in the legacy path, and only to the
    background-classified pixels in the explicit-area path.
    ``lightness_threshold`` additionally requires that flat area to be
    light — a uniform dark or saturated patch is exactly as likely to be art
    as a caption box, and blanking it would be a guess, not cleanup.

    ``outlier_distance``/``max_foreground_ratio`` govern the explicit-area
    (estimate-based) path only: a pixel further than ``outlier_distance``
    (per channel, worst channel) from the robust background estimate is
    classified foreground (ink); if more than ``max_foreground_ratio`` of the
    box is foreground, the area is rejected as not obviously flat enough to
    trust. The median used for the background estimate has a 50% breakdown
    point, so keeping this ratio comfortably under 0.5 is what keeps the
    estimate meaningful.

    ``mask_max_foreground_ratio`` is the equivalent cap for the real-mask
    path, kept separate and higher: a detector's mask makes the ratio a
    measured fact rather than a statistical guess, so it doesn't need the
    same safety margin.
    """

    def __init__(self, inset_ratio: float = 0.14, min_inset: int = 3,
                 uniformity_threshold: float = 18.0, lightness_threshold: float = 175.0,
                 outlier_distance: float = 50.0, max_foreground_ratio: float = 0.35,
                 mask_max_foreground_ratio: float = 0.5):
        self.inset_ratio = inset_ratio
        self.min_inset = min_inset
        self.uniformity_threshold = uniformity_threshold
        self.lightness_threshold = lightness_threshold
        self.outlier_distance = outlier_distance
        self.max_foreground_ratio = max_foreground_ratio
        # A real mask makes the ratio a measured fact, not a statistical
        # guess, so it doesn't need the same safety margin the estimate-based
        # path does — real vertical Japanese dialogue routinely packs more
        # densely into a bubble than 35% and is still perfectly renderable.
        self.mask_max_foreground_ratio = mask_max_foreground_ratio

    def _inset(self, box: BoundingBox) -> BoundingBox | None:
        mx = max(self.min_inset, round(box.width * self.inset_ratio))
        my = max(self.min_inset, round(box.height * self.inset_ratio))
        width, height = box.width - 2 * mx, box.height - 2 * my
        if width <= 0 or height <= 0:
            return None
        return BoundingBox(box.x + mx, box.y + my, width, height)

    def inspect(self, image: Image.Image, box: BoundingBox, apply_inset: bool = True,
               mask: Polygon | None = None) -> CleanupResult:
        """Decide whether ``box`` (or its inset) is safe to blank. Never mutates ``image``.

        A real ``mask`` takes precedence over ``apply_inset`` — ground truth
        beats either estimate.
        """
        if mask is not None:
            return self._inspect_with_mask(image, box, mask)
        if apply_inset:
            return self._inspect_uniform(image, box)
        return self._inspect_with_foreground(image, box)

    # --- real segmentation mask: ground truth, not an estimate ---------
    def _inspect_with_mask(self, image: Image.Image, box: BoundingBox, mask: Polygon) -> CleanupResult:
        pixels = _sample(image, box)
        if not pixels:
            return CleanupResult(ok=False, reason="region has no sampled pixels")

        membership = rasterize_polygon(box, mask)
        background = [p for p, is_fg in zip(pixels, membership) if not is_fg]
        foreground_ratio = 1 - (len(background) / len(pixels))
        if foreground_ratio > self.mask_max_foreground_ratio:
            return CleanupResult(ok=False,
                reason=f"too much non-background coverage ({foreground_ratio:.0%} of the area)")
        if not background:
            return CleanupResult(ok=False, reason="no background pixels found")

        mean = _mean(background)
        stdev = _stdev(background, mean)
        brightness = sum(mean) / 3

        if max(stdev) > self.uniformity_threshold:
            return CleanupResult(ok=False,
                reason=f"background isn't flat (max channel std-dev {max(stdev):.1f})")
        if brightness < self.lightness_threshold:
            return CleanupResult(ok=False,
                reason=f"background isn't light enough (mean brightness {brightness:.1f})")

        return CleanupResult(ok=True, cleaned_box=box,
                              fill_color=tuple(round(c) for c in mean))

    # --- legacy: the whole sampled area must be uniform ---------------
    def _inspect_uniform(self, image: Image.Image, box: BoundingBox) -> CleanupResult:
        inset = self._inset(box)
        if inset is None:
            return CleanupResult(ok=False, reason="region too small to inset safely")

        pixels = _sample(image, inset)
        if not pixels:
            return CleanupResult(ok=False, reason="region has no sampled pixels")

        mean = _mean(pixels)
        stdev = _stdev(pixels, mean)
        brightness = sum(mean) / 3

        if max(stdev) > self.uniformity_threshold:
            return CleanupResult(ok=False,
                reason=f"background isn't flat (max channel std-dev {max(stdev):.1f})")
        if brightness < self.lightness_threshold:
            return CleanupResult(ok=False,
                reason=f"background isn't light enough (mean brightness {brightness:.1f})")

        return CleanupResult(ok=True, cleaned_box=inset,
                              fill_color=tuple(round(c) for c in mean))

    # --- explicit production area: tolerate a bounded foreground ------
    def _inspect_with_foreground(self, image: Image.Image, box: BoundingBox) -> CleanupResult:
        pixels = _sample(image, box)
        if not pixels:
            return CleanupResult(ok=False, reason="region has no sampled pixels")

        # The median is unmoved by a minority of dark ink pixels (its
        # breakdown point is 50%), unlike the mean the legacy path uses —
        # that robustness is exactly what lets this path tolerate real
        # glyphs without the estimate being dragged toward them.
        bg_estimate = tuple(median(p[c] for p in pixels) for c in range(3))

        background = [p for p in pixels
                      if max(abs(p[c] - bg_estimate[c]) for c in range(3)) <= self.outlier_distance]
        foreground_ratio = 1 - (len(background) / len(pixels))
        if foreground_ratio > self.max_foreground_ratio:
            return CleanupResult(ok=False,
                reason=f"too much non-background coverage ({foreground_ratio:.0%} of the area)")
        if not background:
            return CleanupResult(ok=False, reason="no background pixels found")

        mean = _mean(background)
        stdev = _stdev(background, mean)
        brightness = sum(mean) / 3

        if max(stdev) > self.uniformity_threshold:
            return CleanupResult(ok=False,
                reason=f"background isn't flat (max channel std-dev {max(stdev):.1f})")
        if brightness < self.lightness_threshold:
            return CleanupResult(ok=False,
                reason=f"background isn't light enough (mean brightness {brightness:.1f})")

        return CleanupResult(ok=True, cleaned_box=box,
                              fill_color=tuple(round(c) for c in mean))

    def apply(self, image: Image.Image, result: CleanupResult) -> None:
        """Blank exactly ``result.cleaned_box`` with ``result.fill_color``. No-op if not ok.

        Structurally clipped: fills a crop of exactly ``cleaned_box`` and
        pastes it back, so this can never touch a pixel outside that box
        regardless of any coordinate arithmetic elsewhere.
        """
        if not result.ok or result.cleaned_box is None or result.fill_color is None:
            return
        box = result.cleaned_box
        patch = image.crop((box.x, box.y, box.x + box.width, box.y + box.height))
        ImageDraw.Draw(patch).rectangle([0, 0, box.width - 1, box.height - 1], fill=result.fill_color)
        image.paste(patch, (box.x, box.y))


def _sample(image: Image.Image, box: BoundingBox) -> list[tuple[int, int, int]]:
    return list(image.crop((box.x, box.y, box.x + box.width, box.y + box.height))
                .convert("RGB").getdata())


def rasterize_polygon(box: BoundingBox, mask: Polygon) -> list[bool]:
    """Membership test for ``box``'s pixels, in the same row-major order
    ``_sample``/``_region_pixels`` return them, against ``mask`` (page
    coordinates). Shared with ``production_area.py``, which needs the same
    box-relative foreground/background split when a real mask is available."""
    local = [(x - box.x, y - box.y) for x, y in mask.points]
    canvas = Image.new("L", (max(1, box.width), max(1, box.height)), 0)
    if len(local) >= 3:
        ImageDraw.Draw(canvas).polygon(local, fill=255)
    return [value > 127 for value in canvas.getdata()]


def _mean(pixels: list[tuple[int, int, int]]) -> tuple[float, float, float]:
    n = len(pixels)
    return tuple(sum(p[c] for p in pixels) / n for c in range(3))


def _stdev(pixels: list[tuple[int, int, int]], mean: tuple[float, float, float]) -> tuple[float, float, float]:
    n = len(pixels)
    return tuple((sum((p[c] - mean[c]) ** 2 for p in pixels) / n) ** 0.5 for c in range(3))
