"""Automatic inference of a page-production area, independent of ``BubbleCleaner``.

``BubbleCleaner`` decides whether a *given* box is safe to blank. This module
answers a different question: starting from a detected/OCR text box that is
almost always too tight to typeset a translation into, how much of the
surrounding bubble can be confidently claimed for cleanup and layout without
ever touching art, borders, or a neighbouring panel?

The approach is deliberately conservative and grows outward in small,
independently-tested steps rather than inferring a bubble's shape in one
guess: a detected text box's own light/flat pixels establish what "this
bubble's background" looks like, and each new border band is required to
still look like that background before it's accepted. Growth on one side
never depends on whether another side is still growing — an ordinary bubble
where the text sits close to one edge still gets full credit on the other
three, which is why sides are evaluated from a single shared snapshot each
round and applied together, and no side is required to grow at all.

This never inspects the *interior* of the text box the way ``BubbleCleaner``
does — the box is assumed to already be tight on the text. When a real
segmentation mask is available it is used only to get a more accurate
"what is background, ignoring the glyphs already in this box" sample; it does
not change the outward-growth logic itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from PIL import Image

from .cleanup import rasterize_polygon
from .layout import BoundingBox, Polygon

BACKGROUND_PERCENTILE = 0.55       # brightest fraction of text_box used as a background guess, if no mask
FLATNESS_THRESHOLD = 18.0          # max per-channel std-dev to call an area flat (matches BubbleCleaner)
LIGHTNESS_THRESHOLD = 175.0        # min mean brightness to call an area light (matches BubbleCleaner)
BACKGROUND_DISTANCE_THRESHOLD = 24.0  # max per-channel distance from the background estimate
OUTLIER_DISTANCE = 50.0            # per-channel distance beyond which a pixel counts as non-background
MAX_BAND_OUTLIER_RATIO = 0.10      # a growth band must be almost pure background, not just mostly
STEP_RATIO = 0.08                  # one round's growth, as a fraction of the text box's own dimension
MIN_STEP = 2
MAX_GROWTH_RATIO = 2.0             # a side can grow at most this many multiples of the box's own dimension


@dataclass
class ProductionAreaResult:
    ok: bool
    box: BoundingBox | None = None
    reason: str | None = None


def infer_production_area(image: Image.Image, text_box: BoundingBox,
                          page_width: int | None = None, page_height: int | None = None,
                          mask: Polygon | None = None) -> ProductionAreaResult:
    """Grow ``text_box`` outward into a production area, or report why it
    couldn't. Never mutates ``image``."""
    background = _estimate_background(image, text_box, mask)
    if background is None:
        return ProductionAreaResult(False, reason="no confident background could be established "
                                                   "from the text box")

    step_x, step_y = _step(text_box.width), _step(text_box.height)
    cap_x, cap_y = MAX_GROWTH_RATIO * text_box.width, MAX_GROWTH_RATIO * text_box.height

    current = text_box
    growing = {"top": True, "bottom": True, "left": True, "right": True}
    grown = {"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0}

    while any(growing.values()):
        snapshot = current
        accepted: dict[str, float] = {}
        for side in [s for s, active in growing.items() if active]:
            step = step_y if side in ("top", "bottom") else step_x
            cap = cap_y if side in ("top", "bottom") else cap_x
            room = _room(snapshot, side, page_width, page_height)
            actual_step = min(step, cap - grown[side], room)
            if actual_step < 1:
                growing[side] = False
                continue
            actual_step = int(actual_step)
            band = _band_rect(snapshot, side, actual_step)
            if not _region_is_safe(image, band, background):
                growing[side] = False
                continue
            accepted[side] = actual_step
        if not accepted:
            break
        current = _apply_growth(current, accepted)
        for side, step in accepted.items():
            grown[side] += step

    current, grown = _revert_bad_corners(image, text_box, current, grown, background)

    if current == text_box:
        return ProductionAreaResult(False, reason="no usable margin was found around the text box")

    margin_reason = _check_whole_margin(image, text_box, current, background)
    if margin_reason is not None:
        return ProductionAreaResult(False, reason=margin_reason)

    return ProductionAreaResult(True, box=current)


def _step(dimension: int) -> int:
    return max(MIN_STEP, round(STEP_RATIO * dimension))


def _room(box: BoundingBox, side: str, page_width: int | None, page_height: int | None) -> float:
    if side == "left":
        return box.x
    if side == "top":
        return box.y
    if side == "right":
        return float("inf") if page_width is None else page_width - (box.x + box.width)
    return float("inf") if page_height is None else page_height - (box.y + box.height)


def _band_rect(box: BoundingBox, side: str, step: int) -> BoundingBox:
    if side == "left":
        return BoundingBox(box.x - step, box.y, step, box.height)
    if side == "right":
        return BoundingBox(box.x + box.width, box.y, step, box.height)
    if side == "top":
        return BoundingBox(box.x, box.y - step, box.width, step)
    return BoundingBox(box.x, box.y + box.height, box.width, step)


def _apply_growth(box: BoundingBox, accepted: dict[str, float]) -> BoundingBox:
    x, y, width, height = box.x, box.y, box.width, box.height
    if "left" in accepted:
        step = int(accepted["left"])
        x, width = x - step, width + step
    if "right" in accepted:
        width += int(accepted["right"])
    if "top" in accepted:
        step = int(accepted["top"])
        y, height = y - step, height + step
    if "bottom" in accepted:
        height += int(accepted["bottom"])
    return BoundingBox(x, y, width, height)


def _revert_bad_corners(image: Image.Image, text_box: BoundingBox, current: BoundingBox,
                        grown: dict[str, float],
                        background: tuple[float, float, float]) -> tuple[BoundingBox, dict[str, float]]:
    """A rectangle inscribing a round or irregular bubble always has corners
    that can land outside the bubble's true edge, even when every straight
    growth band along the way looked like safe background -- each band only
    samples along one axis, so a corner where two growing sides meet is
    never itself tested by either side's own band. A bad corner is corrected
    by reverting the two sides that formed it back to the original text
    box's edge there, rather than accepting a rectangle that bleeds into
    whatever art sits just past a round bubble's corner.
    """
    x0, y0, x1, y1 = text_box.x, text_box.y, text_box.x + text_box.width, text_box.y + text_box.height
    cx0, cy0, cx1, cy1 = current.x, current.y, current.x + current.width, current.y + current.height
    corners = {
        ("top", "left"): BoundingBox(cx0, cy0, x0 - cx0, y0 - cy0),
        ("top", "right"): BoundingBox(x1, cy0, cx1 - x1, y0 - cy0),
        ("bottom", "left"): BoundingBox(cx0, y1, x0 - cx0, cy1 - y1),
        ("bottom", "right"): BoundingBox(x1, y1, cx1 - x1, cy1 - y1),
    }
    revert: set[str] = set()
    for sides, rect in corners.items():
        if rect.width <= 0 or rect.height <= 0:
            continue
        if not _region_is_safe(image, rect, background):
            revert.update(sides)
    if not revert:
        return current, grown
    surviving = {side: (0.0 if side in revert else amount) for side, amount in grown.items()}
    return _apply_growth(text_box, surviving), surviving


def _estimate_background(image: Image.Image, box: BoundingBox,
                         mask: Polygon | None) -> tuple[float, float, float] | None:
    pixels = _region_pixels(image, box)
    if not pixels:
        return None
    if mask is not None:
        membership = rasterize_polygon(box, mask)
        sample = [p for p, is_fg in zip(pixels, membership) if not is_fg]
    else:
        # No ground truth: assume the brightest pixels in the text box are
        # bubble fill, not glyphs — a real mask (above) is always preferred.
        by_brightness = sorted(pixels, key=sum)
        cut = max(1, round(len(by_brightness) * BACKGROUND_PERCENTILE))
        sample = by_brightness[-cut:]
    if not sample:
        return None
    mean = _mean(sample)
    if max(_stdev(sample, mean)) > FLATNESS_THRESHOLD or sum(mean) / 3 < LIGHTNESS_THRESHOLD:
        return None
    return tuple(median(p[c] for p in sample) for c in range(3))


def _looks_like_background(pixels: list[tuple[int, int, int]],
                           background: tuple[float, float, float]) -> bool:
    outliers = sum(1 for p in pixels if _distance(p, background) > OUTLIER_DISTANCE)
    if outliers / len(pixels) > MAX_BAND_OUTLIER_RATIO:
        return False
    mean = _mean(pixels)
    if max(_stdev(pixels, mean)) > FLATNESS_THRESHOLD:
        return False
    if sum(mean) / 3 < LIGHTNESS_THRESHOLD:
        return False
    if _distance(mean, background) > BACKGROUND_DISTANCE_THRESHOLD:
        return False
    return True


def _region_is_safe(image: Image.Image, rect: BoundingBox,
                    background: tuple[float, float, float]) -> bool:
    """Like ``_looks_like_background``, but tested on square-ish slices along
    ``rect``'s longer axis rather than one pooled average over the whole
    thing. A long, thin growth band (or margin strip) can contain a small
    patch of real art — a stray line, a corner of clothing — that a single
    aggregate mean/std-dev comfortably dilutes away without ever crossing a
    threshold. Every slice has to pass on its own, so a bad patch anywhere
    along the band fails the whole band, however small a fraction it is of
    the total area.
    """
    for chunk in _chunk_rects(rect):
        pixels = _region_pixels(image, chunk)
        if not pixels or not _looks_like_background(pixels, background):
            return False
    return True


def _chunk_rects(rect: BoundingBox) -> list[BoundingBox]:
    size = max(4, min(rect.width, rect.height))
    chunks = []
    if rect.width >= rect.height:
        for x in range(rect.x, rect.x + rect.width, size):
            chunks.append(BoundingBox(x, rect.y, min(size, rect.x + rect.width - x), rect.height))
    else:
        for y in range(rect.y, rect.y + rect.height, size):
            chunks.append(BoundingBox(rect.x, y, rect.width, min(size, rect.y + rect.height - y)))
    return chunks


def _check_whole_margin(image: Image.Image, text_box: BoundingBox, current: BoundingBox,
                        background: tuple[float, float, float]) -> str | None:
    """A final holistic check over the entire newly-added margin, one strip
    at a time (each itself chunked — see ``_region_is_safe``) — catches slow
    drift across many individually-passing bands that no single incremental
    round would have caught on its own."""
    for rect in _margin_rects(text_box, current):
        if not _region_is_safe(image, rect, background):
            return "expanded margin doesn't look like the text box's background as a whole"
    return None


def _margin_rects(text_box: BoundingBox, current: BoundingBox) -> list[BoundingBox]:
    """``current`` minus ``text_box``, as four non-overlapping strips."""
    rects = []
    if text_box.y > current.y:
        rects.append(BoundingBox(current.x, current.y, current.width, text_box.y - current.y))
    bottom0, bottom1 = text_box.y + text_box.height, current.y + current.height
    if bottom1 > bottom0:
        rects.append(BoundingBox(current.x, bottom0, current.width, bottom1 - bottom0))
    if text_box.x > current.x:
        rects.append(BoundingBox(current.x, text_box.y, text_box.x - current.x, text_box.height))
    right0, right1 = text_box.x + text_box.width, current.x + current.width
    if right1 > right0:
        rects.append(BoundingBox(right0, text_box.y, right1 - right0, text_box.height))
    return rects


def _region_pixels(image: Image.Image, box: BoundingBox) -> list[tuple[int, int, int]]:
    if box.width <= 0 or box.height <= 0:
        return []
    return list(image.crop((box.x, box.y, box.x + box.width, box.y + box.height))
                .convert("RGB").getdata())


def _distance(p: tuple[float, float, float], q: tuple[float, float, float]) -> float:
    return max(abs(p[c] - q[c]) for c in range(3))


def _mean(pixels: list[tuple[int, int, int]]) -> tuple[float, float, float]:
    n = len(pixels)
    return tuple(sum(p[c] for p in pixels) / n for c in range(3))


def _stdev(pixels: list[tuple[int, int, int]],
          mean: tuple[float, float, float]) -> tuple[float, float, float]:
    n = len(pixels)
    return tuple((sum((p[c] - mean[c]) ** 2 for p in pixels) / n) ** 0.5 for c in range(3))
