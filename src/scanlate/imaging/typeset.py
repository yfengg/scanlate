"""Deterministic text layout for the export renderer.

Given an approved translation and the box it must fit, this wraps text and
steps the font size down until it fits — or reports that it doesn't, rather
than overflowing the box or guessing.

Layout is per target language, looked up by a small strategy registry (the
same shape as ``OcrRouter``/``BackendRegistry`` elsewhere in this codebase).
Only "en" (horizontal Latin, left-to-right) is implemented, matching the
target languages the frontend currently offers. Any other target language is
explicitly unsupported rather than silently typeset as if it were English —
a different script needs different word-boundary rules, quite possibly a
different writing direction, and almost certainly a different font, none of
which the Latin strategy below gets right by accident.

Fonts are bundled, not read from the machine: Pillow's own built-in default
font is deliberately not used here, because a project/render setting must
produce the same output on every machine regardless of what Pillow version
or OS fonts happen to be around. ``imaging/fonts/Lato-Regular.ttf`` (SIL Open
Font License 1.1, full text in ``imaging/fonts/OFL.txt``) is the one font
shipped today; ``RenderSettings.font_id`` names which bundled font a segment
was laid out with, via ``FONT_FILES`` below.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .layout import BoundingBox, RenderSettings

MIN_FONT_SIZE = 8.0
FONT_STEP = 1.0

FONTS_DIR = Path(__file__).parent / "fonts"
DEFAULT_FONT_ID = "lato-regular"
# RenderSettings.font_id -> bundled font file. An unrecognized id falls back
# to the default rather than erroring — there is only one font today, and
# choosing among several is the font-matching work this milestone defers.
FONT_FILES: dict[str, Path] = {DEFAULT_FONT_ID: FONTS_DIR / "Lato-Regular.ttf"}

# A throwaway 1x1 canvas: PIL's text measurement needs a draw context, but
# measuring never reads or writes any pixels, so one shared canvas is enough.
_MEASURE_CANVAS = Image.new("RGB", (1, 1))


@lru_cache(maxsize=64)
def load_font(font_id: str, size: float) -> ImageFont.FreeTypeFont:
    """The bundled font for ``font_id`` at ``size``.

    Both wrap measurement below and the actual drawing in ``render.py`` call
    this, so what got measured to fit is exactly what gets drawn — a
    different font for each step would silently invalidate the fit.
    """
    path = FONT_FILES.get(font_id, FONT_FILES[DEFAULT_FONT_ID])
    return ImageFont.truetype(str(path), size)


@dataclass
class LayoutResult:
    fits: bool
    lines: list[str]
    font_size: float
    unsupported: bool = False
    reason: str | None = None


def _measure(font, text: str) -> tuple[float, float]:
    box = ImageDraw.Draw(_MEASURE_CANVAS).textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


class TypesetStrategy(ABC):
    """One target language/script's wrapping and fitting rules."""

    @abstractmethod
    def layout(self, text: str, box: BoundingBox, render: RenderSettings) -> LayoutResult: ...


class LatinHorizontalTypesetter(TypesetStrategy):
    """Greedy word-wrap, left-to-right, stepping the font size down to fit.

    Uses the bundled font named by ``render.font_id`` (see module docstring)
    for every measurement, so the same render settings produce the same
    layout on any machine.
    """

    def layout(self, text: str, box: BoundingBox, render: RenderSettings) -> LayoutResult:
        pad_l, pad_t, pad_r, pad_b = render.padding
        available_w = box.width - pad_l - pad_r
        available_h = box.height - pad_t - pad_b
        if available_w <= 0 or available_h <= 0:
            return LayoutResult(False, [], render.font_size,
                                 reason="region is too small for its own padding")

        words = text.split()
        if not words:
            return LayoutResult(True, [], render.font_size)

        size = render.font_size
        lines: list[str] = []
        while True:
            font = load_font(render.font_id, size)
            lines = self._wrap(words, font, available_w)
            widest = max((_measure(font, line)[0] for line in lines), default=0)
            total_h = len(lines) * size * render.line_height
            if widest <= available_w and total_h <= available_h:
                return LayoutResult(True, lines, size)
            if size <= MIN_FONT_SIZE:
                break
            size = max(MIN_FONT_SIZE, size - FONT_STEP)

        return LayoutResult(False, lines, size,
                             reason="translation doesn't fit the region even at the minimum font size")

    def _wrap(self, words: list[str], font, available_w: float) -> list[str]:
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if _measure(font, candidate)[0] <= available_w:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines


class TextLayoutEngine:
    """Looks up the strategy for a target language and delegates to it."""

    def __init__(self, strategies: dict[str, TypesetStrategy] | None = None):
        self.strategies = strategies or {"en": LatinHorizontalTypesetter()}

    def layout(self, text: str, box: BoundingBox, render: RenderSettings,
               target_language: str) -> LayoutResult:
        strategy = self.strategies.get(target_language)
        if strategy is None:
            return LayoutResult(False, [], render.font_size, unsupported=True,
                reason=f"no typesetting strategy for target language {target_language!r}")
        return strategy.layout(text, box, render)
