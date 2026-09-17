"""Page rendering: composites cleaned bubbles and typeset translations (or
SFX captions) onto a copy of the source page image.

Pure image transform — no database, no HTTP, no file I/O. The caller
(``PageService.render_page``) decides which segments are eligible (approved
only) and assembles a ``RenderableRegion`` per segment; it is also
responsible for persisting or serving whatever this returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from PIL import Image, ImageDraw

from .cleanup import BubbleCleaner, CleanupResult
from .layout import BoundingBox, RegionKind, RenderSettings
from .typeset import LayoutResult, TextLayoutEngine, load_font

# A caption is deliberately small and fixed-size — it is a label next to the
# art, not a bubble replacing it.
SFX_CAPTION_FONT_SIZE = 12.0
SFX_CAPTION_GAP = 4
SFX_CAPTION_HEIGHT = 40
TEXT_COLOR = (17, 17, 17)


@dataclass
class RenderableRegion:
    """Everything the renderer needs for one region, assembled by the caller
    from a ``Segment`` and its approved candidate — this module never touches
    storage models directly."""
    segment_id: str
    kind: RegionKind
    box: BoundingBox
    text: str
    render: RenderSettings
    target_language: str


@dataclass
class RegionRenderResult:
    segment_id: str
    kind: RegionKind
    rendered: bool
    reason: str | None = None
    cleanup: CleanupResult | None = None
    layout: LayoutResult | None = None
    # The exact settings that produced this result (font size after any
    # step-down), only when rendered — the caller persists this so the next
    # render starts at the size that already worked. None when flagged.
    applied_render: RenderSettings | None = None


@dataclass
class RenderReport:
    results: list[RegionRenderResult] = field(default_factory=list)

    @property
    def flagged(self) -> list[RegionRenderResult]:
        """Regions left as original artwork and needing manual handling."""
        return [r for r in self.results if not r.rendered]


class PageRenderer:
    def __init__(self, cleaner: BubbleCleaner | None = None,
                 typesetter: TextLayoutEngine | None = None):
        self.cleaner = cleaner or BubbleCleaner()
        self.typesetter = typesetter or TextLayoutEngine()

    def render(self, image: Image.Image,
               regions: list[RenderableRegion]) -> tuple[Image.Image, RenderReport]:
        out = image.convert("RGB").copy()
        report = RenderReport()
        for region in regions:
            if region.kind == RegionKind.SFX:
                report.results.append(self._render_sfx(out, region))
            else:
                report.results.append(self._render_bubble(out, region))
        return out, report

    def _render_bubble(self, image: Image.Image, region: RenderableRegion) -> RegionRenderResult:
        cleanup = self.cleaner.inspect(image, region.box)
        if not cleanup.ok:
            return RegionRenderResult(region.segment_id, region.kind, False, cleanup.reason, cleanup, None)

        layout = self.typesetter.layout(region.text, cleanup.cleaned_box, region.render,
                                        region.target_language)
        if not layout.fits:
            reason = layout.reason or "translation does not fit the region"
            return RegionRenderResult(region.segment_id, region.kind, False, reason, cleanup, layout)

        # Only now, once both steps are known to succeed, does anything about
        # the page actually change — a flagged region is pristine, not
        # half-cleaned.
        self.cleaner.apply(image, cleanup)
        applied = replace(region.render, font_size=layout.font_size)
        _draw_lines(image, cleanup.cleaned_box, layout, applied)
        return RegionRenderResult(region.segment_id, region.kind, True, None, cleanup, layout, applied)

    def _render_sfx(self, image: Image.Image, region: RenderableRegion) -> RegionRenderResult:
        caption_render = replace(region.render, font_size=SFX_CAPTION_FONT_SIZE, alignment="center")
        caption_box = _caption_box(region.box, image.width, image.height)
        layout = self.typesetter.layout(region.text, caption_box, caption_render, region.target_language)
        if not layout.fits:
            reason = layout.reason or "SFX caption does not fit near the region"
            return RegionRenderResult(region.segment_id, region.kind, False, reason, None, layout)

        applied = replace(caption_render, font_size=layout.font_size)
        _draw_lines(image, caption_box, layout, applied, stroke=True)
        return RegionRenderResult(region.segment_id, region.kind, True, None, None, layout, applied)


def _caption_box(box: BoundingBox, image_w: int, image_h: int) -> BoundingBox:
    """A small strip just below the SFX box, clamped to stay on the page."""
    width = min(box.width, image_w) if image_w > 0 else box.width
    x = min(box.x, max(0, image_w - width))
    y = min(box.y + box.height + SFX_CAPTION_GAP, max(0, image_h - SFX_CAPTION_HEIGHT))
    return BoundingBox(x, y, max(1, width), SFX_CAPTION_HEIGHT)


def _draw_lines(image: Image.Image, box: BoundingBox, layout: LayoutResult,
                render: RenderSettings, stroke: bool = False) -> None:
    pad_l, pad_t, pad_r, pad_b = render.padding
    font = load_font(render.font_id, layout.font_size)
    draw = ImageDraw.Draw(image)
    available_w = box.width - pad_l - pad_r
    y = box.y + pad_t
    for line in layout.lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        if render.alignment == "center":
            x = box.x + pad_l + max(0, (available_w - line_w) / 2)
        elif render.alignment == "right":
            x = box.x + pad_l + max(0, available_w - line_w)
        else:
            x = box.x + pad_l
        if stroke:
            # A white outline keeps a small caption legible over dark art
            # without needing a background rectangle.
            draw.text((x, y), line, font=font, fill=TEXT_COLOR,
                      stroke_width=2, stroke_fill=(255, 255, 255))
        else:
            draw.text((x, y), line, font=font, fill=TEXT_COLOR)
        y += layout.font_size * render.line_height
