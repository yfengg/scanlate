"""Geometry and typesetting data structures for the future image pipeline.

Defined now so the segment model and database can carry them, deliberately
inert: nothing in this module reads or writes pixels. The intended flow is
page → regions → OCR → translation → approved text → redraw → typesetting →
export, and these are the shapes that flow will pass around.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RegionKind(str, Enum):
    SPEECH_BUBBLE = "speech_bubble"
    THOUGHT_BUBBLE = "thought_bubble"
    CAPTION = "caption"
    FREE_TEXT = "free_text"
    SFX = "sfx"
    SIGN = "sign"


class TextOrientation(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL_RL = "vertical_rl"      # Japanese vertical setting, right to left


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int


@dataclass
class Polygon:
    points: list[tuple[int, int]] = field(default_factory=list)

    def bounding_box(self) -> BoundingBox | None:
        if not self.points:
            return None
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return BoundingBox(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))


@dataclass
class Region:
    kind: RegionKind = RegionKind.SPEECH_BUBBLE
    box: BoundingBox | None = None
    polygon: Polygon | None = None
    reading_order: int = 0
    orientation: TextOrientation = TextOrientation.HORIZONTAL


@dataclass
class RenderSettings:
    font_id: str = "default"
    font_size: float = 16.0
    line_height: float = 1.2
    alignment: str = "center"
    rotation: float = 0.0
    orientation: TextOrientation = TextOrientation.HORIZONTAL
    padding: tuple[int, int, int, int] = (4, 4, 4, 4)


@dataclass(frozen=True)
class MaskRef:
    """Points at the inpainting mask produced by text removal."""
    path: str
    revision: int = 1


# --- serialization -----------------------------------------------------
# Explicit, so nested dataclasses and enums keep their types across a
# round-trip instead of decaying into whatever ``default=str`` produced.

def box_to_dict(box: BoundingBox | None) -> dict | None:
    return None if box is None else {"x": box.x, "y": box.y,
                                     "width": box.width, "height": box.height}


def box_from_dict(data: dict | None) -> BoundingBox | None:
    return None if not data else BoundingBox(int(data["x"]), int(data["y"]),
                                             int(data["width"]), int(data["height"]))


def region_to_dict(region: Region | None) -> dict | None:
    if region is None:
        return None
    return {
        "kind": region.kind.value,
        "box": box_to_dict(region.box),
        "polygon": [[int(x), int(y)] for x, y in region.polygon.points] if region.polygon else None,
        "reading_order": region.reading_order,
        "orientation": region.orientation.value,
    }


def region_from_dict(data: dict | None) -> Region | None:
    if not data:
        return None
    polygon = data.get("polygon")
    return Region(
        kind=RegionKind(data.get("kind", RegionKind.SPEECH_BUBBLE.value)),
        box=box_from_dict(data.get("box")),
        polygon=Polygon([(int(x), int(y)) for x, y in polygon]) if polygon else None,
        reading_order=int(data.get("reading_order", 0)),
        orientation=TextOrientation(data.get("orientation", TextOrientation.HORIZONTAL.value)),
    )


def render_to_dict(render: RenderSettings | None) -> dict | None:
    if render is None:
        return None
    return {
        "font_id": render.font_id, "font_size": render.font_size,
        "line_height": render.line_height, "alignment": render.alignment,
        "rotation": render.rotation, "orientation": render.orientation.value,
        "padding": list(render.padding),
    }


def render_from_dict(data: dict | None) -> RenderSettings | None:
    if not data:
        return None
    padding = data.get("padding") or [0, 0, 0, 0]
    return RenderSettings(
        font_id=data.get("font_id", "default"), font_size=float(data.get("font_size", 16.0)),
        line_height=float(data.get("line_height", 1.2)), alignment=data.get("alignment", "center"),
        rotation=float(data.get("rotation", 0.0)),
        orientation=TextOrientation(data.get("orientation", TextOrientation.HORIZONTAL.value)),
        padding=tuple(int(p) for p in padding),
    )
