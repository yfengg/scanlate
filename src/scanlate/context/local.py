"""Lightweight translation context.

Deliberately local: the neighbouring lines in reading order, with their
approved translations where they exist. No plot summaries, no project-wide
digest, nothing the user has to write by hand. The shape is stable enough that
richer context can be added later without changing the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.types import SegmentKind
from ..storage.models import LocalContext


@dataclass
class ContextWindow:
    preceding_source: list[str] = field(default_factory=list)
    following_source: list[str] = field(default_factory=list)
    preceding_target: list[str] = field(default_factory=list)
    scene_tags: set[str] = field(default_factory=set)

    @property
    def empty(self) -> bool:
        return not (self.preceding_source or self.following_source)


def build_context(local: LocalContext, window: int = 2) -> ContextWindow:
    preceding = local.preceding[-window:]
    following = local.following[:window]
    return ContextWindow(
        preceding_source=[c.source_text for c in preceding],
        following_source=[c.source_text for c in following],
        preceding_target=[c.approved_translation for c in preceding
                          if c.approved_translation],
        scene_tags={"sfx_nearby"} if any(c.kind is SegmentKind.SFX
                                         for c in preceding + following) else set(),
    )
