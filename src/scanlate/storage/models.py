"""Segment domain model.

Status and origin are independent axes:

* ``status``  — who last touched the text: machine, edited, approved.
* ``origin``  — where the text came from: machine_translation,
  translation_memory, glossary, manual, rule.

There is deliberately **no** segment-level lock. Locking is a property of the
*decision* that produced the text, not of the segment: a glossary entry is
locked, or a translation-memory variant is canonical. Storing a third copy on
the segment gave a field with no independent meaning that could silently
disagree with its source. Whether a segment is protected is therefore derived
at read time from its origin plus the locked glossary/memory records behind it
(see ``pipeline`` and ``api.schemas``), and reported to the UI there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..core.types import SegmentKind
from ..imaging.layout import Region, RenderSettings


class Status(str, Enum):
    MACHINE = "machine"
    EDITED = "edited"
    APPROVED = "approved"


class Origin(str, Enum):
    MACHINE_TRANSLATION = "machine_translation"
    TRANSLATION_MEMORY = "translation_memory"
    GLOSSARY = "glossary"
    MANUAL = "manual"
    RULE = "rule"


@dataclass
class Project:
    id: str
    name: str
    default_source_language: str
    target_language: str


@dataclass
class Chapter:
    id: str
    project_id: str
    number: int
    title: str | None = None


@dataclass
class Page:
    id: str
    project_id: str
    number: int
    chapter_id: str | None = None
    image_path: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass
class SegmentState:
    candidate: str | None = None
    status: Status = Status.MACHINE
    origin: Origin = Origin.MACHINE_TRANSLATION
    updated_at: str | None = None


@dataclass
class Segment:
    id: str
    project_id: str
    seq: int                              # project-wide reading order, unique
    kind: SegmentKind
    source_text: str
    language: str | None = None           # detected or assigned; may differ from project default
    page_id: str | None = None
    state: SegmentState = field(default_factory=SegmentState)
    # Reserved for the image pipeline. Unused by the text-only MVP.
    region: Region | None = None
    render: RenderSettings | None = None
    mask_ref: str | None = None

    def resolved_language(self, project_default: str) -> str:
        return self.language or project_default


@dataclass
class SegmentContext:
    """A neighbouring segment, as local translation context."""
    id: str
    source_text: str
    language: str | None
    kind: SegmentKind
    approved_translation: str | None = None


@dataclass
class LocalContext:
    preceding: list[SegmentContext] = field(default_factory=list)
    following: list[SegmentContext] = field(default_factory=list)

    def source_lines(self) -> tuple[list[str], list[str]]:
        return ([c.source_text for c in self.preceding],
                [c.source_text for c in self.following])
