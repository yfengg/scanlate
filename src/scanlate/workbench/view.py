"""The contract between the pipeline and the workbench UI.

Five outputs per segment — source, candidate, features, terminology, warnings —
plus the state axes and the spans the UI highlights. Nothing here exposes
database internals: no row ids, no SQL shapes, no backend metadata beyond what
a translator can act on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..core.types import Action, SegmentKind, Severity, Span
from ..storage.models import Origin, Status


class AnnotationType(str, Enum):
    TERM = "term"
    ENTITY = "entity"
    AMBIGUITY = "ambiguity"
    CULTURAL = "cultural"


@dataclass
class Annotation:
    """A marked range of the source. Spans, never string matching."""
    span: Span
    type: AnnotationType
    target: str | None = None        # approved or proposed rendering, for terms
    label: str | None = None
    locked: bool = False
    origin: Origin | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Alternative:
    text: str
    note: str | None = None          # "closer tone", "more literal"
    reason: str | None = None        # "register" | "literal" | "memory" | "sfx" | "terminology"


@dataclass
class Warning_:
    code: str
    severity: Severity
    message: str
    fingerprint: str
    span: Span | None = None
    actions: list[Action] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentView:
    id: str
    project_id: str
    order: int
    kind: SegmentKind
    source: str
    candidate: str
    language: str
    target_language: str
    page_id: str | None = None
    annotations: list[Annotation] = field(default_factory=list)
    alternatives: list[Alternative] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    warnings: list[Warning_] = field(default_factory=list)
    status: Status = Status.MACHINE
    origin: Origin = Origin.MACHINE_TRANSLATION
    # Derived, never stored: true when the text comes from a locked project
    # decision (canonical memory or a locked glossary rendering).
    locked: bool = False
    needs_review: bool = False
