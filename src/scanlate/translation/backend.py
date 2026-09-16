"""The TranslationBackend interface.

A backend is one component of the translation system, not the system. It sees
source text, languages, nearby context and terminology constraints, and returns
candidates. It never sees the glossary lock state, translation memory, or the
validation result — those are applied around it by the pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..core.types import LanguagePair


@dataclass(frozen=True)
class TerminologyConstraint:
    """A rendering the backend should use if it can. Advisory, not enforcement."""
    source_term: str
    target_term: str
    locked: bool = False


@dataclass
class TranslationRequest:
    source_text: str
    pair: LanguagePair
    preceding: list[str] = field(default_factory=list)
    following: list[str] = field(default_factory=list)
    constraints: list[TerminologyConstraint] = field(default_factory=list)
    segment_kind: str = "dialogue"
    hints: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranslationCandidate:
    text: str
    confidence: float = 0.5
    note: str | None = None
    reason: str | None = None       # "register" | "literal" | "terminology" | "sfx" | "memory"


@dataclass
class BackendResult:
    primary: TranslationCandidate | None
    alternatives: list[TranslationCandidate] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self.primary is None


class TranslationBackend(ABC):
    name: str = "backend"
    #: True for backends that exist to make tests deterministic rather than to
    #: translate. The UI says so instead of passing their output off as real
    #: machine translation.
    development: bool = False

    @abstractmethod
    def supports(self, pair: LanguagePair) -> bool:
        ...

    @abstractmethod
    def translate(self, request: TranslationRequest) -> BackendResult:
        ...
