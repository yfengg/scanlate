from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


@dataclass(frozen=True)
class LanguagePair:
    source: str
    target: str

    def __str__(self) -> str:
        return f"{self.source}>{self.target}"

    @classmethod
    def parse(cls, value: str) -> "LanguagePair":
        source, _, target = value.partition(">")
        if not source.strip() or not target.strip():
            raise ValueError(f"Invalid language pair {value!r}; expected 'src>tgt'")
        return cls(source.strip(), target.strip())


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end

    def contains(self, other: "Span") -> bool:
        return self.start <= other.start and other.end <= self.end

    def __len__(self) -> int:
        return self.end - self.start


class SegmentKind(str, Enum):
    DIALOGUE = "dialogue"
    THOUGHT = "thought"
    NARRATION = "narration"
    SFX = "sfx"
    SIGN = "sign"
    OTHER = "other"


class Severity(IntEnum):
    INFO = 0
    WARNING = 1
    ERROR = 2


@dataclass(frozen=True)
class Action:
    """An offer the translator can take without leaving the segment."""
    kind: str                      # "use" | "add_rule" | "show_alternatives" | "dismiss"
    label: str
    value: str | None = None


@dataclass
class Issue:
    """Canonical problem record. Validators emit these; the API view renders them."""
    code: str
    severity: Severity
    message: str
    stage: str
    span: Span | None = None
    actions: list[Action] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


def select_non_overlapping(items: list, span_of) -> list:
    """Greedy longest-first selection of items whose spans don't overlap."""
    chosen: list = []
    for item in sorted(items, key=lambda i: (-len(span_of(i)), span_of(i).start)):
        if all(not span_of(item).overlaps(span_of(c)) for c in chosen):
            chosen.append(item)
    return sorted(chosen, key=lambda i: span_of(i).start)


TERMINAL_PUNCT = {
    "question": "?？",
    "exclamation": "!！",
    "trailing": "…‥",
}


def terminal_punctuation(text: str) -> set[str]:
    """Generic terminal-punctuation reading, shared by every language."""
    stripped = text.rstrip().rstrip("」』\"')）]】")
    found = set()
    tail = stripped[-4:]
    for kind, chars in TERMINAL_PUNCT.items():
        if any(c in tail for c in chars):
            found.add(kind)
    if stripped.endswith(("--", "ー", "—")):
        found.add("trailing")
    return found


def issue_fingerprint(code: str, source: str, candidate: str, evidence: str = "") -> str:
    """Stable identity for one *instance* of an issue.

    Built from the factors that made the issue appear, so dismissing today's
    register mismatch cannot silence a different one raised after the
    translation changes. Same situation, same fingerprint; changed situation,
    new fingerprint.
    """
    import hashlib

    digest = hashlib.sha256("\x1f".join((code, source, candidate, evidence)).encode("utf-8"))
    return digest.hexdigest()[:16]
