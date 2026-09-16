"""Project glossary: storage, lookup, and conservative enforcement.

Enforcement never performs naive global replacement. It corrects only when the
substitution is provably unambiguous; otherwise it raises an issue and leaves
the translator's text alone.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from enum import Enum

from ..core.types import Action, Issue, Severity, Span, select_non_overlapping
from ..core.scripts import script_of

STAGE = "glossary"


class Category(str, Enum):
    CHARACTER = "character"
    ALIAS = "alias"
    TITLE = "title"
    LOCATION = "location"
    ORGANIZATION = "organization"
    TECHNIQUE = "technique"
    OBJECT = "object"
    RELATIONSHIP = "relationship_term"
    PHRASE = "recurring_phrase"
    OTHER = "other"


@dataclass
class GlossaryEntry:
    id: int | None
    project_id: str
    source_language: str
    source_term: str
    target_language: str
    target_term: str
    category: Category = Category.OTHER
    locked: bool = False
    notes: str | None = None


@dataclass
class GlossaryHit:
    entry: GlossaryEntry
    span: Span


@dataclass
class EnforcementResult:
    text: str
    issues: list[Issue] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.corrections)


def _is_wordy(ch: str) -> bool:
    """True for scripts that use spaces between words, where boundaries matter."""
    return script_of(ch) in {"Latin", "Cyrillic", "Greek"}


class GlossaryStore:
    """Statements only; the caller's UnitOfWork decides when they become durable."""

    def __init__(self, conn: sqlite3.Connection, uow=None):
        self.conn = conn
        from ..storage.unit_of_work import UnitOfWork
        self.uow = uow or UnitOfWork(conn)

    def add(self, entry: GlossaryEntry) -> GlossaryEntry:
        cur = self.conn.execute(
            "INSERT INTO glossary_entries (project_id, source_language, source_term, "
            "target_language, target_term, category, locked, notes) VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(project_id, source_language, source_term, target_language) "
            "DO UPDATE SET target_term=excluded.target_term, category=excluded.category, "
            "locked=excluded.locked, notes=excluded.notes",
            (entry.project_id, entry.source_language, entry.source_term, entry.target_language,
             entry.target_term, Category(entry.category).value, int(entry.locked), entry.notes))
        entry.id = cur.lastrowid or self._find_id(entry)
        return entry

    def _find_id(self, entry: GlossaryEntry) -> int | None:
        row = self.conn.execute(
            "SELECT id FROM glossary_entries WHERE project_id=? AND source_language=? "
            "AND source_term=? AND target_language=?",
            (entry.project_id, entry.source_language, entry.source_term,
             entry.target_language)).fetchone()
        return row["id"] if row else None

    def update(self, entry_id: int, **fields) -> GlossaryEntry | None:
        allowed = {"target_term", "category", "locked", "notes", "source_term"}
        sets = {k: (int(v) if k == "locked" else v) for k, v in fields.items() if k in allowed}
        if sets:
            assignments = ", ".join(f"{k} = ?" for k in sets)
            self.conn.execute(f"UPDATE glossary_entries SET {assignments} WHERE id = ?",
                              (*sets.values(), entry_id))
            return self.get(entry_id)

    def get(self, entry_id: int) -> GlossaryEntry | None:
        row = self.conn.execute("SELECT * FROM glossary_entries WHERE id = ?", (entry_id,)).fetchone()
        return self._row(row) if row else None

    def list(self, project_id: str, source_language: str | None = None) -> list[GlossaryEntry]:
        sql = "SELECT * FROM glossary_entries WHERE project_id = ?"
        args: list = [project_id]
        if source_language:
            sql += " AND source_language = ?"
            args.append(source_language)
        return [self._row(r) for r in self.conn.execute(sql + " ORDER BY length(source_term) DESC", args)]

    @staticmethod
    def _row(row: sqlite3.Row) -> GlossaryEntry:
        return GlossaryEntry(
            id=row["id"], project_id=row["project_id"], source_language=row["source_language"],
            source_term=row["source_term"], target_language=row["target_language"],
            target_term=row["target_term"], category=Category(row["category"]),
            locked=bool(row["locked"]), notes=row["notes"])

    # --- lookup --------------------------------------------------------
    def find_in(self, text: str, project_id: str, source_language: str,
                target_language: str) -> list[GlossaryHit]:
        """All occurrences, longest term first, overlaps resolved by length."""
        hits: list[GlossaryHit] = []
        for entry in self.list(project_id, source_language):
            if entry.target_language != target_language:
                continue
            for m in re.finditer(re.escape(entry.source_term), text):
                if _is_wordy(entry.source_term[0]):
                    before = text[m.start() - 1:m.start()]
                    after = text[m.end():m.end() + 1]
                    if (before and _is_wordy(before)) or (after and _is_wordy(after)):
                        continue
                hits.append(GlossaryHit(entry, Span(m.start(), m.end())))
        return select_non_overlapping(hits, lambda h: h.span)


def _pattern(term: str) -> str:
    if _is_wordy(term[0]):
        return rf"(?<![\w']){re.escape(term)}(?![\w'])"
    return re.escape(term)


def _count(text: str, term: str) -> int:
    return len(re.findall(_pattern(term), text, re.I))


def _replace_once(text: str, term: str, replacement: str) -> str:
    return re.sub(_pattern(term), replacement, text, count=1, flags=re.I)


def _present(text: str, term: str) -> bool:
    return bool(term) and re.search(_pattern(term), text, re.I) is not None


def enforce(text: str, hits: list[GlossaryHit], known_variants: dict[str, list[str]] | None = None,
            stage: str = STAGE, correct: bool = True) -> EnforcementResult:
    """Make locked terminology hold in the candidate, or explain why it can't.

    A correction is applied only when the wrong rendering appears exactly once
    as a whole word and the approved rendering is absent. Anything less certain
    becomes an issue for the translator.

    With ``correct=False`` nothing is rewritten and conflicts are only
    reported — used where the text is itself an approved decision (canonical
    translation memory), which the pipeline must not silently overwrite.
    """
    known_variants = known_variants or {}
    result = EnforcementResult(text=text)

    for hit in hits:
        entry = hit.entry
        if not entry.locked:
            continue
        if _present(result.text, entry.target_term):
            continue

        variants = [v for v in known_variants.get(entry.source_term, []) if v != entry.target_term]
        matches = sorted((v for v in variants if _present(result.text, v)), key=len, reverse=True)

        if matches:
            longest = matches[0]
            # Shorter variants inside the longest ("bro" within "big bro") are the
            # same occurrence; count only their *standalone* extra uses.
            nested = [v for v in matches[1:] if v.lower() in longest.lower()]
            separate = [v for v in matches[1:] if v.lower() not in longest.lower()]
            base = _count(result.text, longest)
            total = base + sum(_count(result.text, v) for v in separate)
            total += sum(max(0, _count(result.text, v) - base) for v in nested)

            if total == 1 and correct:
                result.text = _replace_once(result.text, longest, entry.target_term)
                result.corrections.append(f"{longest} → {entry.target_term}")
                result.issues.append(Issue(
                    code="terminology.corrected", severity=Severity.INFO,
                    message=f"Applied the locked rendering {entry.target_term} for {entry.source_term}.",
                    stage=stage, span=hit.span,
                    data={"source_term": entry.source_term, "from": longest,
                          "to": entry.target_term}))
                continue
            reason = "several possible replacements"
        else:
            reason = "no candidate rendering found"

        result.issues.append(Issue(
            code="terminology.conflict", severity=Severity.WARNING,
            message=(f"{entry.source_term} is locked to {entry.target_term}, "
                     f"but the translation doesn't use it."),
            stage=stage, span=hit.span,
            actions=[Action("use", f"Use {entry.target_term}", entry.target_term),
                     Action("dismiss", "Leave as is")],
            data={"source_term": entry.source_term, "expected": entry.target_term,
                  "entry_id": entry.id,
                  "reason": reason}))
    return result
