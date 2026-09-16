"""Translation memory, shared across languages.

The rules the rest of the system depends on:

* Only approved translations enter memory. A machine candidate never becomes
  authoritative merely by existing.
* One normalized source may hold **several** approved variants — the same line
  is legitimately translated differently in different scenes.
* At most one variant may be canonical (locked). A canonical variant is
  authoritative and may autofill a segment.
* An unlocked exact match is an approved precedent, not immutable truth: it is
  offered, and where several exist they are all offered.
* Fuzzy matches are suggestions only, never applied automatically.
* Usage is counted from distinct segment occurrences, so reapproving one
  segment cannot inflate the count.

``normalization_version`` records which normalizer produced
``normalized_source``, so a future change to normalization can be migrated
rather than silently mismatching old rows.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher

from ..storage.unit_of_work import UnitOfWork

# RapidFuzz when available, difflib otherwise. The two are similar but not
# identical: difflib's ratio is based on matching blocks and can rank close
# candidates differently, so scores are comparable in spirit, not equal. The
# threshold is applied to whichever is active, and neither result is ever
# authoritative on its own.
try:
    from rapidfuzz.fuzz import ratio as _ratio

    def _similarity(a: str, b: str) -> float:
        return _ratio(a, b) / 100.0
except Exception:  # pragma: no cover - exercised only without rapidfuzz
    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

FUZZY_THRESHOLD = 0.78
NORMALIZATION_VERSION = 1


@dataclass
class MemoryEntry:
    id: int | None
    project_id: str
    source_language: str
    target_language: str
    normalized_source: str
    source_text: str
    target_text: str
    locked: bool = False
    occurrences: int = 0
    normalization_version: int = NORMALIZATION_VERSION
    created_at: str | None = None
    updated_at: str | None = None

    @property
    def canonical(self) -> bool:
        """Locked variants are authoritative and may autofill."""
        return self.locked


@dataclass
class MemoryMatch:
    entry: MemoryEntry
    score: float

    @property
    def exact(self) -> bool:
        return self.score >= 0.999


class CanonicalConflict(Exception):
    """Raised when a second canonical variant is requested for one source."""


class TranslationMemory:
    def __init__(self, conn: sqlite3.Connection, uow: UnitOfWork | None = None,
                 fuzzy_threshold: float = FUZZY_THRESHOLD):
        self.conn = conn
        self.uow = uow or UnitOfWork(conn)
        self.fuzzy_threshold = fuzzy_threshold

    # --- writing -------------------------------------------------------
    def record_approved(self, project_id: str, source_language: str, target_language: str,
                        normalized_source: str, source_text: str, target_text: str,
                        segment_id: str | None = None, locked: bool = False,
                        normalization_version: int = NORMALIZATION_VERSION) -> MemoryEntry:
        """The only way in. Callers must hold an approved translation.

        Recording an occurrence is idempotent per (variant, segment): the same
        segment approved twice counts once, two segments count twice.
        """
        with self.uow.transaction():
            if locked:
                existing = self.canonical(project_id, source_language, target_language,
                                          normalized_source, normalization_version)
                if existing and existing.target_text != target_text:
                    raise CanonicalConflict(
                        f"{normalized_source!r} already has a canonical translation "
                        f"({existing.target_text!r}); unlock it before locking another.")
            self.conn.execute(
                "INSERT INTO translation_memory (project_id, source_language, target_language, "
                "normalized_source, normalization_version, source_text, target_text, locked) "
                "VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(project_id, source_language, target_language, normalized_source, "
                "normalization_version, target_text) DO UPDATE SET "
                "locked=max(translation_memory.locked, excluded.locked), "
                "updated_at=datetime('now')",
                (project_id, source_language, target_language, normalized_source,
                 normalization_version, source_text, target_text, int(locked)))
            entry = self.variant(project_id, source_language, target_language,
                                 normalized_source, target_text, normalization_version)
            if entry and segment_id:
                self.assign_occurrence(entry.id, segment_id)
                entry.occurrences = self.occurrence_count(entry.id)
        return entry

    def set_locked(self, entry_id: int, locked: bool) -> MemoryEntry | None:
        """Promote or demote a variant. Promotion fails if another is canonical."""
        with self.uow.transaction():
            entry = self.get(entry_id)
            if entry is None:
                return None
            if locked:
                existing = self.canonical(entry.project_id, entry.source_language,
                                          entry.target_language, entry.normalized_source,
                                          entry.normalization_version)
                if existing and existing.id != entry_id:
                    raise CanonicalConflict(
                        f"{existing.target_text!r} is already canonical for this source.")
            self.conn.execute("UPDATE translation_memory SET locked=?, updated_at=datetime('now') "
                              "WHERE id=?", (int(locked), entry_id))
        return self.get(entry_id)

    # --- reading -------------------------------------------------------
    def exact(self, project_id: str, source_language: str, target_language: str,
              normalized_source: str,
              normalization_version: int = NORMALIZATION_VERSION) -> list[MemoryEntry]:
        """All approved variants for this source, canonical first, then by use.

        Scoped to one normalization version: memory normalized by an older
        normalizer is not a match for text normalized by the current one.
        """
        rows = self.conn.execute(
            "SELECT * FROM translation_memory WHERE project_id=? AND source_language=? "
            "AND target_language=? AND normalized_source=? AND normalization_version=?",
            (project_id, source_language, target_language, normalized_source,
             normalization_version)).fetchall()
        entries = [self._row(r) for r in rows]
        entries.sort(key=lambda e: (not e.locked, -e.occurrences, e.target_text))
        return entries

    def canonical(self, project_id: str, source_language: str, target_language: str,
                  normalized_source: str,
                  normalization_version: int = NORMALIZATION_VERSION) -> MemoryEntry | None:
        row = self.conn.execute(
            "SELECT * FROM translation_memory WHERE project_id=? AND source_language=? "
            "AND target_language=? AND normalized_source=? AND normalization_version=? "
            "AND locked=1",
            (project_id, source_language, target_language, normalized_source,
             normalization_version)).fetchone()
        return self._row(row) if row else None

    def variant(self, project_id: str, source_language: str, target_language: str,
                normalized_source: str, target_text: str,
                normalization_version: int = NORMALIZATION_VERSION) -> MemoryEntry | None:
        row = self.conn.execute(
            "SELECT * FROM translation_memory WHERE project_id=? AND source_language=? "
            "AND target_language=? AND normalized_source=? AND normalization_version=? "
            "AND target_text=?",
            (project_id, source_language, target_language, normalized_source,
             normalization_version, target_text)).fetchone()
        return self._row(row) if row else None

    def get(self, entry_id: int) -> MemoryEntry | None:
        row = self.conn.execute("SELECT * FROM translation_memory WHERE id=?",
                                (entry_id,)).fetchone()
        return self._row(row) if row else None

    def fuzzy(self, project_id: str, source_language: str, target_language: str,
              normalized_source: str, limit: int = 3,
              normalization_version: int = NORMALIZATION_VERSION) -> list[MemoryMatch]:
        """Similar approved sources, scored. Suggestions only — never applied."""
        rows = self.conn.execute(
            "SELECT * FROM translation_memory WHERE project_id=? AND source_language=? "
            "AND target_language=? AND normalization_version=? AND normalized_source != ?",
            (project_id, source_language, target_language, normalization_version,
             normalized_source)).fetchall()
        scored = [MemoryMatch(self._row(r), round(score, 3)) for r in rows
                  if (score := _similarity(normalized_source, r["normalized_source"]))
                  >= self.fuzzy_threshold]
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[:limit]

    def list(self, project_id: str, limit: int = 200) -> list[MemoryEntry]:
        return [self._row(r) for r in self.conn.execute(
            "SELECT * FROM translation_memory WHERE project_id=? "
            "ORDER BY updated_at DESC LIMIT ?", (project_id, limit))]

    # --- occurrences ---------------------------------------------------
    def assign_occurrence(self, entry_id: int, segment_id: str) -> None:
        """Point a segment at the variant it is approved to *right now*.

        A segment has one current occurrence. Approving it to a different
        variant moves the row, so the abandoned variant stops counting this
        segment. Both statements run in one transaction.
        """
        with self.uow.transaction():
            self.conn.execute("DELETE FROM translation_memory_occurrences WHERE segment_id=?",
                              (segment_id,))
            self.conn.execute(
                "INSERT INTO translation_memory_occurrences (memory_id, segment_id) VALUES (?, ?)",
                (entry_id, segment_id))

    def clear_occurrence(self, segment_id: str) -> None:
        """Used when a segment is reopened and no longer endorses any variant."""
        self.conn.execute("DELETE FROM translation_memory_occurrences WHERE segment_id=?",
                          (segment_id,))

    def current_variant_for(self, segment_id: str) -> MemoryEntry | None:
        row = self.conn.execute(
            "SELECT m.* FROM translation_memory m "
            "JOIN translation_memory_occurrences o ON o.memory_id = m.id "
            "WHERE o.segment_id = ?", (segment_id,)).fetchone()
        return self._row(row) if row else None

    def occurrence_count(self, entry_id: int) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM translation_memory_occurrences WHERE memory_id=?",
            (entry_id,)).fetchone()[0]

    def occurrence_segments(self, entry_id: int) -> list[str]:
        return [r[0] for r in self.conn.execute(
            "SELECT segment_id FROM translation_memory_occurrences WHERE memory_id=? "
            "ORDER BY created_at", (entry_id,))]

    def _row(self, row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"], project_id=row["project_id"], source_language=row["source_language"],
            target_language=row["target_language"], normalized_source=row["normalized_source"],
            source_text=row["source_text"], target_text=row["target_text"],
            locked=bool(row["locked"]), occurrences=self.occurrence_count(row["id"]),
            normalization_version=row["normalization_version"],
            created_at=row["created_at"], updated_at=row["updated_at"])
