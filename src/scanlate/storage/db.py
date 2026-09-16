"""Versioned SQLite schema and connection handling.

Migrations are append-only once a schema has shipped. This one has not: no
database has been persisted or distributed, every instance to date is a
disposable in-memory development database, so migration 1 was consolidated in
place rather than layering a migration 2 onto a schema nobody is running.
From the first stable release, append only.

Migration execution is atomic — the script and its ``user_version`` bump inside
one transaction. A failure rolls the whole thing back and leaves
``user_version`` where it was, so a partially applied schema can never be
marked complete.

Connections run in autocommit mode; multi-statement domain actions take a
transaction explicitly through ``storage.unit_of_work.UnitOfWork``.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

MIGRATIONS: list[str] = [
    # 1 — projects, chapters, pages, segments, state, glossary, memory,
    #     memory occurrences, issue resolutions
    """
    CREATE TABLE projects (
        id                      TEXT PRIMARY KEY,
        name                    TEXT NOT NULL,
        default_source_language TEXT NOT NULL,
        target_language         TEXT NOT NULL,
        created_at              TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE chapters (
        id          TEXT PRIMARY KEY,
        project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        number      INTEGER NOT NULL,
        title       TEXT,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(project_id, number)
    );

    CREATE TABLE pages (
        id          TEXT PRIMARY KEY,
        project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        -- NOT NULL: SQLite would allow many NULL-parent rows to share a number,
        -- so an "unfiled" page gets a real chapter rather than a null one.
        chapter_id  TEXT NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
        number      INTEGER NOT NULL,
        image_path  TEXT,
        width       INTEGER,
        height      INTEGER,
        UNIQUE(chapter_id, number)
    );

    CREATE TABLE segments (
        id          TEXT PRIMARY KEY,
        project_id  TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        -- A page owns its segments: deleting the page deletes them. Re-cropping
        -- or replacing artwork is an UPDATE of the same logical page, which
        -- preserves the segments (see ProjectRepository.replace_page_image).
        page_id     TEXT REFERENCES pages(id) ON DELETE CASCADE,
        seq         INTEGER NOT NULL,
        kind        TEXT NOT NULL,
        language    TEXT,
        source_text TEXT NOT NULL,
        -- Reserved for the image pipeline; unused by the text-only MVP.
        region_json TEXT,
        render_json TEXT,
        mask_ref    TEXT,
        UNIQUE(project_id, seq)
    );

    -- No `locked` column: see storage/models.py. Lock state belongs to the
    -- glossary entry or memory variant that produced the text, and is reported
    -- through origin and annotations rather than copied onto the segment.
    CREATE TABLE segment_state (
        segment_id  TEXT PRIMARY KEY REFERENCES segments(id) ON DELETE CASCADE,
        candidate   TEXT,
        status      TEXT NOT NULL DEFAULT 'machine',
        origin      TEXT NOT NULL DEFAULT 'machine_translation',
        updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE glossary_entries (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id      TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        source_language TEXT NOT NULL,
        source_term     TEXT NOT NULL,
        target_language TEXT NOT NULL,
        target_term     TEXT NOT NULL,
        category        TEXT NOT NULL DEFAULT 'other',
        locked          INTEGER NOT NULL DEFAULT 0,
        notes           TEXT,
        created_at      TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(project_id, source_language, source_term, target_language)
    );
    CREATE INDEX idx_glossary_project ON glossary_entries(project_id, source_language);

    -- One normalized source may hold several approved variants: the same line
    -- can legitimately be translated differently in different scenes.
    CREATE TABLE translation_memory (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id            TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
        source_language       TEXT NOT NULL,
        target_language       TEXT NOT NULL,
        normalized_source     TEXT NOT NULL,
        normalization_version INTEGER NOT NULL DEFAULT 1,
        source_text           TEXT NOT NULL,
        target_text           TEXT NOT NULL,
        locked                INTEGER NOT NULL DEFAULT 0,
        created_at            TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at            TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(project_id, source_language, target_language, normalized_source,
               normalization_version, target_text)
    );
    CREATE INDEX idx_tm_lookup
        ON translation_memory(project_id, source_language, target_language,
                              normalized_source, normalization_version);
    -- At most one canonical (locked) variant per source, per normalization version.
    CREATE UNIQUE INDEX idx_tm_canonical
        ON translation_memory(project_id, source_language, target_language,
                              normalized_source, normalization_version)
        WHERE locked = 1;

    -- CURRENT occurrences only: one row per segment, pointing at the variant
    -- that segment is approved to right now. Reapproving one segment cannot
    -- inflate a count, and switching variants moves the row rather than adding
    -- one. This is not an approval history; if history is ever wanted it needs
    -- its own append-only table.
    CREATE TABLE translation_memory_occurrences (
        memory_id   INTEGER NOT NULL REFERENCES translation_memory(id) ON DELETE CASCADE,
        segment_id  TEXT NOT NULL UNIQUE REFERENCES segments(id) ON DELETE CASCADE,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (memory_id, segment_id)
    );

    -- Keyed by issue fingerprint, not by code: dismissing one register
    -- mismatch must not silence a different one raised later.
    CREATE TABLE issue_resolutions (
        segment_id  TEXT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
        fingerprint TEXT NOT NULL,
        code        TEXT NOT NULL,
        resolution  TEXT NOT NULL DEFAULT 'dismissed',
        resolved_at TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (segment_id, fingerprint)
    );
    CREATE INDEX idx_resolutions_segment ON issue_resolutions(segment_id);
    """,
]


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.isolation_level = None          # autocommit; UnitOfWork owns transactions
    conn.execute("PRAGMA foreign_keys = ON")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations. Each one is all-or-nothing."""
    for index in range(schema_version(conn), len(MIGRATIONS)):
        script = f"BEGIN;\n{MIGRATIONS[index]}\nPRAGMA user_version = {index + 1};\nCOMMIT;"
        try:
            conn.executescript(script)
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    return schema_version(conn)


def schema_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]
