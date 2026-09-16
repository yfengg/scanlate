"""Persistence for projects, chapters, pages, segments and segment state.

These methods issue SQL and nothing else — no commits. The caller's
``UnitOfWork`` decides when work becomes durable, so a domain action spanning
several tables succeeds or fails as a whole. Connections are in autocommit
mode, so a lone write outside a transaction still lands.

The API depends on this module; SQL row shapes never leave it.
"""
from __future__ import annotations

import json
import sqlite3

from ..core.types import SegmentKind
from ..imaging.layout import (region_from_dict, region_to_dict,
                              render_from_dict, render_to_dict)
from .models import (Chapter, LocalContext, Origin, Page, Project, Segment,
                     SegmentContext, SegmentState, Status)
from .unit_of_work import UnitOfWork


class ProjectRepository:
    def __init__(self, conn: sqlite3.Connection, uow: UnitOfWork | None = None):
        self.conn = conn
        self.uow = uow or UnitOfWork(conn)

    # --- projects ------------------------------------------------------
    def create_project(self, project: Project) -> Project:
        self.conn.execute(
            "INSERT INTO projects (id, name, default_source_language, target_language) "
            "VALUES (?, ?, ?, ?)",
            (project.id, project.name, project.default_source_language, project.target_language))
        return project

    def get_project(self, project_id: str) -> Project | None:
        row = self.conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return Project(row["id"], row["name"], row["default_source_language"],
                       row["target_language"]) if row else None

    def list_projects(self) -> list[Project]:
        return [Project(r["id"], r["name"], r["default_source_language"], r["target_language"])
                for r in self.conn.execute("SELECT * FROM projects ORDER BY created_at")]

    # --- chapters and pages --------------------------------------------
    def create_chapter(self, chapter: Chapter) -> Chapter:
        self.conn.execute(
            "INSERT INTO chapters (id, project_id, number, title) VALUES (?, ?, ?, ?)",
            (chapter.id, chapter.project_id, chapter.number, chapter.title))
        return chapter

    def list_chapters(self, project_id: str) -> list[Chapter]:
        return [Chapter(r["id"], r["project_id"], r["number"], r["title"])
                for r in self.conn.execute(
                    "SELECT * FROM chapters WHERE project_id = ? ORDER BY number", (project_id,))]

    def create_page(self, page: Page) -> Page:
        self.conn.execute(
            "INSERT INTO pages (id, project_id, chapter_id, number, image_path, width, height) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (page.id, page.project_id, page.chapter_id, page.number, page.image_path,
             page.width, page.height))
        return page

    def get_page(self, page_id: str) -> Page | None:
        row = self.conn.execute("SELECT * FROM pages WHERE id = ?", (page_id,)).fetchone()
        return self._row_to_page(row) if row else None

    def find_page(self, chapter_id: str, number: int) -> Page | None:
        row = self.conn.execute("SELECT * FROM pages WHERE chapter_id = ? AND number = ?",
                                (chapter_id, number)).fetchone()
        return self._row_to_page(row) if row else None

    @staticmethod
    def _row_to_page(row: sqlite3.Row) -> Page:
        return Page(row["id"], row["project_id"], row["number"], row["chapter_id"],
                    row["image_path"], row["width"], row["height"])

    def set_page_image(self, page_id: str, image_path: str, width: int, height: int) -> None:
        self.conn.execute("UPDATE pages SET image_path=?, width=?, height=? WHERE id=?",
                          (image_path, width, height, page_id))

    def list_pages(self, project_id: str) -> list[Page]:
        """Project-wide reading order: chapter number, then page number."""
        return [Page(r["id"], r["project_id"], r["number"], r["chapter_id"], r["image_path"],
                     r["width"], r["height"])
                for r in self.conn.execute(
                    "SELECT p.* FROM pages p JOIN chapters c ON c.id = p.chapter_id "
                    "WHERE p.project_id = ? ORDER BY c.number, p.number", (project_id,))]

    def replace_page_image(self, page_id: str, image_path: str, width: int | None = None,
                           height: int | None = None) -> None:
        """Swap the artwork on an existing page.

        Re-cropping or re-scanning updates the logical page in place, so its
        segments and their translations survive. Deleting and recreating the
        page would cascade them away — that is not this workflow.
        """
        self.conn.execute("UPDATE pages SET image_path = ?, width = ?, height = ? WHERE id = ?",
                          (image_path, width, height, page_id))

    # --- segments ------------------------------------------------------
    def add_segment(self, segment: Segment) -> Segment:
        """Segment row and state row are one action; neither lands alone."""
        with self.uow.transaction():
            self.conn.execute(
                "INSERT INTO segments (id, project_id, page_id, seq, kind, language, source_text, "
                "region_json, render_json, mask_ref) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (segment.id, segment.project_id, segment.page_id, segment.seq,
                 SegmentKind(segment.kind).value, segment.language, segment.source_text,
                 _dump(region_to_dict(segment.region)), _dump(render_to_dict(segment.render)),
                 segment.mask_ref))
            self.conn.execute(
                "INSERT INTO segment_state (segment_id, candidate, status, origin) "
                "VALUES (?, ?, ?, ?)",
                (segment.id, segment.state.candidate, segment.state.status.value,
                 segment.state.origin.value))
        return segment

    _SELECT = ("SELECT s.*, st.candidate, st.status, st.origin, st.updated_at "
               "FROM segments s LEFT JOIN segment_state st ON st.segment_id = s.id ")

    def _row_to_segment(self, row: sqlite3.Row) -> Segment:
        return Segment(
            id=row["id"], project_id=row["project_id"], seq=row["seq"],
            kind=SegmentKind(row["kind"]), source_text=row["source_text"],
            language=row["language"], page_id=row["page_id"], mask_ref=row["mask_ref"],
            region=region_from_dict(_load(row["region_json"])),
            render=render_from_dict(_load(row["render_json"])),
            state=SegmentState(
                candidate=row["candidate"],
                status=Status(row["status"] or "machine"),
                origin=Origin(row["origin"] or "machine_translation"),
                updated_at=row["updated_at"]))

    def get_segment(self, segment_id: str) -> Segment | None:
        row = self.conn.execute(self._SELECT + "WHERE s.id = ?", (segment_id,)).fetchone()
        return self._row_to_segment(row) if row else None

    def list_segments(self, project_id: str) -> list[Segment]:
        rows = self.conn.execute(self._SELECT + "WHERE s.project_id = ? ORDER BY s.seq",
                                 (project_id,))
        return [self._row_to_segment(r) for r in rows]

    def chapter_of(self, segment: Segment) -> str | None:
        if segment.page_id is None:
            return None
        row = self.conn.execute("SELECT chapter_id FROM pages WHERE id = ?",
                                (segment.page_id,)).fetchone()
        return row["chapter_id"] if row else None

    def local_context(self, segment: Segment, window: int = 2) -> LocalContext:
        """Nearby segments in reading order — the pipeline's only context source.

        Scoped to the segment's own chapter: the last line of chapter 3 and the
        first line of chapter 4 are adjacent by sequence number but are not
        each other's context. Segments with no page fall back to their
        unchaptered siblings.

        Deliberately small: neighbouring lines with their approved translations
        where they exist. No plot summaries, no project-wide digest.
        """
        chapter_id = self.chapter_of(segment)
        if chapter_id is None:
            scope = ("AND s.page_id IS NULL", ())
        else:
            scope = ("AND s.page_id IN (SELECT id FROM pages WHERE chapter_id = ?)",
                     (chapter_id,))

        def fetch(comparison: str, order: str) -> list[SegmentContext]:
            rows = self.conn.execute(
                "SELECT s.id, s.source_text, s.language, s.kind, st.candidate, st.status "
                "FROM segments s LEFT JOIN segment_state st ON st.segment_id = s.id "
                f"WHERE s.project_id = ? AND s.seq {comparison} ? {scope[0]} "
                f"ORDER BY s.seq {order} LIMIT ?",
                (segment.project_id, segment.seq, *scope[1], window)).fetchall()
            return [SegmentContext(
                id=r["id"], source_text=r["source_text"], language=r["language"],
                kind=SegmentKind(r["kind"]),
                approved_translation=r["candidate"] if r["status"] == Status.APPROVED.value else None)
                for r in rows]

        return LocalContext(preceding=list(reversed(fetch("<", "DESC"))), following=fetch(">", "ASC"))

    def next_seq(self, project_id: str) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM segments WHERE project_id=?",
                                (project_id,)).fetchone()
        return row[0]

    def set_source_text(self, segment_id: str, source_text: str) -> None:
        self.conn.execute("UPDATE segments SET source_text = ? WHERE id = ?",
                          (source_text, segment_id))

    def set_kind(self, segment_id: str, kind: SegmentKind) -> None:
        self.conn.execute("UPDATE segments SET kind = ? WHERE id = ?",
                          (SegmentKind(kind).value, segment_id))

    def set_region(self, segment_id: str, region, render=None) -> None:
        """Geometry is stored in image coordinates, never viewport coordinates."""
        self.conn.execute("UPDATE segments SET region_json = ?, render_json = ? WHERE id = ?",
                          (_dump(region_to_dict(region)),
                           _dump(render_to_dict(render)) if render else None, segment_id))

    def delete_segment(self, segment_id: str) -> None:
        self.conn.execute("DELETE FROM segments WHERE id = ?", (segment_id,))

    def detach_from_page(self, segment_id: str) -> None:
        self.conn.execute("UPDATE segments SET page_id = NULL, region_json = NULL WHERE id = ?",
                          (segment_id,))

    def list_page_segments(self, page_id: str) -> list[Segment]:
        rows = self.conn.execute(self._SELECT + "WHERE s.page_id = ? ORDER BY s.seq", (page_id,))
        return [self._row_to_segment(r) for r in rows]

    def save_state(self, segment_id: str, state: SegmentState) -> None:
        self.conn.execute(
            "INSERT INTO segment_state (segment_id, candidate, status, origin, updated_at) "
            "VALUES (?, ?, ?, ?, datetime('now')) "
            "ON CONFLICT(segment_id) DO UPDATE SET candidate=excluded.candidate, "
            "status=excluded.status, origin=excluded.origin, updated_at=datetime('now')",
            (segment_id, state.candidate, state.status.value, state.origin.value))

    def set_language(self, segment_id: str, language: str) -> None:
        self.conn.execute("UPDATE segments SET language = ? WHERE id = ?", (language, segment_id))

    # --- issue resolutions ---------------------------------------------
    def resolve_issue(self, segment_id: str, fingerprint: str, code: str,
                      resolution: str = "dismissed") -> None:
        """Resolutions are keyed by fingerprint, so a later, genuinely different
        issue carrying the same code is not silenced by this one."""
        self.conn.execute(
            "INSERT INTO issue_resolutions (segment_id, fingerprint, code, resolution) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(segment_id, fingerprint) DO UPDATE SET "
            "resolution=excluded.resolution, resolved_at=datetime('now')",
            (segment_id, fingerprint, code, resolution))

    def resolved_fingerprints(self, segment_id: str) -> set[str]:
        return {r[0] for r in self.conn.execute(
            "SELECT fingerprint FROM issue_resolutions WHERE segment_id = ?", (segment_id,))}


def _dump(data: dict | None) -> str | None:
    return json.dumps(data, ensure_ascii=False) if data is not None else None


def _load(raw: str | None) -> dict | None:
    return json.loads(raw) if raw else None
