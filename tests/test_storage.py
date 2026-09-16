"""Storage invariants.

These are the gate on the rest of the backend: pipeline work continues only
while these hold.
"""
from __future__ import annotations

import sqlite3

import pytest

from scanlate.core.types import SegmentKind, issue_fingerprint
from scanlate.imaging import (BoundingBox, Polygon, Region, RegionKind,
                              RenderSettings, TextOrientation)
from scanlate.memory import CanonicalConflict, TranslationMemory
from scanlate.storage import (Chapter, Origin, Page, Project, ProjectRepository,
                              Segment, SegmentState, Status, UnitOfWork, connect)
from scanlate.storage.db import MIGRATIONS, migrate, schema_version


@pytest.fixture()
def db():
    conn = connect()
    uow = UnitOfWork(conn)
    repo = ProjectRepository(conn, uow)
    repo.create_project(Project("p1", "Demo", "ja", "en"))
    return conn, uow, repo


def _segment(repo, seg_id="s1", seq=1, text="テスト", language="ja", kind="dialogue", **kw):
    return repo.add_segment(Segment(seg_id, "p1", seq, SegmentKind(kind), text,
                                    language=language, **kw))


# --- translation memory ------------------------------------------------
def test_same_source_supports_multiple_approved_variants(db):
    conn, uow, repo = db
    _segment(repo, "s1", 1, "行くぞ")
    _segment(repo, "s2", 2, "行くぞ")
    tm = TranslationMemory(conn, uow)
    tm.record_approved("p1", "ja", "en", "行くぞ", "行くぞ", "Let's go!", "s1")
    tm.record_approved("p1", "ja", "en", "行くぞ", "行くぞ", "Here we go.", "s2")

    variants = tm.exact("p1", "ja", "en", "行くぞ")
    assert {v.target_text for v in variants} == {"Let's go!", "Here we go."}
    assert all(not v.locked for v in variants)


def test_only_one_canonical_variant_per_source(db):
    conn, uow, repo = db
    _segment(repo)
    tm = TranslationMemory(conn, uow)
    tm.record_approved("p1", "ja", "en", "行くぞ", "行くぞ", "Let's go!", "s1", locked=True)

    with pytest.raises(CanonicalConflict):
        tm.record_approved("p1", "ja", "en", "行くぞ", "行くぞ", "Here we go.", "s1", locked=True)

    other = tm.record_approved("p1", "ja", "en", "行くぞ", "行くぞ", "Here we go.", "s1")
    with pytest.raises(CanonicalConflict):
        tm.set_locked(other.id, True)

    assert tm.canonical("p1", "ja", "en", "行くぞ").target_text == "Let's go!"
    assert conn.execute(
        "SELECT COUNT(*) FROM translation_memory WHERE normalized_source='行くぞ' AND locked=1"
    ).fetchone()[0] == 1


def test_reapproving_same_segment_does_not_inflate_occurrences(db):
    conn, uow, repo = db
    _segment(repo)
    tm = TranslationMemory(conn, uow)
    for _ in range(3):
        entry = tm.record_approved("p1", "ja", "en", "テスト", "テスト", "Test", "s1")
    assert entry.occurrences == 1
    assert tm.occurrence_segments(entry.id) == ["s1"]


def test_distinct_segments_count_as_separate_occurrences(db):
    conn, uow, repo = db
    _segment(repo, "s1", 1, "テスト")
    _segment(repo, "s2", 2, "テスト")
    tm = TranslationMemory(conn, uow)
    tm.record_approved("p1", "ja", "en", "テスト", "テスト", "Test", "s1")
    entry = tm.record_approved("p1", "ja", "en", "テスト", "テスト", "Test", "s2")
    assert entry.occurrences == 2
    assert sorted(tm.occurrence_segments(entry.id)) == ["s1", "s2"]


def test_fuzzy_matches_are_never_authoritative(db):
    conn, uow, repo = db
    _segment(repo)
    tm = TranslationMemory(conn, uow)
    tm.record_approved("p1", "ko", "en", "저희가 처리하겠습니다.", "저희가 처리하겠습니다.",
                       "We'll take care of it.", "s1", locked=True)

    near = "저희가 처리하겠어요."
    assert tm.exact("p1", "ko", "en", near) == []
    assert tm.canonical("p1", "ko", "en", near) is None

    matches = tm.fuzzy("p1", "ko", "en", near)
    assert matches and not matches[0].exact
    assert 0 < matches[0].score < 1.0


# --- issue resolutions -------------------------------------------------
def test_dismissal_does_not_suppress_a_new_issue_of_the_same_code(db):
    conn, uow, repo = db
    _segment(repo, "s1", 1, "うるせぇな")

    first = issue_fingerprint("register.mismatch", "うるせぇな", "Please be quiet.", "crude/polite")
    repo.resolve_issue("s1", first, "register.mismatch")
    assert first in repo.resolved_fingerprints("s1")

    # Translation changes; the mismatch that reappears is a different instance.
    second = issue_fingerprint("register.mismatch", "うるせぇな", "I would be obliged.", "crude/formal")
    assert second != first
    assert second not in repo.resolved_fingerprints("s1")

    # The identical situation stays dismissed.
    assert issue_fingerprint("register.mismatch", "うるせぇな", "Please be quiet.",
                             "crude/polite") in repo.resolved_fingerprints("s1")


# --- segments ----------------------------------------------------------
def test_region_and_render_survive_round_trip(db):
    conn, uow, repo = db
    region = Region(kind=RegionKind.SFX, box=BoundingBox(10, 20, 30, 40),
                    polygon=Polygon([(10, 20), (40, 20), (40, 60)]),
                    reading_order=3, orientation=TextOrientation.VERTICAL_RL)
    render = RenderSettings(font_id="comic", font_size=18.5, line_height=1.4,
                            alignment="left", rotation=2.5,
                            orientation=TextOrientation.VERTICAL_RL, padding=(1, 2, 3, 4))
    _segment(repo, "s1", 1, "ドクン", kind="sfx", region=region, render=render,
             mask_ref="masks/p1.png")

    loaded = repo.get_segment("s1")
    assert loaded.region == region
    assert loaded.render == render
    assert loaded.mask_ref == "masks/p1.png"
    assert isinstance(loaded.region.kind, RegionKind)
    assert isinstance(loaded.render.padding, tuple)


def test_add_segment_is_atomic_when_state_insert_fails(db):
    conn, uow, repo = db
    _segment(repo, "s1", 1)

    class FailingOnStateInsert:
        """Delegates to the real connection, but the state insert blows up."""

        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *args):
            if "INSERT INTO segment_state" in sql:
                raise sqlite3.IntegrityError("simulated state failure")
            return self._inner.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    repo.conn = FailingOnStateInsert(conn)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            repo.add_segment(Segment("s2", "p1", 2, SegmentKind.DIALOGUE, "失敗"))
    finally:
        repo.conn = conn

    assert repo.get_segment("s2") is None
    assert conn.execute("SELECT COUNT(*) FROM segments WHERE id='s2'").fetchone()[0] == 0
    assert not conn.in_transaction


def test_ordering_uniqueness_is_enforced(db):
    conn, uow, repo = db
    _segment(repo, "s1", 1)
    with pytest.raises(sqlite3.IntegrityError):
        _segment(repo, "s2", 1)                      # duplicate project-wide reading order

    repo.create_chapter(Chapter("c1", "p1", 1))
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_chapter(Chapter("c2", "p1", 1))   # duplicate chapter number

    repo.create_page(Page("pg1", "p1", 1, chapter_id="c1"))
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_page(Page("pg2", "p1", 1, chapter_id="c1"))   # duplicate page in chapter
    with pytest.raises(sqlite3.IntegrityError):
        repo.create_page(Page("pg3", "p1", 9, chapter_id=None))   # a page must have a chapter


def test_mixed_language_segments_persist(db):
    conn, uow, repo = db
    _segment(repo, "s1", 1, "田中先輩", language="ja")
    _segment(repo, "s2", 2, "형, 진짜 괜찮아?", language="ko")
    _segment(repo, "s3", 3, "SALE", language=None)

    segments = repo.list_segments("p1")
    assert [s.language for s in segments] == ["ja", "ko", None]
    project = repo.get_project("p1")
    assert segments[2].resolved_language(project.default_source_language) == "ja"


def test_local_context_carries_approved_translations(db):
    conn, uow, repo = db
    _segment(repo, "s1", 1, "行くぞ")
    _segment(repo, "s2", 2, "テスト")
    _segment(repo, "s3", 3, "형", language="ko")
    repo.save_state("s1", SegmentState("Let's go!", Status.APPROVED, Origin.MANUAL))

    context = repo.local_context(repo.get_segment("s2"))
    assert [c.id for c in context.preceding] == ["s1"]
    assert context.preceding[0].approved_translation == "Let's go!"
    assert [c.id for c in context.following] == ["s3"]
    assert context.following[0].language == "ko"
    assert context.following[0].approved_translation is None   # not approved yet


# --- integrity ---------------------------------------------------------
def test_foreign_key_cascades(db):
    conn, uow, repo = db
    repo.create_chapter(Chapter("c1", "p1", 1))
    repo.create_page(Page("pg1", "p1", 1, chapter_id="c1"))
    repo.create_page(Page("pg2", "p1", 2, chapter_id="c1"))
    _segment(repo, "s1", 1, page_id="pg1")
    _segment(repo, "s2", 2, page_id="pg2")
    tm = TranslationMemory(conn, uow)
    entry = tm.record_approved("p1", "ja", "en", "テスト", "テスト", "Test", "s1")
    repo.resolve_issue("s1", "fp1", "register.mismatch")

    # A page owns its segments.
    conn.execute("DELETE FROM pages WHERE id='pg1'")
    assert repo.get_segment("s1") is None
    assert repo.get_segment("s2") is not None

    # Replacing artwork is an update, so the logical page and its segments survive.
    repo.replace_page_image("pg2", "pages/002-rev2.png", width=1400, height=2000)
    assert repo.get_segment("s2").page_id == "pg2"
    assert repo.list_pages("p1")[0].image_path == "pages/002-rev2.png"

    # Deleting the project removes everything hanging off it.
    conn.execute("DELETE FROM projects WHERE id='p1'")
    for table in ("segments", "segment_state", "chapters", "pages", "translation_memory",
                  "translation_memory_occurrences", "issue_resolutions"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0, table
    assert tm.get(entry.id) is None


def test_failed_migration_leaves_no_partial_schema():
    conn = sqlite3.connect(":memory:")
    conn.isolation_level = None
    MIGRATIONS.append("CREATE TABLE fine (x); CREATE TABLE fine (x);")   # duplicate table
    try:
        with pytest.raises(sqlite3.OperationalError):
            migrate(conn)
        assert schema_version(conn) == len(MIGRATIONS) - 1
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='fine'").fetchone()[0] == 0
    finally:
        MIGRATIONS.pop()


def test_unit_of_work_rolls_back_a_whole_domain_action(db):
    conn, uow, repo = db
    _segment(repo, "s1", 1)
    tm = TranslationMemory(conn, uow)

    with pytest.raises(RuntimeError):
        with uow.transaction():
            repo.save_state("s1", SegmentState("Approved text", Status.APPROVED, Origin.MANUAL))
            tm.record_approved("p1", "ja", "en", "テスト", "テスト", "Approved text", "s1")
            raise RuntimeError("glossary step failed")

    assert repo.get_segment("s1").state.status is Status.MACHINE
    assert tm.exact("p1", "ja", "en", "テスト") == []
