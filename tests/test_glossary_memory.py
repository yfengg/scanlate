"""Glossary and translation-memory behaviour (stage 1 of the backend order)."""
from __future__ import annotations

import pytest

from scanlate.core.types import SegmentKind, Severity
from scanlate.glossary import Category, GlossaryEntry, GlossaryStore, enforce
from scanlate.memory import TranslationMemory
from scanlate.storage import (Chapter, Page, Project, ProjectRepository, Segment,
                              UnitOfWork, connect)

VARIANTS = {"兄貴": ["big bro", "bro", "brother"], "先輩": ["senpai", "upperclassman"]}


@pytest.fixture()
def store():
    conn = connect()
    uow = UnitOfWork(conn)
    repo = ProjectRepository(conn, uow)
    repo.create_project(Project("p1", "Demo", "ja", "en"))
    repo.add_segment(Segment("s1", "p1", 1, SegmentKind.DIALOGUE, "兄貴、助けてくれ！", language="ja"))
    glossary = GlossaryStore(conn, uow)
    glossary.add(GlossaryEntry(None, "p1", "ja", "兄貴", "en", "Aniki",
                               Category.RELATIONSHIP, locked=True))
    glossary.add(GlossaryEntry(None, "p1", "ja", "田中", "en", "Tanaka",
                               Category.CHARACTER, locked=True))
    glossary.add(GlossaryEntry(None, "p1", "ja", "部活", "en", "club", Category.OTHER))
    return conn, uow, repo, glossary


def test_glossary_finds_every_occurrence_with_spans(store):
    _, _, _, glossary = store
    hits = glossary.find_in("田中！田中はどこ？兄貴も来る。", "p1", "ja", "en")
    surfaces = [(h.entry.source_term, h.span.start, h.span.end) for h in hits]
    assert surfaces == [("田中", 0, 2), ("田中", 3, 5), ("兄貴", 9, 11)]


def test_locked_term_beats_machine_wording(store):
    _, _, _, glossary = store
    hits = glossary.find_in("兄貴、助けてくれ！", "p1", "ja", "en")
    result = enforce("Big bro, help me out!", hits, VARIANTS)
    assert result.text == "Aniki, help me out!"
    assert result.corrections == ["big bro → Aniki"]
    assert [i.code for i in result.issues] == ["terminology.corrected"]


def test_unsafe_replacement_warns_instead_of_guessing(store):
    _, _, _, glossary = store
    hits = glossary.find_in("兄貴、助けてくれ！", "p1", "ja", "en")

    ambiguous = enforce("Where's big bro? Ask bro.", hits, VARIANTS)
    assert ambiguous.text == "Where's big bro? Ask bro."      # untouched
    assert ambiguous.issues[0].code == "terminology.conflict"
    assert ambiguous.issues[0].severity is Severity.WARNING

    missing = enforce("Help me out!", hits, VARIANTS)
    assert missing.issues[0].data["reason"] == "no candidate rendering found"
    assert [a.kind for a in missing.issues[0].actions] == ["use", "dismiss"]


def test_unlocked_entries_are_not_enforced(store):
    _, _, _, glossary = store
    hits = glossary.find_in("部活に行く", "p1", "ja", "en")
    result = enforce("Heading to practice.", hits, {})
    assert result.issues == [] and not result.changed


def test_canonical_memory_is_authoritative_and_unlocked_variants_are_precedent(store):
    conn, uow, repo, _ = store
    repo.add_segment(Segment("s2", "p1", 2, SegmentKind.DIALOGUE, "行くぞ！", language="ja"))
    tm = TranslationMemory(conn, uow)

    tm.record_approved("p1", "ja", "en", "行くぞ！", "行くぞ！", "Let's go!", "s2")
    tm.record_approved("p1", "ja", "en", "行くぞ！", "行くぞ！", "Move out.", "s1")
    assert tm.canonical("p1", "ja", "en", "行くぞ！") is None      # precedent, not truth
    assert len(tm.exact("p1", "ja", "en", "行くぞ！")) == 2

    winner = tm.variant("p1", "ja", "en", "行くぞ！", "Let's go!")
    tm.set_locked(winner.id, True)
    canonical = tm.canonical("p1", "ja", "en", "行くぞ！")
    assert canonical.target_text == "Let's go!" and canonical.canonical
    assert tm.exact("p1", "ja", "en", "行くぞ！")[0].target_text == "Let's go!"   # canonical first


def test_memory_lookups_are_scoped_to_the_normalization_version(store):
    conn, uow, repo = store[0], store[1], store[2]
    repo.add_segment(Segment("s3", "p1", 3, SegmentKind.DIALOGUE, "行くぞ！", language="ja"))
    tm = TranslationMemory(conn, uow)
    tm.record_approved("p1", "ja", "en", "行くぞ！", "行くぞ！", "Let\'s go!", "s3",
                       locked=True, normalization_version=1)

    # A later normalizer produces the same string but is a different contract.
    assert tm.exact("p1", "ja", "en", "行くぞ！", normalization_version=2) == []
    assert tm.canonical("p1", "ja", "en", "行くぞ！", normalization_version=2) is None
    assert tm.variant("p1", "ja", "en", "行くぞ！", "Let\'s go!", normalization_version=2) is None
    assert tm.fuzzy("p1", "ja", "en", "行くぞ", normalization_version=2) == []

    # v1 still answers v1, and both versions can hold their own canonical row.
    assert tm.canonical("p1", "ja", "en", "行くぞ！", normalization_version=1).target_text == "Let\'s go!"
    migrated = tm.record_approved("p1", "ja", "en", "行くぞ！", "行くぞ！", "Let\'s go!", "s3",
                                  locked=True, normalization_version=2)
    assert migrated.normalization_version == 2
    assert tm.canonical("p1", "ja", "en", "行くぞ！", normalization_version=2) is not None


def test_segment_counts_only_toward_its_current_variant(store):
    conn, uow, repo = store[0], store[1], store[2]
    repo.add_segment(Segment("s4", "p1", 4, SegmentKind.DIALOGUE, "行くぞ！", language="ja"))
    tm = TranslationMemory(conn, uow)

    a = tm.record_approved("p1", "ja", "en", "行くぞ！", "行くぞ！", "Let\'s go!", "s4")
    assert a.occurrences == 1

    # Reopened and approved to a different variant.
    b = tm.record_approved("p1", "ja", "en", "行くぞ！", "行くぞ！", "Move out.", "s4")
    assert b.occurrences == 1
    assert tm.occurrence_count(a.id) == 0            # no longer endorses the old variant
    assert tm.occurrence_segments(b.id) == ["s4"]
    assert tm.current_variant_for("s4").target_text == "Move out."
    assert conn.execute(
        "SELECT COUNT(*) FROM translation_memory_occurrences WHERE segment_id='s4'"
    ).fetchone()[0] == 1

    # Both variants remain on record; only the occurrence moved.
    assert len(tm.exact("p1", "ja", "en", "行くぞ！")) == 2


def test_pages_list_in_chapter_then_page_order(store):
    conn, uow, repo = store[0], store[1], store[2]
    repo.create_chapter(Chapter("c2", "p1", 2))
    repo.create_chapter(Chapter("c1", "p1", 1))
    repo.create_page(Page("b", "p1", 1, chapter_id="c2"))
    repo.create_page(Page("a2", "p1", 2, chapter_id="c1"))
    repo.create_page(Page("a1", "p1", 1, chapter_id="c1"))
    assert [p.id for p in repo.list_pages("p1")] == ["a1", "a2", "b"]
