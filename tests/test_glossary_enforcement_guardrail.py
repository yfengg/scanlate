"""Guardrail: locked-glossary automatic rewriting must stay exactly as narrow
as documented, in CLAUDE.md and in ``glossary/store.py``'s own docstring —
"a correction is applied only when the wrong rendering appears exactly once
as a whole word and the approved rendering is absent. Anything less certain
becomes an issue for the translator."

This file is not exercising new behaviour. It exists so a future change to
``enforce()`` cannot silently widen automatic rewriting without a very
visible, deliberately-labelled test failure. Every "must NOT auto-rewrite"
case here should fail loudly, not quietly start passing, if that guarantee
is ever relaxed.

Do not broaden what's asserted here to match a future change to
``enforce()``: if one of these starts failing, the fix is almost always in
the code that changed, not in this test.
"""
from __future__ import annotations

import pytest

from scanlate.core.types import SegmentKind, Severity
from scanlate.glossary import Category, GlossaryEntry, GlossaryStore, enforce
from scanlate.storage import Project, ProjectRepository, Segment, UnitOfWork, connect

VARIANTS = {"兄貴": ["big bro", "bro", "brother"], "先輩": ["senpai", "upperclassman"]}


@pytest.fixture()
def glossary():
    conn = connect()
    uow = UnitOfWork(conn)
    repo = ProjectRepository(conn, uow)
    repo.create_project(Project("p1", "Demo", "ja", "en"))
    repo.add_segment(Segment("s1", "p1", 1, SegmentKind.DIALOGUE, "兄貴、助けてくれ！", language="ja"))
    store = GlossaryStore(conn, uow)
    store.add(GlossaryEntry(None, "p1", "ja", "兄貴", "en", "Aniki", Category.RELATIONSHIP, locked=True))
    store.add(GlossaryEntry(None, "p1", "ja", "先輩", "en", "senpai", Category.TITLE, locked=True))
    return store


def hits_for(glossary, source_term_text):
    return glossary.find_in(source_term_text, "p1", "ja", "en")


# --- the one case that IS allowed -----------------------------------------
def test_allowed_the_single_provably_unambiguous_case(glossary):
    """The only shape enforce() may rewrite: one whole-word occurrence of a
    known variant, the locked rendering absent, correct=True."""
    hits = hits_for(glossary, "兄貴、助けてくれ！")
    result = enforce("Big bro, help me out!", hits, VARIANTS)
    assert result.text == "Aniki, help me out!"
    assert result.changed is True
    assert result.corrections == ["big bro → Aniki"]
    assert [i.code for i in result.issues] == ["terminology.corrected"]


# --- guardrails: none of the following may ever auto-rewrite --------------
def test_guardrail_multiple_occurrences_of_the_same_variant(glossary):
    hits = hits_for(glossary, "兄貴、助けてくれ！")
    result = enforce("Bro! Where's bro when you need him?", hits, VARIANTS)
    assert result.text == "Bro! Where's bro when you need him?"
    assert result.changed is False
    assert result.issues[0].code == "terminology.conflict"
    assert result.issues[0].data["reason"] == "several possible replacements"


def test_guardrail_overlapping_nested_and_separate_occurrences(glossary):
    """"bro" is a literal substring of "big bro" (nested), but here it also
    occurs a second time on its own -- one nested nickname plus one separate,
    standalone use is still two real occurrences, not one."""
    hits = hits_for(glossary, "兄貴、助けてくれ！")
    result = enforce("Big bro said bro would come.", hits, VARIANTS)
    assert result.text == "Big bro said bro would come."
    assert result.changed is False
    assert result.issues[0].code == "terminology.conflict"


def test_guardrail_substring_only_match_is_not_a_match_at_all(glossary):
    """"bro" inside "broader" is not a whole-word occurrence of the variant --
    it must not be counted, and must not be silently corrected into
    "Aniikoader" or anything else. The absence of any real occurrence is
    itself reported, not silently ignored."""
    hits = hits_for(glossary, "兄貴、助けてくれ！")
    result = enforce("This is a broader topic than I expected.", hits, VARIANTS)
    assert result.text == "This is a broader topic than I expected."
    assert result.changed is False
    assert result.issues[0].code == "terminology.conflict"
    assert result.issues[0].data["reason"] == "no candidate rendering found"


def test_guardrail_conflicting_entries_in_the_same_text_are_resolved_independently(glossary):
    """Two different locked entries in one candidate: one resolvable, one
    not. The resolvable one must still be corrected, and the unresolved one
    must still be reported -- neither one's outcome may leak into the
    other's."""
    hits = hits_for(glossary, "兄貴、助けてくれ！先輩！")
    variants = {**VARIANTS, "先輩": ["upperclassman", "senior"]}  # neither equals the target "senpai"
    result = enforce("Big bro, help! Also hi upperclassman and senior.", hits, variants)
    assert "Aniki" in result.text            # 兄貴: single unambiguous match, corrected
    assert "big bro" not in result.text.lower()
    assert "upperclassman" in result.text and "senior" in result.text  # 先輩: left alone
    codes = sorted(i.code for i in result.issues)
    assert codes == ["terminology.conflict", "terminology.corrected"]  # both entries reported,
                                                                        # correctly, independently


def test_guardrail_uncertain_morphology_is_not_treated_as_present_or_matched(glossary):
    """"senpai's" (possessive) is not the registered exact term "senpai" --
    enforce() must not treat the inflected form as satisfying the locked
    rendering, and must not mistake it for a correctable variant either."""
    hits = hits_for(glossary, "先輩！")
    result = enforce("Ask senpai's opinion first.", hits, VARIANTS)
    assert result.text == "Ask senpai's opinion first."
    assert result.changed is False
    assert result.issues[0].code == "terminology.conflict"
    assert result.issues[0].data["reason"] == "no candidate rendering found"


def test_guardrail_human_owned_text_is_never_rewritten_even_when_unambiguous(glossary):
    """The exact scenario from test_allowed_the_single_provably_unambiguous_case,
    but with correct=False -- the mode used for text the translator already
    edited or approved (see pipeline.py). Report-only, even for a case that
    would otherwise qualify for automatic correction."""
    hits = hits_for(glossary, "兄貴、助けてくれ！")
    result = enforce("Big bro, help me out!", hits, VARIANTS, correct=False)
    assert result.text == "Big bro, help me out!"
    assert result.changed is False
    assert result.issues[0].code == "terminology.conflict"
