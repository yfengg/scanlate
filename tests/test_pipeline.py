"""Pipeline precedence and approval integration.

These pin the decision order: locked project decisions beat machine output,
precedent is offered rather than imposed, fuzzy memory never autofills.
"""
from __future__ import annotations

import pytest

from scanlate.core.types import LanguagePair, SegmentKind, Severity, issue_fingerprint
from scanlate.fixtures import build_services, seed_demo
from scanlate.glossary.store import Category, GlossaryEntry
from scanlate.memory.store import CanonicalConflict
from scanlate.pipeline.approval import ApprovalError
from scanlate.storage.models import Origin, Segment, SegmentState, Status
from scanlate.workbench.view import AnnotationType


@pytest.fixture()
def app():
    services = build_services()
    seed_demo(services)
    return services


def view(app, segment_id):
    return app.pipeline.process(app.repository.get_segment(segment_id)).view


def add_segment(app, seg_id, seq, text, language="ja", kind=SegmentKind.DIALOGUE):
    return app.repository.add_segment(
        Segment(seg_id, "demo", seq, kind, text, language=language))


# --- precedence --------------------------------------------------------
def test_canonical_memory_bypasses_machine_translation(app):
    result = view(app, "p013-b2")
    assert result.candidate == "We'll take care of it."      # not the backend's "We will handle it."
    assert result.origin is Origin.TRANSLATION_MEMORY
    assert result.locked is True
    assert "backend" not in result.features


def test_unlocked_exact_memory_is_offered_not_imposed(app):
    add_segment(app, "extra", 20, "うるせぇな…今忙しいんだよ。")
    app.memory.record_approved("demo", "ja", "en", "うるせぇな…今忙しいんだよ。",
                               "うるせぇな…今忙しいんだよ。", "Pipe down, I'm swamped.", "extra")

    result = view(app, "p012-b2")
    assert result.candidate == "Please be quiet, I am busy right now."   # MT still wins
    assert result.origin is Origin.MACHINE_TRANSLATION
    assert result.locked is False
    assert "Pipe down, I'm swamped." in [a.text for a in result.alternatives]
    precedent = next(w for w in result.warnings if w.code == "memory.precedent")
    assert precedent.severity is Severity.INFO
    assert any(a.kind == "use" for a in precedent.actions)


def test_locked_glossary_beats_conflicting_machine_wording(app):
    result = view(app, "p013-b3")
    assert result.candidate == "Aniki, help me out!"          # backend produced "Big bro, ..."
    assert result.origin is Origin.GLOSSARY
    assert result.locked is True
    assert any(w.code == "terminology.corrected" for w in result.warnings)
    term = next(a for a in result.annotations if a.type is AnnotationType.TERM)
    assert (term.target, term.locked) == ("Aniki", True)


def test_unsafe_terminology_conflict_warns_without_rewriting(app):
    app.backends.for_pair(LanguagePair("ja", "en")).add_phrase(
        LanguagePair("ja", "en"), "兄貴はどこだ？兄貴！", "Where's big bro? Bro!")
    add_segment(app, "amb", 21, "兄貴はどこだ？兄貴！")

    result = view(app, "amb")
    assert result.candidate == "Where's big bro? Bro!"        # left alone
    conflict = next(w for w in result.warnings if w.code == "terminology.conflict")
    assert conflict.severity is Severity.WARNING
    assert conflict.data["expected"] == "Aniki"


def test_fuzzy_memory_never_autofills(app):
    add_segment(app, "seed", 22, "行くぞ！")
    app.memory.record_approved("demo", "ja", "en", "行くぞ！", "行くぞ！", "Let's move!", "seed")
    add_segment(app, "near", 23, "行くぞ")                     # similar, not identical

    result = view(app, "near")
    assert result.candidate != "Let's move!"
    assert result.origin is not Origin.TRANSLATION_MEMORY
    fuzzy = next(w for w in result.warnings if w.code == "memory.fuzzy")
    assert fuzzy.severity is Severity.INFO
    assert "Let's move!" in [a.text for a in result.alternatives]


# --- language neutrality ------------------------------------------------
def test_japanese_and_korean_pass_through_the_same_pipeline(app):
    japanese, korean = view(app, "p012-b2"), view(app, "p013-b1")
    assert (japanese.language, korean.language) == ("ja", "ko")
    assert japanese.features["register"] == "crude"
    assert korean.features["register"] == "casual"
    # Language-specific detail rides in namespaced features, not the shared scale.
    assert japanese.features["ja.politeness"] == "plain"
    assert korean.features["ko.speech_level"] == "haeche"
    assert "ko.speech_level" not in japanese.features
    assert type(japanese).__name__ == type(korean).__name__


def test_mixed_language_project_resolves_per_segment(app):
    segments = app.repository.list_segments("demo")
    languages = {s.id: app.pipeline.process(s).view.language for s in segments}
    assert languages["p012-b1"] == "ja" and languages["p013-b1"] == "ko"


# --- context ------------------------------------------------------------
def test_approved_neighbour_translations_reach_the_backend(app):
    captured = {}
    backend = app.backends.for_pair(LanguagePair("ja", "en"))
    original = backend.translate

    def spy(request):
        captured["preceding"] = request.preceding
        captured["preceding_target"] = request.hints.get("preceding_target")
        captured["constraints"] = [(c.source_term, c.target_term) for c in request.constraints]
        return original(request)

    backend.translate = spy
    try:
        app.repository.save_state("p012-b1", SegmentState(
            "Tanaka-senpai, do you have a second?", Status.APPROVED, Origin.MANUAL))
        app.pipeline.process(app.repository.get_segment("p012-b2"))
    finally:
        backend.translate = original

    assert captured["preceding"] == ["田中先輩、ちょっといいですか？"]
    assert captured["preceding_target"] == ["Tanaka-senpai, do you have a second?"]


def test_glossary_constraints_are_passed_to_the_backend(app):
    captured = {}
    backend = app.backends.for_pair(LanguagePair("ja", "en"))
    original = backend.translate
    backend.translate = lambda r: (captured.update(
        c=[(x.source_term, x.target_term, x.locked) for x in r.constraints]) or original(r))
    try:
        app.pipeline.process(app.repository.get_segment("p013-b3"))
    finally:
        backend.translate = original
    assert ("兄貴", "Aniki", True) in captured["c"]


# --- SFX ----------------------------------------------------------------
def test_sfx_takes_the_sfx_path(app):
    result = view(app, "p012-s1")
    assert result.candidate == "BA-DUMP"
    assert result.origin is Origin.RULE
    assert "register" not in result.features              # no dialogue register on a sound
    assert result.features["ja.sfx_form"] == "ドクン"
    assert [a.text for a in result.alternatives]          # other renderings offered
    assert not any(w.code.startswith("register.") for w in result.warnings)


def test_unknown_sfx_is_flagged_not_invented(app):
    add_segment(app, "sfx-x", 24, "ズギャギャ", kind=SegmentKind.SFX)
    result = view(app, "sfx-x")
    assert result.candidate == "ズギャギャ"
    assert any(w.code == "sfx.unknown" for w in result.warnings)


# --- validation ---------------------------------------------------------
def test_validator_warnings_survive_into_the_view(app):
    result = view(app, "p012-b2")
    mismatch = next(w for w in result.warnings if w.code == "register.mismatch")
    assert mismatch.data == {"source_register": "crude", "target_register": "polite",
                             "distance": 3, "direction": "target_more_formal"}
    assert [a.kind for a in mismatch.actions] == ["show_alternatives", "dismiss"]
    assert result.needs_review is True


def test_dismissed_fingerprint_suppresses_only_that_issue(app):
    before = view(app, "p013-b1")
    unresolved = next(w for w in before.warnings if w.code == "relationship.unresolved")
    other_codes = {w.code for w in before.warnings} - {"relationship.unresolved"}

    app.repository.resolve_issue("p013-b1", unresolved.fingerprint, unresolved.code)
    after = view(app, "p013-b1")

    assert "relationship.unresolved" not in {w.code for w in after.warnings}
    assert other_codes <= {w.code for w in after.warnings}       # neighbours untouched
    assert after.needs_review is False


def test_changed_translation_raises_the_issue_again(app):
    before = view(app, "p012-b2")
    mismatch = next(w for w in before.warnings if w.code == "register.mismatch")
    app.repository.resolve_issue("p012-b2", mismatch.fingerprint, mismatch.code)
    assert not any(w.code == "register.mismatch" for w in view(app, "p012-b2").warnings)

    # A different polite rendering is a different instance of the same problem.
    app.repository.save_state("p012-b2", SegmentState(
        "Would you kindly be quiet? I am occupied.", Status.EDITED, Origin.MANUAL))
    after = view(app, "p012-b2")
    reraised = next(w for w in after.warnings if w.code == "register.mismatch")
    assert reraised.fingerprint != mismatch.fingerprint


# --- approval -----------------------------------------------------------
def test_approval_persists_and_enters_memory_atomically(app):
    result = app.approvals.approve("p012-b2", "Shut up… I'm busy right now.")

    segment = app.repository.get_segment("p012-b2")
    assert segment.state.status is Status.APPROVED
    assert segment.state.origin is Origin.MANUAL
    assert segment.state.candidate == "Shut up… I'm busy right now."

    entries = app.memory.exact("demo", "ja", "en", "うるせぇな…今忙しいんだよ。")
    assert [e.target_text for e in entries] == ["Shut up… I'm busy right now."]
    assert entries[0].occurrences == 1
    assert result.memory_entry_id == entries[0].id


def test_machine_and_edited_segments_stay_out_of_memory(app):
    app.pipeline.process(app.repository.get_segment("p012-b2"))      # machine candidate exists
    assert app.memory.exact("demo", "ja", "en", "うるせぇな…今忙しいんだよ。") == []

    app.repository.save_state("p012-b2", SegmentState(
        "Shut up, busy.", Status.EDITED, Origin.MANUAL))             # edited, not approved
    assert app.memory.exact("demo", "ja", "en", "うるせぇな…今忙しいんだよ。") == []

    app.approvals.approve("p012-b2")
    assert len(app.memory.exact("demo", "ja", "en", "うるせぇな…今忙しいんだよ。")) == 1


def test_reapproval_is_idempotent_and_reassignment_moves_the_occurrence(app):
    app.approvals.approve("p012-b2", "Shut up… I'm busy.")
    app.approvals.approve("p012-b2", "Shut up… I'm busy.")
    first = app.memory.exact("demo", "ja", "en", "うるせぇな…今忙しいんだよ。")[0]
    assert first.occurrences == 1

    app.approvals.reopen("p012-b2")
    assert app.memory.occurrence_count(first.id) == 0
    app.approvals.approve("p012-b2", "Quit bugging me.")

    variants = {e.target_text: e.occurrences for e in
                app.memory.exact("demo", "ja", "en", "うるせぇな…今忙しいんだよ。")}
    assert variants == {"Quit bugging me.": 1, "Shut up… I'm busy.": 0}
    assert app.memory.current_variant_for("p012-b2").target_text == "Quit bugging me."


def test_failed_approval_rolls_back_state_and_memory(app):
    original = app.memory.record_approved

    def exploding(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("memory write failed downstream")

    app.memory.record_approved = exploding
    try:
        with pytest.raises(RuntimeError):
            app.approvals.approve("p012-b2", "Shut up… I'm busy right now.")
    finally:
        app.memory.record_approved = original

    assert app.repository.get_segment("p012-b2").state.status is not Status.APPROVED
    assert app.memory.exact("demo", "ja", "en", "うるせぇな…今忙しいんだよ。") == []


def test_approval_suggests_a_glossary_rule_without_creating_it(app):
    result = app.approvals.approve("p013-b1", "Hyung, are you really okay?")
    suggestion = result.suggestion
    assert suggestion is not None
    assert (suggestion.source_term, suggestion.target_term) == ("형", "hyung")
    assert suggestion.category is Category.RELATIONSHIP
    # Nothing was written until the user consents.
    assert [e.source_term for e in app.glossary.list("demo", "ko")] == []

    app.approvals.accept_suggestion("demo", suggestion)
    entry = app.glossary.list("demo", "ko")[0]
    assert (entry.source_term, entry.target_term, entry.locked) == ("형", "hyung", True)
    assert not any(w.code == "relationship.unresolved" for w in view(app, "p013-b1").warnings)


def test_approval_does_not_invent_rules_from_free_edits(app):
    result = app.approvals.approve("p012-b2", "Whatever. Go away.")
    assert result.suggestion is None
    assert app.glossary.list("demo", "ja") and all(
        e.source_term in {"田中", "先輩", "兄貴"} for e in app.glossary.list("demo", "ja"))


def test_empty_translation_cannot_be_approved(app):
    with pytest.raises(ApprovalError):
        app.approvals.approve("p012-b1", "   ")


def test_approved_text_survives_reprocessing(app):
    app.approvals.approve("p012-b2", "Shut up… I'm busy right now.")
    result = view(app, "p012-b2")
    assert result.candidate == "Shut up… I'm busy right now."     # pipeline doesn't overwrite
    assert result.status is Status.APPROVED
    assert result.needs_review is False
