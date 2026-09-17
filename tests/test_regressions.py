"""Regressions for the five pipeline/storage corrections."""
from __future__ import annotations

import os

import pytest

from scanlate.core.types import LanguagePair, SegmentKind
from scanlate.fixtures import BACKEND_ENV, build_backends, build_services, seed_demo
from scanlate.pipeline.approval import TermResolution
from scanlate.storage.models import Chapter, Origin, Page, Segment, SegmentState, Status
from scanlate.translation.backends.lexicon import LexiconBackend
from scanlate.translation.backends.opus_mt import OpusMtBackend


@pytest.fixture()
def app():
    services = build_services()
    seed_demo(services)
    return services


def view(app, segment_id):
    return app.pipeline.process(app.repository.get_segment(segment_id)).view


class CountingBackend(LexiconBackend):
    """Wraps the deterministic backend to record whether it was consulted."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def translate(self, request):
        self.calls += 1
        return super().translate(request)


# --- 1. user-owned text bypasses machine translation -------------------
@pytest.mark.parametrize("status", [Status.EDITED, Status.APPROVED])
def test_user_owned_text_never_calls_the_backend(status):
    backend = CountingBackend()
    app = build_services(backend=backend)
    seed_demo(app)
    app.repository.save_state("p012-b2", SegmentState(
        "Shut up… I'm busy right now.", status, Origin.MANUAL))

    backend.calls = 0
    result = view(app, "p012-b2")

    assert backend.calls == 0
    assert result.candidate == "Shut up… I'm busy right now."
    assert result.origin is Origin.MANUAL


def test_user_owned_text_is_validated_against_itself(app):
    app.repository.save_state("p012-b2", SegmentState(
        "Shut up… I'm busy right now.", Status.EDITED, Origin.MANUAL))
    result = view(app, "p012-b2")
    # Crude source, crude translation: the mismatch the machine output produced
    # is gone, because validation now reads the user's text.
    assert not any(w.code == "register.mismatch" for w in result.warnings)
    assert result.needs_review is False


def test_canonical_memory_does_not_overwrite_user_text(app):
    app.repository.save_state("p013-b2", SegmentState(
        "We'll deal with it ourselves.", Status.EDITED, Origin.MANUAL))
    result = view(app, "p013-b2")
    assert result.candidate == "We'll deal with it ourselves."
    assert result.origin is Origin.MANUAL
    assert result.locked is False


def test_locked_glossary_reports_but_does_not_rewrite_user_text(app):
    app.repository.save_state("p013-b3", SegmentState(
        "Big bro, help me out!", Status.EDITED, Origin.MANUAL))
    result = view(app, "p013-b3")
    assert result.candidate == "Big bro, help me out!"          # left exactly as written
    assert any(w.code == "terminology.conflict" for w in result.warnings)


# --- 2. context is scoped to the chapter -------------------------------
def test_local_context_stops_at_the_chapter_boundary(app):
    repo = app.repository
    repo.create_chapter(Chapter("demo-ch2", "demo", 2, "Next Morning"))
    repo.create_page(Page("demo-p014", "demo", 14, "demo-ch2"))
    repo.add_segment(Segment("p014-b1", "demo", 8, SegmentKind.DIALOGUE, "行くぞ！",
                             language="ja", page_id="demo-p014"))

    # p013-b3 (seq 7, chapter 1) and p014-b1 (seq 8, chapter 2) are adjacent by
    # sequence but belong to different chapters.
    last_of_chapter_one = repo.local_context(repo.get_segment("p013-b3"))
    assert [c.id for c in last_of_chapter_one.following] == []
    assert all(c.id.startswith("p01") and not c.id.startswith("p014")
               for c in last_of_chapter_one.preceding)

    first_of_chapter_two = repo.local_context(repo.get_segment("p014-b1"))
    assert [c.id for c in first_of_chapter_two.preceding] == []


def test_context_within_a_chapter_still_flows_across_pages(app):
    # p012-s1 (page 12) and p013-b1 (page 13) share chapter 1.
    context = app.repository.local_context(app.repository.get_segment("p013-b1"))
    assert "p012-s1" in [c.id for c in context.preceding]


def test_unchaptered_segments_see_only_their_own_kind(app):
    app.repository.add_segment(Segment("loose-1", "demo", 30, SegmentKind.DIALOGUE,
                                       "何してるの？", language="ja"))
    app.repository.add_segment(Segment("loose-2", "demo", 31, SegmentKind.DIALOGUE,
                                       "行くぞ！", language="ja"))
    context = app.repository.local_context(app.repository.get_segment("loose-2"))
    assert [c.id for c in context.preceding] == ["loose-1"]      # not the chaptered segments


# --- 3. runtime backend selection --------------------------------------
# These tests are the auto-selection logic itself, so each one explicitly
# removes/overrides the deterministic-by-default SCANLATE_BACKEND that
# tests/conftest.py's autouse fixture sets, and explicitly controls
# OpusMtBackend's own availability -- never relying on whether transformers
# genuinely happens to be installed and reachable in the environment running
# the suite.
def test_backend_defaults_to_the_deterministic_lexicon(monkeypatch):
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    monkeypatch.setattr(OpusMtBackend, "unavailable_reason",
                        lambda self: "transformers not installed (forced for this test)")
    assert build_backends().for_pair(LanguagePair("ja", "en")).name == "lexicon"


def test_backend_is_selected_from_the_environment(monkeypatch):
    monkeypatch.setenv(BACKEND_ENV, "lexicon")
    assert build_services().backends.for_pair(LanguagePair("ja", "en")).name == "lexicon"
    monkeypatch.setenv(BACKEND_ENV, "nonsense")
    with pytest.raises(ValueError):
        build_services()


def test_requesting_opus_without_the_model_fails_loudly(monkeypatch):
    monkeypatch.setenv(BACKEND_ENV, "opus")
    monkeypatch.setattr(OpusMtBackend, "unavailable_reason",
                        lambda self: "transformers isn't installed (forced for this test)")
    with pytest.raises(RuntimeError, match="transformers"):
        build_services()


def test_opus_is_preferred_and_lexicon_covers_the_rest(monkeypatch):
    class FakeOpus(OpusMtBackend):
        def available(self):
            return True

        def supports(self, pair):
            return str(pair) == "ja>en"

    monkeypatch.setattr("scanlate.fixtures.OpusMtBackend", FakeOpus)
    monkeypatch.setenv(BACKEND_ENV, "opus")
    registry = build_backends()
    assert registry.for_pair(LanguagePair("ja", "en")).name == "opus-mt"
    assert registry.for_pair(LanguagePair("ko", "en")).name == "lexicon"


# --- 4. explicit term resolutions --------------------------------------
def test_explicit_resolution_drives_the_glossary_suggestion(app):
    result = app.approvals.approve(
        "p013-b1", "Bro, are you really okay?",
        term_resolutions=[TermResolution("형", "bro")])

    suggestion = result.suggestion
    assert (suggestion.source_term, suggestion.target_term) == ("형", "bro")
    assert suggestion.explicit is True
    assert "You chose bro" in suggestion.reason
    assert app.glossary.list("demo", "ko") == []        # still needs consent


def test_resolution_is_dropped_if_the_user_edited_it_away(app):
    result = app.approvals.approve(
        "p013-b1", "Are you really okay?",               # no rendering survives
        term_resolutions=[TermResolution("형", "bro")])
    assert result.suggestion is None


def test_substring_inference_remains_the_fallback(app):
    result = app.approvals.approve("p013-b1", "Hyung, are you really okay?")
    assert result.suggestion.target_term == "hyung"
    assert result.suggestion.explicit is False


def test_already_decided_terms_are_not_re_suggested(app):
    app.approvals.accept_suggestion(
        "demo", app.approvals.approve("p013-b1", "Hyung, are you okay?").suggestion)
    app.approvals.reopen("p013-b1")
    again = app.approvals.approve("p013-b1", "Hyung, are you okay?",
                                  term_resolutions=[TermResolution("형", "hyung")])
    assert again.suggestion is None


# --- 5. spans in fingerprints ------------------------------------------
def test_same_code_at_different_spans_gets_different_fingerprints(app):
    app.repository.add_segment(Segment("two-terms", "demo", 32, SegmentKind.DIALOGUE,
                                       "형, 누나, 괜찮아?", language="ko"))
    result = view(app, "two-terms")
    unresolved = [w for w in result.warnings if w.code == "relationship.unresolved"]
    assert len(unresolved) == 2
    assert unresolved[0].fingerprint != unresolved[1].fingerprint
    assert {w.span.start for w in unresolved} == {0, 3}


def test_dismissing_one_span_leaves_the_other_open(app):
    app.repository.add_segment(Segment("two-terms", "demo", 33, SegmentKind.DIALOGUE,
                                       "형, 누나, 괜찮아?", language="ko"))
    first, second = [w for w in view(app, "two-terms").warnings
                     if w.code == "relationship.unresolved"]
    app.repository.resolve_issue("two-terms", first.fingerprint, first.code)

    remaining = [w for w in view(app, "two-terms").warnings
                 if w.code == "relationship.unresolved"]
    assert [w.fingerprint for w in remaining] == [second.fingerprint]
    assert remaining[0].span.start == 3
