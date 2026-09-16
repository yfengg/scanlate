"""Approval.

Approval is where the system learns, and the only door into translation
memory. It is deliberately separate from the pipeline: the pipeline proposes,
this commits.

One transaction covers: segment state, the memory variant, the segment's
current occurrence, and any issue resolution. If any step fails, none of it
happened.

What approval will **not** do is create a locked glossary rule from a user's
edit. An edit is one line in one panel; a glossary rule binds the whole
project. Approval may *suggest* one and return it for the user to accept.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.analysis import AnalysisContext
from ..core.types import SegmentKind
from ..glossary.store import Category, GlossaryStore
from ..languages import AdapterRegistry
from ..memory.store import NORMALIZATION_VERSION, TranslationMemory
from ..storage.models import Origin, Segment, SegmentState, Status
from ..storage.repository import ProjectRepository
from ..storage.unit_of_work import UnitOfWork


@dataclass(frozen=True)
class TermResolution:
    """A terminology choice the user made explicitly.

    Produced when they take a ``use`` action on an unresolved-term warning, so
    approval knows exactly which source term got which rendering instead of
    guessing from the prose.
    """
    source_term: str
    target_term: str


@dataclass
class GlossarySuggestion:
    """A terminology decision worth keeping — offered, never applied."""
    source_term: str
    target_term: str
    category: Category
    reason: str
    source_language: str
    target_language: str
    explicit: bool = False      # the user resolved the term, vs. inferred from the text


@dataclass
class ApprovalResult:
    segment: Segment
    memory_entry_id: int | None
    suggestion: GlossarySuggestion | None = None


class ApprovalError(Exception):
    pass


class ApprovalService:
    def __init__(self, repository: ProjectRepository, memory: TranslationMemory,
                 glossary: GlossaryStore, adapters: AdapterRegistry,
                 uow: UnitOfWork | None = None,
                 normalization_version: int = NORMALIZATION_VERSION):
        self.repository = repository
        self.memory = memory
        self.glossary = glossary
        self.adapters = adapters
        self.uow = uow or repository.uow
        self.normalization_version = normalization_version

    def approve(self, segment_id: str, translation: str | None = None,
                term_resolutions: list[TermResolution] | None = None) -> ApprovalResult:
        segment = self.repository.get_segment(segment_id)
        if segment is None:
            raise ApprovalError(f"Unknown segment {segment_id!r}")
        project = self.repository.get_project(segment.project_id)
        if project is None:
            raise ApprovalError(f"Unknown project {segment.project_id!r}")

        text = (translation if translation is not None else segment.state.candidate) or ""
        if not text.strip():
            raise ApprovalError("Nothing to approve: the translation is empty.")

        language = segment.resolved_language(project.default_source_language)
        adapter = self.adapters.require(language)
        normalized = adapter.normalize(segment.source_text).text
        edited = translation is not None and translation != segment.state.candidate
        origin = Origin.MANUAL if edited else segment.state.origin

        with self.uow.transaction():
            self.repository.save_state(
                segment_id, SegmentState(text, Status.APPROVED, origin))
            entry = None
            if segment.kind is not SegmentKind.SFX:
                # record_approved moves this segment's current occurrence onto
                # the variant it now endorses, inside this same transaction.
                entry = self.memory.record_approved(
                    project_id=project.id, source_language=language,
                    target_language=project.target_language,
                    normalized_source=normalized, source_text=segment.source_text,
                    target_text=text, segment_id=segment_id,
                    normalization_version=self.normalization_version)
            suggestion = self._suggest_glossary(segment, project, adapter, text, language,
                                                term_resolutions or [])

        return ApprovalResult(segment=self.repository.get_segment(segment_id),
                              memory_entry_id=entry.id if entry else None,
                              suggestion=suggestion)

    def reopen(self, segment_id: str) -> Segment:
        """Take a segment back out of the approved set.

        Its memory variant stays on record — it was approved once — but the
        segment stops counting as a current occurrence of it.
        """
        segment = self.repository.get_segment(segment_id)
        if segment is None:
            raise ApprovalError(f"Unknown segment {segment_id!r}")
        with self.uow.transaction():
            self.repository.save_state(
                segment_id, SegmentState(segment.state.candidate, Status.EDITED,
                                         segment.state.origin))
            self.memory.clear_occurrence(segment_id)
        return self.repository.get_segment(segment_id)

    def accept_suggestion(self, project_id: str, suggestion: GlossarySuggestion,
                          locked: bool = True):
        """Create the glossary rule the user consented to."""
        from ..glossary.store import GlossaryEntry
        with self.uow.transaction():
            return self.glossary.add(GlossaryEntry(
                None, project_id, suggestion.source_language, suggestion.source_term,
                suggestion.target_language, suggestion.target_term,
                suggestion.category, locked=locked))

    # ------------------------------------------------------------------
    def _suggest_glossary(self, segment, project, adapter, text: str, language: str,
                          resolutions: list[TermResolution]) -> GlossarySuggestion | None:
        """Propose a rule for a term the project hasn't decided.

        An explicit resolution is the good case: the user pressed "Use hyung"
        on a specific warning, so the mapping is known rather than reconstructed.
        Substring inference is the fallback for approvals that arrive without
        one, and stays narrow — a recognised relationship term where exactly one
        legitimate rendering appears in the approved text. Anything vaguer is
        the user's prose, not a rule.
        """
        if segment.kind is SegmentKind.SFX:
            return None

        decided_terms = {e.source_term for e in self.glossary.list(project.id, language)}
        for resolution in resolutions:
            if resolution.source_term in decided_terms:
                continue
            if resolution.target_term.lower() not in text.lower():
                continue                      # the user changed their mind while editing
            return GlossarySuggestion(
                source_term=resolution.source_term, target_term=resolution.target_term,
                category=Category.RELATIONSHIP,
                reason=f"You chose {resolution.target_term} for {resolution.source_term}.",
                source_language=language, target_language=project.target_language,
                explicit=True)
        analysis = adapter.analyze(adapter.normalize(segment.source_text).text,
                                   AnalysisContext(segment_kind=segment.kind))
        decided = decided_terms
        lowered = text.lower()
        for term in analysis.relationship_terms:
            if term.surface in decided:
                continue
            options = [o for o in term.implies.get("options", "").split(", ") if o]
            present = [o for o in options if o.lower() in lowered]
            if len(present) != 1:
                continue
            return GlossarySuggestion(
                source_term=term.surface, target_term=present[0],
                category=Category.RELATIONSHIP,
                reason=f"{term.surface} recurs and has no project decision yet.",
                source_language=language, target_language=project.target_language)
        return None
