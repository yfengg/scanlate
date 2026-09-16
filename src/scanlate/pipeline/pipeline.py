"""The translation decision sequence, in one place.

Precedence, highest first:

1. **Locked glossary terminology** and **canonical (locked) translation
   memory** — approved project decisions. Machine translation never overwrites
   them.
2. **Unlocked exact memory** — precedent. Offered as an alternative; it does
   not override a fresh translation.
3. **Machine translation** — the default source of a candidate.
4. **Fuzzy memory** — suggestion only, never applied.

Validators only ever *report*. The single place text is rewritten automatically
is locked-glossary enforcement, and only when the substitution is provably
unambiguous.

The order below is the whole decision procedure; no other module re-implements
part of it, and API routes call ``process`` rather than reproducing it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..context.local import ContextWindow, build_context
from ..core.analysis import AnalysisContext, CulturalImpact, LinguisticAnalysis
from ..core.types import (Action, Issue, LanguagePair, SegmentKind, Severity,
                          Span, issue_fingerprint)
from ..glossary.store import GlossaryHit, GlossaryStore, enforce
from ..languages import AdapterRegistry
from ..memory.store import NORMALIZATION_VERSION, MemoryEntry, TranslationMemory
from ..sfx.renderer import SfxRendererRegistry
from ..storage.models import Origin, Project, Segment, SegmentState, Status
from ..storage.repository import ProjectRepository
from ..storage.unit_of_work import UnitOfWork
from ..translation.backend import TerminologyConstraint, TranslationRequest
from ..translation.registry import BackendRegistry
from ..validation.validators import (DEFAULT_VALIDATORS, ValidationContext,
                                     Validator)
from ..workbench.view import (Alternative, Annotation, AnnotationType,
                              SegmentView, Warning_)

STAGE = "pipeline"


@dataclass
class PipelineOutcome:
    """Everything the pipeline decided, for callers that need more than the view."""
    view: SegmentView
    analysis: LinguisticAnalysis
    normalized_source: str
    glossary_hits: list[GlossaryHit] = field(default_factory=list)
    canonical_memory: MemoryEntry | None = None
    precedents: list[MemoryEntry] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    backend_metadata: dict = field(default_factory=dict)


class TranslationPipeline:
    def __init__(self, repository: ProjectRepository, glossary: GlossaryStore,
                 memory: TranslationMemory, adapters: AdapterRegistry,
                 backends: BackendRegistry, sfx_renderers: SfxRendererRegistry,
                 validators: list[Validator] | None = None,
                 uow: UnitOfWork | None = None,
                 normalization_version: int = NORMALIZATION_VERSION):
        self.repository = repository
        self.glossary = glossary
        self.memory = memory
        self.adapters = adapters
        self.backends = backends
        self.sfx_renderers = sfx_renderers
        self.validators = validators if validators is not None else DEFAULT_VALIDATORS
        self.uow = uow or repository.uow
        self.normalization_version = normalization_version

    # ------------------------------------------------------------------
    def process(self, segment: Segment, project: Project | None = None,
                persist: bool = True) -> PipelineOutcome:
        # 1 — project
        project = project or self.repository.get_project(segment.project_id)
        if project is None:
            raise LookupError(f"Unknown project {segment.project_id!r}")
        pair = LanguagePair(segment.resolved_language(project.default_source_language),
                            project.target_language)

        # 2, 3 — language and adapter
        language = segment.language or self.adapters.resolve(
            segment.source_text, project.default_source_language)
        pair = LanguagePair(language, project.target_language)
        adapter = self.adapters.require(language)
        target_adapter = self.adapters.get(project.target_language)

        # 4 — normalize
        normalized = adapter.normalize(segment.source_text)
        source = normalized.text

        # 5 — local context (neighbouring lines only)
        context = build_context(self.repository.local_context(segment))

        # 6 — linguistic analysis
        analysis = adapter.analyze(source, AnalysisContext(segment_kind=segment.kind))

        # 7 — glossary annotations
        hits = self.glossary.find_in(source, project.id, language, project.target_language)

        issues: list[Issue] = []
        alternatives: list[Alternative] = []

        # A segment the user has edited or approved is not a translation
        # problem any more. Its text is the answer, so no backend is called and
        # no memory decision overrides it; it is analysed and validated against
        # itself, and the glossary is checked in report-only mode because
        # rewriting someone's approved wording behind their back is worse than
        # telling them it disagrees with a locked term.
        user_owned = (segment.state.status in (Status.EDITED, Status.APPROVED)
                      and bool(segment.state.candidate))

        if user_owned:
            candidate = segment.state.candidate
            origin = segment.state.origin
            check = enforce(candidate, hits, self._variants(hits), correct=False)
            issues.extend(check.issues)
            locked = False
            canonical = self.memory.canonical(project.id, pair.source, pair.target, source,
                                              self.normalization_version)
            precedents = [e for e in self.memory.exact(
                project.id, pair.source, pair.target, source, self.normalization_version)
                if not e.locked and e.target_text != candidate]
            metadata = {"path": "user translation"}
        elif segment.kind is SegmentKind.SFX:
            candidate, origin, locked, sfx_alternatives, sfx_issues = self._sfx_path(
                adapter, source, project, context)
            alternatives.extend(sfx_alternatives)
            issues.extend(sfx_issues)
            canonical, precedents, metadata = None, [], {"path": "sfx"}
        else:
            (candidate, origin, locked, canonical, precedents, metadata,
             mt_alternatives, term_issues) = self._dialogue_path(
                source, pair, project, hits, context, segment)
            alternatives.extend(mt_alternatives)
            issues.extend(term_issues)

        # 13 — validation.
        # Skipped for SFX (no dialogue register to compare) and for text taken
        # verbatim from canonical memory: that wording was already approved for
        # this project, so re-litigating its register every load is noise, not
        # information. Terminology conflicts against it are still reported,
        # because those come from enforcement above, not from here.
        authoritative = canonical is not None and not user_owned
        if segment.kind is not SegmentKind.SFX and not authoritative:
            target_analysis = (target_adapter.analyze(candidate) if target_adapter
                               else LinguisticAnalysis(language=project.target_language,
                                                       text=candidate))
            validation_context = ValidationContext(
                source_analysis=analysis, target_analysis=target_analysis,
                candidate=candidate, glossary_hits=hits,
                has_alternatives=bool(alternatives), segment_kind=segment.kind.value)
            for validator in self.validators:
                issues.extend(validator.validate(validation_context))

        # 14, 15 — fuzzy suggestions and the assembled alternative list
        fuzzy = self.memory.fuzzy(project.id, language, project.target_language, source,
                                  normalization_version=self.normalization_version)
        alternatives.extend(self._memory_alternatives(precedents, fuzzy, candidate))
        issues.extend(self._memory_issues(precedents, fuzzy, candidate))

        # 16 — annotations and warnings
        annotations = self._annotations(analysis, hits)
        warnings = self._warnings(issues, segment, source, candidate)

        # 17 — review state
        resolved = self.repository.resolved_fingerprints(segment.id)
        open_warnings = [w for w in warnings if w.fingerprint not in resolved]
        needs_review = (segment.state.status is not Status.APPROVED
                        and any(w.severity >= Severity.WARNING for w in open_warnings))

        # 18 — persist what the pipeline generated, never what the user owns
        status = segment.state.status
        if persist and not user_owned:
            self.repository.save_state(segment.id, SegmentState(candidate, status, origin))
            if segment.language is None:
                self.repository.set_language(segment.id, language)

        # 19 — the view
        view = SegmentView(
            id=segment.id, project_id=project.id, order=segment.seq, kind=segment.kind,
            source=source, candidate=candidate, language=language,
            target_language=project.target_language, page_id=segment.page_id,
            annotations=annotations, alternatives=_dedupe(alternatives, candidate),
            features=self._features(analysis, segment, metadata),
            warnings=open_warnings, status=status, origin=origin, locked=locked,
            needs_review=needs_review)

        return PipelineOutcome(view=view, analysis=analysis, normalized_source=source,
                               glossary_hits=hits, canonical_memory=canonical,
                               precedents=precedents, issues=issues,
                               backend_metadata=metadata)

    def process_id(self, segment_id: str, persist: bool = True) -> PipelineOutcome:
        segment = self.repository.get_segment(segment_id)
        if segment is None:
            raise LookupError(f"Unknown segment {segment_id!r}")
        return self.process(segment, persist=persist)

    # --- paths ---------------------------------------------------------
    def _dialogue_path(self, source, pair, project, hits, context: ContextWindow, segment):
        """Steps 8–12."""
        # 8, 9 — canonical memory is an approved project decision
        canonical = self.memory.canonical(project.id, pair.source, pair.target, source,
                                          self.normalization_version)
        # 10 — ordinary exact precedents
        precedents = [e for e in self.memory.exact(project.id, pair.source, pair.target, source,
                                                   self.normalization_version) if not e.locked]

        if canonical is not None:
            # Enforcement runs in report-only mode: a locked glossary term that
            # disagrees with an approved memory decision is a conflict for the
            # user to settle, not something to rewrite behind their back.
            check = enforce(canonical.target_text, hits, self._variants(hits), correct=False)
            return (canonical.target_text, Origin.TRANSLATION_MEMORY, True, canonical,
                    precedents, {"path": "canonical memory", "memory_id": canonical.id},
                    [], check.issues)

        # 11 — machine translation
        backend = self.backends.for_pair(pair)
        metadata: dict = {"path": "machine translation"}
        candidate, alternatives = "", []
        if backend is not None:
            request = TranslationRequest(
                source_text=source, pair=pair,
                preceding=context.preceding_source, following=context.following_source,
                constraints=[TerminologyConstraint(h.entry.source_term, h.entry.target_term,
                                                   h.entry.locked) for h in hits],
                segment_kind=segment.kind.value,
                hints={"preceding_target": context.preceding_target})
            result = backend.translate(request)
            metadata.update(result.metadata)
            metadata["development_backend"] = backend.development
            if result.primary is not None:
                candidate = result.primary.text
                alternatives = [Alternative(a.text, a.note, a.reason) for a in result.alternatives]
        else:
            metadata["reason"] = "no backend for this language pair"

        backend_issues = self._backend_issues(backend, candidate, metadata, source)

        # 12 — locked terminology, corrected only where provably safe
        enforcement = enforce(candidate, hits, self._variants(hits))
        origin = Origin.GLOSSARY if enforcement.changed else Origin.MACHINE_TRANSLATION
        locked = any(h.entry.locked for h in hits) and enforcement.changed
        return (enforcement.text, origin, locked, None, precedents, metadata,
                alternatives, backend_issues + enforcement.issues)

    def _sfx_path(self, adapter, source, project, context: ContextWindow):
        """SFX are semantic events, not dialogue: no MT, no register check."""
        match = adapter.lookup_sfx(source)
        renderer = self.sfx_renderers.get(project.target_language)
        if match is None or renderer is None:
            issue = Issue(code="sfx.unknown", severity=Severity.WARNING,
                          message=f"No known sound effect matches {source}.",
                          stage=STAGE, span=Span(0, len(source)),
                          actions=[Action("dismiss", "Dismiss")])
            return source, Origin.MACHINE_TRANSLATION, False, [], [issue]

        renderings = renderer.render(match, context.scene_tags)
        alternatives = [Alternative(r.text, r.rationale, "sfx") for r in renderings[1:]]
        issues: list[Issue] = []
        if match.note:
            issues.append(Issue(code="sfx.note", severity=Severity.INFO, message=match.note,
                                stage=STAGE, span=Span(0, len(source)),
                                actions=[Action("dismiss", "Dismiss")],
                                data={"category": match.category.value}))
        return renderings[0].text, Origin.RULE, False, alternatives, issues

    def _backend_issues(self, backend, candidate: str, metadata: dict, source: str) -> list[Issue]:
        """Be explicit about what produced the candidate.

        A development backend that fell through to a word gloss must not look
        like a real translation that came out badly, and a missing backend must
        not look like an empty result.
        """
        if backend is None:
            return [Issue(
                code="translation.backend_unavailable", severity=Severity.ERROR,
                message=("Machine translation backend unavailable for this language pair. "
                         "Install the MT model or write the translation by hand."),
                stage=STAGE, span=Span(0, len(source)))]
        if not backend.development:
            return []
        if metadata.get("match") == "exact phrase":
            return []
        detail = ("a word-by-word gloss" if candidate else "nothing")
        return [Issue(
            code="translation.development_backend", severity=Severity.WARNING,
            message=(f"Machine translation backend unavailable — the development backend "
                     f"({backend.name}) produced {detail}, not a translation. "
                     f"Install the MT model, or write this line by hand."),
            stage=STAGE, span=Span(0, len(source)),
            actions=[Action("dismiss", "Dismiss")],
            data={"backend": backend.name})]

    # --- assembly ------------------------------------------------------
    def _variants(self, hits: list[GlossaryHit]) -> dict[str, list[str]]:
        """Renderings a backend might have produced for a locked term."""
        variants: dict[str, list[str]] = {}
        for hit in hits:
            known = variants.setdefault(hit.entry.source_term, [])
            if hit.entry.notes:
                known.extend(v.strip() for v in hit.entry.notes.split("|") if v.strip())
        return variants

    def _memory_alternatives(self, precedents, fuzzy, candidate) -> list[Alternative]:
        out = [Alternative(e.target_text,
                           f"approved before, used in {e.occurrences} segment"
                           f"{'s' if e.occurrences != 1 else ''}", "memory")
               for e in precedents if e.target_text != candidate]
        out += [Alternative(m.entry.target_text,
                            f"similar line, {int(m.score * 100)}% match", "memory")
                for m in fuzzy if m.entry.target_text != candidate]
        return out

    def _memory_issues(self, precedents, fuzzy, candidate) -> list[Issue]:
        issues = []
        if precedents and all(e.target_text != candidate for e in precedents):
            issues.append(Issue(
                code="memory.precedent", severity=Severity.INFO,
                message=("This line was approved differently before: "
                         f"{precedents[0].target_text}"),
                stage=STAGE,
                actions=[Action("use", "Use previous", precedents[0].target_text),
                         Action("dismiss", "Dismiss")],
                data={"variants": [e.target_text for e in precedents]}))
        if fuzzy and not precedents:
            best = fuzzy[0]
            issues.append(Issue(
                code="memory.fuzzy", severity=Severity.INFO,
                message=(f"A similar line ({int(best.score * 100)}% match) was approved as: "
                         f"{best.entry.target_text}"),
                stage=STAGE,
                actions=[Action("use", "Use it", best.entry.target_text),
                         Action("dismiss", "Dismiss")],
                data={"score": best.score, "source": best.entry.source_text}))
        return issues

    def _annotations(self, analysis: LinguisticAnalysis,
                     hits: list[GlossaryHit]) -> list[Annotation]:
        annotations = [Annotation(span=h.span, type=AnnotationType.TERM,
                                  target=h.entry.target_term, locked=h.entry.locked,
                                  origin=Origin.GLOSSARY,
                                  data={"category": h.entry.category.value,
                                        "entry_id": h.entry.id})
                       for h in hits]
        decided = {h.entry.source_term for h in hits}
        for term in analysis.relationship_terms:
            if term.surface in decided:
                continue      # the glossary annotation above already covers this span
            annotations.append(Annotation(
                span=term.span, type=AnnotationType.AMBIGUITY,
                label=f"{term.relation.replace('_', ' ')}"
                      + (", may be non-kin" if term.may_be_non_kin else ""),
                data={"options": term.implies.get("options", "")}))
        for flag in analysis.ambiguities:
            if flag.span.start == 0 and len(flag.span) == len(analysis.text):
                continue                      # whole-line flags aren't underlinable
            annotations.append(Annotation(span=flag.span, type=AnnotationType.AMBIGUITY,
                                          label=flag.explanation,
                                          data={"kind": flag.kind.value}))
        for hit in analysis.cultural_terms:
            if hit.impact >= CulturalImpact.MEDIUM:
                annotations.append(Annotation(span=hit.span, type=AnnotationType.CULTURAL,
                                              label=hit.note, data={"key": hit.key}))
        for entity in analysis.entities:
            annotations.append(Annotation(span=entity.span, type=AnnotationType.ENTITY,
                                          label=entity.kind.value,
                                          data={"confidence": entity.confidence}))
        return annotations

    def _warnings(self, issues: list[Issue], segment: Segment, source: str,
                  candidate: str) -> list[Warning_]:
        seen: set[str] = set()
        warnings = []
        for issue in issues:
            # The span is part of the issue's identity: the same code raised at
            # a different place in the line is a different problem, and
            # dismissing one must not silence the other.
            location = f"@{issue.span.start}:{issue.span.end}" if issue.span else "@-"
            evidence = location + "|" + "|".join(
                f"{k}={v}" for k, v in sorted(issue.data.items())
                if k in {"source_register", "target_register", "source_term",
                         "expected", "kind", "key", "options", "direction"})
            fingerprint = issue_fingerprint(issue.code, source, candidate, evidence)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            warnings.append(Warning_(code=issue.code, severity=issue.severity,
                                     message=issue.message, fingerprint=fingerprint,
                                     span=issue.span, actions=list(issue.actions),
                                     data=dict(issue.data)))
        return warnings

    def _features(self, analysis: LinguisticAnalysis, segment: Segment, metadata: dict) -> dict:
        features: dict = {}
        if segment.kind is not SegmentKind.SFX:
            features["register"] = _REGISTER_WORDS.get(analysis.register.level, "neutral")
            features["register_confidence"] = analysis.register.confidence
            if analysis.register.tone_flags:
                features["tone"] = ", ".join(sorted(analysis.register.tone_flags))
        else:
            if "sfx_category" in metadata:
                features["sfx_category"] = metadata["sfx_category"]
        features["adapter"] = analysis.language
        features.update(analysis.adapter_features)
        if metadata.get("backend"):
            features["backend"] = metadata["backend"]
        if metadata.get("development_backend"):
            features["backend_mode"] = "development"
        return features


_REGISTER_WORDS = {-2: "crude", -1: "casual", 0: "neutral", 1: "polite", 2: "formal"}


def _dedupe(alternatives: list[Alternative], candidate: str) -> list[Alternative]:
    seen = {candidate}
    out = []
    for alternative in alternatives:
        if alternative.text in seen:
            continue
        seen.add(alternative.text)
        out.append(alternative)
    return out
