"""Validators.

Each validator is an independent component that sees the source analysis and
the target candidate, and returns structured Issues. Validators never rewrite
text — the pipeline decides what to apply. The bar for emitting an issue is
that it would change what a translator does.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core.analysis import (AmbiguityKind, CulturalImpact, LinguisticAnalysis,
                             RegisterLevel)
from ..core.types import Action, Issue, Severity, Span

if TYPE_CHECKING:  # pragma: no cover
    from ..glossary.store import GlossaryHit

STAGE = "validation"

# Ambiguity kinds that can change meaning, characterization or relationships.
MATERIAL_AMBIGUITY = {
    AmbiguityKind.OMITTED_ARGUMENT, AmbiguityKind.REFERENT, AmbiguityKind.GENDER,
    AmbiguityKind.NUMBER, AmbiguityKind.SOCIAL_RELATION, AmbiguityKind.WORDPLAY,
    AmbiguityKind.INTERLINEAR_GLOSS,
}
REGISTER_WORDS = {
    RegisterLevel.CRUDE: "crude", RegisterLevel.CASUAL: "casual",
    RegisterLevel.NEUTRAL: "neutral", RegisterLevel.POLITE: "polite",
    RegisterLevel.FORMAL: "formal",
}


@dataclass
class ValidationContext:
    source_analysis: LinguisticAnalysis
    target_analysis: LinguisticAnalysis
    candidate: str
    glossary_hits: list["GlossaryHit"] = field(default_factory=list)
    has_alternatives: bool = False
    segment_kind: str = "dialogue"
    resolved_codes: set[str] = field(default_factory=set)


class Validator(ABC):
    code_prefix: str = ""

    @abstractmethod
    def validate(self, ctx: ValidationContext) -> list[Issue]:
        ...


class RegisterValidator(Validator):
    """Flags a target that misses the source's register by two levels or more.

    One level apart is normal translation latitude. Two is a character change.
    """
    code_prefix = "register"
    threshold = 2

    def validate(self, ctx: ValidationContext) -> list[Issue]:
        source, target = ctx.source_analysis.register, ctx.target_analysis.register
        if source.confidence < 0.5 or target.confidence < 0.5:
            return []
        distance = int(source.level) - int(target.level)
        if abs(distance) < self.threshold:
            return []
        source_word, target_word = REGISTER_WORDS[source.level], REGISTER_WORDS[target.level]
        actions = [Action("dismiss", "Dismiss")]
        if ctx.has_alternatives:
            actions.insert(0, Action("show_alternatives", "Show alternatives"))
        return [Issue(
            code="register.mismatch", severity=Severity.WARNING,
            message=(f"Source is {source_word}, the translation reads {target_word}. "
                     f"{abs(distance)} levels apart."),
            stage=STAGE, actions=actions,
            data={"source_register": source_word, "target_register": target_word,
                  "distance": abs(distance),
                  "direction": "target_more_formal" if distance < 0 else "target_more_casual"})]


class PunctuationValidator(Validator):
    """Only meaningful losses: a question becoming a statement, trailing speech
    flattened, heavy emphasis dropped. Never a comma."""
    code_prefix = "punctuation"

    def validate(self, ctx: ValidationContext) -> list[Issue]:
        source = ctx.source_analysis.terminal_punctuation
        target = ctx.target_analysis.terminal_punctuation
        issues = []
        if "question" in source and "question" not in target:
            issues.append(Issue(
                code="punctuation.question_lost", severity=Severity.WARNING,
                message="The source asks a question; the translation doesn't.",
                stage=STAGE, actions=[Action("dismiss", "Dismiss")]))
        if "trailing" in source and "trailing" not in target:
            issues.append(Issue(
                code="punctuation.trailing_lost", severity=Severity.INFO,
                message="The source trails off; the translation closes cleanly.",
                stage=STAGE, actions=[Action("dismiss", "Dismiss")]))
        if "emphatic" in ctx.source_analysis.register.tone_flags \
                and "emphatic" not in ctx.target_analysis.register.tone_flags:
            issues.append(Issue(
                code="punctuation.emphasis_lost", severity=Severity.INFO,
                message="Strong emphasis in the source isn't carried over.",
                stage=STAGE, actions=[Action("dismiss", "Dismiss")]))
        return issues


class RelationshipValidator(Validator):
    """Relationship and address terms with no project decision.

    Surfaces the legitimate options rather than picking one silently.
    """
    code_prefix = "relationship"

    def validate(self, ctx: ValidationContext) -> list[Issue]:
        decided = {hit.entry.source_term for hit in ctx.glossary_hits}
        issues = []
        for term in ctx.source_analysis.relationship_terms:
            if term.surface in decided:
                continue
            options = [o for o in term.implies.get("options", "").split(", ") if o]
            actions = [Action("use", f"Use {option}", option) for option in options[:3]]
            actions.append(Action("add_rule", "Add rule"))
            issues.append(Issue(
                code="relationship.unresolved", severity=Severity.WARNING,
                message=f"{term.surface} has no approved rendering in this project.",
                stage=STAGE, span=term.span, actions=actions,
                data={"source_term": term.surface, "relation": term.relation,
                      "options": options, "note": term.note}))
        return issues


class AmbiguityValidator(Validator):
    """Only ambiguity that could change meaning, characterization or referent."""
    code_prefix = "ambiguity"
    min_confidence = 0.55

    def validate(self, ctx: ValidationContext) -> list[Issue]:
        issues = []
        seen: set[tuple[str, int]] = set()
        for flag in ctx.source_analysis.ambiguities:
            if flag.kind not in MATERIAL_AMBIGUITY or flag.confidence < self.min_confidence:
                continue
            # A relationship term already raised its own, more actionable issue.
            if flag.kind is AmbiguityKind.SOCIAL_RELATION:
                continue
            key = (flag.kind.value, flag.span.start)
            if key in seen:
                continue
            seen.add(key)
            issues.append(Issue(
                code=f"ambiguity.{flag.kind.value}", severity=Severity.INFO,
                message=flag.explanation, stage=STAGE, span=flag.span,
                actions=[Action("dismiss", "Dismiss")],
                data={"options": flag.options, "confidence": flag.confidence}))
        return issues


class CulturalValidator(Validator):
    """Cultural notes only where they materially affect the rendering."""
    code_prefix = "cultural"

    def validate(self, ctx: ValidationContext) -> list[Issue]:
        decided = {hit.entry.source_term for hit in ctx.glossary_hits}
        issues = []
        for hit in ctx.source_analysis.cultural_terms:
            if hit.impact < CulturalImpact.MEDIUM:
                continue
            if hit.impact is CulturalImpact.MEDIUM and hit.surface in decided:
                continue
            issues.append(Issue(
                code="cultural.note", severity=Severity.INFO, message=hit.note,
                stage=STAGE, span=hit.span, actions=[Action("dismiss", "Dismiss")],
                data={"key": hit.key, "domain": hit.domain,
                      "suggestions": hit.suggestions}))
        return issues


class EmptyCandidateValidator(Validator):
    """The backend couldn't translate this. Say so rather than shipping silence."""
    code_prefix = "translation"

    def validate(self, ctx: ValidationContext) -> list[Issue]:
        if ctx.candidate.strip():
            return []
        return [Issue(code="translation.missing", severity=Severity.ERROR,
                      message="No translation candidate was produced for this segment.",
                      stage=STAGE, span=Span(0, len(ctx.source_analysis.text)))]


DEFAULT_VALIDATORS: list[Validator] = [
    EmptyCandidateValidator(), RegisterValidator(), PunctuationValidator(),
    RelationshipValidator(), AmbiguityValidator(), CulturalValidator(),
]
