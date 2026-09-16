"""Generic linguistic-analysis vocabulary shared by all language adapters.

Adapters map language-specific phenomena onto these concepts. Anything that
does not fit may be exposed through ``LinguisticAnalysis.adapter_features``
using a namespaced key such as ``"xx.feature_name"``.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any

from .types import SegmentKind, Span


class RegisterLevel(IntEnum):
    """Ordered formality scale; comparable across languages."""
    CRUDE = -2
    CASUAL = -1
    NEUTRAL = 0
    POLITE = 1
    FORMAL = 2


@dataclass
class Evidence:
    span: Span
    surface: str
    reason: str


@dataclass
class RegisterAnalysis:
    level: RegisterLevel = RegisterLevel.NEUTRAL
    confidence: float = 0.3
    tone_flags: set[str] = field(default_factory=set)
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class RegisterSignal:
    evidence: Evidence
    level: RegisterLevel | None
    flags: tuple[str, ...] = ()
    weight: float = 1.0


def aggregate_register(
    signals: list[RegisterSignal],
    default_level: RegisterLevel = RegisterLevel.NEUTRAL,
    default_confidence: float = 0.3,
) -> RegisterAnalysis:
    flags: set[str] = set()
    evidence: list[Evidence] = []
    weights: dict[RegisterLevel, float] = defaultdict(float)
    for s in signals:
        flags.update(s.flags)
        evidence.append(s.evidence)
        if s.level is not None:
            weights[s.level] += s.weight
    if not weights:
        return RegisterAnalysis(default_level, default_confidence, flags, evidence)
    total = sum(weights.values())
    level, weight = max(weights.items(), key=lambda kv: (kv[1], abs(int(kv[0]))))
    if any(l > 0 for l in weights) and any(l < 0 for l in weights):
        flags.add("mixed_register")
    confidence = min(0.95, 0.5 + 0.45 * (weight / total) * min(1.0, total / 2))
    return RegisterAnalysis(level, round(confidence, 3), flags, evidence)


class HierarchyDirection(str, Enum):
    UPWARD = "upward"        # speaker elevates the referent/addressee
    PEER = "peer"
    DOWNWARD = "downward"
    INTIMATE = "intimate"
    HOSTILE = "hostile"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class SocialMarkerKind(str, Enum):
    ADDRESS_AFFIX = "address_affix"            # attached to a name
    TITLE = "title"                            # role/rank used as address or reference
    SELF_REFERENCE = "self_reference"          # first-person choice carrying social meaning
    ADDRESSEE_REFERENCE = "addressee_reference"
    RESPECTFUL_FORM = "respectful_form"        # grammatical elevation of a referent
    HUMBLE_FORM = "humble_form"                # grammatical lowering of the speaker


@dataclass
class SocialMarker:
    kind: SocialMarkerKind
    span: Span
    surface: str
    direction: HierarchyDirection = HierarchyDirection.UNKNOWN
    attached_to: str | None = None
    convention_key: str | None = None   # matched against glossary honorific conventions
    register: RegisterLevel | None = None
    flags: tuple[str, ...] = ()
    note: str | None = None


@dataclass
class RelationshipTerm:
    span: Span
    surface: str
    relation: str
    addressive: bool = True
    may_be_non_kin: bool = False
    implies: dict[str, str] = field(default_factory=dict)
    note: str | None = None


class AmbiguityKind(str, Enum):
    OMITTED_ARGUMENT = "omitted_argument"
    REFERENT = "referent"
    GENDER = "gender"
    NUMBER = "number"
    LEXICAL = "lexical"
    WORDPLAY = "wordplay"
    INTERLINEAR_GLOSS = "interlinear_gloss"
    SOCIAL_RELATION = "social_relation"


@dataclass
class AmbiguityFlag:
    kind: AmbiguityKind
    span: Span
    surface: str
    explanation: str
    options: list[str] = field(default_factory=list)
    confidence: float = 0.5


class EntityKind(str, Enum):
    PERSON = "person"
    PLACE = "place"
    ORGANIZATION = "organization"
    TITLE = "title"
    TECHNIQUE = "technique"
    OBJECT = "object"
    OTHER = "other"


@dataclass
class EntityMention:
    span: Span
    surface: str
    kind: EntityKind
    glossary_entry_id: int | None = None
    confidence: float = 0.5
    source: str = "adapter"


class CulturalImpact(IntEnum):
    LOW = 0      # never surfaced automatically
    MEDIUM = 1   # surfaced unless a project decision already covers it
    HIGH = 2     # surfaced when present


@dataclass
class CulturalTermHit:
    span: Span
    surface: str
    key: str
    domain: str
    impact: CulturalImpact
    note: str
    suggestions: list[str] = field(default_factory=list)


@dataclass
class SlangHit:
    span: Span
    surface: str
    meaning: str
    register: RegisterLevel | None = None
    note: str | None = None


@dataclass
class Token:
    surface: str
    start: int
    end: int
    lemma: str | None = None
    upos: str = "X"                 # Universal Dependencies POS tag
    features: dict[str, str] = field(default_factory=dict)


@dataclass
class InterlinearAnnotation:
    """Small text printed alongside base text (readings or glosses)."""
    base: str
    annotation: str
    kind: str = "reading"           # "reading" | "gloss"
    base_span: Span | None = None


@dataclass
class OcrCleanup:
    text: str
    interlinear: list[InterlinearAnnotation] = field(default_factory=list)
    edits: list[str] = field(default_factory=list)


@dataclass
class OcrProfile:
    vertical_text_likely: bool = False
    engine_hints: dict[str, Any] = field(default_factory=dict)


class PunctKind(str, Enum):
    QUESTION = "question"
    QUESTION_IMPLIED = "question_implied"   # grammatical question without a mark
    EXCLAMATION = "exclamation"
    TRAILING = "trailing"


@dataclass
class AnalysisContext:
    segment_kind: SegmentKind = SegmentKind.DIALOGUE
    settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class LinguisticAnalysis:
    language: str
    text: str
    sentences: list[Span] = field(default_factory=list)
    tokens: list[Token] = field(default_factory=list)
    register: RegisterAnalysis = field(default_factory=RegisterAnalysis)
    social_markers: list[SocialMarker] = field(default_factory=list)
    relationship_terms: list[RelationshipTerm] = field(default_factory=list)
    ambiguities: list[AmbiguityFlag] = field(default_factory=list)
    entities: list[EntityMention] = field(default_factory=list)
    cultural_terms: list[CulturalTermHit] = field(default_factory=list)
    slang: list[SlangHit] = field(default_factory=list)
    interlinear: list[InterlinearAnnotation] = field(default_factory=list)
    terminal_punctuation: set[PunctKind] = field(default_factory=set)
    adapter_features: dict[str, Any] = field(default_factory=dict)
