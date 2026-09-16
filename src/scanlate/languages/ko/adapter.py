"""Korean adapter.

Korean speech levels are a six-way grammatical system, not a politeness dial.
They are preserved verbatim in ``adapter_features`` and only *approximated*
onto the shared RegisterLevel scale for cross-language comparison.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from ...core.adapter import LanguageAdapter, NormalizedText
from ...core.analysis import (AmbiguityFlag, AmbiguityKind, AnalysisContext,
                              CulturalImpact, CulturalTermHit, EntityKind,
                              EntityMention, Evidence, HierarchyDirection,
                              LinguisticAnalysis, OcrProfile, RegisterLevel,
                              RegisterSignal, RelationshipTerm, SlangHit,
                              SocialMarker, SocialMarkerKind,
                              aggregate_register)
from ...core.scripts import count_in_scripts
from ...core.types import SegmentKind, Span, select_non_overlapping, terminal_punctuation
from ...knowledge import KnowledgeBase
from ...sfx.categories import SfxCategory, SfxMatch, periodic_unit

DATA = Path(__file__).parent / "data"

# Speech level -> (approximate shared register, weight, human label)
SPEECH_LEVELS: dict[str, tuple[RegisterLevel, float, str]] = {
    "hapsyoche": (RegisterLevel.FORMAL, 2.0, "합쇼체 (deferential)"),
    "haeyoche": (RegisterLevel.POLITE, 1.6, "해요체 (polite informal)"),
    "haoche": (RegisterLevel.POLITE, 1.0, "하오체 (dated formal)"),
    "haerache": (RegisterLevel.NEUTRAL, 1.0, "해라체 (plain written)"),
    "haeche": (RegisterLevel.CASUAL, 1.4, "해체 (intimate banmal)"),
    "hageche": (RegisterLevel.NEUTRAL, 0.8, "하게체 (familiar)"),
}

# Checked in order; the first match on a sentence wins.
_LEVEL_RULES: list[tuple[str, str, str]] = [
    (r"(습니다|ㅂ니다|습니까|ㅂ니까|십시오|읍시다)[.!?？！]?$", "hapsyoche", "-seumnida ending"),
    (r"(어요|아요|에요|예요|해요|세요|죠|지요|네요|는데요|거든요)[.!?？！]?$", "haeyoche", "-yo ending"),
    (r"(는다|ㄴ다|다|니|냐|자|라)[.!?？！]?$", "haerache", "plain written ending"),
    (r"(아|어|야|지|네|거든|는데|잖아|을까|ㄹ까)[.!?？！]?$", "haeche", "banmal ending"),
]

_HONORIFIC_INFIX = re.compile(r"(으시|시)(?=[어요다는겠])|께서|드리")
_HUMBLE = re.compile(r"(저희|제가|저는|드립니다|드릴)")
_BLUNT = re.compile(r"(야[,!\s]|임마|새끼|닥쳐|꺼져|미친)")
_SENTENCE_END = re.compile(r"[.!?。！？…]+|\n")
_SUBJECT_MARK = re.compile(r"(은|는|이|가)(?=\s|$)")
_PRONOUN = re.compile(r"(나|저|너|우리|저희|그|그녀|당신)")
_HANGUL = frozenset({"Hangul"})


class KoreanAdapter(LanguageAdapter):
    language = "ko"
    display_name = "Korean"

    def __init__(self, knowledge: KnowledgeBase | None = None):
        self.knowledge = knowledge or KnowledgeBase("ko", DATA)
        self._kiwi = None
        self._kiwi_tried = False

    def capabilities(self) -> frozenset[str]:
        caps = {"normalize", "sentences", "register", "social_markers", "relationships",
                "ambiguity", "entities", "slang", "cultural", "sfx"}
        if self._load_kiwi() is not None:
            caps |= {"tokenize", "morphology"}
        return frozenset(caps)

    def _load_kiwi(self):
        """kiwipiepy if installed; otherwise no morphology is claimed."""
        if not self._kiwi_tried:
            self._kiwi_tried = True
            try:
                from kiwipiepy import Kiwi  # type: ignore
                self._kiwi = Kiwi()
            except Exception:
                self._kiwi = None
        return self._kiwi

    def tokenize(self, text: str):
        from ...core.analysis import Token
        kiwi = self._load_kiwi()
        if kiwi is None:
            return []
        return [Token(t.form, t.start, t.start + t.len, lemma=t.form,
                      upos=_UPOS.get(t.tag[:2], "X"), features={"tag": t.tag})
                for t in kiwi.tokenize(text)]

    def normalize(self, text: str) -> NormalizedText:
        edits = []
        out = unicodedata.normalize("NFC", text)   # compose decomposed jamo
        if out != text:
            edits.append("NFC composition")
        joined = re.sub(r"[ \t]*\n[ \t]*", " ", out).strip()
        if joined != out.strip():
            edits.append("joined wrapped lines")
        collapsed = re.sub(r" {2,}", " ", joined)
        if collapsed != joined:
            edits.append("collapsed spaces")
        return NormalizedText(collapsed, text, edits)

    def ocr_profile(self, text: str) -> OcrProfile:
        return OcrProfile(vertical_text_likely=False,
                          engine_hints={"scripts": "ko", "expect_long_strips": True})

    def segment_sentences(self, text: str) -> list[Span]:
        spans, start = [], 0
        for m in _SENTENCE_END.finditer(text):
            spans.append(Span(start, m.end()))
            start = m.end()
        if start < len(text):
            spans.append(Span(start, len(text)))
        return spans or ([Span(0, len(text))] if text else [])

    # --- SFX -----------------------------------------------------------
    def looks_like_sfx(self, text: str) -> bool:
        stripped = re.sub(r"[!~…\s]", "", text)
        if not stripped or len(stripped) > 6:
            return False
        return count_in_scripts(stripped, _HANGUL) == len(stripped) and stripped in self._sfx_index()

    def _sfx_index(self) -> dict[str, dict]:
        def build():
            index = {}
            for entry in self.knowledge.entries("sfx"):
                index[entry["form"]] = entry
            return index
        return self.knowledge.cached("__sfx_index__", build)

    def lookup_sfx(self, text: str) -> SfxMatch | None:
        index = self._sfx_index()
        raw = text.strip()
        elongated = "~" in raw or "ㅡ" in raw
        core = re.sub(r"[~!…\s]", "", raw)
        unit, reps = periodic_unit(core)
        for candidate, extra in ((core, 1), (unit, reps)):
            entry = index.get(candidate)
            if entry is None:
                continue
            intensity = int(entry.get("intensity", 1))
            if elongated:
                intensity = min(3, intensity + 1)
            if extra > 1:
                intensity = min(3, intensity + 1)
            return SfxMatch(category=SfxCategory(entry["category"]), form=candidate,
                            subtype=entry.get("subtype"), intensity=intensity,
                            repetitions=extra, elongated=elongated,
                            confidence=0.9 if candidate == core else 0.75,
                            note=entry.get("note"))
        return None

    # --- analysis ------------------------------------------------------
    def analyze(self, text: str, context: AnalysisContext | None = None) -> LinguisticAnalysis:
        context = context or AnalysisContext()
        analysis = LinguisticAnalysis(
            language=self.language, text=text,
            sentences=self.segment_sentences(text), tokens=self.tokenize(text),
            terminal_punctuation=set(terminal_punctuation(text)),
        )
        if context.segment_kind is SegmentKind.SFX:
            match = self.lookup_sfx(text)
            if match:
                analysis.adapter_features["ko.sfx_form"] = match.form
            return analysis

        levels, signals = self._speech_levels(text)
        analysis.social_markers = self._social_markers(text)
        analysis.relationship_terms = self._relationship_terms(text)
        analysis.cultural_terms = self._cultural_terms(text)
        analysis.slang = self._slang(text)
        analysis.entities = self._entities(text, analysis.social_markers)

        for marker in analysis.social_markers:
            if marker.register is not None:
                signals.append(RegisterSignal(
                    Evidence(marker.span, marker.surface, f"{marker.kind.value} {marker.surface}"),
                    marker.register, weight=0.8))
        if _BLUNT.search(text):
            m = _BLUNT.search(text)
            signals.append(RegisterSignal(
                Evidence(Span(m.start(), m.end()), m.group(0), "blunt address"),
                RegisterLevel.CRUDE, ("rough",), 1.5))

        analysis.register = aggregate_register(signals)
        analysis.ambiguities = self._ambiguities(text, analysis)

        if levels:
            analysis.adapter_features["ko.speech_level"] = levels[0]
            analysis.adapter_features["ko.speech_level_label"] = SPEECH_LEVELS[levels[0]][2]
            if len(set(levels)) > 1:
                analysis.register.tone_flags.add("mixed_speech_levels")
                analysis.adapter_features["ko.speech_levels_seen"] = ", ".join(dict.fromkeys(levels))
        if _HONORIFIC_INFIX.search(text):
            analysis.adapter_features["ko.subject_honorific"] = "yes"
        if _HUMBLE.search(text):
            analysis.adapter_features["ko.humble_self_reference"] = "yes"
        return analysis

    def _speech_levels(self, text: str) -> tuple[list[str], list[RegisterSignal]]:
        levels: list[str] = []
        signals: list[RegisterSignal] = []
        for sentence in self.segment_sentences(text):
            chunk = text[sentence.start:sentence.end].strip()
            if not chunk:
                continue
            for pattern, level, reason in _LEVEL_RULES:
                m = re.search(pattern, chunk)
                if not m:
                    continue
                levels.append(level)
                register, weight, label = SPEECH_LEVELS[level]
                span = Span(sentence.start + m.start(), sentence.start + m.end())
                signals.append(RegisterSignal(
                    Evidence(span, m.group(0), f"{label}: {reason}"), register, weight=weight))
                break
        return levels, signals

    def _social_markers(self, text: str) -> list[SocialMarker]:
        markers: list[SocialMarker] = []
        for entry in self.knowledge.entries("honorifics"):
            form = entry["form"]
            for m in re.finditer(re.escape(form), text):
                preceding = re.search(r"[\uAC00-\uD7AF]{1,4}$", text[:m.start()])
                markers.append(SocialMarker(
                    kind=SocialMarkerKind(entry.get("kind", "address_affix")),
                    span=Span(m.start(), m.end()), surface=form,
                    direction=HierarchyDirection(entry.get("direction", "unknown")),
                    attached_to=preceding.group(0) if preceding else None,
                    convention_key=entry.get("convention_key"),
                    register=getattr(RegisterLevel, entry.get("register", "NEUTRAL"), None),
                    note=entry.get("note")))
        for m in _HONORIFIC_INFIX.finditer(text):
            markers.append(SocialMarker(
                kind=SocialMarkerKind.RESPECTFUL_FORM, span=Span(m.start(), m.end()),
                surface=m.group(0), direction=HierarchyDirection.UPWARD,
                register=RegisterLevel.POLITE,
                note="Subject honorific: the referent is elevated grammatically."))
        for m in _HUMBLE.finditer(text):
            markers.append(SocialMarker(
                kind=SocialMarkerKind.HUMBLE_FORM, span=Span(m.start(), m.end()),
                surface=m.group(0), direction=HierarchyDirection.UPWARD,
                register=RegisterLevel.FORMAL,
                note="Humble first person: the speaker lowers themselves."))
        return sorted(markers, key=lambda m: m.span.start)

    def _relationship_terms(self, text: str) -> list[RelationshipTerm]:
        hits = []
        for entry in self.knowledge.entries("relationships"):
            form = entry["form"]
            for m in re.finditer(re.escape(form), text):
                # Korean has no spaces inside words; require a boundary so 형 in
                # 형태 ("form") is not read as the kinship term.
                after = text[m.end():m.end() + 1]
                before = text[max(0, m.start() - 1):m.start()]
                if after and re.match(r"[\uAC00-\uD7AF]", after) and not re.match(r"(은|는|이|가|을|를|도|만|의|아|야|님)", after):
                    continue
                if before and re.match(r"[\uAC00-\uD7AF]", before):
                    continue
                hits.append(RelationshipTerm(
                    span=Span(m.start(), m.end()), surface=form, relation=entry["relation"],
                    may_be_non_kin=bool(entry.get("may_be_non_kin")),
                    implies={"options": ", ".join(entry.get("options", []))},
                    note=entry.get("note")))
        return select_non_overlapping(hits, lambda h: h.span)

    def _cultural_terms(self, text: str) -> list[CulturalTermHit]:
        return [CulturalTermHit(span=Span(m.start(), m.end()), surface=entry["form"],
                                key=entry["key"], domain=entry["domain"],
                                impact=CulturalImpact(int(entry.get("impact", 1))),
                                note=entry["note"], suggestions=list(entry.get("suggestions", [])))
                for entry in self.knowledge.entries("cultural")
                for m in re.finditer(re.escape(entry["form"]), text)]

    def _slang(self, text: str) -> list[SlangHit]:
        hits = []
        for entry in self.knowledge.entries("slang"):
            for m in re.finditer(re.escape(entry["form"]), text):
                hits.append(SlangHit(span=Span(m.start(), m.end()), surface=entry["form"],
                                     meaning=entry["meaning"],
                                     register=getattr(RegisterLevel, entry.get("register", ""), None)))
        return hits

    def _entities(self, text: str, markers: list[SocialMarker]) -> list[EntityMention]:
        out = []
        for marker in markers:
            if marker.kind is SocialMarkerKind.ADDRESS_AFFIX and marker.attached_to:
                start = marker.span.start - len(marker.attached_to)
                out.append(EntityMention(span=Span(start, marker.span.start),
                                         surface=marker.attached_to, kind=EntityKind.PERSON,
                                         confidence=0.55, source="ko.honorific_attachment"))
        return out

    def _ambiguities(self, text: str, analysis: LinguisticAnalysis) -> list[AmbiguityFlag]:
        flags: list[AmbiguityFlag] = []
        if not _SUBJECT_MARK.search(text) and not _PRONOUN.search(text) and len(text) > 4:
            flags.append(AmbiguityFlag(
                kind=AmbiguityKind.OMITTED_ARGUMENT, span=Span(0, len(text)), surface=text,
                explanation="No subject is stated; English has to supply one.",
                options=["I", "you", "we", "they"], confidence=0.6))
        for term in analysis.relationship_terms:
            if term.may_be_non_kin:
                flags.append(AmbiguityFlag(
                    kind=AmbiguityKind.SOCIAL_RELATION, span=term.span, surface=term.surface,
                    explanation=f"{term.surface} may address a real sibling or an unrelated senior.",
                    options=term.implies.get("options", "").split(", "), confidence=0.75))
            if term.relation in {"older_brother_of_male", "older_sister_of_male",
                                 "older_brother_of_female", "older_sister_of_female"}:
                flags.append(AmbiguityFlag(
                    kind=AmbiguityKind.GENDER, span=term.span, surface=term.surface,
                    explanation=f"{term.surface} also reveals the speaker's gender, which English drops.",
                    confidence=0.6))
        if "우리" in text:
            m = re.search("우리", text)
            flags.append(AmbiguityFlag(
                kind=AmbiguityKind.NUMBER, span=Span(m.start(), m.end()), surface="우리",
                explanation="우리 covers both an inclusive 'we/our' and a possessive 'my'.",
                options=["our", "my"], confidence=0.55))
        return flags


_UPOS = {"NN": "NOUN", "NP": "PRON", "VV": "VERB", "VA": "ADJ", "MA": "ADV",
         "JK": "ADP", "JX": "ADP", "EF": "AUX", "EC": "AUX", "SF": "PUNCT", "XS": "PART"}
