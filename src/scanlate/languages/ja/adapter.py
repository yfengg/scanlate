"""Japanese adapter.

Everything Japanese-specific lives here: politeness morphology, honorific
affixes, sentence-final particles, kana normalization, furigana. The core sees
only the generic concepts these map onto.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from ...core.adapter import LanguageAdapter, NormalizedText
from ...core.analysis import (AmbiguityFlag, AmbiguityKind, AnalysisContext,
                              CulturalImpact, CulturalTermHit, EntityKind,
                              EntityMention, Evidence, HierarchyDirection,
                              InterlinearAnnotation, LinguisticAnalysis,
                              OcrCleanup, OcrProfile, RegisterLevel,
                              RegisterSignal, RelationshipTerm, SlangHit,
                              SocialMarker, SocialMarkerKind, Token,
                              aggregate_register)
from ...core.scripts import count_in_scripts, script_of
from ...core.types import SegmentKind, Span, select_non_overlapping, terminal_punctuation
from ...knowledge import KnowledgeBase
from ...sfx.categories import SfxCategory, SfxMatch, periodic_unit

DATA = Path(__file__).parent / "data"

# (pattern, register level, weight, flags, reason)
_REGISTER_RULES: list[tuple[str, RegisterLevel | None, float, tuple[str, ...], str]] = [
    (r"(でございま|いたします|申し訳|恐れ入り|でしょうか|ございません)", RegisterLevel.FORMAL, 2.0, (), "humble/formal verb"),
    (r"(?:で)?(?:す|ます)(?:か|ね|よ|が|けど)?[。、！？…]?$", RegisterLevel.POLITE, 1.5, (), "desu/masu ending"),
    (r"(ですか|ますか|ましょう|ください|ませんか|でした|ました|ですね|ですよ)", RegisterLevel.POLITE, 1.5, (), "desu/masu form"),
    (r"(ねぇよ|ねえよ|やがる|てめえ|てめぇ|きさま|貴様|うるせ|うぜ|しやがれ|くたばれ|ばか野郎|馬鹿野郎|クソ|くそ)", RegisterLevel.CRUDE, 2.0, ("rough",), "rough vocabulary"),
    (r"(だろ|かよ|んだよ|じゃねえ|じゃねぇ|ぞ$|ぜ$|や$)", RegisterLevel.CRUDE, 1.2, ("rough",), "blunt sentence ending"),
    (r"(だよ|だね|だな|のか|かな|よね|じゃん|けど$|から$)", RegisterLevel.CASUAL, 1.0, (), "plain sentence ending"),
    (r"(?<![まで])(?:る|た|う|い)[。、！？…]?$", RegisterLevel.CASUAL, 0.6, (), "plain form"),
]

_FEMININE = re.compile(r"(わよ|かしら|のよ$|わね)")
_MASCULINE = re.compile(r"(だぜ|だぞ|ぞ$|ぜ$|おれ|俺)")
_EMPHATIC = re.compile(r"[!！]{2,}|[っッ][!！]")

# Particles that overtly mark a subject. Their absence is what makes the
# omitted-argument reading likely, which English must then supply.
_SUBJECT_PARTICLE = re.compile(r"(は|が|も)")
_TOPIC_PRONOUN = re.compile(r"(私|僕|俺|あたし|わたし|あなた|君|きみ|お前|おまえ|彼|彼女)")

_SELF_REFERENCE = {
    "私": (RegisterLevel.POLITE, "neutral-polite first person"),
    "わたくし": (RegisterLevel.FORMAL, "highly formal first person"),
    "僕": (RegisterLevel.NEUTRAL, "soft masculine first person"),
    "俺": (RegisterLevel.CASUAL, "blunt masculine first person"),
    "あたし": (RegisterLevel.CASUAL, "casual feminine first person"),
    "うち": (RegisterLevel.CASUAL, "regional/casual first person"),
}
_ADDRESSEE = {
    "あなた": (RegisterLevel.POLITE, HierarchyDirection.NEUTRAL),
    "君": (RegisterLevel.CASUAL, HierarchyDirection.DOWNWARD),
    "きみ": (RegisterLevel.CASUAL, HierarchyDirection.DOWNWARD),
    "お前": (RegisterLevel.CASUAL, HierarchyDirection.DOWNWARD),
    "おまえ": (RegisterLevel.CASUAL, HierarchyDirection.DOWNWARD),
    "てめえ": (RegisterLevel.CRUDE, HierarchyDirection.HOSTILE),
    "てめぇ": (RegisterLevel.CRUDE, HierarchyDirection.HOSTILE),
    "貴様": (RegisterLevel.CRUDE, HierarchyDirection.HOSTILE),
}

_SENTENCE_END = re.compile(r"[。！？!?…‥]+[」』）]?|\n")
_NAME_CHARS = re.compile(r"[\u4E00-\u9FFF\u30A0-\u30FF\u3041-\u309F]{1,6}$")
_FURIGANA = re.compile(r"([\u4E00-\u9FFF]+)[（(]([\u3041-\u309F\u30A0-\u30FF]+)[）)]")
_KANA = frozenset({"Hiragana", "Katakana"})
_LEVELS = {l.name: l for l in RegisterLevel}


class JapaneseAdapter(LanguageAdapter):
    language = "ja"
    display_name = "Japanese"

    def __init__(self, knowledge: KnowledgeBase | None = None):
        self.knowledge = knowledge or KnowledgeBase("ja", DATA)
        self._tagger = None
        self._tagger_tried = False

    def capabilities(self) -> frozenset[str]:
        caps = {"normalize", "sentences", "register", "social_markers", "relationships",
                "ambiguity", "entities", "slang", "cultural", "sfx", "ocr_cleanup"}
        if self._load_tagger() is not None:
            caps |= {"tokenize", "morphology"}
        return frozenset(caps)

    # --- optional morphology ------------------------------------------
    def _load_tagger(self):
        """fugashi if installed; otherwise the adapter reports no morphology."""
        if not self._tagger_tried:
            self._tagger_tried = True
            try:
                import fugashi  # type: ignore
                self._tagger = fugashi.Tagger()
            except Exception:
                self._tagger = None
        return self._tagger

    def tokenize(self, text: str) -> list[Token]:
        tagger = self._load_tagger()
        if tagger is None:
            return []
        tokens, cursor = [], 0
        for word in tagger(text):
            surface = word.surface
            start = text.find(surface, cursor)
            if start < 0:
                continue
            cursor = start + len(surface)
            pos = getattr(word.feature, "pos1", None) or "X"
            tokens.append(Token(surface, start, cursor,
                                lemma=getattr(word.feature, "lemma", None) or None,
                                upos=_UPOS.get(pos, "X"), features={"pos": pos}))
        return tokens

    # --- normalization -------------------------------------------------
    def normalize(self, text: str) -> NormalizedText:
        edits: list[str] = []
        out = unicodedata.normalize("NFKC", text)
        if out != text:
            edits.append("NFKC normalization")
        # Full-width Latin/digits fold under NFKC, but CJK punctuation must not.
        for wrong, right, label in (("｡", "。", "half-width period"), ("｢", "「", "half-width bracket"),
                                    ("~", "〜", "wave dash"), ("...", "…", "ellipsis")):
            if wrong in out:
                out = out.replace(wrong, right)
                edits.append(label)
        collapsed = re.sub(r"[ \t\u3000]*\n[ \t\u3000]*", "", out).strip()
        if collapsed != out.strip():
            edits.append("joined wrapped lines")   # comic text wraps for the bubble, not for grammar
        return NormalizedText(collapsed, text, edits)

    def ocr_cleanup(self, text: str) -> OcrCleanup:
        edits: list[str] = []
        out = text
        # Confusions the OCR layer will produce; kept conservative.
        confusions = [("力", "カ", "Han 力 read as katakana カ"), ("口", "ロ", "Han 口 read as katakana ロ"),
                      ("工", "エ", "Han 工 read as katakana エ"), ("二", "ニ", "Han 二 read as katakana ニ")]
        katakana_ratio = (count_in_scripts(text, {"Katakana"}) / len(text)) if text else 0
        if katakana_ratio > 0.5:
            for han, kata, label in confusions:
                if han in out:
                    out = out.replace(han, kata)
                    edits.append(label)
        interlinear = [InterlinearAnnotation(base=m.group(1), annotation=m.group(2), kind="reading",
                                             base_span=Span(m.start(1), m.end(1)))
                       for m in _FURIGANA.finditer(out)]
        if interlinear:
            out = _FURIGANA.sub(r"\1", out)
            edits.append("extracted furigana")
        return OcrCleanup(text=out, interlinear=interlinear, edits=edits)

    def ocr_profile(self, text: str) -> OcrProfile:
        return OcrProfile(vertical_text_likely=True,
                          engine_hints={"scripts": "ja", "expect_furigana": True})

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
        stripped = re.sub(r"[ーっッ！!…・\s]", "", text)
        if not stripped or len(stripped) > 8:
            return False
        return count_in_scripts(stripped, _KANA) == len(stripped) and "Katakana" in {
            script_of(c) for c in stripped}

    def _sfx_index(self) -> dict[str, dict]:
        def build():
            index: dict[str, dict] = {}
            for entry in self.knowledge.entries("sfx"):
                index[entry["form"]] = entry
                # Also index the form with length marks stripped, so ドカーン
                # is found from the de-elongated ドカン.
                index.setdefault(re.sub(r"[ー〜]", "", entry["form"]), entry)
            return index
        return self.knowledge.cached("__sfx_index__", build)

    def lookup_sfx(self, text: str) -> SfxMatch | None:
        index = self._sfx_index()
        raw = text.strip()
        elongated = "ー" in raw or "〜" in raw
        core = re.sub(r"[ー〜！!…・\s]", "", raw)
        glottal = core.endswith("ッ") or core.endswith("っ")
        core = core.rstrip("ッっ")
        unit, reps = periodic_unit(core)

        for candidate, extra_reps in ((core, 1), (unit, reps), (unit + "ッ", reps), (core + "ッ", 1)):
            entry = index.get(candidate)
            if entry is None:
                continue
            intensity = int(entry.get("intensity", 1))
            if elongated or glottal:
                intensity = min(3, intensity + 1)
            if extra_reps > 1:
                intensity = min(3, intensity + 1)
            return SfxMatch(
                category=SfxCategory(entry["category"]),
                form=candidate,
                subtype=entry.get("subtype"),
                intensity=intensity,
                repetitions=extra_reps,
                elongated=elongated,
                confidence=0.9 if candidate == core else 0.75,
                note=entry.get("note"),
            )
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
                analysis.adapter_features["ja.sfx_form"] = match.form
            return analysis

        signals = self._register_signals(text)
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
        for hit in analysis.slang:
            if hit.register is not None:
                signals.append(RegisterSignal(
                    Evidence(hit.span, hit.surface, "slang"), hit.register, ("slang",), 0.9))

        analysis.register = aggregate_register(signals)
        if _FEMININE.search(text):
            analysis.register.tone_flags.add("feminine_speech")
        if _MASCULINE.search(text):
            analysis.register.tone_flags.add("masculine_speech")
        if _EMPHATIC.search(text):
            analysis.register.tone_flags.add("emphatic")

        analysis.ambiguities = self._ambiguities(text, analysis)
        analysis.adapter_features.update(self._adapter_features(text, analysis))
        return analysis

    def _register_signals(self, text: str) -> list[RegisterSignal]:
        signals: list[RegisterSignal] = []
        for sentence in self.segment_sentences(text):
            chunk = text[sentence.start:sentence.end]
            for pattern, level, weight, flags, reason in _REGISTER_RULES:
                m = re.search(pattern, chunk)
                if not m:
                    continue
                span = Span(sentence.start + m.start(), sentence.start + m.end())
                signals.append(RegisterSignal(Evidence(span, m.group(0), reason), level, flags, weight))
        for form, (level, reason) in _SELF_REFERENCE.items():
            idx = text.find(form)
            if idx >= 0:
                signals.append(RegisterSignal(
                    Evidence(Span(idx, idx + len(form)), form, reason), level, weight=0.7))
        return signals

    def _social_markers(self, text: str) -> list[SocialMarker]:
        markers: list[SocialMarker] = []
        for entry in self.knowledge.entries("honorifics"):
            form = entry["form"]
            for m in re.finditer(re.escape(form), text):
                preceding = _NAME_CHARS.search(text[:m.start()])
                attached = preceding.group(0) if preceding else None
                markers.append(SocialMarker(
                    kind=SocialMarkerKind(entry.get("kind", "address_affix")),
                    span=Span(m.start(), m.end()), surface=form,
                    direction=HierarchyDirection(entry.get("direction", "unknown")),
                    attached_to=attached, convention_key=entry.get("convention_key"),
                    register=_LEVELS.get(entry.get("register", ""), None),
                    note=entry.get("note")))
        for form, (level, direction) in _ADDRESSEE.items():
            for m in re.finditer(re.escape(form), text):
                markers.append(SocialMarker(
                    kind=SocialMarkerKind.ADDRESSEE_REFERENCE, span=Span(m.start(), m.end()),
                    surface=form, direction=direction, register=level,
                    note="Second-person choice carries the speaker's stance."))
        for form, (level, reason) in _SELF_REFERENCE.items():
            for m in re.finditer(re.escape(form), text):
                markers.append(SocialMarker(
                    kind=SocialMarkerKind.SELF_REFERENCE, span=Span(m.start(), m.end()),
                    surface=form, direction=HierarchyDirection.NEUTRAL, register=level, note=reason))
        if re.search(r"(いらっしゃ|おっしゃ|なさ(る|い)|ご覧)", text):
            m = re.search(r"(いらっしゃ|おっしゃ|なさ(る|い)|ご覧)", text)
            markers.append(SocialMarker(
                kind=SocialMarkerKind.RESPECTFUL_FORM, span=Span(m.start(), m.end()),
                surface=m.group(0), direction=HierarchyDirection.UPWARD,
                register=RegisterLevel.FORMAL, note="Honorific verb elevating the referent."))
        if re.search(r"(いたし|申し|伺(う|い)|拝見)", text):
            m = re.search(r"(いたし|申し|伺(う|い)|拝見)", text)
            markers.append(SocialMarker(
                kind=SocialMarkerKind.HUMBLE_FORM, span=Span(m.start(), m.end()),
                surface=m.group(0), direction=HierarchyDirection.UPWARD,
                register=RegisterLevel.FORMAL, note="Humble verb lowering the speaker."))
        return sorted(markers, key=lambda m: m.span.start)

    def _relationship_terms(self, text: str) -> list[RelationshipTerm]:
        hits = []
        for entry in self.knowledge.entries("relationships"):
            for m in re.finditer(re.escape(entry["form"]), text):
                hits.append(RelationshipTerm(
                    span=Span(m.start(), m.end()), surface=entry["form"],
                    relation=entry["relation"], may_be_non_kin=bool(entry.get("may_be_non_kin")),
                    implies={"options": ", ".join(entry.get("options", []))},
                    note=entry.get("note")))
        return select_non_overlapping(hits, lambda h: h.span)

    def _cultural_terms(self, text: str) -> list[CulturalTermHit]:
        hits = []
        for entry in self.knowledge.entries("cultural"):
            for m in re.finditer(re.escape(entry["form"]), text):
                hits.append(CulturalTermHit(
                    span=Span(m.start(), m.end()), surface=entry["form"], key=entry["key"],
                    domain=entry["domain"], impact=CulturalImpact(int(entry.get("impact", 1))),
                    note=entry["note"], suggestions=list(entry.get("suggestions", []))))
        return hits

    def _slang(self, text: str) -> list[SlangHit]:
        hits = []
        for entry in self.knowledge.entries("slang"):
            for m in re.finditer(re.escape(entry["form"]), text):
                hits.append(SlangHit(span=Span(m.start(), m.end()), surface=entry["form"],
                                     meaning=entry["meaning"],
                                     register=_LEVELS.get(entry.get("register", ""), None)))
        return hits

    def _entities(self, text: str, markers: list[SocialMarker]) -> list[EntityMention]:
        """Names are inferred from what precedes an address affix, not from a gazetteer."""
        out = []
        for marker in markers:
            if marker.kind is SocialMarkerKind.ADDRESS_AFFIX and marker.attached_to:
                start = marker.span.start - len(marker.attached_to)
                out.append(EntityMention(span=Span(start, marker.span.start),
                                         surface=marker.attached_to, kind=EntityKind.PERSON,
                                         confidence=0.6, source="ja.honorific_attachment"))
        return out

    def _ambiguities(self, text: str, analysis: LinguisticAnalysis) -> list[AmbiguityFlag]:
        flags: list[AmbiguityFlag] = []
        if not _SUBJECT_PARTICLE.search(text) and not _TOPIC_PRONOUN.search(text) \
                and len(text) > 4 and analysis.register.level is not None:
            flags.append(AmbiguityFlag(
                kind=AmbiguityKind.OMITTED_ARGUMENT, span=Span(0, len(text)), surface=text,
                explanation="No subject is stated; English has to supply one.",
                options=["I", "you", "they", "he/she"], confidence=0.6))
        for term in analysis.relationship_terms:
            if term.may_be_non_kin:
                flags.append(AmbiguityFlag(
                    kind=AmbiguityKind.SOCIAL_RELATION, span=term.span, surface=term.surface,
                    explanation=f"{term.surface} may address an actual relative or an unrelated senior.",
                    options=term.implies.get("options", "").split(", "), confidence=0.7))
        for m in re.finditer(r"(彼ら|彼女たち|たち|ら)(?=[はがを])", text):
            flags.append(AmbiguityFlag(
                kind=AmbiguityKind.NUMBER, span=Span(m.start(), m.end()), surface=m.group(0),
                explanation="Plural marking is optional in Japanese; the count may be unclear.",
                confidence=0.5))
        if analysis.interlinear:
            for note in analysis.interlinear:
                if note.kind == "gloss":
                    flags.append(AmbiguityFlag(
                        kind=AmbiguityKind.INTERLINEAR_GLOSS, span=note.base_span or Span(0, 0),
                        surface=note.base,
                        explanation="Printed reading differs from the written word; both carry meaning.",
                        options=[note.base, note.annotation], confidence=0.8))
        return flags

    def _adapter_features(self, text: str, analysis: LinguisticAnalysis) -> dict:
        """Japanese detail the generic scale can't hold, namespaced for the UI."""
        features: dict[str, object] = {}
        if re.search(r"(でございま|いたします|申し訳|伺(う|い)|拝見|いらっしゃ)", text):
            features["ja.politeness"] = "keigo"
        elif re.search(r"(です|ます|ました|ません|でしょう)", text):
            features["ja.politeness"] = "desu/masu"
        else:
            features["ja.politeness"] = "plain"
        particles = [p for p in ("よ", "ね", "な", "ぞ", "ぜ", "さ", "わ", "かな", "かよ")
                     if re.search(re.escape(p) + r"[。！？…]?$", text.rstrip())]
        if particles:
            features["ja.sentence_final_particles"] = ", ".join(particles)
        self_ref = [f for f in _SELF_REFERENCE if f in text]
        if self_ref:
            features["ja.self_reference"] = ", ".join(self_ref)
        if re.search(r"[\u3041-\u309F]", text) and not re.search(r"[\u4E00-\u9FFF]", text):
            features["ja.script"] = "kana only"
        return features


_UPOS = {"名詞": "NOUN", "動詞": "VERB", "形容詞": "ADJ", "助詞": "ADP", "助動詞": "AUX",
         "副詞": "ADV", "接続詞": "CCONJ", "感動詞": "INTJ", "代名詞": "PRON", "記号": "PUNCT"}
