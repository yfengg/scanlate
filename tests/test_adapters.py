"""Adapter behaviour on known examples (stage 2 of the backend order)."""
from __future__ import annotations

import pytest

from scanlate.core.analysis import (AmbiguityKind, AnalysisContext, RegisterLevel,
                                    SocialMarkerKind)
from scanlate.core.types import SegmentKind
from scanlate.languages.ja.adapter import JapaneseAdapter
from scanlate.languages.ko.adapter import KoreanAdapter
from scanlate.sfx.categories import SfxCategory


@pytest.fixture(scope="module")
def ja():
    return JapaneseAdapter()


@pytest.fixture(scope="module")
def ko():
    return KoreanAdapter()


# --- Japanese ----------------------------------------------------------
@pytest.mark.parametrize("text,level", [
    ("申し訳ございません。", RegisterLevel.FORMAL),
    ("田中先輩、ちょっといいですか？", RegisterLevel.POLITE),
    ("うるせぇな…今忙しいんだよ。", RegisterLevel.CRUDE),
    ("てめえ、ふざけんな", RegisterLevel.CRUDE),
])
def test_japanese_register(ja, text, level):
    assert ja.analyze(text).register.level is level


def test_japanese_honorific_attaches_to_a_name(ja):
    analysis = ja.analyze("田中先輩、ちょっといいですか？")
    marker = next(m for m in analysis.social_markers if m.surface == "先輩")
    assert marker.kind is SocialMarkerKind.TITLE
    assert marker.attached_to == "田中"
    assert marker.convention_key == "senpai"
    assert [e.surface for e in analysis.entities] == []      # 先輩 is a title, not an affix


def test_japanese_relationship_term_and_ambiguity(ja):
    analysis = ja.analyze("兄貴、助けてくれ！")
    term = analysis.relationship_terms[0]
    assert (term.surface, term.relation, term.may_be_non_kin) == ("兄貴", "older_brother", True)
    assert AmbiguityKind.SOCIAL_RELATION in {a.kind for a in analysis.ambiguities}


def test_japanese_keeps_language_specific_detail_in_adapter_features(ja):
    features = ja.analyze("うるせぇな…今忙しいんだよ。").adapter_features
    assert features["ja.politeness"] == "plain"
    assert features["ja.sentence_final_particles"] == "よ"
    assert all(key.startswith("ja.") for key in features)


def test_japanese_normalization_and_furigana(ja):
    normalized = ja.normalize("ｱｲｳ\n　です")
    assert "joined wrapped lines" in normalized.edits and normalized.text == "アイウです"
    cleanup = ja.ocr_cleanup("先輩（せんぱい）")
    assert cleanup.text == "先輩"
    assert (cleanup.interlinear[0].base, cleanup.interlinear[0].annotation) == ("先輩", "せんぱい")


def test_japanese_sfx_lookup_scales_with_form(ja):
    assert ja.lookup_sfx("ドクン").category is SfxCategory.HEARTBEAT
    assert ja.lookup_sfx("ドカーン").intensity == 3            # elongation raises intensity
    # A doubled form the lexicon lists outright matches whole.
    assert ja.lookup_sfx("コツコツ").subtype == "hard_sole"
    assert ja.lookup_sfx("コツコツ").repetitions == 1
    # A form repeated beyond its lexicon entry is read as repetition.
    repeated = ja.lookup_sfx("ドクンドクン")
    assert repeated.repetitions == 2 and repeated.intensity == 3
    assert ja.lookup_sfx("知らない") is None
    assert ja.looks_like_sfx("ドクン") and not ja.looks_like_sfx("今忙しい")


def test_sfx_segments_skip_dialogue_analysis(ja):
    analysis = ja.analyze("ドクン", AnalysisContext(segment_kind=SegmentKind.SFX))
    assert analysis.relationship_terms == [] and analysis.social_markers == []
    assert analysis.adapter_features["ja.sfx_form"] == "ドクン"


# --- Korean ------------------------------------------------------------
@pytest.mark.parametrize("text,level,speech_level", [
    ("저희가 처리하겠습니다.", RegisterLevel.FORMAL, "hapsyoche"),
    ("선배님, 이거 봐 주세요.", RegisterLevel.POLITE, "haeyoche"),
    ("형, 진짜 괜찮아?", RegisterLevel.CASUAL, "haeche"),
])
def test_korean_speech_levels(ko, text, level, speech_level):
    analysis = ko.analyze(text)
    assert analysis.register.level is level
    assert analysis.adapter_features["ko.speech_level"] == speech_level


def test_korean_kinship_term_carries_options_and_gender_signal(ko):
    analysis = ko.analyze("형, 진짜 괜찮아?")
    term = analysis.relationship_terms[0]
    assert term.surface == "형" and term.may_be_non_kin
    assert "hyung" in term.implies["options"]
    kinds = {a.kind for a in analysis.ambiguities}
    assert {AmbiguityKind.SOCIAL_RELATION, AmbiguityKind.GENDER} <= kinds


def test_korean_respects_word_boundaries(ko):
    """형 inside 형태 ('form') is not the kinship term."""
    assert ko.analyze("그 형태는 다르다.").relationship_terms == []


def test_korean_honorific_and_humble_markers(ko):
    analysis = ko.analyze("저희가 처리하겠습니다.")
    assert any(m.kind is SocialMarkerKind.HUMBLE_FORM for m in analysis.social_markers)
    assert analysis.adapter_features["ko.humble_self_reference"] == "yes"


def test_korean_sfx(ko):
    assert ko.lookup_sfx("두근두근").category is SfxCategory.HEARTBEAT
    assert ko.lookup_sfx("쾅").intensity == 3
    assert ko.lookup_sfx("괜찮아") is None


# --- shared vocabulary -------------------------------------------------
def test_both_adapters_produce_comparable_register(ja, ko):
    """The shared scale is comparable; the detail underneath is not conflated."""
    japanese = ja.analyze("申し訳ございません。").register.level
    korean = ko.analyze("저희가 처리하겠습니다.").register.level
    assert japanese is korean is RegisterLevel.FORMAL
    assert ja.analyze("申し訳ございません。").adapter_features.keys() != \
           ko.analyze("저희가 처리하겠습니다.").adapter_features.keys()


def test_unsupported_capabilities_return_empty_not_fake(ja, ko):
    for adapter in (ja, ko):
        if "tokenize" not in adapter.capabilities():
            assert adapter.tokenize("テスト") == []
