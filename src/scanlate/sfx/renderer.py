"""Target-language SFX renderers.

A renderer turns a language-neutral ``SfxMatch`` into target-language text.
Rendering is chosen from category, subtype, intensity, repetition and scene
tags, never from the source form itself.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from .categories import SfxCategory, SfxMatch

C = SfxCategory


@dataclass
class SfxRendering:
    text: str
    rationale: str


class SfxRenderer(ABC):
    target_language: str

    @abstractmethod
    def render(self, match: SfxMatch, scene_tags: set[str] | None = None) -> list[SfxRendering]:
        ...


# Tiers are indexed by intensity 0..3.
_EN_TABLE: dict[tuple[SfxCategory, str | None], list[str]] = {
    (C.IMPACT, None): ["tap", "thud", "BAM", "BOOM"],
    (C.IMPACT, "heavy"): ["thud", "THUD", "BOOM", "KA-BOOM"],
    (C.IMPACT, "explosion"): ["pop", "BANG", "BOOM", "KA-BOOM"],
    (C.IMPACT, "sharp"): ["tak", "SMACK", "BANG", "BANG"],
    (C.IMPACT, "door"): ["click", "clack", "SLAM", "SLAM"],
    (C.IMPACT, "shatter"): ["tink", "crack", "CRASH", "KSSHH"],
    (C.IMPACT, "punch"): ["pap", "THWACK", "POW", "KAPOW"],
    (C.IMPACT, "thud"): ["thup", "thud", "THUD", "THOOM"],
    (C.HEARTBEAT, None): ["ba-dum", "ba-dump", "BA-DUMP", "BA-DUMP"],
    (C.HEARTBEAT, "heavy"): ["thump", "THUMP", "THUMP", "THOOM"],
    (C.HEARTBEAT, "light"): ["pit-pat", "pitter-pat", "ba-dum", "BA-DUM"],
    (C.HEARTBEAT, "startled"): ["ba-dmp", "BA-DUMP", "BA-DUMP", "BA-DUMP"],
    (C.FOOTSTEPS, None): ["tap", "step", "stomp", "STOMP"],
    (C.FOOTSTEPS, "hard_sole"): ["tok", "clack", "clack", "CLACK"],
    (C.FOOTSTEPS, "brisk"): ["tap", "tap", "tmp", "TMP"],
    (C.FOOTSTEPS, "heavy"): ["thmp", "stomp", "STOMP", "STOMP"],
    (C.FOOTSTEPS, "weary"): ["shuffle", "trudge", "trudge", "TRUDGE"],
    (C.FOOTSTEPS, "running"): ["pat", "tatata", "TATATA", "TATATATA"],
    (C.MOVEMENT, None): ["shff", "swish", "WHOOSH", "WHOOSH"],
    (C.MOVEMENT, "quick"): ["fft", "swish", "SWISH", "FWOOSH"],
    (C.MOVEMENT, "whoosh"): ["fwip", "whoosh", "WHOOSH", "FWOOSH"],
    (C.MOVEMENT, "slide"): ["shff", "slide", "SLIDE", "SLIDE"],
    (C.MOVEMENT, "rolling"): ["roll", "rumble", "RUMBLE", "RUMBLE"],
    (C.RUSTLING, None): ["rustle", "rustle", "RUSTLE", "RUSTLE"],
    (C.RUSTLING, "dry"): ["skritch", "rustle", "crinkle", "CRINKLE"],
    (C.RUSTLING, "heavy"): ["rustle", "shff", "RUSTLE", "RUSTLE"],
    (C.MACHINERY, None): ["whir", "whirr", "WHIRR", "WHRRRR"],
    (C.MACHINERY, "rattle"): ["clatter", "rattle", "RATTLE", "CLATTER"],
    (C.MACHINERY, "motor"): ["whir", "vwoom", "VRRRM", "VRRRRM"],
    (C.WEATHER, None): ["shhh", "whoosh", "WHOOSH", "WHOOSH"],
    (C.WEATHER, "rain"): ["pitter", "shhh", "SHHHH", "FSHHHH"],
    (C.WEATHER, "thunder"): ["rumble", "rumble", "RUMBLE", "KRAKOOM"],
    (C.WEATHER, "wind"): ["fwoo", "whoosh", "WHOOSH", "WHOOOSH"],
    (C.SILENCE, None): ["...", "...", "(silence)", "SILENCE"],
    (C.ATMOSPHERIC, None): ["hmm", "rumble", "RUMBLE", "RUMBLE"],
    (C.ATMOSPHERIC, "menace"): ["rmmm", "rumble", "RUMBLE", "RRRUMBLE"],
    (C.ATMOSPHERIC, "flash"): ["flash", "flash", "FLASH", "FLASH"],
    (C.ATMOSPHERIC, "dramatic"): ["dun", "dun dun", "DUN DUN", "DUN DUN DUN"],
    (C.EMOTIONAL, None): ["...", "...", "!", "!!"],
    (C.EMOTIONAL, "smirk"): ["heh", "smirk", "smirk", "SMIRK"],
    (C.EMOTIONAL, "smile"): ["hehe", "smile", "grin", "GRIN"],
    (C.EMOTIONAL, "flinch"): ["twitch", "flinch", "FLINCH", "FLINCH"],
    (C.EMOTIONAL, "irritation"): ["grr", "grrr", "GRRR", "GRRRR"],
    (C.EMOTIONAL, "trembling"): ["shiver", "tremble", "shake", "SHAKE"],
    (C.EMOTIONAL, "lazing"): ["flop", "loll", "sprawl", "sprawl"],
    (C.ANIMAL, None): ["chirp", "chirp", "CRY", "CRY"],
    (C.ANIMAL, "dog"): ["yip", "woof", "WOOF", "ARF"],
    (C.ANIMAL, "cat"): ["mew", "meow", "MEOW", "MRROW"],
    (C.VOICE, None): ["hm", "huh", "HUH", "HUH"],
    (C.VOICE, "crowd_murmur"): ["murmur", "murmur", "MURMUR", "CLAMOR"],
    (C.VOICE, "sigh"): ["hah", "sigh", "haah", "HAAAH"],
    (C.VOICE, "scream"): ["eep", "eek", "EEEK", "AAAAAH"],
    (C.VOICE, "gasp"): ["hk", "gasp", "GASP", "GASP"],
}

# Scene-tag driven subtype overrides: (category, tag) -> subtype
_SCENE_OVERRIDES: dict[tuple[SfxCategory, str], str] = {
    (C.HEARTBEAT, "hostile"): "heavy",
    (C.HEARTBEAT, "fear"): "heavy",
    (C.HEARTBEAT, "affectionate"): "light",
    (C.ATMOSPHERIC, "hostile"): "menace",
}

_REPEATABLE = {C.FOOTSTEPS, C.HEARTBEAT, C.RUSTLING, C.MACHINERY, C.ANIMAL}


def _stretch(word: str) -> str:
    m = list(re.finditer(r"[aeiouAEIOU]+", word))
    if not m:
        return word
    last = m[-1]
    return word[: last.end()] + last.group(0)[-1] * 2 + word[last.end():]


class EnglishSfxRenderer(SfxRenderer):
    target_language = "en"

    def _tiers(self, category: SfxCategory, subtype: str | None) -> tuple[list[str], str]:
        if (category, subtype) in _EN_TABLE:
            return _EN_TABLE[(category, subtype)], f"{category.value}/{subtype}"
        return _EN_TABLE[(category, None)], category.value

    def _one(self, category: SfxCategory, subtype: str | None, match: SfxMatch) -> SfxRendering:
        tiers, label = self._tiers(category, subtype)
        word = tiers[max(0, min(3, match.intensity))]
        if match.elongated and word[:1].isalpha():
            word = _stretch(word)
        if match.repetitions >= 2 and category in _REPEATABLE:
            word = " ".join([word] * min(match.repetitions, 3))
        reason = f"{label}, intensity {match.intensity}"
        if match.repetitions > 1:
            reason += f", repeated x{match.repetitions}"
        return SfxRendering(word, reason)

    def render(self, match: SfxMatch, scene_tags: set[str] | None = None) -> list[SfxRendering]:
        scene_tags = scene_tags or set()
        subtype = match.subtype
        for tag in sorted(scene_tags):
            override = _SCENE_OVERRIDES.get((match.category, tag))
            if override and subtype is None:
                subtype = override
        out = [self._one(match.category, subtype, match)]
        if subtype is not None:
            generic = self._one(match.category, None, match)
            if generic.text != out[0].text:
                out.append(generic)
        # Intensity is a judgement about the panel, not a fact about the word,
        # so the neighbouring tiers are offered too.
        for delta, label in ((-1, "quieter"), (1, "louder")):
            level = match.intensity + delta
            if 0 <= level <= 3:
                nearby = self._one(match.category, subtype,
                                   replace(match, intensity=level))
                nearby.rationale = f"{label} reading"
                out.append(nearby)
        for alt in match.alternatives:
            r = self._one(alt, None, match)
            r.rationale = f"alternative reading: {r.rationale}"
            out.append(r)
        seen, unique = set(), []
        for r in out:
            if r.text not in seen:
                seen.add(r.text)
                unique.append(r)
        return unique


class SfxRendererRegistry:
    def __init__(self) -> None:
        self._renderers: dict[str, SfxRenderer] = {}

    def register(self, renderer: SfxRenderer) -> None:
        self._renderers[renderer.target_language] = renderer

    def get(self, language: str) -> SfxRenderer | None:
        return self._renderers.get(language) or self._renderers.get(language.split("-")[0])
