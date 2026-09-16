"""Shared semantic categories for sound effects.

SFX are treated as semantic events. Language adapters map source forms into
these categories; target-language renderers choose a rendering from the
category, intensity and scene context.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SfxCategory(str, Enum):
    IMPACT = "impact"
    HEARTBEAT = "heartbeat"
    FOOTSTEPS = "footsteps"
    MOVEMENT = "movement"
    RUSTLING = "rustling"
    MACHINERY = "machinery"
    WEATHER = "weather"
    SILENCE = "silence"
    ATMOSPHERIC = "atmospheric"
    EMOTIONAL = "emotional"
    ANIMAL = "animal"
    VOICE = "voice"


@dataclass
class SfxMatch:
    category: SfxCategory
    form: str                         # lexicon form that matched
    subtype: str | None = None
    intensity: int = 1                # 0..3
    repetitions: int = 1
    elongated: bool = False
    confidence: float = 0.9
    alternatives: list[SfxCategory] = field(default_factory=list)
    note: str | None = None


def periodic_unit(text: str) -> tuple[str, int]:
    """Smallest repeating unit: 'abab' -> ('ab', 2)."""
    n = len(text)
    for size in range(1, n // 2 + 1):
        if n % size == 0 and text[:size] * (n // size) == text:
            return text[:size], n // size
    return text, 1
