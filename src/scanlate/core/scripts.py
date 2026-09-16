"""Unicode script classification (generic Unicode data, no linguistics)."""
from __future__ import annotations

from collections import Counter

_RANGES: dict[str, list[tuple[int, int]]] = {
    "Latin": [(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0xFF21, 0xFF3A), (0xFF41, 0xFF5A)],
    "Hiragana": [(0x3040, 0x309F)],
    "Katakana": [(0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F)],
    "Han": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF), (0x3005, 0x3005)],
    "Hangul": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F)],
    "Cyrillic": [(0x400, 0x4FF)],
    "Greek": [(0x370, 0x3FF)],
    "Thai": [(0xE00, 0xE7F)],
    "Arabic": [(0x600, 0x6FF)],
    "Devanagari": [(0x900, 0x97F)],
}


def script_of(ch: str) -> str | None:
    cp = ord(ch)
    for name, ranges in _RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return name
    return None


def script_counts(text: str) -> Counter:
    counts: Counter = Counter()
    for ch in text:
        name = script_of(ch)
        if name is None and ch.isalpha():
            name = "Other"
        if name is not None:
            counts[name] += 1
    return counts


def script_profile(text: str) -> dict[str, float]:
    counts = script_counts(text)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def count_in_scripts(text: str, scripts) -> int:
    return sum(1 for ch in text if script_of(ch) in scripts)
