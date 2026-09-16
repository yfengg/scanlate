"""Script-based language detection.

Uses Unicode script distribution only. Han-only text is genuinely ambiguous
between Japanese and Chinese, so it returns None and lets the project default
decide rather than guessing.
"""
from __future__ import annotations

from ..core.scripts import script_counts


def detect_language(text: str) -> str | None:
    counts = script_counts(text)
    total = sum(counts.values())
    if not total:
        return None
    kana = counts.get("Hiragana", 0) + counts.get("Katakana", 0)
    hangul = counts.get("Hangul", 0)
    han = counts.get("Han", 0)
    latin = counts.get("Latin", 0)
    if hangul and hangul >= kana:
        return "ko"
    if kana:
        return "ja"
    if han:
        return None          # ja or zh; caller falls back to the project default
    if latin / total > 0.8:
        return "en"
    return None
