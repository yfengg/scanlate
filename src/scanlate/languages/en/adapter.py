"""English adapter.

Used as a *target*-side adapter: the pipeline reads the register of the
candidate translation so it can be compared against the source reading.
"""
from __future__ import annotations

import re

from ...core.adapter import LanguageAdapter, NormalizedText
from ...core.analysis import (AnalysisContext, Evidence, LinguisticAnalysis,
                              RegisterLevel, RegisterSignal, aggregate_register)
from ...core.types import Span, terminal_punctuation

_RULES: list[tuple[str, RegisterLevel, float, str]] = [
    (r"\b(damn|hell|shut up|bastard|piss off|screw (you|off)|crap|bloody|jerk|idiot|freaking|goddamn)\b",
     RegisterLevel.CRUDE, 2.0, "rough vocabulary"),
    (r"\b(ain'?t|gonna|wanna|gotta|lemme|dunno|yeah|nah|hey|quit (it|bugging)|buzz off|get lost)\b",
     RegisterLevel.CASUAL, 1.2, "colloquial form"),
    (r"\b(would you kindly|if you would|i should be (grateful|obliged)|permit me|allow me to|"
     r"we regret to|sincerely|hereby|furthermore|shall i)\b",
     RegisterLevel.FORMAL, 2.0, "formal construction"),
    (r"\b(please|thank you|excuse me|pardon me|may i|could you|would you mind|sir|ma'?am|apologies)\b",
     RegisterLevel.POLITE, 1.5, "polite marker"),
    (r"\b(do not|cannot|will not|is not|are not|i am|it is)\b",
     RegisterLevel.POLITE, 0.7, "uncontracted form"),
]
_CONTRACTION = re.compile(r"\b\w+'(s|t|re|ll|ve|d|m)\b", re.I)


class EnglishAdapter(LanguageAdapter):
    language = "en"
    display_name = "English"

    def capabilities(self) -> frozenset[str]:
        return frozenset({"normalize", "sentences", "register"})

    def normalize(self, text: str) -> NormalizedText:
        collapsed = re.sub(r"\s+", " ", text).strip()
        return NormalizedText(collapsed, text,
                              ["collapsed whitespace"] if collapsed != text.strip() else [])

    def segment_sentences(self, text: str) -> list[Span]:
        spans, start = [], 0
        for m in re.finditer(r"[.!?]+[\"')\]]?\s+|\n", text):
            spans.append(Span(start, m.end()))
            start = m.end()
        if start < len(text):
            spans.append(Span(start, len(text)))
        return spans or ([Span(0, len(text))] if text else [])

    def analyze(self, text: str, context: AnalysisContext | None = None) -> LinguisticAnalysis:
        signals: list[RegisterSignal] = []
        for pattern, level, weight, reason in _RULES:
            for m in re.finditer(pattern, text, re.I):
                signals.append(RegisterSignal(
                    Evidence(Span(m.start(), m.end()), m.group(0), reason), level, weight=weight))
        contractions = len(_CONTRACTION.findall(text))
        if contractions:
            signals.append(RegisterSignal(
                Evidence(Span(0, len(text)), text, "contractions"),
                RegisterLevel.CASUAL, weight=min(1.0, 0.4 * contractions)))
        analysis = LinguisticAnalysis(
            language="en", text=text, sentences=self.segment_sentences(text),
            register=aggregate_register(signals),
            terminal_punctuation=set(terminal_punctuation(text)))
        if re.search(r"[!]{2,}|[A-Z]{3,}", text):
            analysis.register.tone_flags.add("emphatic")
        return analysis
