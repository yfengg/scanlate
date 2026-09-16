"""The core must stay language-neutral.

This is the constraint the whole design rests on: if Japanese or Korean
concepts leak into shared modules, adding a language later means reworking the
translation system instead of writing an adapter.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "scanlate"
SHARED = ["core", "glossary", "memory", "validation", "pipeline", "translation",
          "sfx", "context", "storage", "workbench", "imaging"]
LANGUAGE_MODULES = ("scanlate.languages.ja", "scanlate.languages.ko", "scanlate.languages.en",
                    ".languages.ja", ".languages.ko", ".languages.en", "ja.adapter", "ko.adapter")


def _imports(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append("." * node.level + (node.module or ""))
    return names


def shared_files():
    return [p for package in SHARED for p in (SRC / package).rglob("*.py")]


@pytest.mark.parametrize("path", shared_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_shared_modules_do_not_import_language_adapters(path):
    offenders = [name for name in _imports(path)
                 if any(marker in name for marker in LANGUAGE_MODULES)]
    assert not offenders, f"{path.relative_to(SRC)} imports {offenders}"


@pytest.mark.parametrize("path", list((SRC / "core").rglob("*.py")),
                         ids=lambda p: p.name)
def test_core_mentions_no_language_specific_concepts(path):
    text = path.read_text(encoding="utf-8")
    forbidden = ["furigana", "keigo", "desu", "masu", "honorific-san", "hiragana-only",
                 "banmal", "hapsyoche", "haeyoche", "chengyu", "成语", "ー particle"]
    hits = [word for word in forbidden if word.lower() in text.lower()]
    assert not hits, f"core/{path.name} references {hits}"


def test_adapters_only_reach_the_core_through_shared_vocabulary():
    """Adapters may import core and knowledge, never each other."""
    for language in ("ja", "ko", "en"):
        for path in (SRC / "languages" / language).rglob("*.py"):
            others = [n for n in _imports(path)
                      if any(f"languages.{other}" in n for other in {"ja", "ko", "en"} - {language})]
            assert not others, f"{path.relative_to(SRC)} imports another adapter: {others}"


def test_every_adapter_satisfies_the_interface():
    from scanlate.core.adapter import CAPABILITIES, LanguageAdapter
    from scanlate.languages import default_registry

    registry = default_registry()
    assert set(registry.languages()) >= {"ja", "ko", "en"}
    for language in registry.languages():
        adapter = registry.require(language)
        assert isinstance(adapter, LanguageAdapter)
        assert adapter.capabilities() <= CAPABILITIES
        # Unsupported features return empty data rather than fabricated results.
        analysis = adapter.analyze("")
        assert analysis.language == language
        assert analysis.tokens == [] or "tokenize" in adapter.capabilities()


def test_pipeline_does_not_special_case_a_language():
    source = (SRC / "pipeline" / "pipeline.py").read_text(encoding="utf-8")
    for marker in ("ja", "ko", "zh", "japanese", "korean"):
        assert f'"{marker}"' not in source.lower(), f"pipeline hardcodes {marker!r}"
