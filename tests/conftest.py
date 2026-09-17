"""Session-wide test fixtures.

Deterministic-by-default backend/OCR selection. ``build_services()`` reads
``SCANLATE_BACKEND``/``SCANLATE_OCR`` from the environment when a test's
fixture doesn't pass an explicit override, and both fall back to whatever
happens to be genuinely installed and reachable (real transformers/torch for
OPUS-MT, a real Tesseract binary, a real manga-ocr model). That means an
ordinary test's behaviour could silently change based on what's installed on
the machine running it, rather than on what the test actually asked for.

The two autouse fixtures below close that gap: every test gets the
deterministic Lexicon translation backend and no OCR backend unless it says
otherwise. A test that specifically exercises auto-selection, a particular
backend, or real OCR content must explicitly override the relevant
environment variable *and* control the backend's own `.available()` (or
`.unavailable_reason()`) — see ``tests/test_regressions.py``'s
``test_opus_is_preferred_and_lexicon_covers_the_rest`` for the established
pattern — or replace ``services.page_service.ocr`` directly after
construction, the way ``tests/test_page_workflow.py`` already does with
``FakeOcr``. Either way, a real model happening to be installed must not be
what decides which path the test takes.
"""
from __future__ import annotations

import pytest

from scanlate.fixtures import BACKEND_ENV


@pytest.fixture(autouse=True)
def deterministic_translation_backend(monkeypatch):
    monkeypatch.setenv(BACKEND_ENV, "lexicon")


@pytest.fixture(autouse=True)
def deterministic_ocr_backend(monkeypatch):
    monkeypatch.setenv("SCANLATE_OCR", "none")
