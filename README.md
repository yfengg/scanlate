# Scanlate

A multilingual comic scanlation workbench: manga, manhwa, manhua. It manages
the whole translation workflow rather than wrapping a chat model — source
analysis, machine translation, project glossary, translation memory, register
and cultural validation, sound effects, and human approval are separate
components with an explicit decision order between them.

The core is language-neutral. Everything specific to a language lives in an
adapter under `languages/`, so adding Chinese is writing an adapter, not
reworking the translation system. `tests/test_architecture.py` enforces that.

## Requirements

* Python 3.11+
* Tesseract, for OCR (optional — regions can be typed by hand)
* Node with jsdom, for the frontend tests only

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,fuzzy]'

# OCR (optional). Debian/Ubuntu:
sudo apt install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-jpn-vert tesseract-ocr-kor
pip install pillow opencv-python-headless pytesseract python-multipart

# Real neural MT instead of the deterministic phrase table (optional, large):
pip install -e '.[mt]'
```

Without Tesseract or OpenCV the app still runs: detection and OCR report
themselves unavailable and you draw regions and type source text by hand.

## Run

```bash
SCANLATE_DB=scanlate.db SCANLATE_MEDIA=media python -m scanlate
# then open http://127.0.0.1:8000
```

A fresh install opens on an empty **Projects** screen — there is no seeded
project. Create one, add a chapter, and import pages into that chapter.
`SCANLATE_DEMO=1` adds the demo fixture project for screenshots and manual
testing.

The workbench page and the API are served from the same process, so the page's
relative `/api/...` requests need no CORS and no separate dev server.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SCANLATE_DB` | `:memory:` | SQLite path. In-memory means nothing persists. |
| `SCANLATE_MEDIA` | `media` | Where imported page images are stored. |
| `SCANLATE_BACKEND` | `auto` | `auto` (OPUS-MT if installed, else the dev backend), `opus`, or `lexicon`. |
| `SCANLATE_OCR` | `auto` | `auto` (manga-ocr for Japanese, Tesseract otherwise), `tesseract`, `manga-ocr`, `none`. |
| `SCANLATE_DEMO` | unset | `1` seeds the demo fixture project. |
| `SCANLATE_DETECTOR` | `opencv` | `opencv` or `none`. |
| `SCANLATE_HOST` / `SCANLATE_PORT` | `127.0.0.1` / `8000` | Bind address. |

Alternatively: `uvicorn scanlate.api.app:create_app --factory`. The module
exposes a factory, not an app instance, because building one at import time
would open a database as an import side effect.

## Using it

**Review** is the chapter-wide mode: every segment with its source, candidate
translation, register signals, terminology and warnings. `j`/`k` move, `e`
edits, `a` approves.

**Page** is spatial. Pick a chapter, import a page, then use the toolbar:
**Select** to click and move regions, **Add region** to drag a new one (Esc
cancels), **Detect** to find text automatically, **OCR page** to read every
empty region. Selecting a region opens it on the right with its source,
translation, Approve, and Previous/Next. `Del` removes the selected region;
`j`/`k` step through them.

**Glossary** and **Memory** hold the project's terminology decisions and its
approved translations.

## How translation decisions are made

Precedence, highest first, implemented in `pipeline/pipeline.py`:

1. **Locked glossary terms** and **canonical translation memory** — approved
   project decisions. Machine translation never overwrites them.
2. **Unlocked exact memory** — precedent, offered as an alternative.
3. **Machine translation** — the default source of a candidate.
4. **Fuzzy memory** — suggestion only, never applied automatically.

Validators only report; the sole automatic rewrite is locked-glossary
enforcement, and only where the substitution is provably unambiguous.
Approval is the only way into translation memory, and it suggests glossary
rules rather than creating them.

## Layout

```
src/scanlate/
  core/          language-neutral vocabulary: spans, issues, register scale, adapter interface
  languages/     per-language adapters (ja, ko, en) with their knowledge data
  translation/   TranslationBackend interface; lexicon (deterministic) and OPUS-MT backends
  memory/        translation memory: variants, canonical locking, occurrences
  glossary/      project terminology and conservative enforcement
  validation/    register, punctuation, relationship, ambiguity, cultural validators
  sfx/           shared semantic categories and target-language renderers
  context/       lightweight local context (neighbouring lines only)
  pipeline/      TranslationPipeline and ApprovalService
  imaging/       page import, region detection, OCR, page workflow service
  storage/       SQLite schema, repositories, unit of work
  api/           FastAPI routes (thin; they call services)
  workbench/     SegmentView contract and the single-file frontend
tests/
  frontend/      jsdom suite driving the real HTML
```

## Translation and OCR backends

The header shows which translation backend is loaded. `lexicon (development)`
means no MT model is installed and the candidates are deterministic fixture
output, not translation — install `.[mt]` for OPUS-MT.

For Japanese OCR, manga-ocr is preferred when installed (`pip install
manga-ocr`: ~2–3 GB of torch, plus ~450 MB of weights downloaded on first use).
Tesseract is the fallback for every language, and on vertical Japanese it is
poor — `python tools/ocr_bench.py` builds a vertical-Japanese page and prints
both engines side by side.

## Tests

```bash
python -m pytest                       # 216 tests
node tests/frontend/test_workbench.mjs # 94 browser checks (needs: npm install jsdom)
```

`tests/frontend/trace_network.mjs` is a scriptable stand-in for watching the
browser's Network tab against a running server.

## Not implemented yet

Page cleaning, redraw, typesetting, SFX captions and image export. The region
and render data structures those need already persist; nothing renders text
into a page yet.
