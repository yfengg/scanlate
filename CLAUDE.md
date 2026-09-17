# Scanlate — project context for Claude Code

A multilingual comic scanlation workbench (manga, manhwa, manhua). It owns the
whole translation workflow; it is **not** a wrapper around a chat model, and no
general-purpose LLM is used for translation, OCR, or region detection.

## How to run and test

```bash
pip install -e '.[dev,fuzzy]'
SCANLATE_DB=scanlate.db SCANLATE_MEDIA=media python -m scanlate   # http://127.0.0.1:8000

python -m pytest                        # 216 tests
node tests/frontend/test_workbench.mjs  # 105 jsdom checks (npm install first)
python tools/ocr_bench.py               # vertical-Japanese page, Tesseract vs manga-ocr
```

Pillow and python-multipart are plain dependencies (page import needs them
unconditionally) and install with the line above. The OCR/detection-backed
tests additionally need `.[detect,ocr]` (opencv-python-headless,
pytesseract) — and, for the two that assert real recognized text rather than
just "ok or a graceful error", the tesseract binary itself on PATH, which pip
cannot install. `.[mangaocr]` adds manga-ocr for Japanese.

Environment: `SCANLATE_DB` (`:memory:`), `SCANLATE_MEDIA` (`media`),
`SCANLATE_BACKEND` (`auto|opus|lexicon`), `SCANLATE_OCR`
(`auto|tesseract|manga-ocr|none`), `SCANLATE_DETECTOR` (`opencv|none`),
`SCANLATE_DEMO=1` (seed fixture project), `SCANLATE_HOST`/`SCANLATE_PORT`.

The frontend is one file: `src/scanlate/workbench/static/index.html`, vanilla
JS, no build step. It is served by the same FastAPI process as the API, so
relative `/api/...` requests are same-origin and no CORS is configured. A
`?api=` query parameter overrides the base for working on the file directly;
**every** request must go through it (there are regressions for this).

## Architecture

```
src/scanlate/
  core/          language-neutral vocabulary: Span, Issue, Action, RegisterLevel,
                 LanguageAdapter interface, Unicode script helpers
  languages/     adapters: ja, ko, en (+ JSON knowledge data per language)
  translation/   TranslationBackend interface, BackendRegistry, lexicon + OPUS backends
  memory/        translation memory: variants, canonical locking, occurrences
  glossary/      project terminology + conservative enforcement
  validation/    register, punctuation, relationship, ambiguity, cultural validators
  sfx/           shared semantic categories + target-language renderers
  context/       local context (neighbouring lines only)
  pipeline/      TranslationPipeline, ApprovalService
  imaging/       page import, region detection, OCR, PageService
  storage/       SQLite schema, repositories, UnitOfWork
  api/           FastAPI routes (thin — they call services)
  workbench/     SegmentView contract + the single-file frontend
```

**The core is language-neutral.** Japanese/Korean grammar, honorifics,
furigana, speech levels — none of it may appear in `core/`, `pipeline/`,
`validation/`, or any shared module. Adapters map their language onto shared
concepts (`RegisterLevel`, `SocialMarker`, `AmbiguityFlag`, `SfxCategory`) and
put anything that doesn't fit into `LinguisticAnalysis.adapter_features` under
a namespaced key (`ja.politeness`, `ko.speech_level`).
`tests/test_architecture.py` enforces this by AST-inspecting imports and
scanning `core/` for language-specific vocabulary. Adding Chinese should mean
writing an adapter, nothing else.

## Translation decision order

Implemented in one place, `pipeline/pipeline.py::process`. Precedence, highest
first:

1. **Locked glossary terms** and **canonical (locked) translation memory** —
   approved project decisions. MT never overwrites them.
2. **Unlocked exact memory** — precedent. Offered as an alternative, never
   imposed.
3. **Machine translation**.
4. **Fuzzy memory** — suggestion only, never auto-applied.

Rules that must survive refactors:

- Validators **only report**. The single automatic rewrite anywhere is locked
  glossary enforcement, and only when the substitution is provably unambiguous
  (one whole-word occurrence, approved rendering absent). Otherwise it raises
  `terminology.conflict` and leaves the text alone.
- **Approval is the only way into translation memory.** A machine candidate
  never becomes authoritative by existing.
- Approval **suggests** glossary rules and never creates them. Explicit
  `TermResolution`s from UI actions are preferred over substring inference.
- Text the user edited or approved **bypasses MT entirely** and is validated
  against itself. Glossary enforcement runs in report-only mode on it.
- Changing source text *or* source language invalidates a translation: the
  segment reopens, releases its TM occurrence, and the candidate regenerates.

## Storage

SQLite, schema version 1, in `storage/db.py`. Migrations are **append-only from
now on** (migration 1 was consolidated before anything shipped; that latitude
is gone).

- Three orthogonal axes: `status` (machine/edited/approved), `origin`
  (machine_translation/translation_memory/glossary/manual/rule). **There is no
  segment-level `locked` column** — lock state belongs to the glossary entry or
  memory variant that produced the text and is derived at read time. Do not
  reintroduce it.
- Translation memory: one normalized source may hold several approved variants;
  uniqueness includes `target_text` and `normalization_version`. At most one
  canonical (locked) variant, enforced by a partial unique index. Every lookup
  filters on `normalization_version` so v1 rows never answer a v2 query.
- `translation_memory_occurrences` holds **current** occurrences only, one row
  per segment. It is not an approval history.
- `issue_resolutions` keys on `(segment_id, fingerprint)`, not on code.
  Fingerprints include the issue's span, so dismissing one register mismatch
  can't silence a different one.
- Hierarchy: project → chapter → page → segment. Pages require a chapter
  (`NOT NULL`; SQLite would let NULL parents share a number). Deleting a page
  cascades to its segments; re-cropping artwork is `replace_page_image`, an
  UPDATE that preserves them.
- **Repositories never commit.** `UnitOfWork.transaction()` owns commits;
  connections are autocommit so single writes still land. Approval spans
  segment state + memory + occurrence in one transaction.

## OCR and MT design

Both are replaceable interfaces; selection is language-aware and lives in one
place.

- `OcrBackend` → `TesseractOcr` (general), `MangaOcr` (Japanese),
  `OcrRouter` (picks per language, `on_error="fallback"|"error"`, records
  `fell_back_from` and the reason). `NullOcr` when nothing is configured.
- manga-ocr is preferred for Japanese: it reads a whole cropped bubble, needs
  no line segmentation, and handles vertical text internally. Cost: torch +
  transformers (~2–3 GB), ~450 MB weights downloaded on first use.
- manga-ocr **reports no confidence score**. `reports_confidence = False` and
  `confidence_available` travel to the UI, which says "no confidence score".
  Absent ≠ high. It also reports **no orientation** — never invent one; carry
  back whatever the caller supplied.
- Low OCR confidence is never a failure; it is flagged. Total failure never
  blocks the workflow — manual source entry is always available.
- `TranslationBackend` → `LexiconBackend` (`development = True`, deterministic
  phrase tables, for tests) and `OpusMtBackend` (real neural MT). When only a
  development backend is loaded the pipeline emits
  `translation.development_backend` ("Machine translation backend
  unavailable…") and the header chip reads `lexicon (development)`. Never let a
  dev backend's word glosses look like real translation.

### Transformers must stay `<5`

`transformers` 5 removed the generic `pipeline("translation")` task the OPUS
backend is built on. `pyproject.toml` pins `transformers>=4.40,<5`, and
`OpusMtBackend.unavailable_reason()` checks the installed major version and
returns an actionable message instead of reporting itself available and failing
at first use. Do not relax either without porting the backend.

## Region and page workflow

- Region geometry is stored in **image coordinates**. The frontend's `page.scale`
  exists only to draw; nothing viewport-shaped is ever sent. `fitIfNeeded` must
  guard against a zero-width container — dividing by it produced a negative
  scale that corrupted every coordinate.
- Initial text orientation is **language-first**: the adapter's `ocr_profile`
  says whether the language sets text vertically, and shape only overrides it
  for a box too wide to hold a vertical line. An aspect-ratio rule alone called
  most vertical Japanese horizontal, because manga bubbles are roughly square.
  A saved/user-corrected orientation always wins and must survive a resize or a
  type change.
- Detection (`OpenCvComicTextDetector`, MSER + morphological grouping) is a
  starting point, not a finished detector. Manual region editing is a required
  path, not a fallback. `DetectorUnavailable` is distinct from "ran and found
  nothing"; both keep manual creation working.
- Deleting a region deletes its segment **unless approved**; an approved
  segment requires `force=true` and is then detached from the page with its
  translation and memory intact. This rule is documented in
  `imaging/service.py` — keep them in sync.
- Detected regions become **ordinary `Segment` records**. There is no parallel
  "OCR object", and anything created here runs through `TranslationPipeline`
  unchanged.

## Frontend workflow

**Projects** (home, shown when no `?project=`) → new project (name, source
language, target language) / open existing. A fresh install has no seeded
project; `SCANLATE_DEMO=1` adds the "Yoru no Kissaten" fixture.

**Page** (spatial): chapter picker, **+ Chapter**, page picker filtered to the
chapter, import (numbered within that chapter). Toolbar tools: **Select**,
**Add region**, **Detect**, **OCR page**. Regions are thin outlines; number and
type appear on hover or selection only. Right panel: source, translation,
Approve, Previous/Next; region settings (type, language, orientation) behind a
disclosure; linguistic diagnostics behind the Diagnostics toggle. `Del` deletes
the selected region, `Esc` cancels drawing then clears selection, `j`/`k` step
regions.

**Review** (chapter-wide): every segment with source, candidate, register
signals, terminology and warnings. `j`/`k`/`e`/`a`.

Panel textareas: `#f-translation` saves on blur; `#f-source` saves on **Save
source** only — changing source discards the translation and reruns the
pipeline, too disruptive to do mid-word. Generic textarea handlers must stay
scoped to `.seg` cards (Review); the Page panel's boxes are not inside one.

## Completed milestones

1. Language-neutral core, ja/ko/en adapters, deterministic MT backend.
2. Glossary, translation memory, validators, pipeline, ApprovalService.
3. FastAPI + frontend on real SegmentViews.
4. Page import, regions, detection, OCR, segment linkage.
5. Projects/chapters UI, no demo data at startup, language-aware OCR routing,
   backend visibility.

## Known bugs (open, diagnosed)

None currently open.

## Recently fixed

1. **Page-panel Approve did nothing.** In `index.html`, the `#view` click
   handler (the one handling `button[data-act]`) resolved the segment with
   `find(card.dataset.id)` where `card = e.target.closest(".seg")`. The Page
   panel is rendered by `translationHTML`, which has no `.seg` ancestor, so
   `card` was null and the handler threw before reaching the approve branch.
   Fixed by resolving the id from the card **or** `page.selected`; it routes
   through the same `/approve` call Review uses, refreshes the Page panel, and
   shows Approved. Regression: `pagePanelApprove` in
   `tests/frontend/test_workbench.mjs`, which clicks the actual Page-panel
   button.
2. **Add Region mode exited after one region.** The `mouseup` handler's draw
   branch set `page.mode = "select"` right after posting the new region.
   Fixed so it stays in draw mode across several regions; only an explicit
   **Select** click or `Esc` leaves it. Regression: extended
   `pageDrawAndDelete`.

Note for jsdom work: exceptions thrown inside event listeners reach the virtual
console as `jsdomError`, not `window.onerror`. `boot()` forwards them; without
that, "throws nothing" checks pass vacuously.

## Real-browser findings

Verified live: same-origin serving, projects → chapter → import → detect → OCR
→ correct source → translate → approve, persistence across restart, a locked
glossary rule changing a later translation.

On a generated vertical-Japanese page (6 bubbles, real CJK glyphs, RTL columns
— `tools/ocr_bench.py`): detection found 8 regions for 6 bubbles (2 false
positives, both one click to reject); orientation came out vertical for all 5
vertical bubbles. **Tesseract managed 1/6 exact, and the one it got right was
the only horizontal line.** That is clean rendered type, so real scans will be
worse — this is the case for manga-ocr.

**CI environment:** manga-ocr and OPUS-MT have never actually run there
(`huggingface.co` is blocked); their behaviour is pinned by fakes and
unmeasured in that environment.

**Maintainer's local Windows machine:** both have been run for real and
verified working — manga-ocr against an actual manga page, OPUS-MT with
transformers 4. Treat CI's fakes as a stand-in for a blocked network, not as
evidence either backend is broken.

## Do not casually reverse

- Language-neutral core; adapters own all language specifics.
- Precedence order and "validators only report".
- Approval as the only door into TM; suggestions never auto-create glossary rules.
- No segment-level lock column.
- Append-only migrations from here.
- Repositories don't commit; UnitOfWork owns transactions.
- Image-space coordinates as the only persisted geometry.
- Deterministic lexicon backend stays — it is what makes the surrounding logic
  testable without model drift.
- `transformers<5`.

## Deferred (do not start unprompted)

Advanced/generative inpainting, font matching, stylized SFX reconstruction,
Korean/Chinese production optimization, automatic long-strip splitting, cloud
storage, collaboration, accounts.

## Next milestone — Basic Page Production

An approved page becomes an exported translated PNG. Deliberately simple.

1. **Cleanup** — cover original lettering in ordinary white/flat-colour bubbles
   only. No generative inpainting. Preserve artwork and borders. If a region
   can't be cleaned safely by the simple method, **flag it for manual handling
   rather than damaging the page**.
2. **Deterministic typesetting** — fit the approved translation inside its
   region: wrapping, margins, alignment, font-size reduction, all deterministic.
   Never overflow silently. Persist settings (`RenderSettings` /`render_json`
   already exists) so the page re-renders identically.
3. **SFX** — do not erase or redraw stylized SFX. Preserve the original and add
   the approved English as a small nearby caption.
4. **Preview** — a Page-workbench toggle between editing view and translated
   preview.
5. **Export** — PNG at original page dimensions, predictable filename, never
   overwriting the imported source image.

Target loop: new project → chapter → import page → detect/draw → OCR → MT →
correct → approve → preview → export PNG.

Also pending, small: an evaluation fixture for translation quality storing
Japanese source, machine candidate(s), preferred reading, and whether a failure
is OCR, segmentation, or MT — so backends can be compared later. Seed it with
`聞いていただけるのですかァ!` (MT gives "Please listen to me!" / "Do you hear
me?"; contextually "You'll hear me out?!"). Record examples; **do not hardcode
phrase-specific rules**.
