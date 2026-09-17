/**
 * Frontend robustness tests.
 *
 * Loads the real workbench HTML in jsdom with a stubbed fetch, then drives the
 * handlers the way a user would. Any uncaught error or unhandled rejection
 * fails the run, which is the actual thing under test: interacting before data
 * arrives, or after the backend refused, must not throw.
 *
 * Run: node tests/frontend/test_workbench.mjs
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM, VirtualConsole } from "jsdom";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PAGE = path.resolve(HERE, "../../src/scanlate/workbench/static/index.html");
const HTML = fs.readFileSync(PAGE, "utf8");

let failures = 0;
let checks = 0;

function ok(condition, label) {
  checks++;
  if (!condition) {
    failures++;
    console.error(`  FAIL  ${label}`);
  } else {
    console.log(`  ok    ${label}`);
  }
}

const SEGMENT = {
  id: "s1", project_id: "demo", page_id: "pg", order: 1, kind: "dialogue",
  language: "ja", target_language: "en", source: "うるせぇな…", candidate: "Please be quiet.",
  annotations: [], alternatives: [{ text: "Shut up…", note: "closer tone", reason: "register" }],
  features: { register: "crude", register_confidence: 0.84, "ja.politeness": "plain" },
  warnings: [{ code: "register.mismatch", severity: "WARNING", message: "Two levels apart.",
               fingerprint: "abc123", span: null,
               actions: [{ kind: "dismiss", label: "Dismiss", value: null }], data: {} }],
  status: "machine", origin: "machine_translation", locked: false, needs_review: true,
};

const PROJECT = {
  project: { id: "demo", name: "Demo", default_source_language: "ja",
             target_language: "en", chapters: [], pages: [] },
  segments: [SEGMENT],
};

/** Boot the page with a given fetch stub, and trap anything that escapes. */
async function boot(fetchImpl, { settle = 30, url = "http://localhost/?project=demo" } = {}) {
  const errors = [];
  // jsdom routes an exception thrown inside an event listener to its virtual
  // console as a "jsdomError" rather than firing window.onerror, so without
  // this a handler that throws on every keystroke would pass silently.
  const virtualConsole = new VirtualConsole();
  virtualConsole.on("jsdomError", error => errors.push(error));
  virtualConsole.on("error", (...args) => errors.push(args.join(" ")));
  const dom = new JSDOM(HTML, {
    runScripts: "dangerously",
    virtualConsole,
    url,
    beforeParse(window) {
      window.fetch = fetchImpl;
      window.addEventListener("error", e => errors.push(e.error || e.message));
      window.addEventListener("unhandledrejection", e => errors.push(e.reason));
      window.HTMLElement.prototype.scrollIntoView = () => {};
    },
  });
  await new Promise(r => setTimeout(r, settle));
  return { dom, window: dom.window, document: dom.window.document, errors };
}

const json = (body, status = 200) => Promise.resolve({
  ok: status < 400, status, statusText: status === 200 ? "OK" : "Error",
  json: () => Promise.resolve(body),
});

function press(window, key) {
  window.document.dispatchEvent(new window.KeyboardEvent("keydown", { key, bubbles: true }));
}

function clickAll(document, selector) {
  document.querySelectorAll(selector).forEach(el => el.click());
}

async function settle(ms = 30) {
  await new Promise(r => setTimeout(r, ms));
}

// --- 1. backend unavailable -------------------------------------------
async function backendUnavailable() {
  console.log("backend unavailable");
  const { window, document, errors } = await boot(() => Promise.reject(new Error("ECONNREFUSED")));

  ok(document.getElementById("view").textContent.includes("Can't reach the backend"),
     "shows an error state instead of a blank page");

  clickAll(document, "#tabs button");
  await settle();
  ok(errors.length === 0, "clicking every tab throws nothing");

  clickAll(document, "#filters button");
  const toggle = document.getElementById("diag-toggle");
  toggle.checked = true;
  toggle.dispatchEvent(new window.Event("change"));
  await settle();
  ok(errors.length === 0, "filters and diagnostics toggle throw nothing");

  for (const key of ["j", "k", "e", "a", "j", "a"]) press(window, key);
  await settle();
  ok(errors.length === 0, "j/k/e/a throw nothing");

  document.getElementById("view").click();
  await settle();
  ok(errors.length === 0, `clicking the view throws nothing (${errors.map(String)})`);

  // The internals are also called directly, since a future caller may reach
  // them before data exists even if no current handler does.
  let threw = null;
  try {
    window.render();
    window.setFilter("flagged");
    ok(window.find("s1") === null, "find() returns null rather than throwing without data");
    ok(window.visible().length === 0, "visible() yields nothing without data");
    window.merge({ id: "s1" });
  } catch (error) {
    threw = error;
  }
  ok(threw === null, `render/setFilter/merge tolerate null data (${threw})`);
}

// --- 2. slow backend, interaction while loading ------------------------
async function slowBackend() {
  console.log("slow backend");
  let release;
  const gate = new Promise(r => { release = r; });
  const { window, document, errors } = await boot(() => gate.then(() => json(PROJECT)));

  ok(document.getElementById("view").textContent.includes("Loading"), "shows a loading state");

  clickAll(document, "#tabs button");
  clickAll(document, "#filters button");
  for (const key of ["j", "k", "a", "e"]) press(window, key);
  await settle();
  ok(errors.length === 0, "interacting during load throws nothing");

  release();
  await settle(50);
  ok(errors.length === 0, "no error once data lands");
  ok(document.querySelectorAll(".seg").length === 1, "renders the segment after loading");
}

// --- 3. empty project --------------------------------------------------
async function emptyProject() {
  console.log("empty project");
  const empty = { project: PROJECT.project, segments: [] };
  const { window, document, errors } = await boot(
    url => (url.endsWith("/pages") ? json([]) : json(empty)));

  ok(document.getElementById("view").textContent.includes("No segments yet"),
     "Review shows an empty state");

  document.querySelector('#tabs button[data-tab="page"]').click();
  await settle(80);
  ok(document.getElementById("view").textContent.includes("No chapters yet"),
     "Page shows an empty state rather than rendering an undefined segment");

  for (const key of ["j", "k", "e", "a"]) press(window, key);
  clickAll(document, "#filters button");
  clickAll(document, "#tabs button");
  await settle();
  ok(errors.length === 0, `no uncaught error on an empty project (${errors.map(String)})`);
  ok(document.getElementById("c-total").textContent === "0", "counts render as zero");
}

// --- 4. failed PATCH can be retried ------------------------------------
async function failedPatchRetries() {
  console.log("failed PATCH");
  const calls = [];
  let failNext = true;
  const fetchImpl = (url, options = {}) => {
    calls.push({ url, method: options.method || "GET", body: options.body });
    if (url.includes("/api/segments/") && options.method === "PATCH") {
      if (failNext) return json({ detail: "database is locked" }, 500);
      return json({ ...SEGMENT, candidate: "Shut up…", status: "edited" });
    }
    return json(PROJECT);
  };
  const { window, document, errors } = await boot(fetchImpl);

  const textarea = document.querySelector("textarea");
  textarea.value = "Shut up…";
  textarea.dispatchEvent(new window.Event("input", { bubbles: true }));
  textarea.dispatchEvent(new window.Event("focusout", { bubbles: true }));
  await settle();

  const patches = () => calls.filter(c => c.method === "PATCH").length;
  ok(patches() === 1, "first save attempted");
  ok(!document.getElementById("error").hidden, "the failure is shown to the user");
  ok(errors.length === 0, "a failed save throws nothing");

  // Same value, second attempt: must not be skipped as already-saved.
  failNext = false;
  const again = document.querySelector("textarea");
  again.value = "Shut up…";
  again.dispatchEvent(new window.Event("input", { bubbles: true }));
  again.dispatchEvent(new window.Event("focusout", { bubbles: true }));
  await settle();
  ok(patches() === 2, "the same edit is retried after a failure");

  // Now that it succeeded, an unchanged value is not re-sent.
  const third = document.querySelector("textarea");
  third.dispatchEvent(new window.Event("focusout", { bubbles: true }));
  await settle();
  ok(patches() === 2, "a successful save is not repeated");
}

// --- 5. failed glossary accept ----------------------------------------
async function failedGlossaryAccept() {
  console.log("failed glossary accept");
  const calls = [];
  const suggestion = {
    source_term: "형", target_term: "hyung", category: "relationship_term",
    reason: "You chose hyung for 형.", source_language: "ko", target_language: "en",
    explicit: true,
  };
  const fetchImpl = (url, options = {}) => {
    calls.push(url);
    if (url.includes("/approve")) {
      return json({ segment: { ...SEGMENT, status: "approved" }, memory_entry_id: 1,
                    suggestion });
    }
    if (url.includes("/glossary/accept")) return json({ detail: "glossary write failed" }, 500);
    return json(PROJECT);
  };
  const { window, document, errors } = await boot(fetchImpl);

  document.querySelector('button[data-act="approve"]').click();
  await settle();
  ok(document.querySelector(".learn"), "the remember prompt appears after approval");

  document.querySelector('button[data-act="remember"]').click();
  await settle();

  ok(!document.getElementById("error").hidden, "the glossary failure is visible");
  ok(document.getElementById("error").textContent.includes("glossary write failed"),
     "the server's reason is shown");
  ok(document.querySelector(".learn"), "the prompt stays open rather than looking accepted");
  ok(!calls.some(u => u.includes("/api/segments/s1?")), "no false success follow-up");
  ok(errors.length === 0, "a failed glossary write throws nothing");
}

// --- 6. the happy path still works ------------------------------------
async function happyPath() {
  console.log("connected backend");
  const calls = [];
  const fetchImpl = (url, options = {}) => {
    calls.push({ url, method: options.method || "GET" });
    if (url.includes("/approve")) {
      return json({ segment: { ...SEGMENT, status: "approved", needs_review: false },
                    memory_entry_id: 1, suggestion: null });
    }
    if (url.includes("/issues/resolve")) return json({ ...SEGMENT, warnings: [], needs_review: false });
    if (url.includes("/api/segments/")) return json({ ...SEGMENT, status: "edited" });
    if (url.endsWith("/pages")) return json([]);
    return json(PROJECT);
  };
  const { window, document, errors } = await boot(fetchImpl);

  ok(document.getElementById("project").textContent === "Demo", "header shows the project");
  ok(document.getElementById("c-flagged").textContent === "1", "review count is right");
  ok(document.querySelector(".source").textContent.includes("うるせぇな"), "source renders");
  ok(document.querySelector(".signal").textContent === "crude", "register signal renders");

  document.querySelector('button[data-act="warn"]').click();
  await settle();
  ok(calls.some(c => c.url.includes("/issues/resolve")), "dismiss calls the resolve endpoint");

  document.querySelector('button[data-act="approve"]').click();
  await settle();
  ok(calls.some(c => c.url.includes("/approve")), "approve calls the approve endpoint");
  ok(document.querySelector(".state")?.textContent === "Approved", "approved state renders");

  document.querySelector('#tabs button[data-tab="page"]').click();
  await settle(60);
  ok(document.querySelector(".page-wrap"), "page tab renders its split layout");

  press(window, "j");
  press(window, "k");
  await settle();
  ok(errors.length === 0, `the happy path throws nothing (${errors.map(String)})`);
}


// --- 7. page workspace -------------------------------------------------
const PAGE_FIXTURE = { id: "pg1", project_id: "demo", chapter_id: "ch1", number: 1,
               width: 500, height: 700, image_url: "/api/pages/pg1/image", has_image: true,
               regions: [{ segment_id: "s1", seq: 1, kind: "dialogue", language: "ja",
                           source_text: "うるせぇな…", status: "machine",
                           region: { x: 40, y: 60, width: 200, height: 80,
                                     kind: "speech_bubble", orientation: "horizontal",
                                     reading_order: 0 } }] };

function pageFetch(calls, overrides = {}) {
  return (url, options = {}) => {
    calls.push({ url, method: options.method || "GET", body: options.body });
    for (const [fragment, responder] of Object.entries(overrides)) {
      if (url.includes(fragment)) return responder(url, options);
    }
    if (url.endsWith("/pages")) return json([PAGE_FIXTURE]);
    if (url.endsWith("/api/projects/demo")) return json({ ...PROJECT.project,
      chapters: [{ id: "ch1", number: 1, title: "One" }], pages: [PAGE_FIXTURE] });
    if (url.includes("/api/pages/pg1")) return json(PAGE_FIXTURE);
    if (url.includes("/api/segments/")) return json(SEGMENT);
    return json(PROJECT);
  };
}

async function openPageTab(fetchImpl) {
  const boot_ = await boot(fetchImpl);
  boot_.document.querySelector('#tabs button[data-tab="page"]').click();
  await settle(80);
  return boot_;
}

async function pageWorkspace() {
  console.log("page workspace");
  const calls = [];
  const { window, document, errors } = await openPageTab(pageFetch(calls));

  ok(document.querySelector("#stage img")?.getAttribute("src") === "/api/pages/pg1/image",
     "renders the page image");
  const region = document.querySelector('.region[data-segment="s1"]');
  ok(region, "draws the region overlay");

  // Image space is authoritative: at 100% the overlay sits at the stored coords.
  const style = region.style;
  ok(style.left === "40px" && style.top === "60px" && style.width === "200px",
     `overlay uses image coordinates (${style.left}/${style.top}/${style.width})`);

  region.dispatchEvent(new window.MouseEvent("mousedown", { bubbles: true, clientX: 50, clientY: 70 }));
  window.document.dispatchEvent(new window.MouseEvent("mouseup", { bubbles: true }));
  await settle(60);
  ok(document.querySelector('.region[data-selected="true"]'), "clicking a region selects it");
  ok(document.querySelector("#f-source")?.value === "うるせぇな…",
     "the panel shows the segment's source text");
  ok(document.querySelector("#f-kind")?.value === "dialogue", "the panel shows the type");
  ok(errors.length === 0, `selection throws nothing (${errors.map(String)})`);
}

async function pageZoomKeepsCoordinates() {
  console.log("page zoom");
  const calls = [];
  const { window, document, errors } = await openPageTab(pageFetch(calls));

  document.querySelector('button[data-page="zoom-in"]').click();
  await settle(40);
  const region = document.querySelector('.region[data-segment="s1"]');
  const left = parseInt(region.style.left, 10);
  ok(left === 50, `overlay scales with the image (40 × 1.25 = ${left})`);

  // Nothing viewport-shaped was ever sent.
  const patches = calls.filter(c => c.method === "PATCH");
  ok(patches.length === 0, "zooming sends no requests");
  ok(errors.length === 0, "zooming throws nothing");
}

async function pageDrawAndDelete() {
  console.log("page draw and delete");
  const calls = [];
  const created = { ...SEGMENT, id: "s2" };
  const { window, document, errors } = await openPageTab(pageFetch(calls, {
    "/regions": () => json(created),
    "/region?force": () => json({ segment_id: "s1", outcome: "detached" }),
  }));

  document.querySelector('button[data-page="tool-draw"]').click();
  await settle(30);
  ok(document.querySelector(".canvas-stage.drawmode"), "Add region turns on draw mode");
  ok(document.querySelector('button[data-page="tool-draw"]').getAttribute("aria-pressed") === "true",
     "the active tool is marked pressed");

  const stage = document.getElementById("stage");
  stage.getBoundingClientRect = () => ({ left: 0, top: 0, width: 500, height: 700 });
  stage.dispatchEvent(new window.MouseEvent("mousedown", { bubbles: true, clientX: 100, clientY: 120 }));
  document.dispatchEvent(new window.MouseEvent("mousemove", { bubbles: true, clientX: 260, clientY: 210 }));
  document.dispatchEvent(new window.MouseEvent("mouseup", { bubbles: true }));
  await settle(80);

  const post = calls.find(c => c.method === "POST" && c.url.includes("/regions"));
  ok(post, "drawing posts a new region");
  if (post) {
    const body = JSON.parse(post.body);
    ok(body.x === 100 && body.y === 120 && body.width === 160 && body.height === 90,
       `posts image coordinates (${JSON.stringify(body)})`);
  }
  ok(errors.length === 0, `drawing throws nothing (${errors.map(String)})`);
  ok(document.querySelector('button[data-page="tool-draw"]').getAttribute("aria-pressed") === "true",
     "Add region stays active after drawing one, so several bubbles can be drawn in a row");
  ok(document.querySelector(".canvas-stage.drawmode"), "the canvas stays in draw mode");

  // A second bubble can be drawn immediately, with no Select click in between.
  const stage2 = document.getElementById("stage");
  stage2.getBoundingClientRect = () => ({ left: 0, top: 0, width: 500, height: 700 });
  stage2.dispatchEvent(new window.MouseEvent("mousedown", { bubbles: true, clientX: 300, clientY: 320 }));
  document.dispatchEvent(new window.MouseEvent("mousemove", { bubbles: true, clientX: 360, clientY: 400 }));
  document.dispatchEvent(new window.MouseEvent("mouseup", { bubbles: true }));
  await settle(80);
  const posts = calls.filter(c => c.method === "POST" && c.url.includes("/regions"));
  ok(posts.length === 2, "a second region can be drawn without reselecting the tool");
  ok(document.querySelector('button[data-page="tool-draw"]').getAttribute("aria-pressed") === "true",
     "still in draw mode after the second region");

  // Only an explicit Select click leaves draw mode.
  document.querySelector('button[data-page="tool-select"]').click();
  await settle(30);
  ok(document.querySelector('button[data-page="tool-select"]').getAttribute("aria-pressed") === "true",
     "clicking Select leaves draw mode");
  ok(!document.querySelector(".canvas-stage.drawmode"), "the canvas is no longer in draw mode");
}

async function pageEscapeAndDelete() {
  console.log("page escape and delete");
  const calls = [];
  const { window, document, errors } = await openPageTab(pageFetch(calls, {
    "/region": (url, options) => (options.method === "DELETE"
      ? json({ segment_id: "s1", outcome: "deleted" }) : json(SEGMENT)),
  }));

  // Esc leaves draw mode without drawing anything.
  document.querySelector('button[data-page="tool-draw"]').click();
  await settle(30);
  press(window, "Escape");
  await settle(40);
  ok(!document.querySelector(".canvas-stage.drawmode"), "Esc cancels drawing");
  ok(calls.every(c => c.method !== "POST"), "Esc creates nothing");

  // Select a region, then Delete removes it.
  document.querySelector('.region[data-segment="s1"]').dispatchEvent(
    new window.MouseEvent("mousedown", { bubbles: true, clientX: 50, clientY: 70 }));
  document.dispatchEvent(new window.MouseEvent("mouseup", { bubbles: true }));
  await settle(50);
  ok(document.querySelector(".region .reject"), "the selected region shows a reject button");

  press(window, "Delete");
  await settle(80);
  ok(calls.some(c => c.method === "DELETE"), "Delete removes the selected region");
  ok(errors.length === 0, `delete throws nothing (${errors.map(String)})`);
}

async function projectsHome() {
  console.log("projects home");
  const calls = [];
  const created = { id: "my-manga", name: "My Manga",
                    default_source_language: "ja", target_language: "en",
                    chapters: [], pages: [] };
  // No ?project= in the URL: this is what a fresh install opens on.
  const { window, document, errors } = await boot((url, options = {}) => {
    calls.push({ url, method: options.method || "GET", body: options.body });
    if (url.endsWith("/api/projects") && options.method === "POST") return json(created);
    if (url.endsWith("/api/projects")) return json([]);
    if (url.includes("/pages")) return json([]);
    if (url.includes("/segments")) return json({ project: created, segments: [] });
    if (url.includes("/health")) return json({ translation: { names: ["lexicon"], development: true },
                                               ocr: { name: "router" } });
    return json({});
  }, { url: "http://localhost/" });

  ok(document.getElementById("view").textContent.includes("New project"),
     "a fresh install opens on the projects screen");
  ok(document.getElementById("view").textContent.includes("No projects yet"),
     "no demo project is present");
  ok(!document.getElementById("view").textContent.includes("Yoru no Kissaten"),
     "fixture data is absent from normal startup");
  ok(document.getElementById("backend")?.textContent.includes("development"),
     "the development translation backend is named in the header");

  document.getElementById("np-name").value = "My Manga";
  document.querySelector('button[data-home="create"]').click();
  await settle(120);
  const post = calls.find(c => c.method === "POST" && c.url.endsWith("/api/projects"));
  ok(post, "creating posts the new project");
  if (post) {
    const body = JSON.parse(post.body);
    ok(body.name === "My Manga" && body.default_source_language === "ja",
       "the form's values are sent");
  }
  ok(errors.length === 0, `the projects screen throws nothing (${errors.map(String)})`);
}

async function pageOcrAndFailures() {
  console.log("page OCR and failures");
  const calls = [];
  const { window, document, errors } = await openPageTab(pageFetch(calls, {
    "/ocr": () => json({ result: { segment_id: "s1", text: "", ok: false,
                                   error: "No text was recognized in this region.",
                                   low_confidence: false, reopened: false, needs_review: true },
                         segment: SEGMENT }),
    "/detect": () => json({ available: false, created: 0,
                            message: "Automatic detection isn't available. Draw regions by hand to continue.",
                            segments: [] }),
  }));

  document.querySelector('button[data-page="detect"]').click();
  await settle(80);
  ok(document.querySelector(".ocr-note[data-level='warn']")?.textContent.includes("by hand"),
     "an unavailable detector is reported, not thrown");

  document.querySelector('.region[data-segment="s1"]').dispatchEvent(
    new window.MouseEvent("mousedown", { bubbles: true, clientX: 50, clientY: 70 }));
  document.dispatchEvent(new window.MouseEvent("mouseup", { bubbles: true }));
  await settle(50);
  document.querySelector('button[data-page="ocr-one"]').click();
  await settle(80);

  ok(document.querySelector(".ocr-note")?.textContent.includes("type the source text"),
     "an OCR failure points at manual entry");
  ok(document.querySelector("#f-source"), "the source box stays editable after OCR fails");
  ok(errors.length === 0, `OCR failure throws nothing (${errors.map(String)})`);
}

async function pageWithNoImage() {
  console.log("page tab with no pages");
  const { window, document, errors } = await openPageTab(url => {
    if (url.endsWith("/pages")) return json([]);
    if (url.endsWith("/api/projects/demo")) return json({ ...PROJECT.project,
      chapters: [{ id: "ch1", number: 1, title: "One" }], pages: [] });
    return json(PROJECT);
  });
  ok(document.getElementById("view").textContent.includes("No pages in this chapter"),
     "shows an empty state when the chapter has no pages");
  document.querySelector('button[data-page="import"]')?.click();
  for (const key of ["j", "k", "a", "e"]) press(window, key);
  await settle(40);
  ok(errors.length === 0, `an empty page tab throws nothing (${errors.map(String)})`);
}


// --- 8. Page panel textareas -------------------------------------------
async function pagePanelTextareas() {
  console.log("page panel textareas");
  const calls = [];
  const { window, document, errors } = await openPageTab(pageFetch(calls, {
    "/source": () => json({ segment: SEGMENT, reopened: false, changed: "source text",
                            message: null }),
  }));

  // Select the region so the panel is showing.
  document.querySelector('.region[data-segment="s1"]').dispatchEvent(
    new window.MouseEvent("mousedown", { bubbles: true, clientX: 50, clientY: 70 }));
  document.dispatchEvent(new window.MouseEvent("mouseup", { bubbles: true }));
  await settle(60);

  // Typing in the source box must not throw: it is not inside a .seg card.
  const source = document.getElementById("f-source");
  ok(source, "the page panel has a source box");
  source.value = "田中先輩、ちょっといいですか？";
  source.dispatchEvent(new window.Event("input", { bubbles: true }));
  source.dispatchEvent(new window.Event("focusout", { bubbles: true }));
  await settle(60);
  ok(errors.length === 0, `typing in the source box throws nothing (${errors.map(String)})`);
  ok(!calls.some(c => c.url.includes("/source")),
     "blurring the source box does not rerun translation on its own");

  // Saving it explicitly does send it.
  document.querySelector('button[data-page="save-source"]').click();
  await settle(80);
  const saved = calls.find(c => c.url.includes("/source"));
  ok(saved, "Save source sends the corrected text");
  if (saved) ok(JSON.parse(saved.body).source_text === "田中先輩、ちょっといいですか？",
                "the edited text is what gets sent");

  // The translation box saves on blur, and also must not throw.
  const translation = document.getElementById("f-translation");
  ok(translation, "the page panel has a translation box");
  translation.value = "Tanaka-senpai, got a sec?";
  translation.dispatchEvent(new window.Event("input", { bubbles: true }));
  translation.dispatchEvent(new window.Event("focusout", { bubbles: true }));
  await settle(80);
  const patched = calls.find(c => c.method === "PATCH" && c.url.endsWith("/api/segments/s1"));
  ok(patched, "blurring the translation box saves it");
  if (patched) ok(JSON.parse(patched.body).translation === "Tanaka-senpai, got a sec?",
                  "the edited translation is what gets sent");
  ok(errors.length === 0, `typing in the translation box throws nothing (${errors.map(String)})`);
}

// --- 8b. Page panel Approve ---------------------------------------------
async function pagePanelApprove() {
  console.log("page panel approve");
  const calls = [];
  // Fresh, self-contained responses for the segments list and approve calls
  // rather than the shared PROJECT/SEGMENT fixtures: load() aliases
  // state.data.segments directly to the response's segments array, so an
  // earlier scenario's approve (which replaces that array's entry in place
  // via merge()) can leave the shared fixture looking pre-approved for every
  // later test that falls back to it. Not part of either bug being fixed
  // here, just avoided so this test's assertions are deterministic.
  const { window, document, errors } = await openPageTab(pageFetch(calls, {
    "/api/projects/demo/segments": () => json({ project: PROJECT.project,
      segments: [{ ...SEGMENT, candidate: "Please be quiet.", status: "machine", needs_review: true }] }),
    "/approve": () => json({ segment: { ...SEGMENT, candidate: "Please be quiet.",
                             status: "approved", needs_review: false },
                             memory_entry_id: 1, suggestion: null }),
  }));

  // Select the region so the panel (not a .seg card) is showing the Approve button.
  document.querySelector('.region[data-segment="s1"]').dispatchEvent(
    new window.MouseEvent("mousedown", { bubbles: true, clientX: 50, clientY: 70 }));
  document.dispatchEvent(new window.MouseEvent("mouseup", { bubbles: true }));
  await settle(60);

  const approveBtn = document.querySelector('.panel button[data-act="approve"]');
  ok(approveBtn, "the page panel has an Approve button");
  ok(!document.querySelector(".seg"), "the page panel button is not inside a .seg card");

  approveBtn.click();
  await settle(80);

  ok(errors.length === 0, `clicking Page-panel Approve throws nothing (${errors.map(String)})`);
  ok(calls.some(c => c.url.includes("/segments/s1/approve")),
     "it calls the same approve endpoint Review uses");
  ok(document.querySelector(".panel .state")?.textContent === "Approved",
     "the panel visibly shows Approved");
  ok(!document.querySelector('.panel button[data-act="approve"]'),
     "the Approve button is replaced by Reopen once approved");
}

// --- 9. chapters --------------------------------------------------------
const CHAPTERS = [{ id: "ch1", number: 1, title: "One" }, { id: "ch2", number: 2, title: "Two" }];
const PAGE_CH2 = { ...PAGE_FIXTURE, id: "pg2", chapter_id: "ch2", number: 1, regions: [] };

function chapterFetch(calls, overrides = {}) {
  return (url, options = {}) => {
    calls.push({ url, method: options.method || "GET", body: options.body });
    for (const [fragment, responder] of Object.entries(overrides)) {
      if (url.includes(fragment)) return responder(url, options);
    }
    if (url.endsWith("/api/projects/demo")) return json({ ...PROJECT.project,
      chapters: CHAPTERS, pages: [PAGE_FIXTURE, PAGE_CH2] });
    if (url.endsWith("/pages")) return json([PAGE_FIXTURE, PAGE_CH2]);
    if (url.includes("/api/pages/pg2")) return json(PAGE_CH2);
    if (url.includes("/api/pages/pg1")) return json(PAGE_FIXTURE);
    if (url.includes("/api/segments/")) return json(SEGMENT);
    return json(PROJECT);
  };
}

async function pageChapters() {
  console.log("page chapters");
  const calls = [];
  const { window, document, errors } = await openPageTab(chapterFetch(calls));

  const chapterPicker = document.getElementById("chapter-picker");
  ok(chapterPicker, "the toolbar has a chapter picker");
  ok([...chapterPicker.options].length === 2, "it lists the project's chapters");
  ok(chapterPicker.value === "ch1", "it starts on the first chapter");

  const pagePicker = document.getElementById("page-picker");
  ok([...pagePicker.options].length === 1,
     "the page list shows only this chapter's pages");

  chapterPicker.value = "ch2";
  chapterPicker.dispatchEvent(new window.Event("change", { bubbles: true }));
  await settle(120);
  ok(document.getElementById("page-picker").value === "pg2",
     "switching chapter switches to that chapter's page");
  ok(errors.length === 0, `switching chapter throws nothing (${errors.map(String)})`);
}

async function pageImportUsesTheSelectedChapter() {
  console.log("page import into the current chapter");
  const calls = [];
  const { window, document, errors } = await openPageTab(chapterFetch(calls, {
    "/api/projects/demo/pages": (url, options) => (options.method === "POST"
      ? json({ ...PAGE_CH2, id: "pg3", number: 2 })
      : json([PAGE_FIXTURE, PAGE_CH2])),
  }));

  document.getElementById("chapter-picker").value = "ch2";
  document.getElementById("chapter-picker").dispatchEvent(
    new window.Event("change", { bubbles: true }));
  await settle(120);

  const form = new window.FormData();
  const sent = [];
  form.append = (key, value) => sent.push([key, value]);
  window.FormData = function () { return form; };
  await window.importPageFile({ name: "p.png" });
  await settle(120);

  const fields = Object.fromEntries(sent);
  ok(fields.chapter_id === "ch2", `imports into the selected chapter (${fields.chapter_id})`);
  ok(fields.number === "2", `numbers within that chapter, not the project (${fields.number})`);
  ok(errors.length === 0, `importing throws nothing (${errors.map(String)})`);
}

// --- 10. every request honours the API base -----------------------------
async function apiBaseIsHonoured() {
  console.log("api base");
  const calls = [];
  const base = "http://api.example/scanlate";
  const { window, document, errors } = await boot((url, options = {}) => {
    calls.push({ url, method: options.method || "GET" });
    const path = url.replace(base, "");
    if (path.endsWith("/api/projects/demo")) return json({ ...PROJECT.project,
      chapters: CHAPTERS, pages: [PAGE_FIXTURE] });
    if (path.endsWith("/pages")) return json([PAGE_FIXTURE]);
    if (path.includes("/api/pages/pg1")) return json(PAGE_FIXTURE);
    if (path.includes("/region")) return json({ segment_id: "s1", outcome: "deleted" });
    if (path.includes("/api/segments/")) return json(SEGMENT);
    return json(PROJECT);
  }, { url: `http://localhost/?project=demo&api=${encodeURIComponent(base)}` });

  document.querySelector('#tabs button[data-tab="page"]').click();
  await settle(150);

  ok(calls.every(c => c.url.startsWith(base)),
     `every request goes to the configured base (${calls.filter(c => !c.url.startsWith(base))
       .map(c => c.url).slice(0, 2)})`);
  const image = document.querySelector("#stage img")?.getAttribute("src");
  ok(image && image.startsWith(base), `the page image uses the base (${image})`);

  // Region deletion is a bare fetch, so it needs the base explicitly.
  document.querySelector('.region[data-segment="s1"]').dispatchEvent(
    new window.MouseEvent("mousedown", { bubbles: true, clientX: 50, clientY: 70 }));
  document.dispatchEvent(new window.MouseEvent("mouseup", { bubbles: true }));
  await settle(60);
  document.querySelector('.panel button[data-page="delete-region"]').click();
  await settle(120);
  const deletion = calls.find(c => c.method === "DELETE");
  ok(deletion, "region deletion was sent");
  if (deletion) ok(deletion.url.startsWith(base), `deletion uses the base (${deletion.url})`);

  // And the multipart upload.
  const sent = [];
  const form = new window.FormData();
  form.append = (k, v) => sent.push([k, v]);
  window.FormData = function () { return form; };
  await window.importPageFile({ name: "p.png" });
  await settle(120);
  const upload = calls.find(c => c.method === "POST" && c.url.includes("/pages"));
  ok(upload && upload.url.startsWith(base), `the page upload uses the base (${upload?.url})`);
  ok(errors.length === 0, `the API base path throws nothing (${errors.map(String)})`);
}

for (const scenario of [backendUnavailable, slowBackend, emptyProject, failedPatchRetries,
                        failedGlossaryAccept, happyPath, pageWorkspace, pageZoomKeepsCoordinates,
                        pageDrawAndDelete, pageEscapeAndDelete, pageOcrAndFailures,
                        pageWithNoImage, projectsHome, pagePanelTextareas, pagePanelApprove, pageChapters,
                        pageImportUsesTheSelectedChapter, apiBaseIsHonoured]) {
  await scenario();
}

console.log(`\n${checks - failures}/${checks} checks passed`);
process.exit(failures ? 1 : 0);
