/**
 * Network trace against a live server — the scriptable equivalent of opening
 * the app and watching the browser's Network tab.
 *
 * Loads the page from the running server's own URL, resolves relative request
 * URLs against that origin exactly as a browser does, and prints every request
 * with its method, resolved URL, status and whether it stayed same-origin.
 *
 *   PYTHONPATH=src SCANLATE_PORT=8035 python -m scanlate &
 *   node tests/frontend/trace_network.mjs
 */
import { JSDOM } from "jsdom";

const BASE = process.env.SCANLATE_URL || "http://127.0.0.1:8035/";
const html = await (await fetch(BASE)).text();
const log = [];

const dom = new JSDOM(html, {
  runScripts: "dangerously",
  url: BASE,                      // document origin, exactly as the browser has it
  beforeParse(window) {
    window.addEventListener("error", e => log.push({ error: String(e.error || e.message) }));
    window.addEventListener("unhandledrejection", e => log.push({ error: String(e.reason) }));
    window.HTMLElement.prototype.scrollIntoView = () => {};
    // Resolve relative URLs against the document's origin, the way a browser does.
    window.fetch = async (input, init = {}) => {
      const resolved = new URL(input, window.location.href).href;
      const res = await fetch(resolved, init);
      log.push({ method: init.method || "GET", url: resolved, status: res.status,
                 type: res.headers.get("content-type"),
                 sameOrigin: new URL(resolved).origin === new URL(BASE).origin });
      return res;
    };
  },
});

await new Promise(r => setTimeout(r, 600));
const { document } = dom.window;

console.log("--- network ---");
for (const entry of log) {
  if (entry.error) { console.log("UNCAUGHT:", entry.error); continue; }
  console.log(`${entry.method.padEnd(5)} ${entry.status} ${entry.url}`
              + `  [${entry.type}] same-origin=${entry.sameOrigin}`);
}
console.log("--- rendered ---");
console.log("project header :", document.getElementById("project").textContent);
console.log("segments drawn :", document.querySelectorAll(".seg").length);
console.log("needs review   :", document.getElementById("c-flagged").textContent);
console.log("first candidate:", document.querySelector("textarea")?.value);
console.log("uncaught errors:", log.filter(e => e.error).length);

// And a real mutation round trip against the live server.
document.querySelector('button[data-act="approve"]').click();
await new Promise(r => setTimeout(r, 400));
console.log("--- after clicking Approve ---");
for (const e of log.slice(1)) if (!e.error) console.log(`${e.method.padEnd(5)} ${e.status} ${e.url}`);
console.log("state now      :", document.querySelector(".state")?.textContent);
