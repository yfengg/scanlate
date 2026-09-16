# Frontend tests

`test_workbench.mjs` loads the real `workbench/static/index.html` in jsdom with
a stubbed `fetch`, then drives the handlers as a user would. Any uncaught error
or unhandled rejection fails the run.

    npm install jsdom     # once
    node tests/frontend/test_workbench.mjs

`tests/test_frontend.py` runs the same suite under pytest and skips when node or
jsdom isn't installed, so the Python suite stays self-contained.

The guards these cover were verified by mutation: reintroducing each bug
(removing the `render()` null check, setting `_lastSaved` before the PATCH
resolves, calling `fetch` directly for the glossary write) makes the
corresponding checks fail.

## Network trace

`trace_network.mjs` is the scriptable stand-in for opening the app and reading
the browser's Network tab. It loads the page from a running server's own URL,
resolves relative requests against that origin the way a browser does, and
prints method, URL, status and same-origin for each one:

    PYTHONPATH=src SCANLATE_PORT=8035 python -m scanlate &
    node tests/frontend/trace_network.mjs

Expected output is a single `GET 200 .../api/projects/demo/segments`
with `same-origin=true`, and one `POST .../approve` after the click.
