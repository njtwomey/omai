# Changelog

Versions are released in lockstep for mdml-engine, mdml-render and mdml-meta.

## 0.1.1

- The agent index covers mdml-render as well as mdml-engine: `packages/render/llms.txt` (with a hand-written
  `llms.header.md`), `packages/render/index/` and `index.json`, published with the package. Render has no tagged
  entries yet; its domains are listed from the folder summaries.
- A generated `llms.txt` at the repository root maps the packages (mdml-engine, mdml-render, and mdml-meta as not
  indexed).
- The collector (`mdml-meta` extract, `scripts/agent-index.ts`) indexes several packages: outputs per package, a check
  of stale files per package and of the root file, the same tag vocabulary for all.
- The headers of the engine's and render's `llms.txt` refer to each other.

## 0.1.0

- First release. mdml-engine and mdml-render were released as separate GitHub releases.
