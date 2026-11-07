# Code review of the engine and render

A structural review: how the code is composed, checked and defined (repeated definitions, breaches of the repository's
rules, layering, gaps in the tests of the core), with the surface (functions, tests, examples) staying flat. Not a
review of whether each function is perfect. Started 2026-11-07.

| file                         | what it holds                                                                                                                                                                                                                          |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [plan.md](plan.md)           | scope (core against constructed), method in phases, the rubric, how "flat" is kept                                                                                                                                                     |
| [findings.md](findings.md)   | what the scans and the reviews found, with numbers (updated as phases finish)                                                                                                                                                          |
| [synthesis.md](synthesis.md) | the ranked list after the six readings: the headline, the work in waves A to E, and the decisions that need the maintainer                                                                                                             |
| [progress.md](progress.md)   | every work item with its status (planned, in progress, complete, dropped), start and end times, and where the work is                                                                                                                  |
| [register.md](register.md)   | every finding, one row each, with its status and the work item that handles it                                                                                                                                                         |
| [decisions.md](decisions.md) | what was decided and why, and what still needs a yes                                                                                                                                                                                   |
| [gates.md](gates.md)         | what each merge was tested with, and whether it reached CI                                                                                                                                                                             |
| [log.md](log.md)             | dated notes: decisions, noteworthy things found, gotchas                                                                                                                                                                               |
| [reports/](reports/)         | outputs worth keeping that the scripts would otherwise regenerate: `duplicates-strict.txt` (verified duplicate function clusters), `tensor-migration-candidates.txt` (local vector and matrix types left alone, to migrate to tensors) |

Findings from the reading agents are written to `findings/<cluster>.md` in this folder when each cluster is done.

## Re-running the scans

All read the engine's source; none changes anything.
- `make api-surface-check`: the public surface (every export's name and signature digest) against `docs/api-surface.txt`.
  `node scripts/api-surface.ts --full <file>` writes the full list; `--compare <file>` diffs against one. A refactor that
  must not change the API compares against a `--full` file taken before it.
- `node scripts/review/names.mjs <report> [similarity] [min tokens]`: function names clustered, bodies compared (the
  strict tier is `0.8 40`).
- `node scripts/review/scan.mjs`: generic-named helper copies, repeated constants, layering, cast density.
- `node scripts/review/surface.mjs <full surface file>`: export-name collisions and core exports nothing else uses.
- `node scripts/review/types.mjs`: type definitions written more than once; the numeric shapes and how they are spelled.
- `node scripts/review/unused.mjs <full surface file> [rows.tsv]`: the core exports no other source file uses, sorted by
  signal into public API, building blocks and dead (`findings/unused-exports.md`).
- `node scripts/review/signatures.mjs`: native arrays in public numeric signatures, per domain.
- `node scripts/review/coverage.mjs <coverage-final.json>`: coverage per domain and file, never-run functions. Make the
  JSON with `vitest run --project engine --coverage --coverage.provider=v8 --coverage.reporter=json
  --coverage.reportsDirectory=<dir> --coverage.include="src/lib/<domain>/**"` (one include per core domain; the path is
  relative to the engine project, a repository-relative path matches nothing).
- `scripts/type-definitions.ts --check` (in `make lint`): the guard against restated vector types and purposeless
  aliases.

- `summary.md`: where the review stands at the end of 2026-11-07 (state, what changed, surface changes, what is not done, before pushing).
