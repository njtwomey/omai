# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working instructions

- When a message starts with `init`, follow `INIT.md`. With `audit`, follow `AUDIT.md`. With `port`, follow `PORT.md`.
  None of these is a skill or slash command.
- `docs/structure.md` is the plan: the table of contents of the whole library (every package, folder and file, with
  what it contains), the shared contracts (section 3), the conventions (section 2) and the order of construction
  (section 12). Put new code where it says; if something has no home, update the document in the same change.

## Commands

All routine work goes through `make` (it wraps mise, pnpm and uv).
- `make verify`: the gate for every commit and merge. It runs the checks that take seconds over the whole repo (barrels,
  layering, lint, tsc), then only what the change touches: `vitest related` on the changed TypeScript (every test
  importing it, transitively), the golden tests of changed fixtures, pyright and a fixture check of the changed oracle
  case modules. Changes to shared infrastructure (dependencies, tool config, `scripts/`, the oracle's helpers) widen it
  to everything. `BASE=<ref>` compares against something other than `main`.
- While iterating, run just the files you are working on (one file:
  `mise exec -- pnpm exec vitest run packages/engine/src/lib/tensor/ops/arithmetic.test.ts`; one test: add `-t "<name>"`).
- `make check` and `make fixtures-check`: everything, every test and every fixture. A periodic full run (after a batch
  of merges and before a release), not a per-commit gate.
- `make setup`: install tools and dependencies.
- `make test`: all Vitest projects.
- `make barrels`: regenerate `index.ts` files after adding or removing a file or folder (never edit them by hand).
- `make layers`, `make lint`, `make format`, `make typecheck`.
- `make fixtures`: regenerate golden fixtures from the Python oracle (`oracle/cases/...`); tests never run Python. Scope
  it to what you changed: `mise exec -- uv run --group oracle-core --group oracle-torch --group oracle-prob --group
  oracle-rl python -m whitebox_oracle generate --only stats/tests/,kernels/` (comma-separated prefixes).
  `make manifest` rebuilds `fixtures/manifest.json` from the fixture files (for merge conflicts) in under a second.
  Fixtures follow the standard in `docs/structure.md` section 9: prebuilt numpy/scipy/etc. functions as references, tiered
  tolerances (1e-12 exact, 1e-10 special functions and direct linear algebra, 1e-6 to 1e-8 iterative), no gradient
  fixtures (gradients are checked by `checkPrimitive`), and parity with the reference as the goal, not extra digits.

## Architecture in brief

- Four packages: `packages/engine` (`whitebox-engine`: the core in `lib/`, headless methods in `methods/` that return
  typed exposition records from `lib/exposition`), `packages/render` (`whitebox-render`, headless figures, views of
  exposition records, React bindings), `packages/meta` (`whitebox-meta`, docs metadata and the example runner),
  `packages/hub` (`whitebox-hub`, declarative entries composing methods and render views; no drawing of its own).
  One private app, `apps/site` (`make dev`, http://localhost:5300), with three sections on a shared shell: the gallery
  (render, hand-written recipes), the hub (generated from its catalogue) and the docs (generated from the TSDoc).
- Engine core is `packages/engine/src/lib/<domain>/`; each domain has a `domain.json` tier and imports only lower tiers.
  Applications are `packages/engine/src/methods/<area>/` with `area.json` listing allowed area imports. `make layers`
  enforces both. Every folder is an import path (`whitebox-engine/numerics/linalg`); subfolders are namespaces.
- Operations are primitives (`lib/tensor/primitive.ts`) with kernels and `jvp`/`vjp` rules. Differentiation is opt-in
  per call (`grad(f)(x)`, JAX style): outside a transform, tensors are plain arrays with no graph. A new primitive
  must pass `checkPrimitive` (finite-difference vjp and jvp, vjp/jvp consistency, no input mutation).
- Iterative algorithms are `Algorithm`s run by `runAlgorithm` (`lib/trace`), which emits typed trace events according to
  a `TracePolicy`. No `console`, `Math.random` or clocks in published source (`scripts/banned.ts`).
- Randomness is explicit: immutable `Key`s (`lib/random`) or a `Context` (`lib/context`) that derives keys by name.

## Conventions

- Strict TypeScript (TS 7) with `isolatedDeclarations`: exported functions declare their return types. No `any`.
- Relative imports use `.ts` extensions; Biome formats at 120 columns, single quotes, no semicolons.
- No local copies of generic helpers: number-array kernels come from `numerics/linalg/blas.ts` (`ddot`, `dnrm2`,
  `daxpy`, `dscal`, `dgemv`), value-level ones from the tensor ops. A missing helper goes into a shared module first.
- Every core export (`lib/`) has TSDoc: summary, `@param`, `@returns`, and an `@example` whose `// => value` lines are
  checked. Methods (`methods/`) need a summary and parameter docs; examples are optional. Methods wrap the core rather
  than re-implement it, and return the data an explanation needs (the exposition contract, `docs/structure.md` 5.0).
- Test depth by layer: the core (`lib/`) gets the full standard below; methods get a smoke test per method (runs on a
  small dataset, exposition records have the right shape) plus goldens only where cheap and important; the hub gets
  light tests (the catalogue builds, every entry's demo runs).
- Tests sit next to code: `foo.test.ts`, `foo.property.test.ts` (fast-check), `foo.golden.test.ts` (fixtures in
  `packages/engine/fixtures/<same path>/`), `foo.test-d.ts`.
- Python is uv-only, configured through pydantic-settings; the oracle mirrors engine paths under `oracle/cases/`.
- Python is fully typed: every function and method annotates all arguments and its return type (`-> None` too), and
  named lambdas become typed `def`s. Ruff `ANN` and pyright strict enforce it in `make lint` and `make typecheck`.
