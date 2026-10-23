# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working instructions

- When a message starts with `init`, follow `INIT.md`. With `audit`, follow `AUDIT.md`. With `port`, follow `PORT.md`.
  None of these is a skill or slash command.
- `docs/structure.md` is the plan: the table of contents of the whole library (every package, folder and file, with
  what it contains), the shared contracts (section 3), the conventions (section 2) and the order of construction
  (section 12). Put new code where it says; if something has no home, update the document in the same change.

## Commands

All routine work goes through `make` (it wraps mise, pnpm and uv). Run `make check` before declaring work finished.
- `make setup`: install tools and dependencies.
- `make check`: barrels, layering, lint, typecheck, tests.
- `make test`: all Vitest projects. One file: `mise exec -- pnpm exec vitest run packages/engine/src/lib/tensor/ops/arithmetic.test.ts`.
  One test by name: add `-t "<name>"`.
- `make barrels`: regenerate `index.ts` files after adding or removing a file or folder (never edit them by hand).
- `make layers`, `make lint`, `make format`, `make typecheck`.
- `make fixtures`: regenerate golden fixtures from the Python oracle (`oracle/cases/...`); tests never run Python.

## Architecture in brief

- Three published packages: `packages/engine` (`whitebox-engine`, compute), `packages/render` (`whitebox-render`,
  headless figures plus React bindings), `packages/meta` (`whitebox-meta`, docs metadata and the example runner). Two
  private apps: `apps/docs`, `apps/gallery`.
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
- Every export has TSDoc: summary, `@param`, `@returns`, and an `@example` whose `// => value` lines are checked.
- Tests sit next to code: `foo.test.ts`, `foo.property.test.ts` (fast-check), `foo.golden.test.ts` (fixtures in
  `packages/engine/fixtures/<same path>/`), `foo.test-d.ts`.
- Python is uv-only, configured through pydantic-settings; the oracle mirrors engine paths under `oracle/cases/`.
- Python is fully typed: every function and method annotates all arguments and its return type (`-> None` too), and
  named lambdas become typed `def`s. Ruff `ANN` and pyright strict enforce it in `make lint` and `make typecheck`.
