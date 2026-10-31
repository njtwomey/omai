# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working instructions

- When a message starts with `init`, follow `INIT.md` (a one-time bootstrap, done: the repository exists). With `audit`,
  follow `AUDIT.md`. Neither is a skill or slash command. (`port` and `PORT.md` are gone: the port of the parked
  methods became the rebuild of the methods layer below.)
- The methods layer is designed in `packages/engine/src/methods/README.md` (read it before writing a method: slots,
  modes, the structure by what a page does, the README table format, the build order). Each chapter folder's
  `README.md` is its to-do table (methods, why they are interesting, their tracks on the tape, what they need from the
  library, status); `LIB.md` and `RENDER.md` beside it are the library and render work plans. Update the table of
  the area you build in the same change.
- `docs/structure.md` is the plan: the table of contents of the whole library (every package, folder and file, with
  what it contains), the shared contracts (section 3), the conventions (section 2) and the order of construction
  (section 12). Put new code where it says; if something has no home, update the document in the same change.
- Agents that need isolation work in a git worktree under `.claude/worktrees/` named `agent-<id>-<word>`: the agent's
  short id and one word for what it is doing (`agent-afa77c-vision`). One worktree per agent, removed once its branch
  is merged; never an unlabelled `worktree-agent-<hash>`.
- Agents throttle everything that runs tests: `nice -n 15 env MDML_TEST_WORKERS=1 make verify` (never `make check`,
  never a second test run while one is going, never a full-repo `vitest run`). The main session alone runs the gate at
  normal priority. Several agents verifying at once at full width stalls the machine for the person using it.

## Commands

All routine work goes through `make` (it wraps mise, pnpm and uv).
- `make verify`: the gate for every commit and merge. It runs the checks that take seconds over the whole repo (barrels,
  layering, lint, tsc), then only what the change touches: `vitest related` on the changed TypeScript (every test
  importing it, transitively), the golden tests of changed fixtures, the `@example`s of the changed engine source files
  (`make examples` with `--only <files>`, a few seconds), pyright and a fixture check of the changed oracle case
  modules. Changes to shared infrastructure (dependencies, tool config, `scripts/`, the oracle's helpers, the meta
  extractor and runner for examples) widen it to everything. `BASE=<ref>` compares against something other than `main`.
- While iterating, run just the files you are working on (one file:
  `mise exec -- pnpm exec vitest run packages/engine/src/lib/tensor/ops/arithmetic.test.ts`; one test: add `-t "<name>"`).
- `make check` and `make fixtures-check`: everything, every test, example and fixture. A periodic full run (after a batch
  of merges and before a release), not a per-commit gate.
- `make setup`: install tools and dependencies.
- `make docs-catalogue`: precompile the docs catalogue into `apps/site/.generated/docs/` (static files the docs pages
  fetch); `make dev` and `make site` run it first, and it skips itself when the output is newer than the sources. The
  dev server regenerates it in the background after engine edits.
- `make test`: all Vitest projects.
- `make barrels`: regenerate `index.ts` files after adding or removing a file or folder (never edit them by hand).
- `make layers`, `make lint`, `make format`, `make typecheck`.
- `make fixtures`: regenerate golden fixtures from the Python oracle (`oracle/cases/...`); tests never run Python. Scope
  it to what you changed: `mise exec -- uv run --group oracle-core --group oracle-torch --group oracle-prob --group
  oracle-rl python -m mdml_oracle generate --only stats/tests/,kernels/` (comma-separated prefixes).
  `make manifest` rebuilds `fixtures/manifest.json` from the fixture files (for merge conflicts) in under a second.
  Fixtures follow the standard in `docs/structure.md` section 9: prebuilt numpy/scipy/etc. functions as references, tiered
  tolerances (1e-12 exact, 1e-10 special functions and direct linear algebra, 1e-6 to 1e-8 iterative), no gradient
  fixtures (gradients are checked by `checkPrimitive`), and parity with the reference as the goal, not extra digits.

## Architecture in brief

- Three packages: `packages/engine` (`mdml-engine`: the core in `lib/`, headless methods in `methods/` that run as
  sessions of the log contract and return typed exposition records from `lib/exposition`), `packages/render`
  (`mdml-render`, headless figures, views of exposition records, React bindings including the session layer: the hook,
  the player, the panels), `packages/meta` (`mdml-meta`, docs metadata and the example runner). One private app,
  `apps/site` (`make dev`, http://localhost:5300), with two sections on a shared shell: the gallery (render,
  hand-written recipes) and the docs (generated from the TSDoc). There is no hub: pages that explain a method are the
  encyclopedia's, composed from the packages.
- A method is a few dozen lines over the library: a configuration (a `Space`), slots filled from tables of library
  options (`lib/trace/modes.ts` for algorithms, `lib/space/tables.ts` for values; `union` spaces carry a mode and the
  knobs of its own), a contract of paths on the tape, and probes. It never implements an algorithm: if a method needs
  something the library lacks, the library gets it first.
- **Lib or methods?** A piece that would make sense to someone who never reads a tape is library code (algorithms, data
  structures, registries, loops that run other things such as trainers, sweeps and streams, tables of options,
  generators with their truth): it goes in `lib/` with the full standard. Nothing is "shared among methods" as such;
  there is no `methods/shared/`. Every iterative routine ships as an `Algorithm`; every generator carries its truth
  and registers with tags in `lib/datasets`.
- **A model is an objective; the solver is a slot.** An estimator fitted by minimising a smooth objective (logistic and
  other GLMs, ordinal, count models, Cox, kernel and linear classifiers) exposes the objective as a `SmoothProblem`
  and takes `solver` as a hyperparameter filled from `smoothSolvers` (`lib/optim/solvers.ts`: gradient descent,
  conjugate gradient, Newton, BFGS, L-BFGS, coordinate descent, and the stochastic `gradient` mode with its optimiser,
  schedule and batch knobs). It is never fused to one solver ("logistic regression" is not "IRLS"); a solver specific to
  one family (IRLS, SMO, EM) is an extra mode beside the generic ones.
- Engine core is `packages/engine/src/lib/<domain>/`, applications are `packages/engine/src/methods/<area>/`. There are
  no tiers or metadata files: domains import one another, and areas one another, without cycles, and the core never
  imports methods (`make layers` reports any cycle as its path). Every folder is an import path
  (`mdml-engine/numerics/linalg`); subfolders are namespaces, or re-exported flat when their summary has `@inline`.
- The code documents itself: each folder's `index.ts` opens with a hand-written doc comment, a short paragraph on what
  the folder is and holds (`make barrels` keeps it above the generated exports), and each source file opens with a
  short paragraph on what it is and what to expect in it (no to-do lists or export lists). `make barrels` fails
  without them; the docs read both.
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
  small dataset, the tape's paths and the probes have the right shape) and no fixtures, since they wrap code that is
  tested already.
- Tests sit next to code: `foo.test.ts`, `foo.property.test.ts` (fast-check), `foo.golden.test.ts` (fixtures in
  `packages/engine/fixtures/<same path>/`), `foo.test-d.ts`.
- Python is uv-only, configured through pydantic-settings; the oracle mirrors engine paths under `oracle/cases/`.
- Python is fully typed: every function and method annotates all arguments and its return type (`-> None` too), and
  named lambdas become typed `def`s. Ruff `ANN` and pyright strict enforce it in `make lint` and `make typecheck`.
