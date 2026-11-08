# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working instructions

- There is no methods layer in the engine. `docs/methods-catalog/` is the catalogue of methods (a taxonomy over the
  library: one folder and one `<area>.md` table per chapter, with the methods, why they are interesting, their tracks on
  the tape and what they need from the library; `methods-catalog.md` is its design, `PLAN.md`, `LIB.md` and `RENDER.md`
  the work plans of the library and render phases, `archive/` the code of the retired methods layer, kept for reference
  and outside every check). A demonstration of a method is a page of the site that composes a library estimator with
  its dataset, slots, panels and narrative; anything with substance goes into `lib/`.
- `docs/structure.md` is the plan: the table of contents of the whole library (every package, folder and file, with
  what it contains), the shared contracts (section 3), the conventions (section 2) and the order of construction
  (section 12). Put new code where it says; if something has no home, update the document in the same change.
- Agents that need isolation work in a git worktree under `.claude/worktrees/` named `agent-<id>-<word>`: the agent's
  short id and one word for what it is doing (`agent-afa77c-vision`). One worktree per agent, removed once its branch
  is merged; never an unlabelled `worktree-agent-<hash>`.
- **Where a demonstration goes.** The gallery (`apps/site`, `src/gallery/recipes/<section>/NN-slug.tsx`) shows what
  omai-render can draw and the controls it offers: one recipe per render capability, toy data only. It is never for a
  model, an estimator or an algorithm (sparse coding, a sampler, a classifier): those get a library module, tests and
  TSDoc and nothing on the gallery, whatever they would look like. Add or extend a recipe only when a render option,
  chart, panel, control or interaction is new. A proof of concept (`apps/pocs`, `make pocs`) is a single live page of
  what the library can do, for development, and exists only because the user asked for it: never add one unasked, and
  never edit one unasked. The docs (`apps/site`, generated from the TSDoc) are where a method is explained. Examples
  (`examples/`) are programs with a figure and a tape, for the front page and the README.
- **Every agent tests only its own changes.** Its own test files, tsc, barrels, layers, biome and its own examples,
  never the full suite, never `make check`, and never a repo-wide vitest. The main session alone decides when the gate
  runs, and does not run it unless asked.
- Agents throttle everything that runs tests: `nice -n 15 env OMAI_TEST_WORKERS=1 make verify` (never `make check`,
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
- `make thumbs`: make the gallery thumbnails in `apps/site/public/thumbs/` with headless Chrome (about 3 minutes for
  all of them). It only makes the missing ones; `ARGS="--force"` redoes all, `ARGS="--only axes"` or
  `ARGS="--only axes/twin-axes"` limits it. Run it with `--force` after a recipe's figure changes.
- `make agent-index`: regenerate the agent index (`llms.txt` at the root, `packages/{engine,render}/llms.txt` and
  `index/`) from the `@index` tags; `make agent-index-check` (part of `make verify` and `make check`) fails on a stale file or an invalid
  tag. See "The agent index" below.
- `make test`: all Vitest projects.
- `make barrels`: regenerate `index.ts` files after adding or removing a file or folder (never edit them by hand).
- `make layers`, `make lint`, `make format`, `make typecheck`.
- `make fixtures`: regenerate golden fixtures from the Python oracle (`oracle/cases/...`); tests never run Python. Scope
  it to what you changed: `mise exec -- uv run --group oracle-core --group oracle-torch --group oracle-prob --group
  oracle-rl python -m omai_oracle generate --only stats/tests/,kernels/` (comma-separated prefixes).
  `make manifest` rebuilds `fixtures/manifest.json` from the fixture files (for merge conflicts) in under a second.
  Fixtures follow the standard in `docs/structure.md` section 9: prebuilt numpy/scipy/etc. functions as references, tiered
  tolerances (1e-12 exact, 1e-10 special functions and direct linear algebra, 1e-6 to 1e-8 iterative), no gradient
  fixtures (gradients are checked by `checkPrimitive`), and parity with the reference as the goal, not extra digits.

## Architecture in brief

- Three packages: `packages/engine` (`omai-engine`: the core in `lib/`; estimators and algorithms run as
  sessions of the log contract and return typed exposition records from `lib/exposition`), `packages/render`
  (`omai-render`, headless figures, views of exposition records, React bindings including the session layer: the hook,
  the player, the panels), `packages/meta` (`omai-meta`, docs metadata and the example runner). One private app,
  `apps/site` (`make dev`, http://localhost:5300), with two sections on a shared shell: the gallery (render,
  hand-written recipes) and the docs (generated from the TSDoc). A second private app, `apps/pocs` (`make pocs`, http://localhost:5310), holds proofs of
  concept: single live pages on the site's own shell, searchable, added only on request (see its README). There is no hub: pages that explain a method are the
  encyclopedia's, composed from the packages.
- An estimator or algorithm is a configuration (a `Space`), slots filled from tables of library options
  (`lib/trace/modes.ts` for algorithms, `lib/space/tables.ts` for values; `union` spaces carry a mode and the knobs of
  its own), a contract of paths on the tape, and probes. Declare the contract with `defineContract` and give the
  method its path type (`Method<S, C, FooPath>`): `ctx.log`, `ctx.emit`, `defaults` and `logs:` keys are then checked.
  What a viewer can choose (the dataset, the kind of shift, the model's parameters) is a hyperparameter of the
  estimator, so a site page only composes.
- **Everything is library.** Algorithms, data structures, registries, estimators on the iterative, trainer, agent and
  sweep bases (which carry `.method` and `.session` for free), loops that run other things such as trainers, sweeps and
  streams, tables of options, generators with their truth: all go in `lib/` with the full standard. Every iterative
  routine ships as an `Algorithm`; every generator carries its truth and registers with tags in `lib/datasets`.
- **A model is an objective; the solver is a slot.** An estimator fitted by minimising a smooth objective (logistic and
  other GLMs, ordinal, count models, Cox, kernel and linear classifiers) exposes the objective as a `SmoothProblem`
  and takes `solver` as a hyperparameter filled from `smoothSolvers` (`lib/optim/solvers.ts`: gradient descent,
  conjugate gradient, Newton, BFGS, L-BFGS, coordinate descent, and the stochastic `gradient` mode with its optimiser,
  schedule and batch knobs). It is never fused to one solver ("logistic regression" is not "IRLS"); a solver specific to
  one family (IRLS, SMO, EM) is an extra mode beside the generic ones.
- Engine code is `packages/engine/src/lib/<domain>/`. There are no tiers or metadata files: domains import one
  another without cycles (`make layers` reports any cycle as its path). Every folder is an import path
  (`omai-engine/numerics/linalg`); subfolders are namespaces, or re-exported flat when their summary has `@inline`.
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

## The agent index

Agents that consume the libraries start from the repository `llms.txt` (a map of the packages), then the package's own
`packages/engine/llms.txt` (omai-engine, domains = the folders of `src/lib`) or `packages/render/llms.txt`
(omai-render, domains = the folders of `src`, `react` nested by its subfolders), then the domain file in that package's
`index/`, then the source: the index lists the main things, not all of them. The durable place for an entry is the
TSDoc of the thing itself: `@index <kind> <description>`, at most about 12 words, no signature. Kinds (closed, the same
for both packages): `function` (also metrics, losses, tests, transforms; a React component or hook is a `function`
whose description starts "React component:" or "React hook:"), `estimator`, `algorithm`, `structure` (a type, record
or class), `table` (option tables, registries), `module` (nn layers), `dataset`, `distribution`, `kernel`, `agent`.
An estimator is indexed instead of the `Algorithm` it wraps; a factory takes the kind of what it returns. A file entry
is `@index` as the last paragraph of the file's opening comment (a hub to open first); a symbol entry sits in the
declaration's TSDoc before its `@example`. When asked to "index this function", add the tag and run
`make agent-index` as the last step, after the final edit to the source: a line added above a tagged declaration
moves its `path:line`, and a stale index fails CI. Never edit the generated files by hand (`llms.txt`, `index/`; only
`packages/<package>/llms.header.md` is hand-written). Details: `docs/structure.md` section 2.

## Conventions

- Strict TypeScript (TS 7) with `isolatedDeclarations`: exported functions declare their return types. No `any`.
- Relative imports use `.ts` extensions; Biome formats at 120 columns, single quotes, no semicolons.
- No local copies of generic helpers: number-array kernels come from `numerics/linalg/blas.ts` (`ddot`, `dnrm2`,
  `daxpy`, `dscal`, `dgemv`), value-level ones from the tensor ops. A missing helper goes into a shared module first.
- Every core export (`lib/`) has TSDoc: summary, `@param`, `@returns`, and an `@example` whose `// => value` lines are
  checked. What an explanation needs is returned as exposition records (the exposition contract, `docs/structure.md` 5.0).
- Test depth: `lib/` gets the full standard below; site pages get a smoke test (they run on a small dataset and the
  tape's paths and probes have the right shape) and no fixtures.
- Tests sit next to code: `foo.test.ts`, `foo.property.test.ts` (fast-check), `foo.golden.test.ts` (fixtures in
  `packages/engine/fixtures/<same path>/`), `foo.test-d.ts`.
- Python is uv-only, configured through pydantic-settings; the oracle mirrors engine paths under `oracle/cases/`.
- Python is fully typed: every function and method annotates all arguments and its return type (`-> None` too), and
  named lambdas become typed `def`s. Ruff `ANN` and pyright strict enforce it in `make lint` and `make typecheck`.
