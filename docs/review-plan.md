# Structural code review: plan

Written 2026-11-07. A review of how the code is composed, not of each function: repeated definitions, breaches of the
repository's own rules, layering, and gaps in the tests of the core. The surface (functions, tests, examples) stays flat.

## Scope

- **Core**: `base`, `tensor`, `random`, `space`, `trace`, `autodiff`, `context`, `numerics`, `optim`, `losses`, `dists`,
  `stats`, `metrics`, `kernels`, `data`, `exposition`, `scene` (chosen by how many domains import them: `base` 65,
  `tensor` 59, `random` 57, `trace` 51, `space` 49, `numerics` 48, `optim` 29, `autodiff` 24, `dists` 23).
- **Constructed** (estimators and algorithms built on the core: `learn`, `nn`, `dynamics`, `pgm`, `rl`, `signal`,
  `vision`, ...) are not reviewed, except where a finding in the core shows in them.
- **Render** is reviewed with the same rubric, as one cluster.

## Method

1. **Phase 0, scripts only** (no model reading): the public-surface snapshot (the guard that keeps things flat);
   coverage per file for the core; exact and near duplicates, local copies of generic helpers, repeated constants;
   layering ranks and the upward imports; name collisions and exports only tests use; cast density; a CPU profile of
   representative examples for the hot paths.
2. **Phase 1, targeted reading**: a few agents, each given the Phase 0 report of its cluster and reading only what it
   flags, capped on the files it opens. Foundations (`base`, `tensor`, `random`, `space`, `trace`, `autodiff`,
   `context`: about 17,600 lines) are read in full. Findings in one shape: kind, evidence at `file:line`, fix, size,
   risk, whether it preserves behaviour.
3. **Phase 2, synthesis**: one ranked list, gone through with the maintainer.
4. **Phase 3, fix waves**: small agents, at most two at a time, one class of finding each (shared helpers first, then
   layering moves, then the few test gaps). A wave merges after the local full gate and an identical surface snapshot.

## Rubric

Repeated definitions; breaches of the rules in `CLAUDE.md` (local copies of generic helpers, models fused to one
solver, clocks or `Math.random` in published source, algorithms off the log contract, missing TSDoc or `@example`);
inconsistent numerical conventions (tolerances, NaN, degenerate inputs); inconsistent validation and error patterns;
invariants nothing tests (a sampler against its CDF, a primitive's gradients).

## Keeping it flat

Refactors preserve behaviour: the surface snapshot, the test count (26,076 on 2026-11-03) and the example count (9,567)
are compared after every wave. New tests only for gaps in the core, offset by merging redundant ones, with the net count
reported.

## Findings

Recorded in `docs/review-findings.md` as the phases produce them.
