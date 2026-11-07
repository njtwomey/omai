# Structural review: findings

Phase 0 of `docs/review-plan.md`: what the scripts found, before any model reading. Every number is from running the
scripts on 2026-11-07 (commit b09e46da plus the scripts). Re-run them with:
- `node scripts/review/names.mjs <out> [similarity=0.6] [min tokens=12]`: names the function definitions, clusters them
  by name, and verifies each cluster by comparing the bodies (token shingles). `0.8 40` is the strict tier below.
- `node scripts/review/scan.mjs`: near-duplicate functions inside the core, generic-named helpers defined in several
  files, constants defined identically in several files, layering, and cast density.
- `node scripts/review/surface.mjs <full surface file>`: export-name collisions across entry points, and core exports no
  other source file uses. The surface file is `node scripts/api-surface.ts --full <file>`.
- `node scripts/api-surface.ts --check` (`make api-surface-check`): the guard that a refactor leaves the public surface
  (17,919 exports in 121 entry points) unchanged; `--compare <full file>` lists what moved.

Core domains: `base`, `tensor`, `random`, `space`, `trace`, `autodiff`, `context`, `numerics`, `optim`, `losses`,
`dists`, `stats`, `metrics`, `kernels`, `data`, `exposition`, `scene`.

## Names and bodies

13,015 functions (at least 12 tokens) in 1,707 engine files fall into 10,649 name clusters (names split into words,
noise words dropped, plurals folded, near names joined); 980 span two or more files. Checking the bodies of those 980:
- 228 clusters have members at least 60% alike; the strict tier (at least 80% alike, at least 40 tokens) leaves **99**,
  of which **26 have identical bodies** and **10 are a core export copied outside the core**.
- Strict-tier clusters worth a look (a first reading, not a verdict):
  - `dot` identical in six files (`dynamics/ode/_bvp.ts`, `gam/extended.ts`, `kernel-methods/online-kernel.ts`,
    `linear/lars.ts`, `metrics/audio.ts`, ...): the "no local copies of generic helpers" rule; `ddot` is in `blas.ts`.
  - `matmul` identical in `stats/_linear.ts` and `stats/tests/_kolmogorov.ts` (the tensor op exists);
    `transpose` in `numerics/fourier/nonuniform.ts`, `numerics/interpolate/smoothing.ts`, `stats/_linear.ts`.
  - `svddDual` and `oneClassDual` in `optim/smo.ts` against `svm/_dual.ts` (51 and 36 lines there): two implementations
    of the same duals in two domains.
  - `nextAfter` identical in `dynamics/ode/_vec.ts` and `optim/minimize/least-squares-trf.ts` (and similar in
    `numerics/approximation/floating-point.ts`); `neighbourSets` identical in `graph/_neighbours.ts` and
    `graph/measures.ts`; `backcastOf` identical in two `timeseries` files; `blank` identical in `dynamics/control` and
    `environments`.
  - `softplus` in `tensor/ops/math.ts` and `nn/functional/activations.ts`; `logit` in `dists/families/links.ts` and a copy
    in `datasets/causal.ts`; `clip` in `tensor/ops/arithmetic.ts` and `stochastic/diffusion-samplers.ts`.
  - `rowsOf`/`rows`/`toRows`/`columnOf`/`matrixOf` in about 25 files (3 to 10 lines each): the same coercion written
    again and again; one shared helper would take them all.
  - `option` (15 lines) identical in `information/dictionary-coding-options.ts`, `logic/algorithm-options.ts` and
    `numerics/approximation/approximation-options.ts`; `knobsOf` in five `*-options.ts` files: the option-table pattern is
    copied per domain instead of built by one helper.
- Near-duplicate functions of 10 or more lines inside the core (a stricter, separate scan): none at 80% similarity.

## Generic helpers and constants

- Generic-named helpers defined in several files (the rule: number-array kernels from `blas.ts`, value-level ones from the
  tensor ops): `norm` in 13 files, `identity` 10, `dot` 10, `mean` 9, `sigmoid` 8, `square` 6, `flatten` 6, `transpose` 6,
  `variance` 5, `meanOf` 4, `outer` 4, `clamp` 3 (one is `base/numbers.ts`), `lerp` 3, `linspace` 3. Not all are
  copies (some are different functions with a common name); the verified clusters above separate them.
- Constants defined identically in three or more files: `EPS = 2.22e-16` in 15 files, `EULER` in 7,
  `LOG_SQRT_2PI` in 4, `EPS = 1e-12` in 4, `EPS = 1e-9` in 3, `TAU = 1e-12` in 3 (the SMO files).

## Layering

`make layers` checks cycles only. With the core ordered `base < context < tensor < random < space < trace < autodiff <
numerics < optim < scene < exposition < losses < dists < kernels < stats < metrics < data`:
- Core files importing a constructed domain (6): `metrics/audio.ts` and `metrics/image-quality.ts` -> `signal`,
  `metrics/text.ts` -> `text`, `kernels/structured.ts` -> `graph`, `losses/large-output.ts` and
  `stats/tests/independence.ts` -> `information`.
- Core imports against that order (the order is a first guess, to be settled): `context` -> `random`, `trace`, `tensor`
  (the rank of `context` is probably wrong), `losses` -> `dists` (2), `autodiff/graph.ts` -> `scene`, `optim/glm.ts` ->
  `dists`.

## Public surface

- 479 engine export names appear in two or more entry points (of 16,140 distinct), 282 with a core definition. Many are
  right (`gamma` the distribution, the function and the sampler); some are the same thing defined again: `softplus` in
  four entry points, `Constraint` in `dists`, `kernels`, `optim`, `search`; `Bounds`, `Schedule` and `Selection` in four
  each; `FeatureMap`, `MonteCarloOptions`, `Kernel`, `Sampler` in three.
- 1,952 of the core's 4,056 exports are mentioned in no other source file (site and pocs included; tests excluded): optim
  505, numerics 327, stats 319, metrics 249, data 126, losses 121, dists 82, kernels 74. By kind: 895 functions, 747
  interfaces, 177 constants, 133 types. For a library much of this is the public surface; the question is which have
  neither a test nor an `@example` (see coverage).

## Casts

3,975 `as T` assertions in the core files over 60 lines, 44 of them `as unknown as`. The densest are tabulated or
dispatching code: `numerics/special/incomplete.ts` (21 per 100 lines), `optim/schedule-table.ts` (19), `data/_columns.ts`
(16), `numerics/special/elliptic.ts`, `bessel.ts`, `gamma.ts`, `zeta.ts` (11 to 15), `space/operations.ts` (59 casts),
`optim/solvers.ts` (40). Worth reading for a missing type, not as a defect count.

## Not yet done

Coverage per file for the core (running), the CPU profile for the hot paths, the cluster reading (Phase 1), and the
ranking (Phase 2).
