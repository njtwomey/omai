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

## Type definitions written more than once

`node scripts/review/types.mjs` reads every `type` and `interface` of the packages and apps (7,859 definitions, 7,253
exported, in 1,798 files), normalises each right-hand side (readonly dropped, `Array<T>` as `T[]`, members sorted) and
groups them by name and by structure.
- **Same name in several files: 355 names**, 11 with an identical structure (the same type written again): `Point2`
  (`[number, number]`) in `graphics/rasterise.ts`, `numerics/geometry/triangulation.ts` and `render/react/interaction/
  projection.ts`; `Pt` in `numerics/geometry/contours.ts`, `render/charts/lines.ts` and `render/svg/paths.ts`; `Vec3` in
  `dynamics/celestial/elements.ts` and `scene/three-d.ts`; `Rgba` in `graphics/textures.ts` and `scene/paint.ts`;
  `Reduction` in `base/options.ts` and `losses/_reduce.ts`; `RiemannRule` in `foundations/calculus.ts` and
  `render/charts/functions.ts`; `C` (a complex pair) in `numerics/polynomial/rational.ts` and `symbolic/_numeric.ts`. The
  other 344 share a name with a different structure: the same word meaning different things.
- **Numeric shapes: 141 definitions in 49 shapes.** One shape under many names: `[number, number]` is `Point2`, `Pt`,
  `Complex`, `ComplexNumber`, `PacmapPair` and 20 more (several are state types, which are a different matter);
  `[number, number, number]` is `Vec3`, `Rgb`, `Point3`, `P3`, `Triplet`; `[n, n, n, n]` is `Rgba`, `Quat`, `Plane4`, `Box`.
  One name under several shapes: `Vec` is `number[]` in some files and `Float64Array` in others (12 definitions of
  `Mat`/`Vec` as `Float64Array`); `Matrix` is `number[][]` in four places and a `{rows, cols, data}` record in two.
- **How vectors are spelled in the engine's source** (occurrences): `number[]` 11,720; `Float64Array` 10,940; `TensorLike`
  6,947; `Value` 6,567; `number[][]` 2,137; `ArrayLike<number>` 1,472. By domain, `optim` writes `Float64Array` 1,640 times
  and `number[]` 211; `numerics` is even (580 and 513); `dists` and `losses` write `TensorLike` and `Value`.
- **What the scan cannot say:** whether two names with one structure mean one thing (a `Point2` and a `Complex` do not).
  The cluster review reads only the vector, matrix, point, colour and complex families, not every shape.

### The rule for these types

Specialised names stay (`Rgba`, `Complex`, `Point2`, `Quat` mean different things in different code); what the review
wants is that each is a specialisation of one underlying type and not a restatement of it: `type Rgba = Vec4`, not
`type Rgba = readonly [number, number, number, number]`. The scan separates the two. Of the base shapes there are 83
definitions (`Vec2` 26, `Vec3` 7, `Vec4` 11, a plain vector 18, a matrix of rows 21), and **none is an alias of
another name**: every one restates the shape, because the engine has no canonical small-vector or matrix type (`Vec3`
lives in `scene/three-d.ts`, `Point2` in `graphics/rasterise.ts`, `Matrix` in seven places). The proposal is a few
canonical types in `base` (`Vec2`, `Vec3`, `Vec4`, `Vector`, `Matrix`, readonly, tuples and arrays of `number`), and the
vector-named definitions (about 42, listed by `types.mjs`) become aliases of them, in the engine and in render (which
already depends on `mdml-engine`). It is a change of types only: no runtime, the same structural types, so nothing
type-checks differently. The signature digest of the declarations that change moves, which `make api-surface-check`
reports; that is the intended, reviewed change.

The hot-path vector spellings (`Float64Array` against `number[]` against `TensorLike`) are a design question, not a
restatement, and are left to the cluster review.

## Coverage of the core

`node scripts/review/coverage.mjs <coverage-final.json>` (the engine's whole suite in CI mode with v8 coverage over the
core: `vitest run --project engine --coverage` with `--coverage.include="src/lib/<domain>/**"`; the engine suite passed,
25,466 tests, in 565 s). 498 core files: **93.9% of statements, 94.4% of functions, 73.5% of branches**; 642 functions
never run. Branches are the weak dimension everywhere. By domain (statements / functions / branches): `exposition`
76.0 / 80.3 / 68.6 (the lowest), `base` 85.0 / 85.7 / 73.3, `tensor` 89.8 / 92.5 / 71.4, `data` 91.4 / 88.2 / 74.2,
`dists` 92.4 / 90.6 / 68.1 (177 functions never run, the most), `numerics` 94.0 / 95.0 / 71.1 (110), `optim` 95.1 / 95.3 /
75.0 (108); `losses`, `random`, `kernels`, `scene`, `space` are at 97 or more.

Where the gaps are:
- **No test reaches** `exposition/subgroups.ts` (43 statements, 12 functions: `subgroupReport`, `subgroupGrid`, ...).
- **Mostly unrun functions:** `dists/compose/structure.ts` (25 of 113), `dists/continuous/heavy-tailed.ts` (18 of 122),
  `tensor/ops/indexing.ts` (15 of 86, among them `nonzero`), `data/archives.ts` (14 of 28), `dists/relaxed.ts` (12 of
  32), `dists/continuous/bounded.ts` (11 of 78), `dists/multivariate/matrix.ts` (10 of 39), `autodiff/custom.ts` (9 of 60),
  `dists/multivariate/simplex.ts` and `copulas.ts`, `data/batching.ts` (8 of 33).
- **Low statement coverage in a sizeable file:** `tensor/format.ts` 14.3% of 42, `stats/shrinkage.ts` 76.2% of 151,
  `data/archives.ts` 77.1% of 153, `base/numbers.ts` 77.2% of 136, `dists/distribution.ts` 73.4% of 128,
  `numerics/special/_incomplete.ts` 78.0% of 354, `numerics/special/_elliptic.ts` 81.1% of 148.

Caveat: this counts the vitest tests, not the `@example` blocks (run by `scripts/examples.ts`), so some functions listed
here may be exercised by an example; a function that neither runs is the finding. Many unrun functions are anonymous
closures (the listing cannot name them).

## Not yet done

The CPU profile for the hot paths, the cluster reading (Phase 1), and the
ranking (Phase 2).
