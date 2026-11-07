# Log

Dated notes, newest at the bottom: decisions, noteworthy things found, gotchas. Times are local.

## 2026-11-07

- **09:44 Scope.** Core (the domains most others import: `base`, `tensor`, `random`, `space`, `trace`, `autodiff`,
  `context`, `numerics`, `optim`, `losses`, `dists`, `stats`, `metrics`, `kernels`, `data`, `exposition`, `scene`,
  about 150,000 of 622,195 engine source lines) before the constructed domains; render later, as one cluster. The
  engine has 67 domains and 159,109 lines of tests.
- **09:51 Finding.** Exact duplicate function bodies are few (16 of 5+ lines); near duplicates at 80% found none inside the
  core. The real repetition is of patterns: a coercion to rows or columns written in about 25 files, the option-table
  helper `option()` copied three times, `dot` identical in six files.
- **09:58 Finding.** Of the 83 definitions of the base numeric shapes (`[n, n]`, `[n, n, n]`, `[n, n, n, n]`, `number[]`,
  `number[][]`) none was an alias of another name: every one restated the shape.
- **10:00 Decision.** Specialised names (`Rgba`, `Complex`, `Point2`, `Quat`) are fine but must specialise one underlying
  type: an alias, or a labelled tuple of the same arity when the element names matter. Same-concept synonyms
  (`type Pt = Point2`) are not wanted: use the global type. Enforced by `scripts/type-definitions.ts` in `make lint`.
- **10:00 Decision.** Plain `number[]`, `number[][]` and `Float64Array` belong only in the primitives (kernels, `blas.ts`,
  private `_` modules); public numeric code takes `TensorLike` and returns `Tensor` or `Value`. So no canonical `Vector` or
  `Matrix` type in `base` (it would bless native arrays); the variable-length `Vec`/`Matrix`/`Rows` definitions are left
  alone as migration candidates (`reports/tensor-migration-candidates.txt`, 52 of them).
- **10:01 Decision.** Fixed-size geometric tuples stay tuples (not tensor-backed). Reason: hot loops (ray tracing,
  rasterising) and data that is serialised or drawn (scenes, palettes, JSON tapes). Differentiable geometry is a tensor
  whose last axis has length 3 or 4 (as in the splat and NeRF code). Not benchmarked: a million 3-vector operations as
  tuples against tensor ops would settle the cost.
- **10:02 Gotcha.** `grad` rejects a raw tuple input ("pytree leaves must be tensors or traced values, got number"). A
  tuple must be wrapped with `tensor(...)` at the boundary; a tensor of `Vec3`-shaped rows `[n, 3]` differentiates fine
  (`sum(square(x))` gives `2x` row by row). Earlier I said a tuple can be passed "wherever a tensor is expected"; that is
  true of `TensorLike` functions, not of `grad`. The error message now names the path of the offending leaf and says
  to wrap it (`b1c32597`).
- **10:03 Gotcha.** Coverage `--coverage.include` is relative to the project root: `src/lib/<domain>/**` works; a
  repository-relative `packages/engine/src/lib/...` matches nothing and gives an empty report (`{}`) after a full run.
- **10:03 Finding.** Coverage of the core: 93.9% of statements, 94.4% of functions, 73.5% of branches; 642 functions never
  run. No test reaches `exposition/subgroups.ts`. Counts the vitest tests, not the `@example` blocks.
- **10:04 Finding.** Native arrays appear in the signature of 310 of 2,344 exported core functions (13%), 1,257 (54%) mention
  a tensor type: the raw source counts (11,720 `number[]`, 10,940 `Float64Array`) are mostly inside bodies. An upper
  bound: `number[]` is also a shape, an axis list, indices.
- **10:17 Result.** Type unification merged (`44f5bd50`): `Vec2`/`Vec3`/`Vec4` in `base`; restatements of the three shapes
  from 44 to 0; surface comparison against the baseline 3 added, 9 removed (all synonyms: `graphics#Point2`,
  `scene#Vec3`, `graphics#Rgba`, `fourier#ComplexNumber`, `numerics/geometry#Point3`, `dynamics/celestial#Vec3`,
  `render/react#Point2`, `render/charts#RiemannRule`, `render/charts#XY`), 28 changed (converted or relabelled
  declarations and signatures that name a replaced alias). `Rgba` is defined once with a doc covering its two scales
  (parsed paint 0 to 255 with alpha 0 to 1; texture samplers linear 0 to 1).
- **10:17 Noted by the unification agent.** Same name, different meaning (not changed): `ComplexNumber` is a tuple in
  `fourier` (removed) and a `{re, im}` interface in `numerics/differentiate`; `Matrix` is `number[][]` in four places, readonly
  rows in four more and a record or interface in three (`metrics/metric.ts`, `optim/sparse.ts`, `optim/dictionary.ts`);
  `Mat` is a `Float64Array` in two files and an interface in `causal/instrumental.ts`; `Vec` is `number[]` in `agents` and
  `Float64Array` in ten other files; `Box` is a 4-tuple whose meaning depends on `BoxFormat`.
- **10:17 Process.** At most two agents at a time; read-only reviewers work in the main checkout, no worktree.
- **10:50 Result.** Foundations read in full (37 findings, `findings/foundations.md`). The core is sound: per-op dispatch on
  tiny tensors 0.24 µs, `grad` of a small MLP 3x its forward pass, Philox passes its known-answer tests, no clock, `Math.random`
  or `console` in scope. Not sound: degenerate inputs (the live twins of the flaky tests: `matmul` skips zero entries so `0 * Infinity`
  is 1; `tril`/`triu` turn `Infinity` into NaN; `all([])` is false; `cumsum` of a bool wraps at 256; `pow(1, Infinity)` is NaN;
  gradients of `prod`, `norm`, `std`, `pow(x, 0)` are NaN at zero; `scatterAdd` out of range corrupts a neighbouring lane).
  Accidental-quadratic paths: `scheduler.fired` (20k firings 1.9 s, 40k 6.9 s), `gammaVariates`, `einsum`. The three Algorithm
  drivers disagree on the end status. The layering finding: the guessed order was wrong for `context` and `scene`, not the code;
  the real debt is file-level cycles inside `tensor` that `make layers` cannot see. Native arrays: 56 public functions in scope,
  six real findings, all in `random` (`dirichlet`, `categorical`, `multinomial`, `aliasTable`, `choice` weights, `randomSplit`).
- **11:15 Scope.** The maintainer asked whether render is covered. It was only in the plan as the reading P1.6; the
  surface digest and the type scan already covered it, the other scans had run on the engine only. The scripts take
  `--package=render` now and render's Phase 0 is done (P0.7). Render is clean on duplicated helpers and constants; its gap
  is tests of the React layer (57.9% of statements).
- **11:25 Result.** Numerics read (29 findings, `findings/numerics.md`; about 50 of 112 files, flag-led). Most valuable: scale and
  degenerate inputs give silent wrong answers (`svd` wrong above about 1e77 and below 1e-155; `bisect` reports a tiny-scale or NaN function as
  converged; `adaptiveSimpson` on NaN would take about 4.5e15 evaluations; `erf(Infinity)` and `ndtri(5e-324)` are NaN); `lstsq`
  and `polyfit` on a rank-deficient system return garbage of size 2e15 instead of the minimum-norm solution (20 callers outside
  numerics); `delaunay` of 8 or more cocircular points returns overlapping triangles. No single failure convention (throw,
  `converged: false`, a flag, or nothing; `fixedPoint` and `fsolve` have the same return type and opposite behaviour). The
  approximate-nearest-neighbour code in `spatial` (about 1,850 lines) looks like a retrieval domain, filed as a placement
  question. The reader ran no Python: every quoted scipy or numpy value needs the oracle.
