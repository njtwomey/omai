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
- **10:58 Result (P3.3, `agent-dd01-helpers`, `f1e64336`..`e651ebc0`).** Duplicated helpers replaced by shared ones, one
  commit per family; behaviour preserved, every touched domain's test directory run (1,098 sibling tests, then 8,200 more
  by domain, all green; 556 `@example`s of the touched files pass). Replaced: `dot` x8 by `ddot` (blas), `matmul` of the
  Kolmogorov tables by `stats/_linear`, two local `transpose`s by the tensor op, `nextAfter` x2 by `numerics/approximation`,
  `neighbourSets`, `backcastOf` and `studentConstant` (new `timeseries/_volatility.ts`), `stridesOf`/`strides` by the tensor
  one, `complete` x2 by `completeOption` (moved to `space/tables.ts` beside `OptionTable`; `learn` still re-exports it),
  `poly`, `planeOf` (`signal/image/_plane.ts`), `toLive`, `clipToBox`, `observedLogProb`, `thetaBox`, `isMatrixView` (now
  exported from `learn/validation.ts`), `fmix32` of text by the one of sketches, `roundHalfEven` x4 (new in `base/numbers.ts`),
  `logitOf` by `numerics/special` `logit`, `argmax` of the stream generator by `argmaxOf`, `norm` x2 by `dnrm2`.
  Constants: `EULER_GAMMA`, `LOG_2PI`, `LOG_SQRT_2PI` in `base/numbers.ts` (replacing 7 + 9 + 4 local definitions);
  `EPS = 2.220446049250313e-16` (15 files) is now `Number.EPSILON`, the same double; the libsvm `TAU` of the three SMO
  files is defined once in `optim/_smo-update.ts`.
- **10:58 Noted: not bit-identical.** `norm` to `dnrm2` (scaled accumulation) and `logitOf` to `logit`
  (`log p - log1p(-p)`) agree to the last ulp or so, not exactly; no test moved.
- **10:58 Noted: surface.** `--compare` gives 6 added (`base#EULER_GAMMA`, `LOG_2PI`, `LOG_SQRT_2PI`, `roundHalfEven`,
  `learn#isMatrixView`, `space#completeOption`) and 1 removed (`learn#completeOption`): the extractor lists a re-export of a
  public declaration under its declaring module only, so `learn#completeOption` is gone from the digest although
  `import { completeOption } from 'mdml-engine/learn'` still works (an `@example` and a test use it). Digest regenerated.
- **10:58 Left, with the reason.** Same name, different meaning: `entropy` (nats) against `entropyOf` (bits) in
  `metrics/clustering.ts` and `timeseries/shapelets.ts`; `concrete` x5 (different error codes and messages, two convert to
  float64); `columnOf`/`column`/`columns` (different record types); `sampleWeightsOf` against `sampleWeights` (one checks
  sign and finiteness); `distance`, `defaultNames`, `asColumns` (different signatures); `softplus` (the nn one has beta and a
  threshold); `solverConfig` of `learn/density-ratio.ts` against `solverConfigOf` (the latter also fills the `gradient`
  mode's optimiser and schedule, which would change configurations); `clipTo` of `stochastic/diffusion-samplers.ts`
  against the tensor `clip` (it broadcasts the bounds through `x * 0`, so NaN for infinite `x`); `mean`/`variance` against
  `meanOf`/`varianceOf` (pairwise summation, NaN for empty); `klBernoulli` (clamps 1e-12 against 1e-15); `isZero` x3 (tolerances
  and normal forms differ); `startingLogLambda`, `tableOf` x2 (differ in a clamp or an option); `isMatrixView` of
  `linear/bayesian.ts` (a looser structural test); `LOG_SQRT_2PI` of `numerics/special/_kernels.ts` (a literal one ulp from
  the expression); `EPS = 1e-12` in `graph` (two meanings), `tune/*`, `EPS = 1e-9` x3 (local tolerances).
  Needing a public home or a design: `cadd`/`cmul`/`cdiv` (`symbolic` keeps its own so it does not depend on `numerics`),
  `blank`/`basisRows`/`scatterColumn` (and the whole of `trajectoryDerivatives` against `stepDerivatives`, in `environments`
  and `dynamics/control`, which import each other's neither: `ControlModel` satisfies `DiscreteModel`, so one could call the
  other once `stepDerivatives` is exported), `allFinite` (`dynamics/ode`, `sampling`), `codec` (`dynamics/automata`, `graph`;
  the pattern is inline in dozens of places), `minorOf`, `toRows`/`rowsOf` coercions across domains, scalar `sigmoid` x4
  (`learn/calibration`, `embedding`, `logic`, `kernel-methods`), `knobsOf` and `option` (P3.4).
