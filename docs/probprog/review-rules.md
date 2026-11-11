# Design review of `lib/message-passing` (the EP / VMP / BP engine and its operators)

Written 2026-11-11. Read-only review of `packages/engine/src/lib/message-passing` at commit `0a1c56f9`: 16.8k lines of
source (66 registered operators in 32 files under `operators/`, 31 files at the top level) and 9.4k lines of tests. The
question from the owner: "it is a bit patched, did we design for this?". Short answer: the **spine is designed** (the
`Family` algebra, the JSON factor graph with ids, the engine as a serialisable `Algorithm`, the brute-force reference tests),
but **everything between the spine and the 66 operators grew by accretion**: the operator contract has no owner of its
conventions, so each package re-implemented the same five or six conventions inside its own operators, and the numerical
integration core exists in about seventeen variants. None of it is broken today (support and hooks agree for all 66
operators, there are no TODOs, no layering cycles), but every new operator costs a copy of the boilerplate and carries a
fresh set of magic constants.

Everything below was measured (grep counts, a registry-introspection script, two small experiments); the scratch scripts
are in the session scratchpad. File:line references are to the commit above.

## 0. Top findings, ranked by what they will cost future work

| # | Finding | Evidence | Cost |
|---|---------|----------|------|
| 1 | **The operator protocol is re-implemented inside operators.** Turning a numeric tilted moment into a message (project, subtract the cavity, divide by the power, widen a flat cavity, catch "improper" and return NaN evidence) is the engine's job (`engine.ts:334-338` does it for the `tilted` hook) but most of the 30 operators with a numeric EP rule (`support.ep = 'numeric'`) do it by hand. | `c.power === 1 ? site : site.map((v) => v / c.power)` copied 9 times (`functions.ts:341,345,353,361,524`, `log-scale.ts:76,91`, `likelihoods.ts:430`, `gate.ts:222,247`); `siteMessage`/`gaussianSiteMessage` called 16 times; the `logEvidence: try { tilt } catch (ImproperMessageError) { return NaN }` block 17 times (`concentration.ts:246,355`, `function-scale.ts:128`, `gamma-scale.ts:158,227`, `correlation.ts:188`, `covariance.ts:304`, `functions.ts:329,515`, `pairs.ts:329`, `gaussian.ts:303`, `gate.ts:282`, `linear.ts:370`, `plates.ts:146,182`, `scale.ts:303`) while `evidenceOf` (`engine.ts:556`) has no catch for the EP branch; four hand-written `isFlat` (`engine.ts:120`, `extremum.ts:47`, `functions.ts:113`, plus the `1e-12` test in `readable`, `engine.ts:271`) and two more private copies of the "show a flat cavity wide" policy (`_sides.ts:65`, `prior-sites.ts:139`) next to the engine's "one policy" (`types.ts:240-247`). The gate (`gate.ts:140-163`), `glmRow` (`rows.ts:90-125`) and the plates (`plates.ts:118-250`) each re-dispatch to inner operators by hand instead of calling `operatorMessage`. | Every new operator pays tens of lines of boilerplate and can get one convention wrong; fixing a convention means touching every file that has the copy (the counts in the evidence column). |
| 2 | **Numerical integration exists in ~17 variants (~1,650 lines)**, with near-identical panel construction, window finding, moment accumulation and evidence normalisation, and with magic constants that differ without reason (section 2). Eight copies of "window of a gamma cavity in log space" use spans 9, 12/8, 14, 15, 15, 18, 20 and clamps 40, 45, 60. Non-centred variance `E[x^2] - mean^2` on a zoomed grid is used in 12 places with four different floors (0, 1e-300, `FLOOR`, 1e-12) and in three places with none. | `_grid.ts` 272, `_gridn.ts` 169, `_quadrature.ts` 200, `_tilted.ts` 185, `numerical.ts:60-160`, `variational-numeric.ts:60-160`, `multiclass.ts:20-200`, `gaussian.ts:147-246`, `gamma-scale.ts:20-82`, `concentration.ts:100-190`, `scale.ts:20-190`, `function-scale.ts:50-100`, `priors.ts:165-262`; 19 hand-rolled log-sum-exp (`let top = Number.NEGATIVE_INFINITY`) although `logSumExpOf` is in `base`; Legendre rule cache written 5 times; Cholesky written 3 times (`_dense.ts` via tensors, `_bartlett.ts:18`, `multiclass.ts:64`); softplus 4 times, `logSigmoid` twice. | Accuracy is a per-operator accident; an improvement (a better window, a centred variance) has to be made 17 times; reviewing correctness means reading 17 grids. |
| 3 | **Parameter errors are reported as numerical failures.** `ImproperMessageError extends OmaiError('value')` (`args.ts:12`) and the engine treats every `OmaiError` with code `value` as "improper message" (`engine.ts:341`, `:573`). `numberParam`/`vectorParam` (`args.ts:105-117`), 36 other `new OmaiError('value', ...)` in operators, and the families (`families-matrix.ts` 7, `families-vector.ts` 11) throw the same code. 31 of 66 operators declare no `check`, so a bad parameter is first met inside the run. | Experiment: a graph with `times-constant` and no `a` compiles and runs to status `failed` with `improper: {events: 28, skipped: 28}`; a `gaussian` factor given `variance: -2` (the operator wants `precision`) runs to status **`converged`** with the reason "converged after 1 damped restart, but 1 message update could not be computed even then (an improper cavity...)". `probit` accepts `noise: -1` silently (`truncation.ts:217`). | A user's typo is diagnosed as "improper cavity", and a model can be reported converged with an update that never ran. This is the most dangerous finding in the folder. |
| 4 | **The engine is one 1,069-line module with seven concerns and six overlapping mechanisms for "no usable message yet".** Concerns: argument/cavity construction (`argsOf`, `rawCavity`, `pulledCavity`, `readable`), dispatch (`operatorMessage`), the alpha fixed point, the repair policy (`resolve`), evidence (`evidenceOf`, `evidenceDefects`), verdict and restart (`mpVerdict`, `restartDamping`, `restartDiffers`, `stalled`), and the `Algorithm` with its tape (`messagePassingAlgorithm`, 300 lines, four closures over shared state). Mechanisms for improper/flat input: `FactorOperator.readsTarget` (blocked), `needsDensity` (widen), `FactorSpec.start` (seed messages), `EngineSettings.initial` (a second way to seed), `ImproperMessageError -> flat | failed`, the `improper` policy (`skip/damp/reset/error`), and `restarts` with raised damping. | See section 3. The improper policy block is copied a third time in `stochastic.ts:330-350`; `damping must be in [0, 1)` is validated four times (`engine.ts:762`, `stochastic.ts:297`, `regions.ts:611`, `consistent.ts` through the engine). | Each new numerical failure mode (and the EP work is mostly that) is patched into one of the six places; the interactions are not stated or tested together. |
| 5 | **`FactorOperator` has accreted: 16 members, several used by one or two operators.** `unsupported` (1 operator: `likelihood-pair`), `tilted` (2 registry operators, plus the `numericOperator` factory), `bp` (4 operators of 66, so the third "projection" is a stub; `regions.ts` and `pgm` carry the real discrete BP), `noRule` (5), `needsDensity` (4), `ctx.rule` (read by `numeric.ts:64` only; passed through by `gate.ts:94`), `support` (66 declarations, read by `gate.ts:106` and `rows.ts`/`plates.ts` only, otherwise documentation), `defines` (14, VMP only). `readsTarget` and `power` are each `boolean | function`, `ports` is `array | function` (9 operators). | Section 1. | The contract cannot be learned from the type; the same fact is declared twice (`support` vs which hooks exist); the capability check reads hooks and `unsupported`, not `support`. |
| 6 | **Layout is by package of origin, not by concern, and the layering is inverted in places.** The top level mixes the core (engine, families, graph, types), eight algorithm variants (`alpha`, `amp`, `anytime`, `clutter`, `consistent`, `power`, `regions`, `stochastic`, `numerical`: about 3,000 lines of which 2,300 have no consumer outside their own tests), and numerics (`_grid`, `_gridn`, `_quadrature`, `_tilted`, `_scalar`, `_dense`, `_bartlett`). `operators/` is 32 files named after the work package that created them (`gamma-scale`, `function-scale`, `log-scale`, `scale`, `precision-scale` tests; `likelihoods`, `pairs`, `robust`, `prior-sites`, `rows`, `plates`). Inversions: `engine.ts:14` and `_tilted.ts:5` import from `operators/args.ts`; `operators/numeric.ts`, `operators/likelihoods.ts` import `../numerical.ts`, a file that also holds `numericEpAlgorithm` and the Laplace/sampling rules (so an operator depends on an algorithm file). No cycles (checked), but only by luck. | Section 4. | Finding the right place, and the right helper, takes a grep; helpers get copied because the real one is in the wrong layer. |
| 7 | **Functional duplicates among operators.** `gaussian-log-scale` (`gaussian.ts:360`, fixed `loc`, integrated with `tiltedLogGrid`) is `gaussian-scale` (`log-scale.ts:59`, random mean, integrated with `tiltedByQuadrature`) with a constant mean. `likelihood-pair` has a `beta-shapes` entry (`pairs.ts:163`) next to the `beta-two-shapes` operator (`concentration.ts:323`, check whether one is an observed-value special case of the other). `negative-binomial-dispersion` is a `numericOperator` with a fixed 24-point Gauss-Hermite rule (`likelihoods.ts:510-515`) while every other two-variable likelihood goes through the panelled grid of `pairs.ts`. `mixture-site`, `gaussian-mixture` and `select` overlap. `rows.ts` `glmRow` copies its rank-one lift twice (`rows.ts:94` and `:113`) and `plates.ts` a third time (`plates.ts:78`). | Section 4. | Two integrators, two sets of constants and two bug surfaces for one density. |
| 8 | **Hidden state and unsafe memoisation.** `gaussian.ts:177` `tiltCache` (module-global Map, 64 entries, string key of the numbers, FIFO eviction) and `multiclass.ts:90` `last` (module-global, key `JSON.stringify(args, params, power)`: NaN and +-Infinity both stringify to `null`, `rule` is not in the key, a stringify per call). They are pure functions of their key, so values are correct today, but they exist only because the contract asks for one port at a time while the integral is shared by all ports (the gate comment says so: "its evidence, then the message to each variable"). | Section 5. | Cross-graph, cross-test state; any future non-pure input silently returns stale tilts. |
| 9 | **Documentation of the registry has drifted and is hand-maintained in three places.** `registry.ts` imports (50 lines), the `all` array, and a single 1,800-character sentence in the TSDoc; `docs/structure.md` says "46" (line 615), "48" (3919, 3922) operators, `ep-plan.md` says 35, the registry has 66. `Object.fromEntries` (`registry.ts:145`) silently overwrites a duplicate name. | Section 4. | Stale numbers mislead agents who start from the docs. |
| 10 | **Tests: strong brute-force references, weak structure.** `operators.test.ts` (839 lines, 8 `fc.assert` with run-time-random seeds, `numRuns: 12`; the author left an `OMAI_FC_SEED` escape hatch and a comment about a counterexample of seed 1395539355, `operators.test.ts:90`) carries the closed-form operators; 17 of 32 operator files have no test file of their own (`args`, `conjugate`, `discrete`, `extremum`, `gaussian`, `linear`, `links`, `numeric`, `plates`, `registry`, `rows`, `scale`, `spin`, `truncation`, `variational-numeric`, `_sides`); 989 `as number` casts from `noUncheckedIndexedAccess` in the source. | Section 5. | A refactor has weak local nets for the oldest operators; flaky seeds will eventually fail a merge gate. |

What is fine (keep it): `defineFamily` (`families.ts:55`) is a real abstraction (product, quotient, power, KL, entropy
derived from five pieces) and `Family` is a coherent interface; the graph as JSON with the id convention (`types.ts` header,
`graph.ts`); the engine's state is plain numbers, serialisable and resumable (`MpState`), and the schedule slot
(`schedules.ts`, 116 lines) and the projection slot (`projections.ts`) are clean; `glmRow`, `glmPlate`, `likelihoodSite`,
`unary`, `numericOperator` are the right idea (combinators over operators); `support` agrees with the hooks for all 66
operators today (checked by script) and `evidence.test.ts:362` enforces that an EP operator has a `logEvidence`; zero
`TODO`/`FIXME`/history comments in 16.8k lines (the only history-flavoured code is the back-compat in `deserialise`,
`engine.ts:1052-1062`, and the optional `MpSnapshot.unresolved`); the brute-force oracles in `src/testing/message-passing.ts`
(`bruteTilted`, `integrateBox`) are exactly what the refactor needs; file-level imports are acyclic.

---

## 1. The `FactorOperator` contract (`types.ts:216-326`)

### 1.1 What it is today

```
FactorOperator { name, description, ports | (params)=>ports,
                 support: {ep?, vmp?, bp?}: exact | bound | numeric,
                 readsTarget?: bool | (to)=>bool,  needsDensity?, power?: bool | (params)=>bool,
                 defines?: port,  noRule?: {mode: reason},  unsupported?(mode, params),  check?(params, families),
                 ep? vmp? bp? tilted?  : (ctx) => message | tilted moments,
                 logEvidence?(ctx), expectedLog?(ctx) }
```

Usage across the 66 registered operators (script over `defaultOperators()`):

| member | operators | comment |
|--------|-----------|---------|
| `name`, `description`, `ports`, `support` | 66 | uniform. `ports` is a function for 9. |
| `ep` / `vmp` | 63 / 59 | the two real rule hooks |
| `bp` | **4** (`prior`, `equal`, `table`, `mvn-normal-wishart`) | engine `bp` projection fails with `not-implemented` on the other 62 |
| `tilted` | **2** (`negative-binomial-dispersion`, `likelihood-pair`) and the `numericOperator()` factory | the engine's own "project the moments" path serves 2 operators; about 20 others hand-roll it in `ep` (16 `siteMessage` calls) |
| `logEvidence` | 65 (all but the VMP-only `gaussian-mixture`) | |
| `expectedLog` | 58 | 8 have none (`equal`, `greater-than`, `is-between`, `spin`, `spin-row`, `apply-product`, `gaussian-mv-function-scale`, `mvn-sd-corr`) |
| `readsTarget` | 41 | `bool | function` |
| `power` | 19 | `bool | function`; the division by the power is done in the operator (finding 1) |
| `defines` | 14 | VMP only (`engine.ts:150-163`, `refreshDefined`) |
| `check` | 35 (**31 missing**, including `gaussian`, `product`, `sum`, `probit`, `logistic`, `poisson`, `binomial`) | |
| `needsDensity` | 4 (`gaussian-gamma-precision`, `mixture-site`, `prior-site`, `gate`) | |
| `noRule` | 5 | `unsupported`: 1 (`likelihood-pair`); both only consumed by `operatorsWithout` in `probprog/model/factors.ts:5411,5718` |
| `OperatorContext.rule` | 1 reader (`numeric.ts:64`) and one pass-through (`gate.ts:94`) | a slot of the engine (`Projection.numeric`, `numerical.ts`) for one operator factory |

### 1.2 Judgement

- **Coherent core, incoherent periphery.** `ep`/`vmp`/`logEvidence`/`expectedLog`/`ports`/`check` are a clean contract.
  `readsTarget`, `needsDensity`, `power`, `defines`, `noRule`, `unsupported`, `support`, `rule` are patches: each answers a
  question the engine (or the compiler) needed once. They are optional, mostly `false`, and have `boolean | function`
  overloads, so there is no uniform declaration and no place that checks it. `takesPower` (`types.ts:322`) exists
  to paper over the overload.
- **The unit of work is wrong for numeric operators.** EP's natural unit is one tilted distribution per factor, from which
  every port's message follows; the contract is "one message to port `to`". Operators that integrate (30 with
  `support.ep = 'numeric'`) either recompute the integral for each port (the common case; with `evidenceOf` that is one more
  integral per factor per observation) or memoise by hand (finding 8).
- **`support` is documentation with one accidental consumer.** `gate.ts:106` uses `support.ep === 'exact'` plus "does not read
  the target" to decide whether a branch can be raised to a power, which is a different question from "is the rule exact".
- **`bp` is a stub in the engine.** The registry has four `bp` rules. A Gaussian tree under the `sum-product` projection
  throws. The engine claims "EP, VMP and BP as settings of one engine" (`engine.ts:2`, `structure.md`), but the BP part
  lives in `regions.ts` (794 lines) and `pgm/`. Either make `bp` derive from `ep` for the exact operators (the exact EP message on a tree is the BP message) or
  drop the claim.

### 1.3 Proposed contract

One declaration shape, enforced by a `defineOperator()` constructor that throws at construction time (so the registry test
becomes the constructor):

```ts
defineOperator({
  name, description,
  ports,                              // static array, or (params) => array
  params?: ParamSpec,                 // { a: vector(len=port 'w'), noise: positive(default 1), ... }: parsed once by compileGraph;
                                      //   replaces numberParam/vectorParam throws and most `check`s (finding 3)
  check?: (params, families) => string | undefined,    // structural checks only
  rules: {
    ep?:  site(tilt) | message(fn),   // site(tilt): tilt(c) -> { moments | {mean, variance}, logZ }; the ENGINE projects, divides by
                                      //   power, widens a flat cavity and catches Improper. message(fn): pushforwards and exact conjugate rules.
    vmp?: message(fn),
    bp?:  'from-ep' | message(fn),
  },
  evidence?: { ep?: (c) => number, vmp?: (c) => number },   // engine catches ImproperMessageError -> NaN, once
  traits?: { readsTarget?: (to) => boolean, needsDensity?: true, power?: 'site' | 'none', defines?: port },
  noRule?: { vmp: 'a hard step has no finite KL' | (params) => string | undefined },   // unsupported folded in
})
```

- `support` is derived (`site` => numeric, `message` => the operator says `exact` or `bound`), never declared twice.
- `power` becomes a property of the update kind: every `site(tilt)` rule supports a power for free (the engine divides), and
  `message(fn)` rules declare `power: 'none'`. The ten copies of `site.map(v => v / c.power)` disappear.
- `ctx.rule` is replaced by passing the integration profile (section 2) through the context; `numericOperator` becomes a
  `site(tilt)` over a generic log density.
- The registry (`defaultOperators()`) is built from per-folder `operators` arrays with a duplicate-name check; the 1,800-character
  prose list is generated (or removed and `docs/structure.md` links to the registry).

---

## 2. Duplication of the numerical integration core

### 2.1 The census

| Implementation | Where | Lines | Does what | Window / panels |
|----------------|-------|-------|-----------|-----------------|
| `integratePair` | `_grid.ts:105-215` | 272 file | 2-D panelled Gauss-Legendre, zoom, moments of both axes and `E exp(x)`; axes: Gaussian value, gamma log | window = cavity +-9 sd, gamma `centre -12 sd .. +8 sd`, clamp +-45; COARSE 8x5, FINE 10x6, 6 passes, zoom threshold `1e-9`, shrink if `< 0.7` |
| `integrateAxes` | `_gridn.ts:67-170` | 169 file | the same for 1 to 3 axes, plus beta-logit axis | window `centre +-10 sd`, clamp +-40; COARSE 4x4, FINE 6x4, 6 passes, zoom `1e-9`, `0.7`; **no kink handling, uniform panels** (the 2-D one has graded panels) |
| `tiltedByQuadrature` | `_tilted.ts:132-185` | 185 file | 1-D tilted moments with kink-split panels, up to 6 zooms, `fn` and `more` extra moments | window cavity +-9 sd narrowed by a `WindowHint` (`reach` default 120), PANELS 80, ORDER 8, zoom on `+-10 sdT`, stop at `0.6` |
| `gaussianNodes`, `tiltedGaussian` (Gauss-Hermite 48) | `_quadrature.ts:20-70` | 200 file | moments by Hermite | n = 48 (links 64) |
| `gaussianGrid`, `tiltedGrid` | `_quadrature.ts:80-130` | | panelled Legendre, density folded into weights | cavity +-9 sd, 24 x 10 |
| `gaussianGridOn`, `tiltedLogGrid` | `_quadrature.ts:140-200` | | the same on a given window, log weights | 12 x 8 default |
| `quadratureRule`, `gaussHermiteRule`, `unscentedRule`, `sampleRule`, `laplaceRule` | `numerical.ts:60-300` | ~250 of 551 | the `TiltedRule` slot: product rules over up to 3 arguments | 64 or 24 x 8 coarse; zoom to `mu +- 8 max(sd, 1e-3 spread)` |
| `quadraturePort(s)` + `gridWeights` | `variational-numeric.ts:42-160` | ~120 of 348 | VMP: nodes/weights per port: Hermite, composite Legendre for kinks, 161-point grid in log (gamma) or logit (beta) | gamma: `mode - min(max(9 sd, 20/shape), 45) .. mode + min(9 sd, 8)`; beta similarly |
| multiclass integrator | `multiclass.ts:90-200` | ~110 of 274 | r-dimensional Hermite product with 3 rounds of Gaussian adaptation, memoised in a global | NODES 48/24/8/7/6/5 by dimension |
| `precisionTiltGrid` | `gaussian.ts:190-246` | 57 | uniform grid in `log lambda` | `TILT_SPAN 20`, `TILT_POINTS 801`, clamp 60 |
| `pairTilt` | `gamma-scale.ts:20-82` | 65 | same, two gamma variables | `SPAN 18`, `POINTS 241`, clamp 60 |
| dirichlet/beta concentration tilts | `concentration.ts:100-190` (and the 2nd at 270-300) | ~110 | same | `SPAN 14`, `POINTS 241`, clamp 40 |
| `nodeAt` + `summarise` + coarse/fine search | `scale.ts:40-190` | ~150 | same for a vector with a dense conditional | `half 15`, `COARSE_POINTS 81`, `FINE_POINTS 121`, clamp 60 |
| function-scale grid | `function-scale.ts:50-100` | ~50 | same | `half 15` / 12, clamp 60 |
| `massWindow` + `densityMoments` | `priors.ts:165-262` | ~100 | window finder by scanning 801 points and cutting at `top - 40`, then a 6001-point trapezoid | a different (and better: automatic) window algorithm |
| `truncation.ts:35-60` mixture accumulations, `clutter.ts:158` grid | | ~40 | | |

About **1,650 lines in 17 places**, of which three (`_grid`, `_gridn`, `_tilted`) are the same algorithm (panel, evaluate
log weights, shift by the maximum, accumulate moments, find where the mass is, shrink the window, repeat) written three times with
different constants and different feature sets (kinks in two of them, extra functions in one, three dimensions in one).

### 2.2 Near-duplicate code, side by side

Window finding (identical logic, `_grid.ts:196-210` vs `_gridn.ts:147-160`):

```ts
while (first < g.length && (marginal[first] as number) < 1e-9 * top) first++
while (last > first && (marginal[last] as number) < 1e-9 * top) last--
const spacing = (hi - lo) / g.length
const nlo = Math.max(a.lo, (g[Math.max(first - 1, 0)] as number) - spacing)
```

Gamma window (eight copies: `_grid.ts:243-249`, `concentration.ts:122-127`, `gaussian.ts:201-204`, `function-scale.ts:55-61`,
`gamma-scale.ts:47-50`, `scale.ts:201-210`, `variational-numeric.ts:122-125` (a different centre: the mode), and `_gridn.ts:28-33` for beta):

```ts
const centre = digammaOf(shape) - Math.log(rate)
const half = SPAN * Math.sqrt(trigammaOf(shape))      // SPAN = 12/8, 14, 15, 15, 18, 20 depending on the file
const lo = Math.max(centre - half, -60)               // clamp = 45, 40, 60 depending on the file
```

Moment accumulation (`_tilted.ts:86-117`, `_quadrature.ts:56-70`, `:117-130`, `:187-199`, `gaussian.ts:225-246`, `scale.ts:104-135`,
`gamma-scale.ts:60-82`, `concentration.ts:130-175`, `truncation.ts:50-57`, `numerical.ts:40-58`): shift by the maximum
log weight, sum `exp`, sum `w x`, then a second pass for the centred moment (the stable ones) or `E x^2 - mean^2` (the others).

Non-centred variance on a grid, which cancels when the tilted distribution is narrow relative to its mean (the case the zoom is
designed to produce): `gaussian.ts:241-242` (no floor), `gaussian.ts:536` (no floor), `truncation.ts:270` (no floor), `functions.ts:191-192` (floor `1e-300`),
`robust.ts:68`, `truncation.ts:57`, `variational-numeric.ts:337` (`1e-300`), `log-scale.ts:90` (`1e-300`), `function-scale.ts:123`
(`1e-300`), `extremum.ts:44`, `functions.ts:91,103`, `scale.ts:131` (floor 0), `spin.ts:47` (`1e-12`). A precision floor of `1/1e-300` is a
point mass presented to the next update as a legitimate message (`gaussianSiteMessage` only rejects non-finite or non-positive).

Rule caches (`_grid.ts:36`, `_gridn.ts:12`, `_quadrature.ts:6,68` (twice), `_bartlett.ts:69`): five Maps, four of them
`Map<number, {x: Float64Array, w: Float64Array}>` for the same Gauss-Legendre nodes.

Per-likelihood hand tuning of the window: `likelihoods.ts` (`ordinalLocate`, `locate`), `prior-sites.ts:60-110` (`reach: 400`, `reach: 40, zoom: false`,
`reach: 12`, ...), `WindowHint` in `_tilted.ts:10-22`. `priors.ts:massWindow` shows the hint is unnecessary: an automatic scan
of the log density finds the window.

### 2.3 The single shared integration layer (`message-passing/numerics/`)

One module, one algorithm, profiles for the constants; operators describe the integrand and the coordinates, never the grid.

```
numerics/
  rules.ts      gaussLegendre / gaussHermite / gaussLaguerre node caches (one Map per rule), nothing else
  axis.ts       Axis = { point?, lo, hi, logWeight(x), kinks[], coordinate }; constructors, each with its window and clamp in ONE place:
                  gaussianAxis(mean, var), gammaLogAxis(shape, rate), betaLogitAxis(a, b), pointAxis(x), (later) lognormal, dirichletLogit
  window.ts     findWindow(logDensity, lo, hi, start): the priors.ts massWindow algorithm (scan, cut at top-40, narrow onto a spike);
                  zoom(marginals, window): the 1e-9 / 0.7 rule, once
  integrate.ts  integrate(axes, logf, profile?) for 1..3 axes: panels graded to kinks, log-weight shift, centred moments of every axis,
                  optional extra statistics (E exp x, fn(x), conditional moments): returns { logMass, nodes, weights, marginals, moment(axis, fn) }
  tilt1d.ts     tilt(cavity: Gaussian | Axis, logf, { hint?, fn?, more? }) = integrate([axis], ...) : the `_tilted.ts` API on top of integrate()
  moments.ts    centredMoments(xs, logw) with the stable second pass; `varianceOf` returns {variance, floor flag} so a collapse is reported, not floored to 1e-300
  profiles.ts   named profiles: 'fast' (links), 'default', 'precise'; the per-dimension node budgets that today are NODES / COARSE / FINE / POINTS constants
  hermite.ts    the Gauss-Hermite product rules (`gaussHermiteRule`, `tiltedGaussian`) over axes (Gaussian axes only)
  quad-ports.ts the VMP side: `quadraturePorts(args)` built on the same Axis objects (nodes + weights + sufficient statistics)
  dense.ts      one small-dense SPD kernel (`spd`, `choleskyInPlace`, solves) over `numerics/linalg`; _dense.ts + _bartlett.ts' Cholesky + multiclass `chol`
  wishart-rule.ts   the Bartlett rule (`_bartlett.ts` 80-197) on top of rules.ts
```

How each operator uses it:

- `gaussian-gamma-precision`, `gamma-rate`, `gamma-product`, `dirichlet-concentration`, `beta-two-shapes`,
  `gaussian-mv-precision-scale`, `gaussian-mv-function-scale`: one `gammaLogAxis` (or `gaussianAxis` for the `log: true` form), an integrand
  returning the log weight, and `integrate(...).moment(axis, fn)` for `E[log lambda]`, `E[lambda]`, the conditional moments. The `nodeAt` of `scale.ts`
  stays as the integrand (it is the genuinely operator-specific part); `summarise`, the window, the clamp, the `step` and the
  accumulators go.
- `gaussian-scale`/`gaussian-log-scale`, `likelihood-site`, `student-t-site`, `laplace-site`, `apply`, `prior-site`, `probit`/`logistic` under a power: `tilt1d`.
  `logistic` stays on Gauss-Hermite only for the `fast` profile.
- `likelihood-pair`, `negative-binomial-dispersion` (so the 24-point rule goes), `beta-shapes`, `mvn-sd-corr`: `integrate` with 2 or 3 axes.
- `multiclass-site`: the Gaussian-adaptation rounds stay (they are a proposal, not a grid); the node tables and `chol` go to `rules`/`dense`.
- `numerical.ts` rules: `quadratureRule` becomes `integrate` over the problem's bases; `gaussHermiteRule` is `hermite.ts`; Laplace, sample, unscented stay (they are not grids).
- VMP: `quadraturePorts` is built from the same axes; `numericMessage`/`numericExpectedLog` stay (the natural-gradient algebra is VMP's own).
- Families and the model compiler (`priors.ts densityMoments`) use `findWindow` + `tilt1d`.

The families (`families*.ts`) do not need this treatment: `defineFamily` already supplies the shared algebra, and the three files
are cohesive (scalar, vector/discrete/Dirichlet, matrix). The messages' algebra is the part scattered across `args.ts`
(`gaussianMessage`, `gaussianSiteMessage`, `siteMessage`, `rawSum`, `logGaussianIntegral`), `_sides.ts`, `plates.ts:78`
(`sitesMessage`), `rows.ts` (the rank-one lift, twice) and 20 local `const arg = (c, i) => c.args[i] as Arg` definitions
(`operators/*.ts`: conjugate, correlation, concentration, gaussian, ...). Proposal: one `messages.ts` in the operator kernel
(`arg(c, i)`, `site(...)`, `rankOne(a, nu, lambda)`, `wide(arg)`); the 20 local `arg`s go.

### 2.4 VMP (`variational-numeric.ts`) vs EP

The two sides share nothing but `ImproperMessageError`. They apply the same coordinate transforms (log for gamma, logit for beta)
with different windows (VMP: centre at the mode `log(shape/rate)`, left `min(max(9 sd, 20/shape), 45)`, right `min(9 sd, 8)`; EP: centre at `digamma(shape) - log(rate)`,
`-12 sd .. +8 sd`). `numericMessage` is hard-wired to two sufficient statistics (`stats: [number, number][]`,
`variational-numeric.ts:55-135`), so a Dirichlet or Wishart port cannot take a numeric VMP message. After section 2.3 the
VMP quadrature port is an `Axis` plus its statistics vector (length-free), and the gamma/beta/Gaussian windows are the same objects.

---

## 3. The engine: separable policies or one tangle?

### 3.1 Where each concern lives (`engine.ts`)

| Concern | Lines | Verdict |
|---------|-------|---------|
| argument and cavity construction: `argsOf` (`:178`), `rawCavity` (`:221`), `rawBelief`, `beliefOf`, `pulledCavity`, `readable` (`:262`) | 120-290 | a clean unit; needs its own module (`cavity.ts`) and the "flat cavity" policy should live here only (4 `isFlat` + 2 private widenings elsewhere) |
| dispatch `operatorMessage` (`:299-346`) | | a clean unit, but gate/rows/plates do not call it |
| alpha fixed point (`:351-392`, `ALPHA_ITERATIONS = 200`, `1e-12`) | | separable; alpha is already a `Projection` field |
| repair `resolve` (`:420-485`) | | a clean unit; copied in `stochastic.ts:330-350`; damping is applied in it (`family.power(old, d)`), so "damping" and "improper repair" are one function with two policies |
| evidence `evidenceOf`/`evidenceDefects` (`:526-640`) | | the EP and VMP branches share nothing; the EP branch builds a context three times (`:551,:617`) with the same four fields |
| restart and verdict `restartDamping`, `restartDiffers`, `mpVerdict`, `stalled` | 650-760 | **the verdict has five exits and three behaviours that depend on `restarts` and `damping`** (`replays`); it takes `Partial<Pick<EngineSettings, 'damping' | 'restarts'>>`, so a caller that omits them (`verdict.test.ts:151`, any new caller) gets a different verdict from the engine's |
| the `Algorithm` (`messagePassingAlgorithm`, `:761-1069`) | 300 | residual, flooding and serial are interleaved in `advance`; `state.damping` shadows `settings.damping` by rebuilding the settings object each step (`:906`); `observe` computes the evidence (an integral per factor), the verdict (also computed by `done`, so twice) and all marginals at every step, including `granularity: 'message'` |

### 3.2 Knobs: orthogonality and undocumented interactions

- `damping` (natural-parameter interpolation, `resolve`) x `improper: 'damp'` (halving up to 10 times, magic `10`): two dampings in sequence; when the 10 halvings
  fail the update is silently skipped and counted under `skipped`, not `damped`.
- `restarts` raises `state.damping` (0.5, then +0.3, cap 0.9), serialised in the state, so `settings.damping` and the damping in force differ; `restartDiffers`
  suppresses a restart that would repeat the run; the verdict then says so in prose. Orthogonal in intent, entangled in code (`stalled`, `advance`, `mpVerdict` all read both).
- `alpha` (projection) x `power` (per factor): the alpha update runs only if `takesPower(op) && f.power === 1` (`engine.ts:407`); a factor with its own power silently keeps it, and `evidenceOf` returns NaN for any alpha != 1 (`:536`). Not stated in `alphaProjection`'s contract.
- `initial` (settings) vs `FactorSpec.start` (graph): two ways to seed messages (`init`, `:793-806`), applied in sequence.
- `needsDensity` x `power`: the widened target adds `FLAT_CAVITY_PRECISION / power` (`engine.ts:331`), correct but unexplained and tested only through the atoms.
- `tol` is one number for two incommensurable measures: `family.change` falls back to `maxAbsDifference` of the natural parameters whenever
  either message is improper (`families.ts:84-86`), so the convergence measure jumps units at the moment of interest.
- `granularity: 'message'` with `schedule: 'flooding'` or `'residual'` is silently coarsened (`schedules.ts:106`).
- `EngineSettings.order` only means something for `schedule: 'custom'`; `restarts` is 1 for EP and 0 for VMP/BP by default (`named.ts:90-112`) with no stated reason; the `improper` default is `damp` for EP and `skip` for VMP/BP.
- Dead or near-dead: `Projection.numeric` (one reader), `FactorOperator.support` (documentation), `MpSnapshot.unresolved?` (optional only for old states), the `deserialise` defaults (`engine.ts:1052-1062`, "a state saved before the run counted..."), `Support: 'bound'` (4 operators, never read).

### 3.3 Proposed separation (all in `engine/`)

`cavity.ts` (arguments, widening, the one flat-cavity policy), `update.ts` (`propose` = dispatch + alpha, `repair` = damping + improper policy as a
function returning `{eta, event}`; shared by `stochastic.ts`), `evidence.ts`, `verdict.ts` (pure function of `(graph, state, settings)`, settings required, not `Partial`), `restart.ts`, `schedule.ts` (the
plan plus the three advance strategies as functions `advanceSweep`, `advanceFlooding`, `advanceResidual`), `run.ts` (the `Algorithm`, the tape, `observe` that computes the
evidence only when the tape asks for it: a `evidenceEvery` knob or per sweep).

---

## 4. The registry and the folder layout

How operators are found: `FactorSpec.operator` is a string; `compileGraph(spec, registry)` looks it up in a `Record<string, FactorOperator>`
(`graph.ts`); `defaultOperators()` (`registry.ts:93`) builds the record from a hand-written array of 66 imports and then adds the gate, which needs the
registry (`registry.ts:145-148`: a post-construction mutation). `likelihoods()` and the pair tables are separate registries inside operators
(`likelihoods.ts:330`, `pairs.ts:196`, `functions.ts:56`), each with its own `name`/`check`/`density` shape (three more declaration shapes beside `FactorOperator`).
Fine as a mechanism; the problems are listed in findings 5, 7, 9.

Naming: the same stem names unrelated things and related things have unrelated stems.
`gaussian-*` (10 operators: `-gamma-precision`, `-log-scale`, `-normal-gamma`, `-joint`, `-mixture`, `-scale`, `-mv-precision-scale`, `-mv-function-scale`, `-transition`) vs `mvn-*` (3) for
the same Gaussian-vector idea; the suffix `-site` (a scalar factor), `-row` (the site on a weight vector), `-plate` (a block of rows) is a convention
but not applied uniformly (`linear-regression-row`/`-plate` are not lifts of a link; `mixture-weight` is not a mixture site; `likelihood-pair` is not a site);
`prior` vs `prior-site`. These names are public (the model compiler `probprog/model/factors.ts` and 92 atoms use them), so rename only with aliases.

### 4.1 Proposed layout

Organise by what the factor is (the concern), not by the package that added it; put the algorithm variants and the numerics in their own folders.

```
message-passing/
  index.ts                         (generated barrel)
  core/
    types.ts                       Family / FactorOperator (the new contract) / FactorGraphSpec / ids   (types.ts split in three files)
    errors.ts                      ImproperMessageError, ParameterError, FLAT_CAVITY_PRECISION          (out of operators/args.ts)
    families/ scalar.ts vector.ts matrix.ts define.ts   (families.ts, families-vector.ts, families-matrix.ts)
    graph.ts  priors.ts  truncated.ts
  numerics/                        the single integration layer of section 2.3 (+ scalar.ts, dense.ts, wishart-rule.ts)
  engine/                          cavity.ts update.ts evidence.ts verdict.ts restart.ts schedule.ts run.ts projections.ts schedules.ts
  operators/
    define.ts  registry.ts  kernel/ (args.ts messages.ts sides.ts)
    gaussian/    prior, gaussian, product, equal, linear (+plus, difference, sum, times-constant, dot/matmul-constant, joint), transition,
                 normal-gamma, gamma-precision, log-scale (one file with both operators' merged), extremum
    links/       probit, greater-than, is-between, logistic, spin, multiclass, truncation helpers
    likelihoods/ site (likelihood-site), pairs, robust, prior-sites, apply (functions.ts), negative-binomial-dispersion
    scales/      scale, function-scale, gamma-scale, concentration, stick-breaking
    covariance/  covariance (mvn-wishart), correlation (mvn-sd-corr), normal-wishart
    discrete/    conjugate (bernoulli/discrete/poisson/binomial/...), discrete (table/select/gaussian-mixture), gate
    lifts/       rows (glmRow), plates (glmPlate, linear-regression-plate)
  algorithms/  named (ep/vmp/bp), alpha, power, numerical (numeric EP + rules), stochastic, consistent, regions, amp, anytime, clutter
```

Each operator folder has an `index.ts` exporting `operators: FactorOperator[]` and its tests next to the files (split `operators.test.ts` along the same lines).

---

## 5. Correctness risks to flag

1. **Silent fallbacks (finding 3).** `engine.ts:341` and `:573` swallow every `OmaiError('value')`; 36 operator throws and 18 family throws share that code. Also `resolve`'s `damp` branch falls through to `skip` after 10 failed halvings; `mpVerdict` reports an update that never ran as "an improper cavity".
2. **Magic tolerances.** 57 numeric literals of the `1e-N` kind outside tests (`grep`), none named except `FLAT_CAVITY_PRECISION`: `engine.ts:271` flat test `1e-12` vs `isFlat` exact zero at `engine.ts:120`; `ALPHA` `1e-12`; zoom `1e-9`; precision floor `1e-300` in 9 places and `1e-12` in `spin.ts:47`;
   `correlation.ts:160` weight cut `1e-13`, `covariance.ts:176` `1e-14`, `variational-numeric.ts:215` `1e-14`; `plates.ts` inner loop `INNER_SWEEPS 100`, `INNER_TOL 1e-11` (an EP engine nested in an operator, with its own convergence rule); `numerical.ts:379-400` ridge/step tolerances; `_tilted.ts` `PANELS 80`, `ZOOMS 6`, `reach 120`, `0.6`.
3. **NaN handling.** Operators return NaN evidence and the engine sums with a `NaN` guard in some branches (`evidenceOf` VMP branch catches; EP branch does not catch, so a throwing `logEvidence` would escape `observe`); `evidenceDefects` calls `logEvidence` unguarded (`engine.ts:617`), which throws for the operators that raise instead of returning NaN.
4. **State and memoisation.** `gaussian.ts:177` and `multiclass.ts:90` (finding 8); `numerical.ts:229` `drawCache` Map keyed by `seed/pairs/d` is deterministic but unbounded. `MpState` itself does not leak: every step builds new arrays (`slice()`), `frozen` is reset at the end of a sweep, `restartOf` rebuilds from `init()` and carries only counters. One thing to watch: `advance` copies `messages` per step (`state.messages.slice()`), `O(edges)` per message at `granularity: 'message'`.
5. **Non-centred variance on grids** (section 2.2): cancellation when the tilted sd is below about 1e-8 of the mean, floors of different sizes, three call sites with no floor.
6. **Tests that depend on a seed.** `operators.test.ts` property tests run on random seeds (`fc.assert`, no fixed seed; the escape hatch `OMAI_FC_SEED` exists, the seed-1395539355 comment records one flake). The `sampleRule(n, seed)` tests (`numerical.test.ts`) are seeded and deterministic. Recommend a fixed default seed with `OMAI_FC_SEED=random` for stress.
7. **Evidence cost.** `observe` computes `evidenceOf` (one integral per numeric factor) and `marginalsOf` on every observation; at `granularity: 'message'` that is `O(E * F)` integrals per sweep. Not measured; plausible 10-100x slowdown for teaching-granularity runs of the numeric atoms.

Dead code and unused exports (`python` scan over 370 exports, word-match across `packages`, `apps`, `examples`, `scripts`): 58 have no reference outside their own file even in tests (most are interfaces
that are part of an API surface, e.g. `AmpState`, `EcModel`, `RegionPath`; the genuinely dead values are `_bartlett.gammaRule` and `normalRule`, `_scalar.erfcxOf` and `inverseDigamma`, `_grid.gaussianAxis`,
`families.scalarFamilies`, `families.maxAbsDifference` (only used inside `families.ts`), `operators/functions.applyProduct`, `operators/likelihoods.likelihoodOperators`, `families-matrix.logMultiGamma`/`multiDigamma`/`wishartShapeFromMoments` (only inside the file)). 39 more are referenced only by tests:
`alphaEpAlgorithm`, `powerEpAlgorithm`/`Method`, `ecAlgorithm`/`ecMethod`, `stochasticEpAlgorithm`/`Method`, `regionBp*`, `amp*`, `numericEpMethod`, `unscentedRule`, `sampleRule`: the whole algorithm zoo (about 2,300 lines) has no consumer but its tests, which is consistent with the project's
"everything is library" rule but means these files can be moved out of the core's way without a user noticing. `TODO`/`FIXME`/"added by" comments: none. The one history-shaped spot is the back-compat in `deserialise` (`engine.ts:1052`).

---

## 6. Target structure and the stepwise, behaviour-preserving plan

Target structure: section 4.1 (folders), section 1.3 (contract), section 2.3 (integration layer), section 3.3 (engine split).

### 6.0 How to prove "no change"

- **Baseline (step S0).** `verifyAtom(atom)` (`probprog/ep/ep-atoms.ts:737`) over `epAtoms` (92 atoms, all `ran`, 91 with evidence; ~68 s single-threaded with `nice -n 15`)
  gives `run.beliefs`, `run.evidence`, `run.sweeps`, `run.converged` for every atom. S0 writes a small script that dumps them at full precision to a JSON file (kept out of the repo or under `fixtures/`), and a comparer with
  two modes: **exact** (`Object.is` on every number, for the steps declared bit-identical) and **tolerant** (beliefs relative 1e-9, evidence relative 1e-12, plus identical sweep counts and improper counts).
  Also compare the improper/skipped/restart counters, so a changed failure path shows even when the beliefs agree.
- **Per-operator reference.** The existing brute-force tests (`operators.test.ts`, `*.test.ts` of each operator folder, `src/testing/message-passing.ts`) stay green; add for the 17 untested operator files one test each (message and evidence vs `bruteTilted`) in S0 so the net exists before anything moves.
- **Be honest about the integration step.** Replacing one quadrature by another changes the last digits (different node sets): that step is split into 5a (parametrised to reproduce today's nodes exactly: bit-identical) and 5c (deliberate change of constants and windows, judged by error against a high-order reference, not by sameness).
- Per step: its own test files, `tsc`, `make barrels`, `make layers`, biome (CLAUDE.md: never the full suite; the main session runs the gate).

### 6.1 The steps

| Step | What moves or merges | Proof | Size / risk | Files touched | Parallel with |
|------|----------------------|-------|-------------|---------------|---------------|
| **S0 Baseline and nets** | script dumping the atoms' beliefs/evidence/sweeps/improper counters; comparer; one test per untested operator file (17); fixed default fast-check seed (`OMAI_FC_SEED=random` to stress) | the snapshot compares equal to itself; new tests pass | S (1 day) / none | new files only; `operators.test.ts` header | nothing else should start before it |
| **S1 Folder move** | the layout of 4.1: `git mv` only, `make barrels`, import paths, `docs/structure.md` paragraph and the `llms` index (`make agent-index`) | exact snapshot; `make layers` | S-M (1 day) / low, but a huge diff: **do it in a quiet window, no agents on `message-passing`** | every file's path | none; do before S2 |
| **S2 Mechanical helper dedupe** | add `kernel/messages.ts` with `arg`, `rankOne`, `softplus`/`logSigmoid` (to `base`), `isFlat`, one rule-cache (`numerics/rules.ts`), one Cholesky (`numerics/dense.ts`) and swap the 20 `arg`s, 4 `softplus`es, 2 `logSigmoid`s, 4 `isFlat`s, 5 Legendre caches, 3 Choleskys, the doubled rank-one lift. Not the 19 log-sum-exps yet (order of summation differs): those go in 5a | **exact** | M (1 day) / low | operator files (light touches) | three agents in parallel by file group: (A) gaussian/links/extremum/discrete, (B) likelihoods/scales/pairs/concentration, (C) covariance/lifts/multiclass/transition; the shared helper is added by the lead first |
| **S3 Error taxonomy** | S3a: `core/errors.ts` with `ImproperMessageError` and a new `ParameterError`; make families, numerics and operators throw `ImproperMessageError` where they mean improper (36 + 18 throws audited one by one) and `ParameterError` for parameters; S3b: the engine catches only `ImproperMessageError`; `compileGraph` runs `check` plus the new declarative `params` parse for **all 66** operators (the 31 missing `check`s) | tolerant on valid graphs (identical), **intentionally different** on misconfigured ones: a new test pins the two experiments of finding 3 (missing `a`, `variance: -2`) to a clear `ParameterError` at compile time | M (1-1.5 days) / medium: any `value` throw that was relied upon will now abort a run; the 92 atoms and the model compiler's tests find them | `args.ts`, `engine.ts` (2 lines), every operator's param reads, `graph.ts` | S3a can run beside S4; S3b after both |
| **S4 Engine split** | `engine.ts` into `engine/{cavity,update,evidence,verdict,restart,schedule,run}.ts`; the one flat-cavity policy in `cavity.ts` (delete `_sides.ts` and `prior-sites.ts` widenings by calling it); `repair()` shared with `stochastic.ts`; `mpVerdict` takes complete settings; evidence/verdict computed once per observe; damping validated once | **exact** | M (1-1.5 days) / low | `engine.ts`, `stochastic.ts`, `named.ts` imports, `_sides.ts`, `prior-sites.ts` | with S3a and S5 (different files) |
| **S5 Integration layer** | **5a**: build `numerics/{axis,window,integrate,tilt1d,moments,profiles}.ts` with profiles that reproduce today's constants exactly, then migrate one operator per commit and delete its private grid: `gamma-scale`, `concentration`, `gaussian.ts precisionTilt`, `scale`+`function-scale`, `pairs`/`correlation` (already on `_grid`), `numerical.ts` rules, `variational-numeric`, `multiclass` tables, `priors.ts` window, then delete `_grid`, `_gridn`, `_quadrature`, `_tilted` or leave them as 10-line re-exports. Replace the 19 hand-rolled log-sum-exps (here, with a bit-identical `shiftedSum` helper that keeps the order). **5b**: VMP quadrature ports built on the same axes (still the same nodes). **5c** (non-preserving, separate PR after 5a/5b are green): centred variance everywhere and one floor policy that reports collapse; one gamma/beta window and clamp; replace hand `WindowHint`s by `findWindow`; unify profiles; remove the 24-point rule from `negative-binomial-dispersion` | 5a/5b exact; 5c: every operator's brute-force test at 1e-9 (the references at `testing/message-passing.ts`), atoms in tolerant mode, and a per-operator "error vs high-order reference does not increase" table | XL (4-5 days; 5a is ~10 independent operator migrations) / medium for 5c, low for 5a | one file per migration | 5a migrations are independent of each other (different operator files) but must run after S2 and S1; they can run beside S3 if S3 touches different lines (coordinate: both touch operators, so serialise S3b) |
| **S6 Contract trim** | `defineOperator` and the new `FactorOperator` (section 1.3): migrate the hand-rolled numeric `ep` rules (about 20 of the 30 numeric-EP operators) to `site(tilt)`; delete the 9 power divisions, 16 `siteMessage` calls, the 17 evidence catch blocks (the engine catches); `gate`, `glmRow`, `glmPlate` call the shared `invoke`; derive `support`; fold `unsupported` into `noRule`; drop `Projection.numeric`'s operator-side slot except for `numericOperator`; per-folder `operators` arrays with a duplicate-name check and a generated list in the docs; alias `gaussian-log-scale` -> `gaussian-scale` (compiler keeps the old name) | tolerant (a `(v - own) / power` moved into the engine reorders no sums; expect exact, accept tolerant), plus the registry test becomes the constructor | L (2-3 days) / medium: the largest diff in the operators; the contract change touches the `probprog/model/factors.ts` compiler's `operatorsWithout` (2 places) | every operator; `types.ts`; `gate.ts`; `rows.ts`; `plates.ts` | after S2, S3, S4 and 5a; two or three agents by the same folder groups as S2 |
| **S7 Memo and global state** | after S6, an operator that integrates once for all ports exposes `tilt(c)` and the engine caches per `(factor, cavities)` for the duration of one update (and one evidence evaluation); delete `tiltCache` and `last` | exact on the atoms (same values), counters of integrations drop (assert it in one test) | S (0.5 day) / low | `gaussian.ts`, `multiclass.ts`, engine `update.ts` | after S6 |
| **S8 Docs and tests** | `docs/structure.md` (counts, the 100-line operator paragraph shortened to the folder map), `ep-plan.md` numbers, registry TSDoc generated, `operators.test.ts` split by folder, `bp` either derived from `ep` for exact operators or the claim in `engine.ts:2` narrowed; move algorithm variants to `algorithms/`; remove dead exports of section 5 | `make agent-index-check`, `make barrels`, `make layers` | S (1 day) / none | docs, tests | any time after S1 |

Dependencies: S0 -> S1 -> S2 -> {S3a, S4} -> S5a (parallel migrations) -> S3b -> S6 (parallel by folder) -> S7 -> S5c, S8 anywhere after S1.
Two agents can usefully work at once at most points (project rule: 2-3 parallel agents, always throttled with `nice -n 15 env OMAI_TEST_WORKERS=1`): the natural pairs are (S3a, S4), (S2 groups), (S5a migrations), (S6 groups).

### 6.2 What to decide first

1. Accept the **error-taxonomy change** (S3) as the one deliberate behaviour change among the "preserving" steps: it turns misconfigured graphs from silent into loud.
2. Decide whether `bp` stays a projection of this engine (then derive it from `ep` for exact rules) or leaves the claim (S8).
3. Decide whether the algorithm variants stay in the package root or move to `algorithms/` (a move, no code change).
4. Choose the tolerance policy for S5c up front (suggest: operator references at 1e-9 relative, atoms at the existing per-atom `tolerance`, evidence 1e-10 where the references allow it, since a changed window may move evidence at the 1e-11 level).

## 7. Method and limits of this review

Read in full: `types.ts`, `engine.ts`, `projections.ts`, `_grid.ts`, `_gridn.ts`, `_quadrature.ts`, `_tilted.ts`, `_scalar.ts`, `registry.ts`, `numeric.ts`, `rows.ts`, `variational-numeric.ts` (half), parts
of `gaussian.ts`, `gate.ts`, `multiclass.ts`, `gamma-scale.ts`, `concentration.ts`, `scale.ts`, `priors.ts`, `numerical.ts`, `named.ts`, `families*.ts`. Measured by script: the operator-declaration histogram (instantiating `defaultOperators()`), the import graph (file-level
cycles and layer inversions), unused exports, run time of the 92-atom suite (67.9 s), and two experiments on silent failure (finding 3). Not read in depth: `regions.ts`, `amp.ts`, `consistent.ts`, `clutter.ts`, `covariance.ts`, `correlation.ts`, `transition.ts`, `stick-breaking.ts`, `plates.ts` internals, and
the 5,763-line `probprog/model/factors.ts` compiler (out of scope; it is the consumer of the registry and will need the alias map in S6). The counts are exact for the grep patterns quoted; "about N lines" figures for grids are the line spans of the functions named, rounded.

## How a step uses the digest (step 0, built)

The safety net that this plan's step 0 asks for exists: `scripts/ep-digest.ts` (`make ep-digest`, `make ep-digest-check`; the
committed baseline is `docs/probprog/ep-digest.json`, section 10 of `docs/structure.md` says what it holds). It covers every
atom (92) with its variational twin, every benchmark model (27) and the EP programs of the pocs (15 runs), as the compiled
factor-graph hashes (EP's, rows expanded, variational), the verdicts and counts, the beliefs, the evidence, the check rows, the
coverage rows, and the operator registry with the knob spaces. A step of this plan runs `make ep-digest` once on the unchanged
branch (it fills `.cache/ep-digest/graphs/`, so a later report can show the graph that changed, not only its hash), and after
every step `make ep-digest-check ARGS="--exact"` (about 2.5 minutes; `--only <atom or model prefix>` while iterating, 10 to 20
seconds). **Every step that the plan calls preserving must pass EXACT**: graphs, verdicts, sweep, restart and improper counts
identical and numbers equal to 12 digits (a compiler step that leaves the graph hashes alone gets identical beliefs and
evidence by construction). A step the plan allows to reorder sums or change a quadrature (the rules review's 5a is bit-identical,
5c is not) states that it passes TOLERANT (beliefs 1e-9 relative, evidence 1e-12 absolute, check statistics 1e-9) and why. A
deliberate behaviour change (the error taxonomy of S3, a typo in a knob now throwing, a wider window) says so in its commit
message, shows the moves that `make ep-digest-check` printed, and runs `make ep-digest-check ARGS="--accept"` in a commit of its
own, so the baseline's history is the list of intended changes. A step that adds an atom, model or program adds its item the same
way. The digest is not part of `make verify` or `make check`; it is the gate of this refactoring.
