# Design review: the model-to-factor-graph compiler (`probprog/model/factors.ts`)

Read-only review, 2026-11-11, at commit `0a1c56f9`. Scope: `packages/engine/src/lib/probprog/model/factors.ts` (5,763 lines)
and its neighbours (`operators.ts`, `compile.ts`, `analysis*.ts`, `bind.ts`, `scan.ts`, `event-dims.ts`, `ids.ts`), their tests,
and the consumers (`inference/mp.ts`, `_support.ts`, `ep-grid.ts`, `ep-multistart.ts`, the `ep/` suite). Line numbers are for
this commit. Everything below was read or counted (grep/awk), nothing was run or edited.

## 0. The short version

The compiler is a 4,760-line class (`class Compiler`, lines 514-5275) with 142 methods and 32 mutable fields. It has the right
core idea (a symbolic value algebra of affine forms over graph atoms, then one mapping per model idiom), and the cell algebra,
the id contract (`ids.ts`) and the table-driven numeric likelihoods (`likelihoods()`, `pairLikelihoods()`) are good. The mess is
everything around that core: the file is organised by the order in which packages landed, not by concern; the mappings share no
interface; the same four or five helper patterns are written out by hand 5 to 20 times each; refusal is a string; and the
compiler's knowledge of the rules is string literals in 140+ places. 60 commits touched the file in two days (9 on 11-10, 51 on
11-11). It is patched, as the owner suspects, but it is patched in a very regular way, which is what makes a safe refactor
possible: the near-duplicates are mechanical, and a graph-level snapshot corpus (section 5, step 0) can prove "no change" exactly,
stricter than 1e-9 on beliefs.

## 1. Findings ranked by what they cost future work

1. **Every new mapping is a new private method plus edits to two or three if-chains, and nothing says which.** `latent()`
   (1832-1944) is a 113-line if-chain with six early returns, `observed()` (3700-3772) a prefix chain, `observedElement()`
   (3801-3899) a 20-case switch with nested condition chains, `latentNormal()` (2620-2705) a chain on what the scale is. Order
   inside the chains is semantics (normal-gamma is tried before the family lookup, 1861-1870; `valueMovedToMean` before everything
   else in `observed`, 3708-3712) and is documented nowhere. A new agent cannot tell where an idiom belongs without reading all of
   4,760 lines; that is the cost the owner is sensing.
2. **Recognition and emission are fused, and recognition is done by exception.** Mappings test "does my pattern apply" by
   evaluating and catching `Refusal` (12 `catch` sites, 6 of them probes that fall back to another route: 2885 `latentRandomScale`, 3562
   `denseNonCentred`, 4684 `switchedCells`, 4831 `observedSwitchLogScale`, 4894 `observedLogScale`, 4932 `observedRandomScale`). Yet `Refusal` is also
   the final "report and give up" (186 `throw new Refusal`). One class means two things ("not me, try the next" and "stop"). And
   evaluation emits graph structure as a side effect (`gate()` 1399, `special()` 954, `mulAff` 1168, `applyChain` 1248 call
   `addVariable`/`addFactor`), with no rollback. `prune()` (1544) removes dead `apply` factors only, by operator name, so any
   other residue of a probe that later falls back (a `product`, `select`, `gamma-product`, `dot-constant` component) would stay in
   the graph. I could not construct a failing case in the time (a refused node makes the whole compilation refuse anyway, which
   hides this), so treat it as an unproven risk, but the design gives no guarantee.
3. **The compiler re-implements the same handful of helpers by hand, with inconsistent conventions** (section 2 has the list with
   line numbers). The worst: the gate factor is hand-assembled 7 times and "put this variable into the gate's argument list"
   (`let at = vars.indexOf(v); if (at < 0) { at = vars.length; vars.push(v) }`) 4 times, with two different index conventions
   (`mean = at + 1` at 2680 because slot 0 is the selector; `mean = at` at 4781 and 4861 because it is not). One off-by-one
   there is a silent wrong posterior.
4. **Nothing guards the contract with the rules.** 46 distinct operator names appear as string literals (`operator: '...'` 79
   times, `this.need('...')` 37 times); `FactorSpec.operator` is `string`; gate branches are `Record<string, Json>` built with
   `as unknown as Json` (1216, 1263, 2847, 3595) and `as never` (2841). A typo or a changed port order fails at
   `compileGraph` time in a test, or at run time. The compiler does not call `FactorOperator.check` (`message-passing/graph.ts:209`
   does it later, at `compileGraph`, as an `OmaiError` instead of a `Refusal`), so each mapping re-validates parameters itself
   (`must be positive` is checked in 13 places by hand).
5. **Refusals are strings that tests and people parse.** `FactorIssue = { node, kind, reason: string }`. `kind` has 7 values and
   is nearly meaningless: `operator` is 91 of 186 throws (49%) and covers "a random value where a known one is needed", "not the
   idiom" and "no rule"; downstream it is used only in `factorGraphVerdict` (5705-5709) to choose between two canned `missing`
   strings. at least 13 test assertions match substrings of the reason text, so the prose is de facto API and any wording fix breaks tests.
   Three other places also hold "what cannot be mapped" knowledge: `inference/mp.ts` `staticVerdict` (params, factors, continuous
   for BP, unbounded discrete), `ep/ep-gaps.ts` (curated prose) and the stale lists inside messages (below).
6. **The file mixes five jobs**: the value algebra (227-440, 727-1520), the graph builder (1522-1650), the node mappings
   (1651-5270, 70% of the file), the driver (5276-5386), and the other half of the contract (reading results back:
   `factorGraphPosterior` 5449-5665, `graphOnSpec`, `factorGraphVerdict`, `compileFactorGraph`, 5388-5763). The readback is keyed
   by `SiteLayout.family` plus three optional flags (`reciprocal`, `part`, `correlation`, 163-194) that were added one at a time:
   adding a family means editing the compiler site, a flag, and a `switch` case 4,000 lines away.
7. **Pattern detection that is not in the mappings is in the constructor and in lazily-run mappings, with implicit order.**
   The constructor runs `findPacked` -> `findNonCentred` -> `findGroups` (578-580; `findNonCentred` reads `matmulRead` that
   `findPacked` fills, `findGroups` reads `packed`; a comment says so, nothing enforces it). The normal-gamma pair is a "plan"
   detected inside the emission of the precision latent (`latentNormalGamma` 1990-2043), which stores into three maps
   (`normalGammaByMean`, `normalGammaObserved`, and a write-only `normalGammaOf`) that later latents and observations read
   (1861, 3731). It only works because the builder orders the gamma before the mean before the observations, and because
   `isConst` (796) depends on `consts` filled by earlier nodes in the same walk (`derived` fills it at 3523). So "is this known?"
   is not a property of the model but of how far the walk has got.
8. **Hidden coupling through names and counters.** Auxiliary ids `y#0[3]` come from `auxCount` (1561), so ids depend on the order
   in which expressions are evaluated (first element or all elements: `cellsOf`, 3786). Any reorganisation that changes evaluation
   order changes ids, which key the tape, the posteriors and the tests. `this.current` (563, set at 1654 and juggled at
   3500-3504) is dynamic scope for naming. `lookupMode` (564) changes what `gate()` returns (1414), set and reset around calls at
   4667-4673 and 4827-4835. `this.cells` (3783) is a side channel: `gatedMean` pokes `${node.name}|loc` into it (3695) so that
   `observed` later reads it through `paramCell`.
9. **The "analysis" module is not used by the compiler.** `toFactorGraph` takes `options.analysis` only for `opaque` (5332).
   `dependsOnLatent` (717) is `usesLatent` of `analysis-exprs.ts:94` without scans; `sharedGamma` (697) is `latentsIn`
   (analysis-exprs.ts:72) plus a family check; the normal-gamma detection (1990) is a second, independent conjugacy detector next
   to `analysis-conjugacy.ts`. Two definitions of "depends on a latent" can disagree on scans.
10. **The header is an inventory in landing order, not a contract.** Lines 2-136 are one 135-line prose list ("Expectation
    propagation also maps...", "Discrete structure under Dirichlet-type weights and linear dynamics:...") and are already wrong
    in places (section 4). It cannot be kept right by hand; it should be generated from the mapping table.

## 2. Where the same shape is written many times

All counts are on `factors.ts`.

**A. Intern a variable into a gate's argument list, and assemble the gate.**
- Interning copies: 2675-2681 (`latentNormal`, selector at slot 0, so `mean = at + 1`), 2828-2833 (`latentSpikeSlabSwitch`),
  4776-4790 (`observedSwitch`, twice: the mean and the gamma), 4856-4862 (`observedSwitchLogScale.argOf`).
- Gate factor assembly `{ operator: 'gate', args: [z, ...], params: { arity, branches } }`: 2685, 2845, 3333, 3990, 4803,
  4871, 5148 (7). Of these, 3333 (`latentCategorical`, Dirichlet rows) and 5148 (`observedRowGate`) build the same
  `discrete`-branch gate for a latent and for an observed outcome; 3300-3335 and 5113-5155 are the same mapping written twice.
- Fix: one `GateBuilder` (`addShared(variable): number`, `branch(operator, args, params)`, `emit(selector)`), and a typed
  `GateBranch` exported from `message-passing/operators/gate.ts` (its `branchesOf`, line 25, is the parser of exactly this shape).

**B. A Gaussian mean/regressor port: "a constant, else a materialised variable".**
`isConstCell(l) ? { const: l.aff.offset } : this.materialise(affOf(l), node.name, index)` hand-written at 3008, 4192, 4258,
4938, 4960, and as variants in `latentNormal` (2693-2696) and `observedGaussian` (4313-4340). Fix: `portOf(cell, node, index)`.

**C. A Gaussian whose scale/precision/mean is random** (this is the largest family, 14 methods, ~900 lines):
- latent side: dispatch at 2625-2632 -> `latentSharedPrecision` 2920, `latentRandomScale` 2877, `latentRandomPrecision` 2995,
  `latentGammaRate` 2955, `latentNormalGamma` 1990, `latentSpikeSlab(Switch)` 2744/2801.
- observed side: `observedLogScale` 4881, `observedRandomScale` 4911, `observedRandomPrecision` 4947, `observedSwitch` 4702,
  `observedSwitchLogScale` 4815, `observedOutlierSwitch` 4426, `observedNormalGamma` 2054.
- `gaussian-gamma-precision` is built by the helper `precisionFactor` (478) at 3010 and 4963 but by hand at 2835 and 4790.
- The three recognisers for "the log variable behind a positive scale" are separate: `scaledExp` 3905, `positivePort` 4209-4247
  (reads `this.logScale` at 4234), `observedRandomScale` 4915-4934 (reads it at 4929). `op === 'exp'` is matched at 3820, 3916,
  3926, 3928, 4217, 4824, 4922. Recognition is done twice in two styles: syntactically on the expression tree (`precisionExpr`
  1723, `stripClip` 2859, `scaledExp`, `multipleOf` 1945, `precisionMultiple` 1970, `isRecip` 1749) and semantically on cells
  (`scaledOf` 328, `recipArithmetic` 379, `scaledArithmetic` 399, cell kinds `scaled`/`recip`). Both ask "is this a known
  multiple of a gamma variable".

**D. Observed X whose parameter must be a latent of family F: check, declare the observed variable, add one factor.**
`observedConjugate` 5187 (poisson, binomial, categorical), `observedGammaRate` 5013, `observedFailures` 5081,
`observedMixtureWeight` 5047, the beta branch of `observedBernoulli` 4994, `observedHurdle` 5161. Five near-identical messages
"must be a ... latent for the conjugate '...' operator, it is not" (3365, 3449, 5028, 5098, 5216). An observed count is declared
with `family: { kind: 'gaussian' }` as a placeholder in 5031, 5101, 5219, 5227 (a Poisson count, a binomial count, failures): a
typing lie that works because the port accepts it. A table `{ family, parameter, requiredFamily, factor, valueTransform }` would
replace ~350 lines.

**E. The per-element latent skeleton**: `const cells: Cell[] = []; const vars: string[] = []; for ... { addVariable; addFactor;
cells.push; vars.push }; this.syms.set(node.name, { dims, cells }); this.layout(node, 'latent', fam, dims.length === 0 ?
'scalar' : 'elements', vars, true)`. 20 `const cells` declarations, 33 `layout(...)` calls, 24 `syms.set`, 17 scalar/elements ternaries.
Fix: `finishLatent(node, family, cells, vars, extra)`.

**F. Size of a discrete variable**: `(this.variables.find((u) => u.name === v)?.family.size ?? 2) as number` at 961, 992, 1026,
3279, 3312, 3322, 3367, 5128, 5138, 5230 (10). A linear search each time, and `?? 2` silently turns "unknown variable" into
"binary". Fix: `builder.sizeOf(name)` that throws.

**G. Known-parameter-or-refuse**: `known()` closures in `observedNumeric` (4066) and `observedHeavy` (4352), `constAt` 3900,
`paramIsConst` 4389, `constParam` 1707; the message "the X of 'n' is random: the F likelihood has a message rule for a known X
only" is written at 3953, 4073, 4080, 4356, 4436.

**H. The `lookup` cell literal** `{ kind: 'lookup', variable, values, items }` is built by hand in `gate()` three times (1403-1409,
1411-1417 and 1418-1426: the first two with an identical `values`/`items` body) and again in `lookupOf` (1470-1478) and
`choose` (1341-1360 calls); plus 6 more literals in the arithmetic helpers (344, 366, 372, 997, 1030s).

**I. `switch`-on-family emitters that overlap**: `observedNumeric` (4056-4165) is the good shape (a registry name + a predictor +
`emitSite`), but it still contains one `if` per family with the idiom recogniser inline (`negativeBinomial` 4137-4163 is 27 lines
of tree matching). `observedPoissonGate` (3941), `observedSwitch` (4702, `pdf` at 4742) and `observedMixtureWeight` (5047, `density`
at 5063) compute likelihood densities inside the compiler (a Poisson pmf at 3963, two Gaussian pdfs) where the registry's
`Likelihood.density` already exists.

## 3. Contracts between the compiler and the rules

What exists, and is good: `FactorOperator` has `ports` with `accepts`, `check(params, families)`, `unsupported(mode, params)`,
`noRule`, `defines`; the numeric likelihood registries carry `check(y, params)`; `OperatorDef.factor` in the model registry
lets a model operator name its factor. `compileGraph` validates arity, port families and `check`.

What is ad hoc:
- **A mapping declares nothing.** It cannot say "I need operators X, Y", "I apply to family F with parameter idiom P", "my
  start message is S", "my layout is L". The only declaration is the order of statements. A refusal is a `Refusal(kind, string)`
  thrown at the point of failure.
- **Start messages and layout are implicit in the emission**: vague priors (`gaussianMvPrior`, `priorFactor`, `gaussianPrior`
  imports at the top) are added next to the variable; the layout is a call to `this.layout(...)` with five positional arguments
  (1810) that every mapping must remember to make; forgetting it silently drops the site from the posterior.
- **Dangling metadata**: `OperatorDef.factor` names operators that do not exist in the registry: `'matmul'` (operators.ts:201),
  `'at'` (236), `'pick'` (258), `'times'` (383). Of the 13 declared, only `add`, `sub`, `sigmoid`, `gt`, `probit` are ever read
  (`factorName`, factors.ts:473, called at 1635, 1636, 4977). `spec.ts:27` says these "are the names a message-passing factor
  registry is keyed by", which is false for four. No test asserts the mapping is total.
- **Two naming collisions**: `model/operators.ts` (the model-layer expression registry) vs `message-passing/operators/` (factor
  rules); `factors.ts` imports `defaultOperators` twice under two meanings (`defaultOperators` and
  `defaultOperators as factorOperators`), and `modelOps` / `ops` / `operators` / `options.modelOperators` name the two tables.
  `compile.ts` is `toProgram`, not the factor compiler.
- **Three positional booleans** in `new Compiler(bound, modelOps, ops, vectorise, packScalars, denseScales)` (566-573).

Can the registry self-describe what it can compile? Not completely: layering forbids `message-passing` from naming
`ExprSpec` (probprog imports message-passing, never the reverse). The right split:
1. In `message-passing`: export `operatorNames` as a `const` tuple, so `FactorSpec.operator` can be the union `OperatorName`
   (typos fail `tsc`), and export a typed `GateBranch`. Add an optional `idiom: string` (documentation: the model idiom the
   operator serves, in words) and make `check` the single validator.
2. In `probprog/model/factor-graph`: a `Mapping` record `{ name, kind, requires: OperatorName[], applies(ctx, node) -> Plan | Refusal,
   emit(ctx, plan), reads/writes }` in one ordered list. A test asserts (a) every `requires` is in `defaultOperators()` (kills
   the dangling names), (b) every operator of the registry is claimed by at least one mapping or listed in `HAND_BUILT_ONLY` (the
   operators reachable only from hand-built graphs: `spin`, `gaussian-joint`, `is-between`, `matmul-constant`, ...), so the
   coverage gap between rules and compiler is a failing test, not a page of prose. `ep/ep-gaps.ts` can then be derived.
3. `addFactor` calls `op.check` and converts its string to a structured `Refusal('data'...)`. Do this behind a shadow flag first
   (assert in the corpus test that it never fires on a graph that compiles today), because it moves a failure earlier.

Structured refusals: `Refusal { code, node, subject?, facts }` with `reason` rendered from the code by one catalogue (messages
stay byte-identical in step 9 so the substring assertions keep passing; wording changes happen afterwards in one place).
Keep the 7 `kind`s for compatibility; add `code` (e.g. `random-where-known`, `not-precision-idiom`, `missing-size`,
`needs-positive`, `no-rule`). "Try the next mapping" becomes a separate return value (`NoMatch`), not an exception.

## 4. Dead code, stale text, unused exports (with lines)

- Write-only field `normalGammaOf`: declared 542, set 2041, never read.
- Two consecutive doc comments, the first orphaned from its field (`wisharts`): 549-550 (the Wishart comment sits above the
  `correlations` field and `wisharts` has none).
- A field declared mid-class: `private readonly cells` at 3783 (between methods), a different kind of "cells" from `Sym.cells`.
- Section markers are no longer true: `// ----- matrix-valued latents ...` at 2133 sits above `latentTransition` 2442,
  `latentProjected` 2500, `latentNormal` 2620, which are not matrix latents; `observedNormalGamma` (2054) sits in the latent
  region, `observedMvnWishart` (2282) and `observedBivariateNormal` (2410) in the matrix region. Only 6 markers for 142 methods.
- Stale reason: 1772 says the precision idiom is "variational message passing only", but the header (21-26) and
  `observedRandomPrecision` map it for EP (`gaussian-gamma-precision` has an EP rule for observed Gaussians).
- Stale reason lists: 3894 ("the likelihoods mapped are normal, logNormal, ... categorical") omits `beta` (3832 maps it through
  `likelihood-pair`), `zeroTruncatedPoisson`, `hurdlePoisson`; 1871 ("the engine has message families gaussian, gamma, beta,
  bernoulli, discrete and dirichlet") omits `gaussian-mv`, `normal-gamma`, `wishart`.
- Stale TSDoc of `toFactorGraph` (5246-5248): "a random scale that is not the precision idiom (a product of two random precisions is
  not mapped)" is wrong since `gamma-product` (`special`, 976-985); "a gate over a mix of known and random components" is wrong
  since `gate()` 1409-1415 and 1424-1432 return lookups with `items`.
- "first version" wording that no longer holds: 146 (`FactorIssueKind` structure), 445 (`MESSAGE_FAMILY`), and "not expanded
  yet" in 3433 and 3743 (the message promises a feature; the two copies are also the same refusal written twice).
- Exported but used only by tests and docs: `graphOnSpec` (5667), `compileFactorGraph` (5753); `operatorsWithout` (5406) used only
  inside `factorGraphVerdict`. Types `SiteLayout`, `FactorIssue`, `FactorIssueKind` are not imported anywhere outside the file
  (they are public through the barrel, so keep, but they have no external consumer to guide their shape).
- `MESSAGE_FAMILY.normalSpikeSlab` and `PROJECTED_PRIORS` are the only table-driven decisions in 4,700 lines; the dispatch that
  uses them is still an if-chain (1840-1925).
- History narration: little. `gate-log-scale.test.ts:3` ("Before this the two were separate selects...") is the one real case.
  No `TODO`/`FIXME` in the compiler, `inference/mp.ts`, `ep/` or `model/`.
- Unverified duplicate: `dependsOnLatent` 717 vs `analysis-exprs.ts:94 usesLatent` (scan handling differs).

## 5. Target structure and the stepwise plan

### 5.1 Modules (folder `probprog/model/factor-graph/`, an `@inline` namespace folder; `factors.ts` stays one line longer as a re-export until importers move)

Public surface unchanged: `toFactorGraph`, `factorGraphPosterior`, `factorGraphVerdict`, `compileFactorGraph`, `graphOnSpec`,
`operatorsWithout`, `FactorGraphCompilation`, `SiteLayout`, `FactorIssue(Kind)`, `MessageRule` (importers: `inference/mp.ts`,
`inference/_support.ts`, `inference/ep-grid.ts`, `inference/ep-multistart.ts`, `model/index.ts`, ~27 test files).

| Module | Concern | From (lines) | ~Lines |
|---|---|---|---|
| `types.ts` | `FactorIssue`, `SiteLayout` (with a tagged `reading`), `FactorGraphCompilation` | 139-215 | 100 |
| `refusal.ts` | `Refusal`, codes, the message catalogue, `NoMatch` | 216-226 + 186 throw sites | 150 |
| `cells.ts` | `Atom`, `Aff`, `Cell`, `Sym`, `Precision`, affine/lookup/recip/scaled arithmetic, `prod`, `unflatten` (pure, no class) | 227-440 | 220 |
| `builder.ts` | `GraphBuilder`: variables, factors, aux naming, components, `materialise`, `prune`, `sizeOf`, `GateBuilder`, `precisionFactor`, `portOf` | 441-512, 1522-1650 | 300 |
| `known.ts` + `plan.ts` | pass 1 constants (`known` values of data and constant derived nodes, pure); pass 2 `LayoutPlan` = packed, nonCentred, groups, normalGamma, shared gamma, matmulRead as one immutable value | 584-726, 796-847, 1990-2043 (detection half) | 350 |
| `idioms.ts` | syntactic recognisers as pure functions: `precisionExpr`, `isRecip`, `stripClip`, `multipleOf`, `scaledExp`, `transitionRow`, `unwrap`, `mentionsDiscrete`; uses `analysis-exprs` for latent dependence | 1723-1766, 1945-1990, 2859-2876, 3638-3652, 3905-3940, 3406-3423, 4644-4653 | 300 |
| `symbolic/affine.ts`, `symbolic/gates.ts`, `symbolic/functions.ts` | the evaluator (`evalSym`, `symOp`, reductions, `at`, `matmul`; `choose/pick/gate/lookupOf/where/shiftedByState/compare`; apply chains, extremum, `applyGamma`, `quotientByChain`). `lookupMode` becomes an explicit `EvalMode` argument | 727-1520 | 650 |
| `mappings/latent-gaussian.ts` | `latentNormal`, grouped, packed, random scale / precision / shared precision, spike-slab(+switch), truncate, projected | 2569-2705, 2706-3018 | 700 |
| `mappings/latent-positive.ts` | gamma, beta, exponential, `latentGammaRate`, `latentNormalGamma` (emission half), concentration | 1832-1944 (the `else` tail), 2955-2994, 3064-3208 | 450 |
| `mappings/latent-discrete.ts` | dirichlet, categorical, discrete, transition rows, stick-breaking readers | 3019-3063, 3209-3492 | 450 |
| `mappings/matrix.ts` | MVN, Wishart, MVN-Wishart, correlation, bivariate normal, vector mean, `plateVectors`; latent and observed together | 2086-2441 | 360 |
| `mappings/derived.ts` | `derived`, `denseNonCentred`, `tiedToVector` | 3493-3637 | 150 |
| `mappings/observed-gaussian.ts` | Gaussian with known/random scale, log scale, heavy tails, outlier switch, normal-gamma rows | 2054-2085, 3700-3772, 4302-4399, 4400-4473, 4881-4966 | 450 |
| `mappings/observed-numeric.ts` | `observedNumeric`, NB dispersion, pair, multiclass, `positivePort`, `emitSite`, hurdle | 4056-4300, 4474-4509, 5161-5186 | 400 |
| `mappings/observed-conjugate.ts` | the table of section 2.D | 4967-5112, 5187-5262 | 250 -> ~120 |
| `mappings/observed-gates.ts` | switch, switch log scale, Poisson gate, row gate, `switchedCells`, `gatedMean`, `valueMovedToMean` | 3653-3699, 3941-3999, 4654-4880, 5113-5160 | 450 |
| `mappings/plates.ts` | `designRows`, `plate`, `plateNormal`, `plateBernoulli`, `addPlated` (the one-factor-per-plate path) | 4510-4643 | 140 |
| `compiler.ts` | node loop, the ordered `mappings` list (priority = order, written down), Refusal capture, `failed` | 514-580, 1651-1700 | 250 |
| `driver.ts` | `toFactorGraph`: `eventToDims` -> `unrollScans` -> passes -> opaque filter -> compilation record | 5276-5386 | 130 |
| `readback.ts` | `factorGraphPosterior` and `graphOnSpec`, one reader per `SiteLayout.reading` kind | 5417-5693 | 270 |
| `verdict.ts` | `MessageRule`, `operatorsWithout`, `factorGraphVerdict`, `compileFactorGraph` | 5388-5415, 5695-5763 | 110 |

Explicit pass order (today implicit): 0 `eventToDims`, 1 `unrollScans`, 2 `known` (constants), 3 `plan` (packed -> nonCentred ->
groups -> normalGamma; each reads only the previous), 4 node walk in spec order through the mapping list, 5 `prune`, 6 opaque
filter and compilation record. Each pass is a function with typed input and output; the `Compiler` receives the plan instead of
mutating its own fields.

`SiteLayout.reading`: replace the optional flags with one tagged union (`gaussian | gaussian-mv | gamma{reciprocal} |
normal-gamma{part,reciprocal} | beta | correlation | wishart{reciprocal} | bernoulli | discrete | dirichlet`), so a family adds
one compiler site and one reader, found by exhaustiveness.

### 5.2 Steps (behaviour-preserving, each independently mergeable)

Proof of no change used in every step, in this order: (1) the graph snapshot corpus of step 0 is byte-identical (variables,
factors with ids, params, sites, issues, auxiliary, plated, expanded); (2) the model test files and `ep-atoms.test.ts` (92 atoms,
each `verifyAtom`: beliefs and check rows) pass; (3) tsc, barrels, layers, biome. Beliefs identical to 1e-9 and evidence
identical follow from an identical graph, so (1) is the strong check and (2) the cheap confirmation. Run throttled
(`nice -n 15 env OMAI_TEST_WORKERS=1 make verify`), only the model, ep and inference-mp files.

| # | Step | What moves | Proof | Size / risk | Parallel? |
|---|---|---|---|---|---|
| 0 | Graph snapshot corpus | New test util (no production edit): `vi.mock` of `factors.ts` records every `toFactorGraph` call's JSON during the model tests, `ep-atoms.test.ts`, `ep-coverage.test.ts`, `inference/mp.test.ts`, `models/*.test.ts`; plus a script that dumps `epAtoms` + benchmark models to `snapshots/*.json`. Before/after diff is the oracle. | self: run twice, identical | 150 lines; none | Must go first |
| 1 | `types.ts`, `refusal.ts`, `cells.ts` | pure leaves; `factors.ts` imports them | corpus + tsc | 450 lines moved; low | with 2, 11 |
| 2 | `readback.ts`, `verdict.ts` | 5388-5763 out; `inference/*` import paths updated | corpus + `mp.test.ts`, `ep-grid`, `ep-multistart` tests | 380 lines; low | with 1 (disjoint regions; only the import header conflicts) |
| 3a | `builder.ts` (move) | `addVariable/addFactor/need/auxName/component/materialise/prune/atoms` into `GraphBuilder`; `Compiler` holds `this.g` | corpus (aux ids identical) | 250 lines; low-medium | after 1 |
| 3b | `builder.ts` (dedupe) | `GateBuilder` (section 2.A, 7+4 sites), `portOf` (2.B, 5), `sizeOf` (2.F, 10), `finishLatent` (2.E, ~20), `lookup()` constructor (2.H), `precisionFactor` everywhere (2.C) | corpus; this is the step with real edits, do one helper per commit | 400 lines touched; medium (index conventions) | after 3a; the helpers are independent of each other, but touch the same methods, so one agent |
| 4 | `known.ts` + `plan.ts` | constants pass and `LayoutPlan`; constructor `find*` become pure functions; normal-gamma detection split from emission. `isConst` no longer depends on walk position | corpus (the plan must equal today's maps: also assert it in the test) | 350 lines; highest risk of the plan (the `isConst` ordering) | after 3a; independent of 5, 6 |
| 5 | `symbolic/*` | the evaluator extracted with explicit `EvalMode`; `lookupMode` flag removed | corpus + the gate tests (`gates`, `gate-precisions`, `shifted-gates`, `gate-log-scale`, `mixture-*`) | 650 lines; medium-high (touches `this.current`, `variables.find`, side-effecting emission) | with 4 and 6 only if 4 and 5 are on different files: they are |
| 6 | `idioms.ts` | the syntactic recognisers as pure functions over `{ nodes, known }`; `dependsOnLatent` replaced by `usesLatent` (scan-aware: confirm identical on the corpus) | corpus | 300 lines; low | with 4, 5 |
| 7 | mapping files (10 of them, one commit each: derived, matrix, plates, latent-discrete, latent-positive, latent-gaussian, observed-numeric, observed-conjugate, observed-gates, observed-gaussian) | `this.x` -> `cx.x`; no logic change | corpus | 100-700 lines each; low (mechanical) | yes: up to 3 agents at once (per the standing limit), disjoint files; start only after 3b, 4, 5, 6 so the context interface is stable |
| 8 | Mapping table (**done**, phase 3: `_mapping.ts`, `_table.ts`; the six probes use `attempt`/`probe` with a builder checkpoint) | `latent()`, `observed()`, `observedElement()` if-chains become an ordered `mappings` list with `applies`/`emit` and `requires`; write the priority order down | corpus; add a test that reorders nothing | 300 lines; medium (priority) | after 7 |
| 9 | Structured refusals (**done**, phase 3: `RefusalCode`, the catalogue `_reasons.ts`; the stale sentences of section 4 fixed in their own commit) | `Refusal(code, node, facts)`; reasons byte-identical (catalogue); `NoMatch` replaces probe-by-exception in the 12 catch sites one at a time | the reason assertions + corpus issues | 186 throw sites; low per site, long | can start per file inside step 7; finish after 8 |
| 10 | Registry contract (**done**, phase 3: `CompilerFactor`, `requires`, `operator-contract.test.ts`, `addFactor` validates) | `operatorNames` tuple and typed `FactorSpec.operator`; typed `GateBranch`; `requires` tests both ways (every required name exists; every operator claimed or `HAND_BUILT_ONLY`); fix the 4 dangling `OperatorDef.factor`; shadow `check` in `addFactor` | tsc + new tests; `check` shadow must never fire on the corpus | 150 lines across 3 packages; low, but it touches `message-passing` types | the typed names part can start after 3b |
| 11 | Tests relayout (section 6) (**done**, phase 3, with the differences listed at the end of this page) | test files only | same tests, same assertion count (`vitest --reporter` count before/after) | 8,193 test lines moved; low | any time, parallel with everything |
| 12 | Docs (**done**, phase 3: `docs/probprog/compiler-mappings.md` from the tables) | header shrunk to the contract and pass order; "what it maps" generated from the mapping list into `docs/probprog/`; fix the stale items of section 4; `docs/structure.md` 3961-3962 and 4.32 updated | `make docs-catalogue`, agent index | 200 lines; low | after 8 |

Total: about 12 agent sessions of one to four hours each; steps 0, 3a/3b, 4, 5 are the critical path (about 60% of the risk);
steps 1, 2, 6, 11 and the ten files of 7 are where parallel agents are safe. Do not run more than the standing two or three at
once, and never two agents on `factors.ts` itself at the same time: until step 7 is done, only one branch may edit the class.

## 6. Tests

Today: 27 files import `toFactorGraph` (306 `it`s, 8,193 lines in `model/*.test.ts`); about 20 are `*-factors.test.ts` or a single
feature (`gate-weights`, `gate-precisions`, `gate-log-scale`, `shifted-gates`, `gates`, `mixture-random-factors`,
`mixture-regression`, `scale-mixtures`, `random-concentration`, `stick-breaking`, `sv-parameters`, `transition-rows`,
`vector-state`, `topic-models`...). They are named for the feature increment that added them, not for the code concern;
`factors.test.ts` (1,236 lines, 41 `it`s) is a grab-bag of twelve unrelated `describe`s (`probit regression`, `conjugate`,
`TrueSkill`, `random precision`, `categorical gate`, `max and min`, `plated`, `what cannot be mapped`).

What they are: almost all are end-to-end accuracy tests (compile, run EP, compare with an exact or grid reference) and they are
strong at that: they are why the capability can be trusted. What they are not: tests of the compiler's output. Only about 75
lines in 24 files mention `.operator` (factor names), nothing asserts a graph fragment per mapping, nothing tests the cell
algebra (`cells.ts` functions have no direct test), and no test fails when a refusal's reason, id or layout changes unless it
happens to substring-match. EP tolerance (1e-3 to 1e-2 against a grid) means a changed graph can pass.

Duplication in the tests: 19 files define their own `run`/`ep`/`runEp` (compile, `compileGraph`, `runAlgorithm(epAlgorithm)`,
`marginalsOf`), 5 define `grid`/`grid2`, and several define `post`/`belief`/`marginal` readers, each with the same 6-line import
header.

Proposed layout (mirrors the modules; each file gets a one-line header "covers mappings/X"):
```
model/factor-graph/
  _support.ts                 runEp(bound, {damping, plates, packScalars}), grid1/grid2, belief(), factorsOf(c, op)
  cells.test.ts               NEW: arithmetic, lookup, recip, scaled, chains (unit, fast, property-based with fast-check)
  graph-corpus.test.ts        NEW: the snapshot corpus of step 0 (exact graph JSON, one file per atom/model under snapshots/)
  compiler.test.ts            driver: pass order, ids (`ids.ts`), plates vectorise vs expand, scans/event unrolling, opaque
  refusals.test.ts            every Refusal code once, with node and reason (from the 'cannot be mapped' describe + scattered ones)
  latent-gaussian.test.ts     prior-factors, constraints, ordered, truncate, projected priors, sv-parameters, random precision
  latent-positive.test.ts     random-concentration, normal-inverse-gamma, gamma rate hierarchies (horseshoe part of scale-mixtures)
  latent-discrete.test.ts     transition-rows, stick-breaking, topic-models (latent half)
  matrix.test.ts              matrix-factors, vector-state
  observed-gaussian.test.ts   likelihood-factors (Gaussian), robust-factors, scale-mixtures (Student-t), mixture-random (outliers)
  observed-numeric.test.ts    likelihood-factors (counts), ordinal-quantile, pair-factors, multiclass
  observed-conjugate.test.ts  conjugate-likelihoods, hurdle
  gates.test.ts               gates, gate-weights, gate-precisions, gate-log-scale, shifted-gates, mixture-regression, topic-models (observed)
  programs.test.ts            scan.test.ts parts that compile (vector-state, SV unrolling)
```
Each mapping test has two parts: the graph fragment (operators and arities it must produce, by a `fragmentOf(c, site)` helper
that returns the operator names of the factors of a site) and the accuracy check that exists today. The existing tests move
verbatim, so assertion counts are comparable before and after.

## 7. What is fine (keep)

- The cell algebra (`Aff` over atoms, `Cell` kinds, `Sym` over named dims) is a sound representation; its arithmetic helpers are
  pure and short (282-427). It is the part to extract first, unchanged.
- `ids.ts` is a clean contract (`elementId`, `auxId`, `factorId`, `siteOfId`), documented, and used consistently; the posterior
  readback and the structure pictures depend on it correctly.
- The numeric likelihood registries (`likelihoods()`, `pairLikelihoods()`) are the model for how mappings should look: a name,
  a `check`, a `density`; `observedNumeric` and `emitSite` already use them well.
- `toFactorGraph`'s TSDoc and `@example`s are checked and illustrate real graphs; keep the examples, move the inventory out.
- No `any`, no `TODO`, no dead private method (all 142 are called), no `console`. The refusals do tell the user what to change.
- The `ep/` atom registry (92 atoms, each run against an exact or grid reference) is an excellent regression corpus; step 0
  reuses it rather than inventing one.

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

## What phase 3 did differently from this plan

- The tables have five levels, not three chains: node kinds, latents, observed nodes as a whole, observed rows, observed elements (the element table is keyed by family
  and holds the nested chains of the `normal` case as separate entries). A failed probe restores the graph builder's checkpoint, including the counters of auxiliary
  names, so a later route makes the ids it would have made without the probe (the digest is bit-identical); the caches of the evaluator are restored only when
  the probe made something. The evaluator still signals "no" by throwing `Refusal` inside a probe: `probe` turns it into `NOT_MINE` at the mapping's edge.
- There are 14 refusal codes and 168 catalogue functions for the 187 throw sites; `kind` stays, derived from the code. `FactorIssue` gained `code`, which the digest does not hash.
- `FactorSpec.operator` is still a string in `message-passing` (hand-built graphs write it); the compiler's factors are typed by `CompilerFactor`, and the engine's constructors
  (`gaussianPrior`, ...) go through `addLibraryFactor`, which checks the name at run time.
- `addFactor` validates the factor with `compileGraph` on the factor and the variables it names. It was clean over the corpus and every compiler test, so it is enforced (a
  `registry` refusal), not shadowed.
- The tests were moved, not merged: the 27 files of `model/` keep their `describe`s verbatim in `factor-graph/` under concern names, and only `factors.test.ts` was split. The review's
  concern files that merge several features would need the per-file helpers (each with its own tolerance) merged first. The shared support is in `src/testing/factor-graph.ts`
  (the build excludes it), not `factor-graph/_support.ts`. The five local grids stay local: their defaults differ.

