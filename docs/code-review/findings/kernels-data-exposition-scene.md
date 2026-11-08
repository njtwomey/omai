# Kernels, data, exposition and scene review: kernel contract and catalogue, datasets and splits, exposition records, the scene graph

Read-only review of `packages/engine/src/lib/kernels` (9 non-test files, 3,300 lines), `lib/data` (32 files, 7,300 lines), `lib/exposition`
(54 files, 5,800 lines) and `lib/scene` (10 files, 2,500 lines). All paths below are relative to `packages/engine/src/lib/`. Probes cited as
"probe" are throwaway node scripts in
`/private/tmp/claude-501/-Users-nialltwomey-workspace-whitebox-js/d4034594-54d3-484e-a26a-331816a1f3f5/scratchpad/review/probe3/`
(`k1.ts` PSD/symmetry table over 32 kernels, `k2.ts`/`k3.ts` degenerate kernel input, `k4.ts`/`k5.ts` timings, `k6.ts` random features against the
exact kernel, `d1.ts`...`d5.ts` data, `e1.ts` exposition, `s1.ts`/`s2.ts` scene, `cov.mjs`/`cov2.mjs`/`sigmine.mjs` coverage and signatures),
run one at a time at `nice -n 15` on small sizes against the source; nothing in the repository was changed and no test, tsc or gate was run.
Coverage figures are Phase 0's (`coverage-final.json`, commit b09e46da); line numbers are from the tree at commit 158659d4. About 55 source files
were opened (some in part). Scipy and sklearn behaviour quoted is from memory (no Python was run).

## Summary

1. Healthy: **the kernel catalogue is numerically sound.** All 32 kernels probed (`k1`) are exactly symmetric, `gram(x)` equals `crossGram(x, x)` to
   0 (1.4e-12 for the depth-2 arc-cosine; white noise differs on purpose, as scikit-learn), the diagonal agrees with `diag(x)`, the minimum eigenvalue of
   every kernel documented as PSD is above -1e-13 on 8 random points in 1 to 3 dimensions, and the gradients at coincident points are finite for all
   stationary kernels (`k2`). The random-feature maps match the exact kernel to the Monte Carlo rate for all nine spectral samplers (`k6`: 4e-3 to 2e-2 at
   40,000 features). Data is also healthy where it counts: every sampler, split and batcher takes a `Key` and is deterministic (same key, same split;
   the stratified split does not depend on row order), no `Math.random`, clock or `console` in the four folders, TSDoc is complete (311 exported
   functions: 0 missing doc, `@example`, `@param` or `@returns`), and data hot paths are fast (`d2`: 50,000 rows split in 10 ms, 1,563 batches in 22 ms,
   100,000 x 10 CSV in 113 ms, 1e6-double `.npy` in 32 ms). Scene projection is cheap (20,000 points 9 ms) and every `SceneNode` kind has a case in render's
   `svg/svg.ts` and `react/SceneView.tsx`; every exposition kind has a constructor and render's registry is a mapped type over `ExpositionKind`, so a missing
   view fails at compile time.
2. Not healthy, 1: **the data boundary casts silently.** `trainTestSplit({ testSize: NaN })` returns an empty train set and a full test set (F2); `take(data, [NaN])`
   returns row 0 and `[1.7]` row 1; `xyClasses` turns a NaN label into class 0; `table({ a: [1, 'x', 3] })` is numeric with a NaN, `['x', 2, 3]` is categorical with
   a numeric level, `[true, 2]` is boolean (F4); `padSequences` and `collate` pick `int32` for integral values and wrap 3e9 to -1294967296 (F3); `concatRows` keeps
   the first part's `classes` whatever the others say (F8); `trainTestSplit` without a key splits in row order and has no notion of groups, queries, episodes or time
   (F7). The same NaN-into-a-finite-answer pattern as the foundations, numerics and dists reviews (F16).
3. Not healthy, 2: **a few kernel entry points have no validation**: `orthogonalRandomFeatures({ inputDim: 0 })` never returns and exhausts the heap in 100 s (F1);
   `gram(k, [1, 2], { jitter })` of one point returns `[]` (F6); `resolveKernel` with `gamma: 'scale'` returns 1 for data offset by 1e8 where the right value is 0.153
   (F5); a `positive` hyperparameter of exactly 0 is silently held fixed and gives an all-NaN Gram (F12). `periodic` and `cosine` in 2 or more dimensions and `sigmoid` are
   not PSD (minimum eigenvalue -0.24 to -1.95, probe `k1`; documented in prose), and the `Kernel` contract has no field to say so (F18).
4. Not healthy, 3: **no builder of a record or a scene checks finiteness, and nothing checks the shape of a Scene.** Exposition constructors all check lengths
   (61 kinds, one shape of contract), but only 7 of 54 files check finite values, inputs are aliased and not copied (`curveSet` shows a later mutation of the caller's array),
   and `isExposition` trusts a hand-copied list of kinds (F9). Scene builders check nothing: `scene(NaN, -5, [])`, `rect` with negative size, `structure` with duplicate ids and
   dangling edges, `grid(.., 0, ..)` (NaN height), `timeline` with NaN steps and negative fps all succeed and serialise to `null` (F13, F14). `groupReport` counts an empty
   group as a group with all rates 0, which makes up disparities of 0.8 to 1 (F10).
5. Hot paths: one real, `rbfKernel` and every stationary kernel build their distances from the `[m, n, d]` difference tensor of `cdist`: a Gram matrix of 1500 x 20 takes 2.65 s, the
   linear kernel of the same size 64 ms (F11). Everything else measured is fine (F35).
6. Layering: no cycle. `kernels/structured.ts` imports `graph` and `trace` for three graph kernels that nothing else uses (F23). By what they import, `scene` needs only `base` and
   `exposition` only `base` and (types of) `scene`, so both belong right above `base`, not between `optim` and `losses` as the first-guess order has it (F24).
7. Native arrays (F32-F34): data 17 functions (13 indices, labels, lengths or ragged token rows: not vectors; 3 to migrate: `weightedRandomSampler`, `balancedWeights`,
   `noiseTransition`; 1 decoder), exposition 11 (8 record data, where plain arrays are the format; 3 computational), kernels 2 (both allowed), scene 1 (a drawing builder: allowed).
8. The three highest-value fixes: (a) **F1 + F2 + F3 + F4 + F8** (argument and dtype validation at the data boundary and in the random-feature constructors: each a few
   lines, one table-driven test, F20); (b) **F9 + F13 + F16 + F19** (a `finite`/`copy` helper in `exposition/_check.ts` shared with `scene/build.ts`, `KINDS` as an object keyed by
   `ExpositionKind`, one contract test over the 61 record kinds, +3 tests); (c) **F11** (a matmul-based squared distance for the stationary kernels, 40x on 1500 points).
9. Not read or not verified: `data/modifiers/features.ts` (650 lines), `images.ts` (515), `labels.ts`, `leaks.ts`, `data/schema/*` beyond `common.ts`, `_coerce.ts` and the table
   part of `tabular.ts`, `data/streams.ts` bodies (only the validators), `recipes.ts` bodies, `learn/model-selection/splitters.ts` (the second stratified implementation, F27), `scene/paint.ts`,
   `scene/three-d.ts` beyond `surface3d`, `exposition/` files other than the dozen named, render's use of the records and of `Timeline`, and the SSK/WL/graph kernels against a
   reference (only property tests were read). The tar claims in F25 are from reading, not from a GNU-tar fixture. No test, tsc or gate was run; coverage is the engine
   suite only (the `@example` blocks and render's tests may reach some of the 12 exposition files that show 0%).

Entries by kind: consistency 21, hot-path 2, duplicate 3, layering 2, native-array 3, test-gap 4, rule 1 (36 in all, F1 to F36).

---

### F1: `orthogonalRandomFeatures` never returns for `inputDim <= 0`, and the random-feature constructors validate none of their sizes

kind: consistency
evidence: `kernels/expansion.ts:217` (`for (let b = 0; b * d < n; b++) blocks.push(...)`: with `d = 0` the condition is always true); probe `k2` ran it: heap exhausted after 100 s (`FATAL ERROR: Ineffective mark-compacts near heap limit`). Same file: `randomFourierFeatures({ inputDim: -1 })` throws a raw `RangeError: Invalid typed array length: -4` (no `OmaiError`), `nComponents: 0` returns a `[2, 0]` map, `polynomialCountSketch({ gamma: -1 })` returns NaN features (`Math.sqrt(g)` at `:547`), `degree: 0` fails with `ragged nested array`, `tensorSketch({ nComponents: 0 })` fails inside `randint`; `additiveChi2Sampler` accepts NaN (`:326`: `where(greater(NaN, 0), ..., 0)` makes it a zero feature).
what: the sampler constructors take `inputDim`, `nComponents`, `degree`, `gamma`, `coef0`, `sampleSteps` and check at most the last (`additiveChi2Sampler`), so a bad size surfaces as a hang, a typed-array error or NaN inside the first `transform`. The hang is the serious one: a UI knob set to 0 freezes the tab.
fix: one `positiveInt(name, v)` guard at the top of each constructor (`streams.ts:182` in `data` already has the idiom; a shared one belongs in `base`), `gamma > 0`, `coef0 >= 0`; loop `b < Math.ceil(n / d)`.
size: S (<1h)
risk: low
preserves behaviour: yes for valid input
changes public surface: no

### F2: sizes and indices in `data/rows.ts` are not validated: `testSize: NaN` leaves the train set empty, `take` accepts NaN and fractions, `bootstrap` and `stratifiedIndices` take any number

kind: consistency
evidence: `data/rows.ts:90-93` (`sizeOf`: `size < 1 ? ceil(size * n) : round(size)`, then `m < 0 || m > n`; NaN passes both tests); probe `d1`: `trainTestSplit(ds, { testSize: NaN })` gives train shape `[0]`, test `[8]`; `testSize: 2.5` is rounded to 3 rows; `take(ds, [NaN])` returns row 0 and `take(ds, [1.7])` row 1 (`rows.ts:13-17`, `Int32Array.from` truncates, and the range check `i < 0 || i >= n` is false for NaN); `bootstrap(ds, key, -1)` throws a raw `RangeError: Invalid typed array length: -2`, `size: 2.5` a `shape` error from the reshape, an empty dataset a message from `categorical` (`rows.ts:279-293`); `stratifiedIndices([0, 1], 5, key)` answers `{ train: [], test: [0, 1] }` and `nTest = -1` answers `test: []` (`rows.ts:111-143`, public, no check; only `stratifiedSplit` validates through `sizeOf`).
what: the split family is the one place where a wrong number gives a plausible dataset, not an error. `take` is also what `batches`, `bucketBy`, `shuffleRows`, `bootstrap` and the shift modifiers call, so a NaN index from a computed order reaches training as a silent copy of row 0. The size rule (fraction below 1, count at or above 1: `testSize: 1` is one row, `0.99` is 99%) is the sklearn rule and is documented, but `1.0` meaning one row is a trap worth a sentence.
fix: in `indexArray` reject non-integers (`Number.isInteger`) with `value`; in `sizeOf` reject `!Number.isFinite(size)` and non-integer counts; give `stratifiedIndices` and `bootstrap` the same check; make `sizeOf` shared (F27).
size: S (<1h)
risk: low (only invalid input changes)
preserves behaviour: yes for valid input
changes public surface: no

### F3: `padSequences` and `collate` choose `int32` for any integral value and wrap on overflow

kind: consistency
evidence: `data/batching.ts:148-149` (`parts.every((p) => Number.isInteger(p)) ? 'int32' : 'float64'`) and `:218-221` (`allInt = values.every(Number.isInteger)` then `Int32Array.from(values)`); probe `d5`: `padSequences([[3e9, 1]])` gives dtype `int32`, values `[-1294967296, 1]`; `collate([{ a: 3e9 }, { a: 1 }])` the same. Also `collate([{ a: true }, { a: 2 }])` is decided by the first item only (`bool`, so 2 becomes true) and `padSequences` of NaN is `float64` (fine).
what: silent corruption of large ids, timestamps (epoch milliseconds are 1.7e12) and counts, as the dtype is inferred from values and not from range. The fix is the inference, not the cast.
fix: choose `int32` only when `|v| <= 2^31 - 1` for all values (else `float64`), and decide `bool` for every item, not the head; a test with 3e9 and with epoch milliseconds.
size: S (<1h)
risk: low
preserves behaviour: yes except for the overflow cases
changes public surface: no

### F4: `table()` infers a column's type from its first non-null value, and `xyClasses` casts NaN and fractions into classes

kind: consistency
evidence: `data/tensors.ts:197-225` (`toColumn`: `const first = c.find((v) => v !== null)`, then a boolean, string or numeric branch; the rest of the column is never checked); probe `d3`: `table({ a: [1, 'x', 3] })` is `numeric` with values `[1, null, 3]` (the string became NaN), `['x', 2, 3]` is `categorical` with levels `[2, 3, 'x']` (numbers as levels), `[true, 2]` is `boolean` `[1, 1]`; `data/tensors.ts:92-` `xyClasses` goes through `concrete(y, 'int32')` (`schema/_coerce.ts:6-10`, `cast`): `xyClasses(x, [NaN, 0])` has labels `[0, 0]`, `[1.5, 0]` has `[1, 0]`. Negative labels are kept (a negative label is the "unlabelled" sentinel of `semiSupervisedSplit`, `modifiers/shift.ts`, and `classCounts` skips `c >= 0`), which is by design but written nowhere in `schema/`.
what: the schema constructors are the validation point of the dataset kinds (CLAUDE.md: "schema coercion"), and they coerce instead of failing. A label column with a NaN is a normal outcome of a join or a CSV read; it becomes class 0.
fix: `toColumn` checks every non-null value against the inferred type (`value` error naming column and row); `xyClasses`/`xyMultilabel` check integrality and finiteness of labels before the cast (allow the documented -1); document the sentinel in `schema/tabular.ts`.
size: S (<1h)
risk: low
preserves behaviour: yes for well-typed input
changes public surface: no

### F5: `resolveKernel` with `gamma: 'scale'` computes the variance as E[x^2] - E[x]^2, which is wrong for offset data

kind: consistency
evidence: `kernels/named.ts:48-58` (`variance = s2 / values.length - (s / values.length) ** 2`, `gamma = variance > 0 ? 1 / (d * variance) : 1`); probe `d5`: 50 rows around 1e8 with spread 7 give `gamma = 1` (the constant-data fallback) where `1 / (d Var(X)) = 0.1529`. Every SVM, kernel ridge and one-class estimator reaches this through `kernelSpace`/`resolveKernel` (10 call sites, e.g. `svm/classifier.ts:497`).
what: unstandardised features with a large offset (years, timestamps, prices) make the single-pass formula cancel to zero or a negative number, and the code answers with the fallback 1 without a sign. It is also a local copy of a variance (the rule: value-level statistics come from tensor ops or `base`).
fix: two-pass or Welford (`base` has `quantileOf`; the tensor `variance` op exists), `variance > 0` as before; one test with offset 1e8.
size: S (<1h)
risk: low
preserves behaviour: no (the fitted gamma changes for offset data, towards the right one)
changes public surface: no

### F6: `gram` of a single point returns a scalar, and with a jitter it returns `[]`

kind: consistency
evidence: `kernels/kernel.ts:549-553` (`const k = kernel.k(x); ... add(k, mul(eye(k.shape[0] ?? 0), jitter))`); `kernel.k` of `[d]` drops the axis (`kernel.ts:174-176`); probe `k2`: `gram(rbf, [1, 2])` is the scalar `1`, `gram(rbf, [1, 2], { jitter: 0.1 })` is `[]` (shape `[0, 0]`); the same for a scalar input. The documented return is "`[n, n]`". Also `jitter: NaN` makes an all-NaN matrix and `jitter: -1` is accepted (`:551`).
what: a GP with one training point (the first step of an active-learning loop, a prior draw) gets an empty covariance. `gram` is the contract's matrix entry point and the shape rule is not honoured at n = 1.
fix: `gram` calls `asRows` and `k(rows)` so the result is always `[n, n]` (and `[0, 0]` for n = 0, which already works); `jitter` must be finite and non-negative.
size: S (<1h)
risk: low (only n = 1 changes)
preserves behaviour: no (scalar becomes `[1, 1]`)
changes public surface: no

### F7: `trainTestSplit` splits in row order unless a key is given, is not stratified by default, and knows nothing of groups, queries, episodes or time

kind: consistency
evidence: `data/rows.ts:173-195` (`options.key === undefined ? Array.from({ length: n }, (_, i) => i) : permutation(...)`); `SplitOptions.key` is optional and `stratify` needs one (`:182-183`); probe `d1`: on labels `[0 x 6, 1 x 6]` with no key the test set is `[1, 1, 1]` and the train set has all of class 0. `take`/`trainTestSplit` split by `perRowFields` (`schema/common.ts:100-125`): for `queries` that is the documents (`x`, `y`, `queries`) so one query lands in both parts, for `interactions` the (user, item) pairs, for `trajectories` the transitions (an episode in both parts), for `timeseries` and `events` a random permutation of the times (future in train, past in test). There is no `groups` option and no temporal split in `data` (`learn/model-selection/blocked-splitters.ts` and `splitters.ts` have `groupKFold`, `timeSeriesSplit`, but on indices, not datasets).
what: the leakage-safe path exists in `learn` and is not reachable from the dataset splitters, and the default of `data` (ordered, no shuffle) differs from `train_test_split` (shuffles). For the iid kinds (`regression`, `classification`, `multilabel`, `multioutput`, `table`) the key-less default is a footgun on sorted data; for the grouped and ordered kinds a split is wrong by construction and silent.
fix: (1) `key` required unless `shuffle: false` is passed (sklearn's `shuffle`); (2) refuse the kinds that are not iid (`queries`, `interactions`, `trajectories`, `timeseries`, `events`, `sequence`, `graph`) unless the caller gives `groups` (a per-row group id: split whole groups, deterministic with the key) or `ordered: true`; (3) a warning sentence in the TSDoc of `trainTestSplit`/`trainValTestSplit`/`stratifiedSplit`.
size: M (1-3h)
risk: medium (callers that split ordered or grouped kinds start to throw; today only `data` tests and `modifiers/shift.ts` call these)
preserves behaviour: no
changes public surface: no (options added)

### F8: `concatRows` keeps the first part's metadata and does not check that the parts agree

kind: consistency
evidence: `data/rows.ts:307-329` (`const out = { ...first }` then only `perRowFields` are concatenated); probe `d1`: stacking `xyClasses(.., { classes: ['a', 'b'] })` with `xyClasses(.., { classes: ['x', 'y', 'z'] })` gives `classes = ['a', 'b']`, so the second part's labels now index a different list; `featureNames` `['a']` and `['b']` keep `['a']`. The table columns do check that categorical levels agree (`_columns.ts:49-53`), so the check exists for one kind only.
what: stacking datasets is how folds, augmentations and federated parts are put together; labels that mean different classes in the two halves are a silent mislabelling.
fix: compare `classes`, `featureNames`, `targetNames`, `sensitive`, `labels` (and the per-kind vocabularies) with `JSON.stringify`/length and throw `value` on a disagreement, as `_columns.ts` does for levels; take `meta` from the first part and say so.
size: S (<1h)
risk: low
preserves behaviour: yes for consistent parts
changes public surface: no

### F9: the exposition record contract: lengths are checked in all 61 kinds, finiteness in 7 files, nothing is copied, and the list of kinds is a hand copy

kind: consistency
evidence: every kind has a constructor `(record: Omit<R, 'kind'>) => R` that calls `sameLength`/`rowsOf` (`exposition/_check.ts`) and returns `{ kind, ...record }` (e.g. `curves.ts:45-56`, `thresholds.ts:79-94`); `isFinite` appears in 7 of 54 files (`complex-plane.ts`, `equation-fit.ts`, `fairness.ts`, `fields.ts`, `landscape-slice.ts`, `stability-region.ts`, `transcript.ts`); `concepts.ts`, `diagrams.ts`, `scene-steps.ts` and `rules.ts` check one or two things each (3-5 lines); probe `e1`: `curveSet` with `y: [NaN, 1]` and `x: [0, Infinity]` is accepted and JSON-serialises to `[null, 1]` and `[0, null]`; `curveSet`'s `series[0].y` shows a later `y[0] = 99` of the caller's array (the record spreads its input, nothing is copied); `thresholdSweep` accepts a NaN threshold (the "strictly ascending" test `!(a > b)` catches it only between neighbours; with one threshold it passes); `isExposition` (`record.ts:154-156`) tests only `kind` against `KINDS` (`record.ts:171-231`), a `Set<ExpositionKind>` written out by hand: TypeScript checks every entry is a kind, not that every kind is an entry (the list is complete today: 61 and 61).
what: the contract the render package relies on ("a malformed record fails where it is made") holds for lengths and not for values; NaN is sometimes legitimate (a missing rate) and sometimes a bug, and the constructor cannot tell. A record deserialised from the tape or a file is never re-checked, since `isExposition` is only a tag test.
fix: (1) `finite(kind, field, xs)` and `copy` helpers in `_check.ts` (shared with F13); constructors state which fields may hold NaN (a `nullable` list) and reject the rest; (2) `KINDS` as `const KINDS: Record<ExpositionKind, true>` so a new kind that is not listed fails to compile, as render's registry does; (3) one contract test over all kinds (F19).
size: M (1-3h for the helpers and the 61 call sites; most are one line)
risk: low-medium (strictness could reject a record a method produces today; the contract test finds them)
preserves behaviour: no for non-finite input
changes public surface: no

### F10: exposition fairness counts an empty group as a group with all rates 0, and the "difference" gaps include it

kind: consistency
evidence: `exposition/fairness.ts:172` (`ratio = (a, b) => (b === 0 ? 0 : a / b)`), `:186-199` (`tableRates`), `:201-203` (`spread`), `:220-238` (`groupDisparities`); probe `e1`: groups with tables `[[40,10],[10,40]]`, `[[0,0],[0,0]]` (empty) and `[[50,0],[0,0]]` (no positives) give accuracy difference 1, equal-opportunity difference 0.8, predictive-parity difference 0.8, disparate impact 0, and the empty group's rates are all 0 (`selectionRate`, `truePositiveRate`, `precision`, `accuracy`). `subgroupReport` has a `minSize` for this reason (`subgroups.ts:41`) and `groupReport` has none.
what: the same "rate with no denominator" decision as `metrics` F13 (precision of no predictions) is taken a third way here (0, where metrics returns NaN, 0 or throws), and it flows into headline numbers a fairness page shows. A group with no positives has an undefined true-positive rate, not 0; the gap should skip it (or the record should say it was skipped).
fix: `tableRates` returns `NaN` for an undefined rate (the record already allows `Number.NaN` in `subgroupGrid`), `spread`/`min`/`max` skip NaN and empty groups, and `groupReport` takes the `minSize` rule of `subgroupReport`; one test with an empty group and one without positives.
size: S (<1h)
risk: low-medium (the numbers of a report with an empty group change; views must tolerate NaN, they already do for the grid)
preserves behaviour: no for degenerate groups
changes public surface: no

### F11: every stationary kernel builds distances from the `[m, n, d]` difference tensor: a 1500 x 20 Gram matrix takes 2.65 s

kind: hot-path
evidence: `kernels/stationary.ts:46-58` (`scaledSquared`/`scaledDistance` via `cdist(..., 'sqeuclidean')`), `numerics/geometry/distances.ts:275-287` (`cdist` reshapes to `[m, 1, d]` and `[1, n, d]` and reduces the broadcast difference); probe `k5`: `gram(rbfKernel(), x)` n = 1000, d = 10 takes 522 ms, n = 1500, d = 20 takes 2,652 ms against 16 ms and 64 ms for `linearKernel` (a matmul) on the same points; probe `k4`: n = 400, d = 5: rbf 51 ms, matern 79 ms, general-nu matern 136 ms, spectral mixture (Q = 4, n = 200) 138 ms, `neuralTangentKernel` depth 3 340 ms. Memory is O(m n d) per intermediate (1500 x 1500 x 20 x 8 B = 360 MB).
what: GP regression and kernel ridge are the main users (`gp` 17 files, `kernel-methods` 22 files), and they call the Gram matrix once per marginal-likelihood evaluation; a 33x gap to the matmul identity is the dominant cost. The difference form is exact at coincident points, which is why it is safe; the identity `|x|^2 + |y|^2 - 2 x.y` needs a clamp at 0 and care for the derivative, and `_safe.ts` already holds the "double where" idiom for that.
fix: a `squaredDistances(x, y)` in `kernels/_safe.ts` (norms + `matmul` + `max(., 0)`, exact zero on the diagonal for the Gram case through the `unitDiagonal`-style `where(eye)`), used by `scaledSquared` and `scaledDistance`; keep `cdist` for the non-Euclidean kernels (laplacian, chi2). Check against the existing golden and property tests and `gramGradient` at coincident points.
size: M (1-3h)
risk: medium (values change at 1e-8 relative for nearly coincident points; the gradients need the same care as `cdist`'s)
preserves behaviour: no (within rounding)
changes public surface: no

### F12: hyperparameter validation in `defineKernel` admits 0, Infinity and NaN settings

kind: consistency
evidence: `kernels/kernel.ts:126-135` (`values.some((v) => v < 0 || Number.isNaN(v))` throws; `values.every((v) => v === 0)` makes the constraint `fixed`); probes `k2`/`k3`: `rbfKernel({ lengthScale: 0 })` is accepted and its Gram matrix is all NaN (0/0 on the diagonal); `lengthScale: Infinity` gives the constant 1 matrix; `periodicKernel({ period: 0 })` all NaN; `rationalQuadraticKernel({ alpha: 0 })` is the constant 1 (the wrong limit: alpha -> 0 is not a kernel); `powerKernel(k, NaN)` is accepted (`kernel.ts:262`, no check of the exponent) and a non-integer exponent of a kernel with negative entries (`linear`) gives NaN; `scaleKernel(k, 0)` and `lengthScale: 0` become `fixed` (so `theta` skips them) with no message. `polynomialKernel`, `arcCosineKernel` and `maternKernel` do validate their settings, so the pattern exists and is used unevenly (same finding as foundations F9/F30 for `space` and dists F9).
what: "positive" is documented as strictly positive for length scales and periods but enforced as non-negative, and the zero case is turned into `fixed` so that `log(0)` never appears, which also hides the error. Validation only sees concrete tensors (`isTensor(leaf)`), by design for traced parameters.
fix: a per-parameter `lower` bound in the spec (`strictlyPositive` for length scales, periods, alpha; `nonNegative` for variances, noise levels), `Infinity` rejected for `positive`, `powerKernel` checks `Number.isFinite(exponent)`; the zero-to-fixed rule stays for variances only (white noise 0, constant 0), where it is meaningful.
size: S (<1h)
risk: low
preserves behaviour: yes for valid input (a zero length scale throws instead of producing NaN)
changes public surface: no

### F13: no scene builder checks anything: sizes, finiteness, ids, edge endpoints, `columns`

kind: consistency
evidence: `scene/build.ts:43-56` (`scene` returns its arguments), `:67-` (`rect`, `circle`, `path`, ... likewise), `scene/structure.ts:105-120` (`structure`), `scene/geometry.ts:267-275` (`grid`), `scene/timeline.ts:51-57`; probe `s1`/`s2`: `scene(NaN, -5, [])` (JSON `"width":null`), `rect(0, 0, -10, 5)`, `circle` with `r = -1`, `path` with a NaN point, `group` with `scale: 0` are all built; `structure('g', [{id:'a'},{id:'a'}], [{from:'a', to:'zzz'}])` (duplicate id, dangling edge) is built; `grid(scenes, 0, [10, 10])` has `height: NaN`; `bounds` of a NaN rect is NaN; `fit` of an empty or zero-size set returns a group at scale 1; `timeline` with `fps: -1` and a NaN step sorts with an inconsistent comparator (`timeline.ts:56`). The exceptions are `camera` (distance, fov, extent: `projection.ts:43-60`), `face3d` (3 vertices), `toPixels`, `invertTransform` (zero scale) and `mergeStructures` (prefix count). `parsePaint` throws on a bad colour only when rasterising (`s1`: `rasterise ... '#zzz'` throws `value`), the SVG path draws the string as is.
what: the Scene is the contract with render, and it is the only record-like type with no gate; a NaN width becomes `width="NaN"` in SVG and `null` in JSON, and a dangling edge is discovered by the layout code of render, far from where it was made. The cost of checking at the builders is small (a number compare per field) next to the building of the node.
fix: share the helpers of F9: `finite` for coordinates and sizes, `size >= 0` for width/height/radius, unique ids and known endpoints in `structure()` (O(n) with a set), `columns >= 1` in `grid`, `fps > 0` and finite steps in `timeline`; plus an exported `validateScene(scene)` for scenes assembled by hand or deserialised (one recursive walk), which render calls at its entry points.
size: M (1-3h)
risk: low-medium (strictness; an environment that draws with width 0 would throw, so check `environments`/`worlds` callers with the examples)
preserves behaviour: no for invalid input
changes public surface: yes (`validateScene` added)

### F14: camera degeneracies: `fitCamera` of coincident points, a face or segment with one vertex behind the eye, NaN points, unclamped elevation

kind: consistency
evidence: `scene/projection.ts:109-127` (`extent = margin * Math.max(1e-9, hypot(...))`, `distance = extent / sin(fov/2)`); probe `s1`: `fitCamera(cam, scene3d([point3d([1,1,1])]))` returns `distance: 3.7e-9`, `extent: 1.1e-9`, a camera that places the eye nanometres from the point (every other point is then behind it); `projectPoint` returns `undefined` when `depth <= 1e-9 * distance` (`:173`) and `projectItem` drops a segment, polyline piece or face if any vertex is undefined (`:223-260`), so a segment from 0 to 100 along the view axis with `distance: 3` disappears (probe: 0 children) where a clipped segment would show its visible part; NaN coordinates give `{x: NaN, y: NaN, depth: NaN}` and enter the depth sort (`:285`, `b.depth - a.depth` with NaN is an inconsistent comparator); `camera({ elevation: 200 })` and `azimuth: NaN` are accepted (`orbit` clamps elevation to +-90, `camera` does not); `fitCamera` ignores a NaN vertex because `v < lo[k]` is false for NaN (silent, fine).
what: single-point scenes (a first sample, a one-point trajectory) are the common case that makes `fitCamera` degenerate; the clipping is a visual rather than numerical defect but a 3-D axis or a long arrow seen from near is exactly where it shows.
fix: `fitCamera` with a floor on `extent` relative to the cam (`max(extent, 1)` when the sphere radius is 0, or keep the camera's own distance); clip segments against the near plane (one parametric intersection) and drop only faces with all vertices behind; reject non-finite points in `scene3d` builders (F13); `camera` clamps or rejects elevation outside [-90, 90].
size: S (<1h) for the floor and the validation, M for the near-plane clip
risk: low
preserves behaviour: no for the degenerate cases
changes public surface: no

### F15: `clusterAgreement` matches clusters greedily and does not check lengths

kind: consistency
evidence: `data/truth.ts:227-250` (pairs sorted by count, each class and cluster used once); probe `d4`: truth `[0,0,0,0,0,1,1]` against labels `[0,0,0,1,1,0,0]`: greedy takes (0,0) = 3 and leaves 1 unmatched, answer 0.4286; the best assignment (0 -> 1, 1 -> 0) matches 4 rows, 0.5714. `clusterAgreement(truth, [0, 1])` for 7 truth rows answers 0.4286 with no error (`b[i]` is `undefined` for `i >= 2`, key `"0,undefined"`).
what: the function reports an agreement with the planted clusters (the truth every cluster generator carries) and underestimates it whenever the greedy choice is not the optimal one; `optim/assignment.ts` has `linearSumAssignment` and `metrics` has the ARI family for exactly this.
fix: use `linearSumAssignment` on the (negated) contingency table, check `labels.length === a.length`; or document it as a lower bound and point to ARI. Pick one.
size: S (<1h)
risk: low
preserves behaviour: no (values go up in the cases above)
changes public surface: no

### F16: NaN, Infinity and empty input: what each of the four domains does (the cross-cutting entry)

kind: consistency
evidence: the same defect class as foundations F3-F7, numerics F11, optim F5/F16, dists F2/F8/F13: no policy, and the idiom behind it is `where(x > 0, f(x), edge)` / `x < 0 || NaN` / `b === 0 ? 0 : ...` / a typed-array cast. Here:
- kernels: NaN and Infinity in the points propagate to the affected entries only (`k2`: `[[NaN, NaN], [NaN, 1]]`, correct); NaN hyperparameters are rejected, Infinity and 0 are not (F12); empty input is fine (`gram` of 0 rows is `[0, 0]`); degenerate statistics return NaN without a message: `hsic([[1]], [[1]])` (n = 1, `(n-1)^2 = 0`), `hsic(.., { normalise: true })` of a constant Gram, `kernelAlignment` of a zero matrix (`matrices.ts:142-156`); `additiveChi2Sampler` maps NaN to 0 (F1); `chi2Kernel`/`histogramIntersectionKernel` accept negative features and give 1 or 0 (`structured.ts:138-165`, `:251`; `additiveChi2Sampler` throws `domain` for the same input, `expansion.ts:326`).
- data: NaN in features is legal (missing) and handled well by `describe`/`columnStats` (counted as missing, probe `d1`); NaN in labels, indices and sizes is cast (F2, F4); empty datasets throw a `value` error from the split but `bootstrap` and `randomSampler(0, replacement)` leak messages of `categorical` and `randint` (F26); `describe` of 0 rows works.
- exposition: NaN is stored (F9), and the fairness ratio is 0 not NaN (F10); `sweepIndex(NaN)` returns the last index; `sceneStepAt` of no steps returns `undefined`.
- scene: NaN becomes `null` in JSON and `NaN` in SVG; `rasterise` skips shapes whose mask is empty (`maskFor` with NaN bounds returns `undefined`) and `samples: NaN` yields `Math.max(1, NaN) = NaN`, an empty buffer (`raster.ts:480`).
  what: the library has no stated rule, and each domain is somewhere on a line from "propagate NaN" (kernels, elementwise) through "cast to a number" (data) to "replace by 0" (exposition). The decision for these four is cheap: kernels propagate (document it, add the `hsic` guards), data rejects at the schema boundary (F2, F4), records and scenes reject non-finite unless the field is declared `nullable` (F9, F13).
  fix: write the rule once (suggested: "a function that returns a number may return NaN for a degenerate statistic and says so in `@returns`; a constructor of data, records or scenes rejects non-finite values unless declared nullable; sizes and indices are validated integers") in `docs/structure.md` section 2 and apply it through F1-F4, F9, F10, F13.
  size: M (the rule is 1h; the applications are the entries above)
  risk: low
  preserves behaviour: no for degenerate input
  changes public surface: no

### F17: the failure convention: the same condition has three error codes, and inner calls leak their messages

kind: consistency
evidence: error codes by use (grep of `new OmaiError('<code>'` in non-test files): kernels `shape` 19, `domain` 13, `value` 7, `not-implemented` 2; data `value` 80, `shape` 14, `schema` 5, `not-implemented` 4; exposition `value` 53, `shape` 41; scene `value` 9, `shape` 6, `singular` 1. A parameter outside its range is `domain` in `kernels/kernel.ts:132`, `dot-product.ts` (degree, order, depth) and `stationary.ts` (nu, power), and `value` in `data/` (`streams.ts:184`, `rows.ts:94`) and `scene/projection.ts:46-52`. Messages from inner calls: `weightedRandomSampler` with a zero weight total reports `categorical: weights must be non-negative...`, `randomSampler(0, key, { replacement: true })` reports `randint: need high > low, got [0, 0)`, `onDimensions(k, [5]).k(x)` reports `take: index 5 out of range for length 2`, `onDimensions(k, [])` `reshape: cannot view [3, 0] as [-1, 1, 0]`, `bootstrap` of an empty dataset `categorical: ...`. Raw platform errors: `parseNpy` of a truncated file (`Offset is outside the bounds of the DataView`), `bootstrap` with `size: -1` and `inputDim: -1` (`Invalid typed array length`), `untar`'s `Number.parseInt` NaN size ends the loop silently.
what: the dists review F20 found `domain` against `value` and inner-call names in messages in `dists`/`stats`/`metrics`; it is the same here, and the code a caller catches differs for the same cause. Not a runtime problem; a documentation and test one (`@throws` lists `value` in some of these and `domain` in others).
fix: one table in `base/errors` of when each code is used (`domain` = a number outside its mathematical range, `value` = a bad choice or combination, `shape` = dimensions, `schema` = parsing data), applied by a sweep of the 19 `domain` and the `value` sites that mean `domain`; validate the arguments in the public function so the message names it (F1, F2, F26).
size: M (1-3h for the sweep)
risk: low (code strings change; `toThrow(/message/)` tests may name them)
preserves behaviour: no (error codes)
changes public surface: no

### F18: the `Kernel` contract promises positive semi-definiteness and has no way to say when it does not hold

kind: consistency
evidence: `kernels/kernel.ts:50-52` ("A positive semi-definite kernel with its hyperparameters as data"); probe `k1`: minimum eigenvalue of an 8-point Gram matrix: `sigmoid` -0.93, -0.90, -1.12 (d = 1, 2, 3), `periodic` d = 2 -0.24 and d = 3 -0.73, `cosine` d = 2 -1.41 and d = 3 -1.95; all other 29 kernels at or above -1.4e-13 (the zero-scale linear, cosine-similarity and `arc*` ones at rounding level). The prose says so in three places (`stationary.ts:274-276` for periodic, `dot-product.ts:150` for sigmoid; `cosineKernel` says nothing about dimension) and `stationary.property.test.ts:64-67` encodes the "PSD in one dimension" list by name; `Kernel.stationary` is set by 20 kernels and read by none outside `kernels/` (grep); `ard`, `formatKernel` and `spectralFrequencies` dispatch on the `name` string (`kernel.ts:386-401`, `:627-650`, `expansion.ts:110`), so a user kernel named `'scale'` or `'sum'` is treated as the library's.
what: consumers (`gp`, `svm`, `kernel-methods`) factorise a Gram matrix with a jitter and have no way to ask whether that is justified. A `psd` flag would let `gp` warn or add the jitter that makes a non-PSD Gram usable, and would let the property test iterate the catalogue instead of naming exceptions.
fix: `psd: 'always' | 'conditional' | 'never'` (and for the conditional ones a `psdDimension?: number`) on `KernelSpec`/`Kernel`; combinators compute it from their parts (sum and product of PSD: PSD; power with an integer exponent: PSD; otherwise `conditional`); drop the unread `stationary` or make `randomFourierFeatures` use it instead of `name`; the combinator dispatch moves to a `kind` field of the spec. The property tests then become one table over all kernels (F21).
size: M (1-3h)
risk: low (additive fields)
preserves behaviour: yes
changes public surface: yes (fields added to `Kernel` and `KernelSpec`)

### F19: test gap: 12 of 54 exposition files have 0% statement coverage, and no test checks the contract of the record kinds

kind: test-gap
evidence: coverage (Phase 0, engine suite): `exposition/subgroups.ts` (43 statements, 12 functions), `agents.ts`, `concepts.ts`, `data-bias.ts`, `diagrams.ts`, `embeddings.ts`, `epicycles.ts`, `examples.ts` (32 statements, 6 functions), `explanation-quality.ts`, `frontiers.ts`, `neighbours.ts`, `sequences.ts` all at 0%; `rules.ts` 44% (3 of 5 functions never run), `surfaces.ts`, `fields.ts`, `stability-region.ts` (84%, 2 functions), `trust.ts` 88%; domain total 76.0% of statements and 68.6% of branches, the lowest core domain. Tests exist per record for 14 kinds (`record.test.ts`, `attributions.test.ts`, `fairness.test.ts`, ...), none for the contract as a whole. Caveat: render's tests and the `@example` blocks (run by `scripts/examples.ts`) may reach some; the engine suite does not.
what: the contract is the same for all 61 kinds (build, check lengths, tag), so it is testable as a table.
fix: `exposition/records.contract.test.ts`: a table `{ kind, make: () => valid record, bad: () => [mutations that must throw] }` for the 61 kinds (the valid minimal records are the `@example`s already written: lift them), and three checks over the table: `isExposition(make())`, `JSON.parse(JSON.stringify(r))` equals `r` (no functions, typed arrays or undefined-vs-missing surprises) and each `bad()` throws `OmaiError` of `shape` or `value`. 3 tests (`it.each`) of ~180 rows. It would have caught F9/F10's non-finite cases once the rule exists and closes the 12 files. Offset: the 14 per-kind validation tests in `record.test.ts`/`fairness.test.ts`/`panels.test.ts` that only assert a throw can move into the table (about -6 tests).
size: M (1-3h; the rows are the work)
risk: low
preserves behaviour: yes
changes public surface: no

### F20: test gap: no table for the degenerate arguments of splits, samplers and batchers, and the readers of `archives.ts` run only on their happy path

kind: test-gap
evidence: coverage: `data/archives.ts` 14 of 28 functions never run (the `.npy` reader table for `f4`, `i8`, `i2`, `i1`, `u*`, `b1` at `:177-187`, and the Fortran-order branch at `:240-241`), 77.1% of statements; `data/batching.ts` 8 of 33 (`collate`'s bool branch `:155`, `padSequences` dtype branches `:209`, `sequenceLengths` for corpus, parallel and sequence-labelling `:246-258`), 79% of statements, 64% of branches; `data/_columns.ts` 3 (text column take, boolean and text concat); `data/rows.ts` 67% of branches; `data/modifiers/leaks.ts` 61%. The probes of this review (`d1`-`d3`, `d5`) found F2, F3, F4, F8, F15, F25, F26 with arguments that no test passes: NaN sizes, an index of NaN, a single-class dataset, an empty dataset, a rare class of one member, `batchSize: 2.5`, unsorted `boundaries`, mixed-type table columns, values above 2^31.
what: the data tests are example-based and cover the useful paths; what is missing is the boundary table, and the unrun readers are the ones most likely to be wrong (an `i8` read through `Number(BigInt)` loses precision above 2^53, no test says so).
fix: (1) `data/rows.contract.test.ts` with two properties over the split family (for each of `trainTestSplit`, `stratifiedSplit`, `trainValTestSplit`: parts are disjoint, their union is all rows, same key same result, a stratified part keeps class proportions within one row) and one table of invalid arguments (`it.each` of ~25 rows: each must throw `value`/`shape`, none may return); (2) one `.npy` table built with a tiny writer for 6 dtypes and both orders, one `parseCsv` table (quotes, CRLF, BOM, ragged, unterminated), one `untar`/`unzip` fixture; (3) `collate`/`padSequences` dtype table with 3e9 and epoch milliseconds. About +6 tests, offset by folding `data.test.ts` cases that repeat the happy path of the same functions (about -3).
size: M (1-3h)
risk: low
preserves behaviour: yes
changes public surface: no

### F21: test gap: kernels are at 99.7% of statements and 87% of branches, but the laws are tested in three files by name, and no test passes a degenerate input

kind: test-gap
evidence: `dot-product.property.test.ts:49,65`, `stationary.property.test.ts:64-86` and `structured.test.ts:109` each loop over their own kernels with `expectKernelLaws` and each carries its own exceptions list (`PSD_IN_ONE_DIMENSION`, sigmoid `{ psd: false }`); spectral mixture, arc-cosine and the neural tangent kernel are in the dot-product file, the chi2 family in the structured one; no test calls a kernel with a zero length scale, a NaN, a single point through `gram` with a jitter, or compares `gram(x)` with `k(x, x)` (probe `k1` did, all 32 agree except white noise by design); no test checks that a hyperparameter of 0 or Infinity fails (F12) or that `gram` returns `[n, n]` at n = 1 (F6). Branch coverage is 87% (the 13% missed is the validation branches).
what: the catalogue is the place where one table over all kernels pays: it is the same law for each.
fix: after F18 (the `psd` field), one `kernels/catalogue.test.ts`: a table of ~35 `[name, kernel, inputs]` rows and four properties (`symmetric and gram == k(x,x)` except white noise, `eigmin >= -1e-9 * trace` iff `psd !== 'never'` for d in {1, 2, 3}, `diag(x) == diag(gram)`, `finite gradient in theta and x at duplicate points`) plus one `it.each` of degenerate inputs (zero scale, n = 0, n = 1 with jitter, NaN row). 5 tests; fold the three loops into it (-3). Net +2.
size: M (1-3h)
risk: low
preserves behaviour: yes
changes public surface: no

### F22: test gap: scene has no test of invalid input, and the camera has no table of degenerate configurations

kind: test-gap
evidence: scene is at 97.0% of statements and 87.5% of branches (no function unrun), and `projection.property.test.ts`/`geometry.property.test.ts` test the laws (orbit, project/unproject, bounds); nothing passes `fitCamera` a single point or an empty scene, a point behind the camera with `cull`, NaN coordinates, `grid(.., 0, ..)`, a structure with a dangling edge, or `rasterise` with `samples: NaN`; probes `s1`, `s2` found F13 and F14 with six one-line calls.
what: the invariants of the drawing model (finite, non-negative sizes, ids unique) are untested because they are unenforced (F13).
fix: with F13/F14, one `scene/validation.test.ts`: a table `{ build: () => ..., throws: boolean }` of ~20 rows (builders on invalid input) and a `fitCamera`/`projectPoint` table of ~8 degenerate cameras (single point, empty, coincident, behind the eye, ortho). 2 tests.
size: S (<1h)
risk: low
preserves behaviour: yes
changes public surface: no

### F23: layering: `kernels/structured.ts` imports `graph` and `trace` for three graph kernels that only it uses, and the folder holds two kinds of "kernel"

kind: layering
evidence: `kernels/structured.ts:3-14,42` (`heatKernel`, `isGraph`, `laplacian`, `normalisedLaplacian`, `weisfeilerLehmanAlgorithm` from `graph`; `runAlgorithm` from `trace`); `diffusionKernel` (`:530`), `regularisedLaplacianKernel` (`:552`), `wlSubtreeFeatures` and `wlSubtreeKernel` (`:588`, `:627`) are used by no other file (grep of `lib/`: tests only); `heatKernel` already lives in `graph`; Phase 0 lists it as one of 6 core-to-constructed imports. Beyond layering: `spectrumKernel`, `subsequenceKernel`, `jaccardKernel`, `setKernel`, `diffusionKernel`, `regularisedLaplacianKernel`, `wlSubtreeKernel` return a Gram matrix (a `Tensor` or `Value`) and are not `Kernel` objects: they have no `params`, `withParams`, `diag` or `theta`, and take the data (strings, graphs) as arguments (`structured.ts:325,408,449,485,530,552,627`), while `whiteNoiseKernel`, `constantKernel`, `chi2Kernel` etc. in the same file are `Kernel`s.
what: the `kernels` domain depends on a constructed domain for the sake of three functions, and a reader of the folder cannot tell from the name `...Kernel` which contract a function follows. The graph ones are graph methods that happen to be PSD matrices.
fix: move `diffusionKernel`, `regularisedLaplacianKernel`, `wlSubtree*` to `graph/kernels.ts` (so `kernels` drops its `graph` and `trace` imports), and rename the string/set functions to `...Gram` (`spectrumGram`, `subsequenceGram`, `jaccardGram`, `setGram`) or put them in `kernels/structured-gram.ts` under that name; the `Kernel`-contract ones stay in `structured.ts`. Run `make api-surface-check --compare` as the reviewed break; `docs/structure.md` for the new file.
size: M (1-3h)
risk: low (moves and renames; the entry points change)
preserves behaviour: yes
changes public surface: yes

### F24: layering: by their imports `scene` and `exposition` sit right above `base`, not between `optim` and `losses`

kind: layering
evidence: imports counted per file (non-test): `scene` imports `base` only (9 files; the prose of `build.ts` mentions tensors, no import); `exposition` imports `base` (41) and `scene` (4: type imports of `Structure`, `StructureOverlay`, `Scene`); `data/truth.ts:12` imports the type `Structure` from `scene`; `kernels` imports `tensor`, `numerics`, `base`, `random`, `autodiff`, `space`, `context`, `graph`, `trace`. `docs/code-review/findings.md` orders the core `... optim < scene < exposition < losses ...` as a first guess and lists `autodiff/graph.ts -> scene` as an upward import; foundations F17 already found the code right and the rank wrong for `scene`; this adds `exposition`.
what: nothing imports upward; the ranking document is what is out of date. `make layers` checks cycles only, so the order has no effect, but the review's counts of "upward imports" are wrong by it.
fix: set the order in the plan to `base < scene < exposition < context < tensor ...` (or `scene` and `exposition` as leaves above `base`), regenerate the Phase 0 layering section; consider enforcing it in `make layers` (the cheap form: `scene` may import only `base`; `exposition` only `base` and `scene`).
size: S (<1h)
risk: low
preserves behaviour: yes
changes public surface: no

### F25: `parseCsv`, `untar` and `parseNpy` accept malformed input silently or fail with a platform error

kind: consistency
evidence: `data/archives.ts:133-190` (`parseCsv`), `:51-66` (`untar`), `:209-` (`parseNpy`); probe `d2`: `parseCsv('a,b\n"x,1\ny,2\n')` (unterminated quote) returns one row `["x,1\ny,2\n"]` with no error; ragged rows (`1,2,3` and `4` under two headers) are returned as they are; a UTF-8 BOM stays in the first column name (`"﻿a"`); `parseCsv('')` returns no columns; `parseNpy` of a file cut short throws `RangeError: Offset is outside the bounds of the DataView` (an `i8`/`u8` array is read through `Number(BigInt)` and loses precision above 2^53, `:171-172`); `untar` reads only entry types `0` and NUL (`:57-61`), so GNU long-name (`L`) and pax (`x`, `g`) headers, which `tar` writes for names over 100 bytes and for metadata, are skipped as non-files and the long name is lost (from reading; no fixture run), a NaN size parsed from a corrupt header ends the loop and returns the entries so far; `unzip` supports methods 0 and 8 only and has no zip64 (it throws `value` with the method number: fine).
what: the three readers are the entry of every remote dataset (`load`, checksum-pinned), where a correct hash makes truncation unlikely, and of user files, where it is not. The failure convention is a mix of silence and platform errors (F17).
fix: `parseCsv` throws `schema` for an unterminated quote and (option `strict`, default true) for ragged rows, strips a BOM; `parseNpy` checks `bytes.length >= offset + count * size` and throws `schema`; `untar` handles `L`/`x` records (or throws `not-implemented` on them, which is already better than losing a name) and rejects a NaN size.
size: S (<1h) for csv and npy, M for tar long names
risk: low
preserves behaviour: no for malformed input
changes public surface: no

### F26: batching and sampler conventions: three shuffle conventions, fractional batch sizes, unvalidated boundaries, inner messages

kind: consistency
evidence: `data/batching.ts:44-50` (`batches` with a key: `permutation(options.key, n)`, the same order on every call), `:70-74` (`epochBatches`: `foldIn(key, epoch)`), `:108-118` (`dataLoader`: `randomSampler(n, key)`, which also folds the epoch in, `samplers.ts:56-66`); probe `d1`: the same key gives `[[2,3,1],[5,0,4]]` from `batches`, and `[[0,1,2],[4,5,3]]` from both `epochBatches(.., 0, ..)` and `dataLoader(..).epoch(0)`. `batchSampler(sampler, 2.5)` is accepted (`samplers.ts:157`, `!(batchSize >= 1)`) and returns uneven batches `[[0,1],[2,3,4],[5,6]]` (probe); `bucketBy` with `boundaries: [10, 5]` (unsorted) puts length 11 in bucket 2 and nothing in bucket 1 (`batching.ts:301`, no sorted check); `weightedRandomSampler` takes negative or NaN weights through to `categorical`, whose error is raised (`d1`), `balancedWeights([])` returns `[]`; `randomSampler(5, key, { numSamples: NaN })` returns `[]`.
what: small, but they are the places a training loop is made, and the first one means that `batches` with a key and `dataLoader` with the same key do not shuffle the same way, so a run cannot be reproduced across the two APIs.
fix: one rule (`batches` uses `foldIn(key, epoch)` with `epoch ?? 0`, i.e. equal to the other two), integer `batchSize`, sorted boundaries, a `weights` check in `weightedRandomSampler` (finite, non-negative, positive total) with its own message.
size: S (<1h)
risk: low-medium (the order of `batches(data, { key })` changes, so seeds pinned in tests and recipes move)
preserves behaviour: no (shuffle order)
changes public surface: no

### F27: data: the size rule, a column-statistics helper and a logistic written locally; two stratified implementations; a name shared with `sampling`

kind: duplicate
evidence: the size rule `size < 1 ? Math.ceil(size * n) : Math.round(size)` at `data/rows.ts:91` and `data/modifiers/shift.ts:307` (private in the first, so the second re-wrote it); `data/modifiers/_shared.ts:128-153` `columnStats` (NaN-aware mean and population standard deviation per column, 25 lines, native arrays) and `:155-157` `logistic` (a sigmoid; Phase 0 counts `sigmoid` in 8 files); `columnStats` has a third relative in `data/tensors.ts:416-450` (`summarise`/`column`); `learn/model-selection/splitters.ts:677` `stratifiedShuffleSplit` and `:169` `stratifiedKFold` implement stratification again on indices, next to `data/rows.ts:111` `stratifiedIndices` (not read; the two cannot agree on tie-breaks of the leftover rows: `rows.ts:127` allots them by largest remainder like sklearn); `defaultNames` at `data/schema/_coerce.ts:43` and `sampling/target.ts:71` (Phase 0 "similar"; not the same function: a prefix + index list against a six-line variant).
what: the rule in CLAUDE.md is "no local copies of generic helpers"; these are small but they are what drifts. `semiSupervisedSplit` also draws its own shuffle (`shift.ts:309-314`, `uniform` + sort) next to `shuffleRows` and `permutation`.
fix: export `sizeOf` from `rows.ts` (as `splitSize`) and use it in `shift.ts`; `columnStats` and `describe`'s column pass call one function (the NaN-aware mean/std of the tensor ops or `base`); `logistic` from the `sigmoid` op; make `learn/model-selection` `stratifiedShuffleSplit` call `stratifiedIndices` (or record why not).
size: S (<1h) for the first three, M for the `learn` one
risk: low
preserves behaviour: yes
changes public surface: no

### F28: kernels and the neighbouring domains: a one-line `rowsOf`, `concrete` twice, window kernels written twice, three parallel kernel-name lists

kind: duplicate
evidence: `kernels/expansion.ts:59-61` `rowsOf` is `asRows` renamed (used 8 times in the file); `:67-69` `concrete` (`primalOf(promoteFloat(x)[0])`) is one of the five `concrete` of register F-041 (with `data/schema/_coerce.ts:6`); the window kernels (`gaussian`, `epanechnikov`, `uniform`, `tricube`, `triangular`) are a table in `stats/smoothing/smoothers.ts:213-225` (`SmoothingKernel`, `kernelWeight`) and again in `causal/discontinuity.ts:17-26` (`Kernel`, `kernelWeight`), and the second exports a type named `Kernel` (a string union) that collides with `kernels.Kernel` (the interface) and the `Kernel` of the register's F-013; the kernel names as three lists: `kernels/named.ts:~88` `NAMES` (`rbf`, `linear`, `poly`, `sigmoid`, `laplacian`), `matrices.ts:340-349` `PairwiseKernelMetric` (nine, with `additive_chi2` in snake case where everything else is camel case) and the `switch` in `pairwiseKernels` (`:375-405`); `kernelSpace` offers 5 of the 9, so `cosine` and `chi2` cannot be selected from a UI knob. Checked and not duplicates: `formatNumber` (`kernels/kernel.ts:607`, 3 lines of `toPrecision(4)`) against `tensor/format.ts:21` (dtype-aware, 10 lines), and `corners` (`scene/geometry.ts:18`, private, the four corners of a box) against `worlds/_topology.ts:109` (exported, the corners of a map cell): same name, different functions; `asRows` in `kernels/kernel.ts:112` against `distances/evaluate.ts` is register F-007 and was not re-read.
what: small, with one user-facing effect (the kernel choice list) and one naming trap (`Kernel`).
fix: delete the two aliases; one `windowKernels` table (`numerics/` or `stats/smoothing`) used by `causal`, and rename its type; derive `NAMES` from one table that also holds the `pairwiseKernels` constructor (`{ name, make(gamma, degree, coef0), usesGamma, usesDegree }`) so the list, the switch and the knobs cannot disagree; rename `additive_chi2` to `additiveChi2` (it is a string value in the surface: reviewed break).
size: M (1-3h)
risk: low
preserves behaviour: yes
changes public surface: yes (the `Kernel` type rename in `causal` and the snake-case name)

### F29: exposition re-implements confusion counts and rates, and the counting function is used by no one

kind: duplicate
evidence: `exposition/evaluation.ts:69-80` `confusionCounts(truth, predicted, classes)` against `metrics/classification.ts:545` `confusionMatrix`; `confusionCounts` is referenced only by `exposition/record.test.ts` (grep over `packages/` and `apps/`); `exposition/fairness.ts:172-199` `ratio`/`tableRates` (TPR, FPR, precision, accuracy, selection rate from a 2 x 2 table) against `metrics/_labels.ts:132` `ratio(a, b, zeroDivision)` and the rate family of `metrics/classification.ts`; `exposition/fairness.ts:201` `spread` (`Math.max(...values) - Math.min(...values)`, spread arguments: breaks at about 125,000 groups) beside `stats`' range.
what: exposition sits below `metrics` (F24), so it cannot import it, and has therefore grown a second, 0-for-undefined (F10) definition of the rates. The counting helper is dead, which shows the callers (constructed domains) compute their counts with `metrics`.
fix: remove `confusionCounts` (or move it to `metrics` if wanted); keep `tableRates` as the exposition-side rates of a table but make its undefined-rate convention the one chosen for `metrics` F13 (NaN), and say in its TSDoc why it is not `metrics`.
size: S (<1h)
risk: low
preserves behaviour: yes (removal of an unused export), no for the NaN change (F10)
changes public surface: yes (one export removed)

### F30: the scene data model: plain data except for pixels, a rasteriser that draws no text, and no walk that checks a whole scene

kind: consistency
evidence: `scene/types.ts:124-128` `Pixels.rgba: Uint8ClampedArray`: `JSON.stringify` of a scene with an image gives `{"0":0,"1":0,...}` objects (probe `s1`: `toPixels(...)` printed that way), while the trace codec writes `{ $typed: 'Uint8ClampedArray', data }` (`trace/serialise.ts:19,47-50`), so a scene survives the tape and not a plain `JSON.stringify`; `isScene` (`build.ts:372-374`), `isStructure` (`structure.ts:217`) and `isTimeline` (`timeline.ts:92`) test the `kind` tag only; `Paint` is a template-literal type (`#${string}`, `rgba(${string})`) that is checked at rasterising only (`paint.ts:85`, `parsePaint` throws `value`; SVG output writes the string); `scene/raster.ts:434-437` skips `text` and `tex` ("a headless rasteriser has no fonts"), and render's `animation/frames.ts:68` rasterises with it, so GIF and PNG frames made headless have no labels (the browser path through `svgToPng` has them); `Rgba` has two scales in the codebase (register F-019: `scene/paint.ts:13` 0 to 255 with alpha 0 to 1; `graphics/textures.ts` 0 to 1); `bounds` uses the anchor point of `text` and `tex` and ignores stroke width (`geometry.ts:46-50`), so `fit` can clip a label or a thick stroke at the box edge (from reading).
what: the model is complete for render (every kind is drawn: `svg.ts:59-110`, `SceneView.tsx:110-220`), immutable by `readonly` types only (no freeze, fine), and not strictly "plain data" because of the image bytes. None of this is a bug; what is missing is one place that says what a valid Scene is (F13) and what survives which serialiser.
fix: document in `scene/types.ts` that images are typed arrays and `sceneToJson`/`sceneFromJson` through the trace codec (or `Pixels` as base64 in a `data` field, a model change); a `Rasterise` doc line that text is not drawn; `bounds` with an optional `pad` for strokes (and text extents by `size * length` as an estimate).
size: S (<1h) for the docs, M for the helpers
risk: low
preserves behaviour: yes
changes public surface: yes if helpers are added

### F31: two models of "frames": `scene/timeline.ts` and the exposition step records disagree on order and validation

kind: consistency
evidence: `scene/timeline.ts:17-57` (`Frame` = a scene or a structure with an overlay at a `step`; `timeline()` sorts by `a.step - b.step`, silently, and accepts any `fps`); `exposition/scene-steps.ts:30-46` (`sceneSteps`: a scene per `step`, throws `value` when steps are out of order) and `exposition/structure-steps.ts` (structure with overlays per step, 111 lines); `StructureOverlay` is defined in `scene/timeline.ts:10` and imported by `exposition/diagrams.ts:7`; `sceneStepAt` (`scene-steps.ts:49`, `findLast ... ?? steps[0]`) and render's timeline playback each look up a frame by step. Probe `s2`: `timeline` with `fps: -1` and a NaN step is accepted.
what: the same idea (a sequence of scenes/structures indexed by step) is a scene-domain type for the headless animation path and an exposition record for the session path, with opposite handling of unordered steps (sort against throw) and no shared validation. Not read: how render converts one to the other.
fix: pick the exposition rule (throw on unordered or non-finite steps, `fps > 0`) for `timeline` too, with the validation in a helper shared with F13; consider building `sceneSteps` and `structureSteps` on `Timeline`'s `Frame` (a model change, not proposed without reading render's side).
size: S (<1h) for the validation; M for the shared model
risk: low (validation); medium (model)
preserves behaviour: no for unordered input (sorted today)
changes public surface: no

### F32: native arrays in `data`: 17 functions classified (5 in `rows.ts`, 3 in `samplers.ts`, 9 elsewhere)

kind: native-array
evidence: `signatures.mjs` (probe `sigmine.mjs`) lists, by file:
- indices (not a vector): `rows.ts` `take` (`indices: readonly number[] | Int32Array | Value`: already accepts a tensor), `bootstrap` (returns `indices: readonly number[]`), `stratifiedIndices` (labels in, `{train, test}` index lists out), `stratifiedSplit` (`labels?: readonly number[]`), `samplers.ts` `subsetRandomSampler(indices)`, `modifiers/shift.ts` `shiftedSplit` and `semiSupervisedSplit` (return `trainIndices`, `testIndices`, `labelledIndices`, ...).
- labels or lengths (not a vector): `rows.ts` `labelsOf(data): number[]` (the class of each row; `Array.from` of an int32 tensor, 50,000 rows copied each call), `batching.ts` `sequenceLengths(data): number[]`, `truth.ts` `clusterAgreement(truth, labels: Value | readonly number[])`, `padSequences(seqs: readonly (readonly number[] | Tensor)[])` and `schema/sequence.ts` `sequences`, `schema/corpus.ts` `sequenceLabelling` (ragged token sequences: rows of records, the right type under the rule).
- decoder (allowed, a primitive): `archives.ts` `parseNpy` returns `values: Float64Array` with a shape (the raw data from which `fromStorage` builds a tensor).
- public vector or matrix (migrate): `samplers.ts` `weightedRandomSampler(weights: readonly number[], ...)` (-> `TensorLike`; `categorical` accepts one already, the without-replacement branch reads `weights.map`), `balancedWeights(labels): number[]` (labels in, a weights vector out -> `Tensor<'float64'>`, which `weightedRandomSampler` then takes), `modifiers/labels.ts:53` `noiseTransition(k, rate, mode): number[][]` (a k x k stochastic matrix -> `Tensor`; it is documented with a nested-array `@example` and consumed at `:86`, `options.transition` is a `readonly (readonly number[])[]`).
  what: data's data is rows of tensors and the rule is mostly met (datasets hold tensors; `Dataset` fields are `Tensor`); the remaining native arrays are indices and labels, which are index vectors in the rule's sense. Three functions are real vectors/matrices.
  fix: migrate the three (`weightedRandomSampler` and `balancedWeights` together: accept `TensorLike`, return `Tensor<'float64'>`; `noiseTransition` returns a `Tensor` and `transition` accepts `TensorLike`); keep the index/label ones, and make `labelsOf` return the int32 tensor (`Tensor<'int32'>`) instead of copying to an array (the callers inside `data` read it as an array; not checked one by one).
  size: S (<1h) for the three, S for `labelsOf`
  risk: low (public signatures of 4 functions change)
  preserves behaviour: yes
  changes public surface: yes (4 signatures)

### F33: native arrays in `exposition`: 11 functions classified (8 record data, 3 computational)

kind: native-array
evidence: probe `sigmine.mjs`; `exposition/` records are plain JSON by contract (no typed arrays, no functions: grep of `Float64Array|Int32Array|Map<|Set<|=>` in field types found none), so `number[]` and `number[][]` are the record format, not a lapse of the rule. By function:
- record data (allowed): `fairness.ts` `tableRates(table: number[][])` and `groupReportAt(record, thresholds: number[])`, `evaluation.ts` `confusionCounts` (labels in, count matrix out: a record field), `equation-fit.ts` `fitResidual` (returns arrays for a record), `trust.ts` `setSizes`, `subgroups.ts` `subgroupGrid` (returns `values: number[][]`, `counts`), `fields.ts` `sampleVectorField(f, x: number[], y: number[])` and `landscape-slice.ts` `sampleLandscape(loss, x, y)` (grids of axis values plus a callback on two scalars: a sampler of a drawing).
- computational (could migrate): `examples.ts:101` `changedFeatures(a, b: readonly number[]): number[]` (two feature vectors in, indices out), `landscape-slice.ts:164` `sliceCoordinates(iterates: number[][], origin: number[], directions: number[][]): number[][]` (a projection of iterates on two directions: a matmul), `stability-region.ts:98` `explicitRungeKutta(a: number[][], b: number[])` (a Butcher tableau, constants of a method).
  what: the exposition rule is "plain, serialisable"; a `Tensor` in a record would break it. The computational three could take `TensorLike` and return arrays only where the result goes into a record.
  fix: leave the 8; `sliceCoordinates` -> `TensorLike` in, `number[][]` out (it is a matmul of `[k, d]` by `[d, 2]`); `changedFeatures` and `explicitRungeKutta` stay (constants and a comparison). Net one migration.
  size: S (<1h)
  risk: low
  preserves behaviour: yes
  changes public surface: yes (one signature)

### F34: native arrays in `kernels` (2) and `scene` (1): all allowed

kind: native-array
evidence: `kernels/kernel.ts:352` `onDimensions(kernel, dims: readonly number[])` (column indices: an index list); `kernels/named.ts:85-` `kernelMatrix(...)`: `Float64Array` (the Gram matrix flat, row-major) for the SMO solvers of `svm/*` (8 call sites: `svm/one-class.ts:130,230,250,257`, `regressor.ts:138,293`, `learn/semi-supervised.ts:930,1051,1329,1388`) which want a flat buffer; `scene/three-d.ts:207` `surface3d(x: number[], y: number[], z: number[][] | (x, y) => number)` (grid coordinates of a drawing builder). In bodies (not signatures): `Array.from(leaf.data)` in `kernel.ts:128`, `Array.from(rows.data).some(...)` in `expansion.ts:326,400` (a full copy to test a minimum), `theta()` builds a `number[]` then a `Float64Array` (`kernel.ts:~498`).
what: the kernels folder is where the rule allows raw arrays, and this folder uses tensors almost everywhere (a kernel is a `Value -> Value` function, so gradients reach hyperparameters and points: that is the design, and F11's gain is a reason to keep it). `kernelMatrix` is the one bridge, and calls `pairwiseKernels` (a full tensor graph) once per row at prediction time (`svm/one-class.ts:257` inside a per-row closure): a row costs an `asRows`, a kernel build and a `toFlat`.
fix: none for the signatures; replace the three `Array.from(...).some` by a min over the tensor (`min(rows) < 0`), and consider a `kernelRow(choice, resolved, x, support)` for the per-row prediction path once SVM prediction is profiled (not measured here).
size: S (<1h)
risk: low
preserves behaviour: yes
changes public surface: no

### F35: hot-path notes: `subStructure` is quadratic in the worst order, string kernels fill both triangles, `mmdSquared` linear loops in tensor ops

kind: hot-path
evidence: `scene/structure.ts:235-247` (`while (changed) { for (const e of s.edges) ... }`): descendants of the root of a 4,000-node chain take 2.2 ms with edges in order and 325 ms with the edges reversed (probe `s2`); `kernels/structured.ts:271-290` `pairwise` computes all m x n pairs for a Gram matrix (both triangles, the diagonal twice when normalising: `f(s, s)` is called for the diagonal and again for the norms): 3.3 ms for 100 strings with the spectrum kernel and 27 ms for 60 strings with the subsequence kernel of order 3 (probe `k4`), fine at these sizes, a 2x and a 4x factor at n = 2,000; `kernels/matrices.ts:233-247` `mmdSquared` with `estimator: 'linear'` runs `n / 2` iterations of 5 `sliceAxis`/`sum` tensor ops (2.6 ms for n = 200, linear in n, with large constants at n = 1e4); `kernels/kernel.ts:603` `gramGradient` uses `jacfwd` through the kernel: 21 ms for rbf at n = 100, 99 ms for a sum of three kernels, memory O(n^2 p). Measured fine: rasterising a 5,000-point polyline at width 2 on 800 x 600 takes 140 ms (each stroke piece is scanned over the mask's whole height, `raster.ts:317`, so cost grows with pieces x height: 4x for 20,000 points), 2,000 circles 20 ms, projecting 20,000 3-D points 9 ms, `mergeStructures` of two 4,000-node structures 5 ms, `batches` of 50,000 rows 22 ms, `padSequences` of 2,000 sequences 19 ms.
what: none is a problem at the sizes of the pages; they are the first things to look at if a larger input shows.
fix: `subStructure`: a queue over a precomputed adjacency map; `pairwise`: fill the upper triangle and mirror when `b === undefined`, and reuse the diagonal; `mmdSquared` linear: gather the pair entries with `take` and sum once; the rasteriser: cull each piece to its own bounding rows.
size: S each
risk: low
preserves behaviour: yes
changes public surface: no

### F36: `data/_columns.ts` has 16 casts per 100 lines because tensor `take` and `concat` lose the dtype

kind: rule
evidence: `data/_columns.ts:14-27,44-75` (`takeRows(c.values, ...) as Tensor<'float64'>`, `as Tensor<'int32'>`, `as Tensor<'bool'>`, `concat(parts.map(p => (p as typeof first).values), 0) as Tensor<'float64'>`, 14 casts in 98 lines); `tensor/ops` `take` and `concat` return `Value`/`Tensor` of the general dtype, so each caller that knows its dtype asserts it; Phase 0 lists the file as the third-densest in the core (16 per 100); `schema/_coerce.ts:6-10` and `batching.ts:218-228` (`typed as Float64Array`, `fromStorage(typed as Float64Array, ..., dtype as 'float64')`) repeat the pattern.
what: one cause, as in numerics F25 and optim F12: the dtype is in the type of the input and is dropped by the op. A typed overload `take<D>(x: Tensor<D>, ...): Tensor<D>` in `tensor/ops/indexing.ts` removes about 20 casts in `data` and some in `kernels`.
fix: dtype-preserving signatures for `take`, `concat`, `sliceAxis` when the input is a concrete `Tensor<D>`; then delete the casts. A foundations change (`tensor`), listed here because `data` is its main user.
size: M (1-3h, mostly checking the overloads against `isolatedDeclarations`)
risk: low (types only)
preserves behaviour: yes
changes public surface: yes (signature digest of 3 functions; the intended reviewed change)
