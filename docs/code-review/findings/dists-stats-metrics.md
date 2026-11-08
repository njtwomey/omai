# Dists, stats and metrics review: distributions, hypothesis tests and descriptive statistics, evaluation metrics

Read-only review of `packages/engine/src/lib/dists` (46 non-test files, about 13,000 lines), `lib/stats` (49 files, about
18,800) and `lib/metrics` (32 files, about 12,000). All paths below are relative to `packages/engine/src/lib/`. Probes
cited as "probe" are throwaway node scripts in
`/private/tmp/claude-501/-Users-nialltwomey-workspace-whitebox-js/d4034594-54d3-484e-a26a-331816a1f3f5/scratchpad/review/probe2/`
(`d1.ts` ... `d7.ts` distributions, `b1.ts`/`b2.ts` bijectors, `v1.ts`/`v2.ts` multivariate, `k1.ts` divergences, `s1.ts`...`s3.ts`,
`sp.ts`, `c1.ts`, `r1.ts` stats, `m1.ts`...`m7.ts` metrics, `h1.ts`...`h3.ts` timings, `unrun2.mjs`/`unrun3.mjs` coverage),
run one at a time at `nice -n 15` on small sizes against the source; nothing in the repository was changed and no test, tsc or
gate was run. Scipy, sklearn and torch behaviour quoted here is from memory, not from running them (no Python); treat it as an
expectation to confirm with the oracle before a fix. Coverage figures are Phase 0's (`coverage-final.json`, commit b09e46da);
line numbers are from the tree at commit 47df9bee. About 45 source files were opened (some in part).

## Summary

1. Healthy: **the numerical core of the common distributions is good.** Over 37 continuous families (probe `d1`, `d6`) and 12
   discrete ones (`d3`, `d7`) the density integrates to 1, `sf`, `logSf` and `logCdf` are accurate far into the upper tail
   (`normal.sf(30) = 4.9067e-198`, `gamma(2).sf(50)`, `gumbel.cdf(-4)`, `poisson.sf(60)`, `binomial.sf(40)` all to 4 digits), the
   discrete `ppf` brackets `p` everywhere, sample moments, `entropy` and `mean/variance` agree with quadrature (`d6`: only
   `beta`, the singular densities and the heavy-tailed ones differ), and `cdf(-Inf)`, `cdf(Inf)` are right for most. Stats: the
   p-value routines are fast and sound on ordinary input (a Mann-Whitney U on 2 x 2000 in 1.5 ms, Shapiro-Wilk on 2000 in 3 ms,
   permutation test 2000 resamples in 68 ms, BCa bootstrap 2000 x 200 in 79 ms), Welford/Pebay running moments and their `merge`
   are accurate at an offset of 1e9 (`r1`), `pValueFromTails` follows scipy, and `hoeffdingBound`, `markovBound` and the other
   concentration bounds validate their arguments. Metrics: the ten most-used metrics (accuracy, F1, confusion matrix, ROC AUC, AP, MSE, R2, log loss, ARI, Spearman) run in 11-60 ms at n = 1e5; sample-weight support is
   uniform where it exists (`columns`, `perOutputMetric`, the rate family). No `Math.random`, clock or `console` in the three
   folders; TSDoc is complete for all but 11 of 570 exported functions (F29).
2. Not healthy, 1: **no policy for NaN, Infinity and empty input, and the idiom behind it is `where(x > 0, f(x), edge)`**, which
   is false for NaN. `chdtr(k, NaN) = 0` and `chdtrc(k, NaN) = 1` (`numerics/special/cdf.ts:135,154`), so every chi-square and F
   based test answers `p = 1` for a NaN statistic (kruskal, fOneway, chisquare, jarqueBera, levene, ljungBox, chi2Contingency); the
   rank-based tests (`mannWhitneyU`, `wilcoxon`, `kruskal`, `ks2samp`) report ordinary p-values for samples containing NaN, and
   `spearman([1,2,NaN,4],[1,2,3,4])` and `kendallTau` return **1** (stats and metrics); distributions answer `cdf(NaN)` with 0,
   0.578, 0.5, 1 or NaN depending on the family (F2, F3, F8, F13).
3. Not healthy, 2: **`sampleWeight` is silently ignored by 33 of 99 numeric metrics** (cosine similarity, concordance, SNR/PSNR,
   Kendall, Spearman, calibration errors, ...) while the shared `MetricOptions` type accepts it (F1); there is no convention for
   the empty or degenerate input of a metric (156 metrics probed: 60 return NaN, 52 a number, 39 throw, with `informedness([]) =
   -1`, `ARI([]) = 1`, `maxError([]) = Infinity`; F13); the failure convention of the tests is five different shapes with no
   status field (F4).
4. Not healthy, 3: **sampling is 100 times slower than it needs to be for the gamma family and the count families**:
   `gamma`, `beta` (two draws), `chi2`, `studentT`, `dirichlet`, `F` cost 1.0 s per 1e5 draws against 8 ms for `normal`, and `poisson`,
   `binomial` 1.0 s against 25 ms for the batch sampler in `random` (F12; the dists-side face of foundations F2).
5. A handful of **small certain bugs** each have a one-line reproduction: `fligner` of constant groups returns `stat = 7, p = 0.008`
   (F3); `beta(3, 0.5).mode()` is 1.33, outside the support, `binomial(20, 1).mode()` is 21, `vonMises(loc = 3)` draws and a CDF
   outside its declared support with `cdf(-3) = -0.43` (F6); discrete KL and `categorical.entropy` are NaN when a category has
   probability 0 (the masked-logits case of RL; F7); `correlationCholesky.forward` is NaN for `|u| >= 20` and `stickBreaking` does
   not round-trip from `u = 20` (F11); `shortTimeObjectiveIntelligibility` never returns when `fs` is NaN or missing (F14).
6. Parameter validation is uneven in the way foundations F9/F30 found: `uniform(0, -1)`, `triangular` with the mode outside, `normal(0,
   Infinity)`, `gamma(Infinity)`, `geometric(0)`, `negativeBinomial(3, 1)` and a non-symmetric covariance are all accepted (F9).
7. Layering: no cycle. `metrics/{audio,image-quality,text}` import `signal` and `text`, `stats/tests/independence` imports one function
   of `information`, and `weightedPercentile`, a generic statistic, lives in `metrics` and is used by `trees` and `ensembles` (F19).
   Duplicates: a `spec` builder in 18 metrics files, bootstrap/BCa/percentile/Wilson written again in `metrics/foundations.ts`
   next to `stats/resampling.ts` and `stats/tests/proportions.ts` (F16, F17, F18).
8. Native arrays: dists 6 functions, none to migrate (all shapes and indices); stats 28 (9 index or level, 6 kernel, 13 migrate);
   metrics 21 (5 kernel, 5 index, 11 migrate), plus `ConfusionMatrix.matrix` and other record fields the script cannot see (F23, F24).
9. The three highest-value fixes: (a) **F1 + F2 + F13 together**: one rule for NaN and empty input implemented at the four choke
   points (`stats/_util.ts#sample`, `ranksOf`, the metrics readers `columns`/`encodeLabels`/`flat`, the `where(x > 0, ...)` special
   functions), and `sampleWeight` either honoured or rejected by every metric; (b) **F5 + F25 + F26 + F27**, three table-driven
   contract tests (about 45 distribution rows x 10 generic checks, one degenerate-input table for the tests, one for the metrics;
   +24 test cases in all) that would have caught F2, F3, F6, F7, F8 and F11 and close the 168 never-run `mode`/`entropy`/`sf`/`isf`
   closures of dists; (c) **F12** (batch sampling for the gamma family and the count families, on top of foundations F2).
10. Not read or not verified: `dists/bijectors/flows.ts` (764 lines; only its unrun lines were listed), `compose/conditional.ts`,
    `compose/mixture.ts`, `relaxed.ts` bodies (unrun lines listed, not probed), `conjugacy.ts`, `exponential-family.ts`,
    `families/*` fitting code; in stats `design.ts` (1487 lines), `experiments.ts`, `missing.ts`, `smoothing/*`, `density.ts`,
    `robust.ts`, `extremes.ts`, `fit.ts` (read only around the catches), the sequential, kernel, Bayesian and equivalence tests; in
    metrics `text.ts`, `image-quality.ts`, `segmentation.ts`, `detection.ts`, `generative.ts`, `uplift.ts`, `operating-points.ts` bodies
    (probed through the generic sweep of `m2`, not read). Every scipy/sklearn/torch value quoted is from memory. No test, tsc or
    gate was run.

Entries by kind: consistency 14, hot-path 3, duplicate 3, rule 3, layering 1, native-array 2, test-gap 4 (30 in all, F1 to F30).

---

### F1: `sampleWeight` is accepted by every metric's options type and silently ignored by 33 of 99 numeric metrics

kind: consistency
evidence: `metrics/metric.ts:13-17` (`MetricOptions.sampleWeight`, inherited by `Metric<O>` for every metric); probe `m7` calls each metric twice on the same 6 samples, without and with `sampleWeight [1,1,1,2,0.5,0]`, and lists those whose value is unchanged: boundaryF1, brierDecomposition, calinskiHarabaszScore, concordanceCorrelation, cosineSimilarity, cramersV, daviesBouldinScore, diceCoefficient, distsFromFeatures, entityF1, imageMeanSquaredError, kendallTau, krippendorffAlpha, lpipsFromFeatures, maskHausdorffDistance, maskIou, maxError, maximumCalibrationError, meanIou, medianAbsoluteError, minkowskiDistance, peakSignalNoiseRatio, precisionAtFixedRecall, recallAtFixedPrecision, scaleInvariantSdr, scaleInvariantSnr, signalDistortionRatio, signalNoiseRatio, spearmanCorrelation, spectralAngleMapper, theilsU, theilsU2, topKAccuracyScore. `m6`: with weights `[1,1,1,0,0]` `cosineSimilarity`, `kendallTau`, `spearmanCorrelation`, `peakSignalNoiseRatio` return the unweighted value (0.906, 0.738, 0.821, 4.72) while "the first three samples only" gives 0.991, 1, 1, 10.8; `pearsonCorrelation` honours them (0.982), `meanAbsoluteError`, `r2Score`, `accuracyScore` too. With weights all 0: `accuracyScore`, `meanSquaredError`, `weightedMean` give NaN (0/0, `metric.ts:195`), `sampleWeights` (`metric.ts:101-106`) checks only the length: weights `[1, -1]` (sum 0) give `accuracyScore = Infinity` (probe `m1`) and a NaN weight gives NaN.
what: the contract (`MetricOptions`) promises per-sample weights and 1 metric in 3 does not read them, with no error, so a caller who passes weights (class frequency, a mask) gets the wrong number and no sign of it. Some of the 33 are legitimately unweightable (`maxError`, structural image/audio metrics, `medianAbsoluteError` may be a coincidence of this probe's data); the list needs a pass, not a blanket edit. The weights are also not validated (negative, NaN, zero total).
fix: (1) `sampleWeights()` rejects negative and non-finite weights (`value`) and a zero total only where the metric divides by it; (2) every metric either honours the weight or declares `weights: false` in its `MetricSpec` and `defineMetric` throws `value` when `sampleWeight` is passed (the spec already carries `inputs`, `range`, `direction`); (3) one generic test (F26) that repeats rows by integer weights and compares. The cheap ones to honour: cosine, concordance, SNR family (weighted energies), calibration errors, Spearman/Kendall (weighted ranks are what scipy does not offer: reject), IoU/Dice (weighted counts).
size: M (1-3h for the check and the generic test; 33 decisions, most one line)
risk: low (a weighted call that used to ignore weights now changes or throws)
preserves behaviour: no (weighted calls of those metrics)
changes public surface: no (adds a spec flag)

### F2: NaN in rank-based routines: tests report p-values, correlations return 1, `histogram` drops NaN and throws on Infinity

kind: consistency
evidence: `stats/_util.ts:10-14` (`sample` checks only the length, so no test or statistic sees a NaN), `stats/ranks.ts:14-16` (`order` compares with `(x[a] - x[b]) || a - b`: a NaN difference is falsy, so NaN ties with everything in the sort and never with itself in the tie loop at `:33`; the doc says "(no NaN)" and nothing enforces it). Probes `m7`, `m6`, `c1`: `mannWhitneyU([1,2,NaN,4],[3,5,6,7])` gives `stat 1, p 0.057`; `wilcoxon` with a NaN gives `p 0.0625`; `kruskal([[1,2,NaN],[3,5,6]])` gives `p 0.0495`; `ks2samp` with a NaN gives `p 0.77`; `S.spearman([1,2,NaN,4],[1,2,3,4])` = 1 and `S.kendallTau(...)` = 1 while `pearson` gives NaN; the metrics `spearmanCorrelation` and `kendallTau` (delegating, `metrics/correlation-agreement.ts:9`) = 1 (probe `m2`, nan column); `rocAucScore([0,1,0,1],[0.1,NaN,0.3,0.9])` = 0.75 (and 0.5 on other data), `averagePrecisionScore`, `fprAtTpr`, `equalErrorRate` the same, while `brierScoreLoss`, `logLoss`, `expectedCalibrationError` return NaN and `precisionScore`/`f1Score` throw `value` on a NaN label and `accuracyScore` counts NaN as a label (0.75). `S.histogram([1,NaN,3],{bins:2})` counts 2 of 3 values with no sign of the NaN, `S.histogram([1,Infinity,3])` throws `value: data must be finite`. `p2Push` (`stats/running.ts:226`) with a NaN increments the marker positions and moves no height (probe `r1`). The t-tests, `pearson`, `mean`, `quantile` and `median` propagate NaN, as scipy's default `nan_policy='propagate'` does.
what: the same input gives a p-value, a perfect correlation, a NaN or an exception depending on whether the routine sorts. A NaN in a data column turns a significant-looking test into a quiet one, which is the flaky-CI class (a generator that occasionally emits NaN passes the test it should fail).
fix: one rule, stated in the `stats` and `metrics` index summaries: **a statistic of data containing NaN is NaN (propagate) or throws `value`; it is never a finite number.** Implement at the choke points: `sample()` and `groupsOf()` get a `nanPolicy: 'propagate' | 'omit' | 'raise'` (default `propagate`: the caller returns NaN for statistic and p), `ranksOf` returns NaN ranks for NaN (or throws), the metrics readers `flat`/`columns`/`positives`/`encodeLabels` check finiteness once, `histogram` treats NaN as Infinity does. Then the oracle's scipy `nan_policy` cases become fixtures.
size: M (1-3h for the rule and the choke points; the per-test returns of NaN are mostly one line)
risk: medium (callers that relied on NaN being ranked; grep `ranksOf` first: `sampling`, `learn` and `cluster` use it)
preserves behaviour: no (NaN inputs only)
changes public surface: no

### F3: p-values from special functions for NaN, and small wrong answers on degenerate input in the tests

kind: consistency
evidence: (a) `numerics/special/cdf.ts:135-138` (`chdtr`: `where(greater(xx, 0), ..., 0)`) and `:154-158` (`chdtrc`: `... , 1)`), the same shape in `fdtr/fdtrc` just below: a NaN statistic is not greater than 0, so `chi2Sf(1, NaN) = 1`, `chi2Cdf(1, NaN) = 0`, `fSf(1, 6, NaN) = 1`, `fSf(1, 6, -1) = 1` (probe `sp`); this is why `kruskal` of identical groups (stat NaN), `fOneway` of identical groups, `jarqueBera` of a constant, `levene` of constants, `ljungBox` of a constant, `chi2Contingency` with a zero row, `chisquare([5, NaN])` all report `p = 1` with a NaN statistic (probe `s1`, `s2`); `tSf`, `normSf` and `chdtrc(NaN, x)` correctly give NaN. (b) `stats/tests/result.ts:316-338` `fligner`: constant groups give deviations all 0, all ranks tie, `a` is constant up to rounding, `varsq ~ 1e-32`, so `s / varsq` is rounding noise: `fligner([[3,3,3,3],[3,3,3,3]])` and `fligner([[3,3,3,3],[5,5,5,5]])` both return `statistic 7, p 0.00815` (probe `s2`), a confident rejection for identical samples. (c) `stats/tests/means.ts:253-278` `fOneway` computes `etaSquared(gs)` and two assumption tests as part of the result; for constant groups `F = Infinity` and `stats/effect-size.ts:243` (`etaBounds`) passes it to `noncentralityBounds` (`:164-170`), `rootAbove` and the noncentral-F CDF (an `arange` with a NaN length): the call throws `shape: data of length 0 does not fill shape [NaN]` (probe `s3`) although the F test itself (statistic Infinity, p 0) is computable; `fOneway([[1],[2],[3]])` throws `levene: needs at least 2 observations`, the name of an inner call. (d) `chisquare([5, -5])` returns `stat Infinity, p 0` (negative counts are not rejected; `distribution.ts:104-112` checks only the totals), `lilliefors` of a constant returns `stat NaN, p 0.0001` (the table edge), `durbinWatson` of a constant returns `stat 0, p 0.0455`.
what: three of these are the same root (comparisons that are false for NaN select the edge value), one (fligner) is a plain cancellation bug, and one (fOneway) lets an optional decoration take down the headline result.
fix: (a) in `cdf.ts` write `where(lessEqual(xx, 0), edge, where(isnan(xx), NaN, ...))` or use `greater` on the negation (`where(not(xx <= 0), ...)`), and add NaN rows to the special-function table (numerics F26 asks for the same table); (b) in `fligner` return NaN statistic and p when `varsq <= eps * abar^2`, and in the other variance-ratio tests (`levene`, `bartlett`, `fOneway`) return the documented limit (Infinity, p 0) or NaN, not a rounded ratio; (c) compute `effectSize`/`assumptions` of `fOneway` inside a guard that drops them (as `undefined`) when they cannot be computed, and label inner errors with the outer name; (d) reject negative counts in `powerDivergence`, return NaN p for a NaN statistic everywhere.
size: M (1-3h)
risk: low (a, b, d change degenerate outputs only) to medium (c changes what `fOneway` returns for constant groups from an exception to a result)
preserves behaviour: no (degenerate inputs only)
changes public surface: no

### F4: the failure convention of the tests: five shapes, no status, and `rejects` reads them all as "not significant"

kind: consistency
evidence: `base/results.ts:29-39` (`TestResult` has `statistic`, `pValue`, `df`, ...: no `status`, no `valid`); probes `s1`, `s2`. Degenerate input produces (1) an exception `value`: `ttest1samp` n < 2, `shapiro` constant ("the sample is constant"), `adfuller` constant, `wilcoxon` all differences zero, `binomTest(0, 0)`, `ks2samp` empty; (2) NaN statistic with `p = 1`: `kruskal` identical, `fOneway` identical, `jarqueBera` constant, `chi2Contingency` zero row, `fisherExact([[0,0],[0,0]])`, `levene`/`ljungBox` constant, `bartlett` of two constants (see F3a); (3) NaN statistic and NaN p: `ttestInd` of equal constants, `ttestRel` identical, `andersonDarling` constant, `proportionsZtest([0,0],[10,10])`, `mannWhitneyU(c, c)` (statistic 8, p NaN); (4) a finite or infinite statistic with a made-up p: `lilliefors` constant (NaN, 0.0001), `durbinWatson` constant (0, 0.0455); (5) `Infinity` statistic with p 0 for a zero variance: `ttestInd` of two different constants (`-Infinity`, 0) with the pooled variance but `p NaN, df null` with `equalVar: false`, `bartlett([constant, varied])`. `rejects()` (`stats/tests/result.ts:221`) is `p < alpha`, so NaN is "not rejected"; `checkEqualVariances` (`:384`) maps a NaN p to 1 ("equal"); the `assumptions` of `ttest1samp` for a NaN sample read `normal: false`. `runsTest` of a constant returns `statistic -Infinity, p 0.25`, which mirrors statsmodels' exact one-run probability.
what: a caller (or a gate over p-values) cannot tell "no evidence" from "test undefined" from "wrong"; `p = 1` with a NaN statistic is the worst one because it is a valid-looking answer.
fix: the rule, written in `stats/tests/index.ts`'s summary and enforced by `testResult()`: **the statistic is NaN exactly when the test is undefined for the data; then `pValue` is NaN and `method` says why; `rejects` returns false and `formatTest` prints "undefined"; fewer observations than the test needs throw `value` naming the test.** `testResult` can assert `Number.isNaN(statistic) === Number.isNaN(pValue)`. Optionally an `undefined?: string` field on `TestResult` (reason) in place of a `status` enum, which keeps the record flat. The table test of F25 pins one case per shape.
size: M (1-3h: about 25 tests need a degenerate branch; `testResult` guard is five lines)
risk: medium (callers reading `p = 1` of identical groups; grep `pValue` in `learn`, `fairness`, `causal`)
preserves behaviour: no (degenerate inputs only)
changes public surface: maybe (one optional field)

### F5: test gap: no contract test across the distributions; 168 anonymous closures and 9 named functions never run

kind: test-gap
evidence: `dists` has 543 lines of hand-written tests (`dists.test.ts` 248, `continuous/continuous.test.ts` 157, `discrete/discrete.test.ts` 95, `continuous/continuous.golden.test.ts` 43) for 13,000 lines of source; the sampling and gradient tests are tables, but nothing checks `ppf` against `cdf`, `sf` against `cdf`, `isf` against `ppf`, `logCdf`, `mode`, `entropy`, the support edges or non-finite input per family. Coverage (probe `unrun3`): the never-run closures are almost all methods of the contract: `mode` of about 30 families (`heavy-tailed.ts:108,152,194,251,294,348,473`, `gamma.ts`, `counts.ts:93,148,206,276`, `binary.ts:96,156,208`, `normal.ts:128,167,226`, `extreme.ts:103,184,248,295`, ...), `entropy` of about 14 (`bounded.ts:86,137,261`, `gamma.ts:100,159,212,282`, `normal.ts:228`, `categorical.ts:186`, `copulas.ts:323`, ...), the `sf`/`logSf`/`logCdf`/`isf`/`ppf` closures of `johnsonSU`/`sinhArcsinh` (`heavy-tailed.ts:465-469`), `bounded.ts:184-187,311-315` (kumaraswamy and logitNormal `logSf`, `ppf`, `rsample`), `compose/structure.ts:100-102,169-171,390-399,483,552-558` (transformed, truncated, censored, empirical, deterministic), `relaxed.ts:80-105` (`variance`, `mode`, `entropy`), `copulas.ts:310-323`, `matrix.ts:298-344` (LKJ moments). Branch coverage of dists is 68.1% (494 missed; `_numeric.ts` 66%, `distribution.ts` 53%, `mixture.ts` 50%, `structure.ts` 61%). The two bugs F6 and F7 sit on unrun lines (`bounded.ts:136`, `binary.ts:156`, `categorical.ts:144,186`).
what: a table of (distribution, parameters) rows through generic checks is exactly what probes `d1`, `d3`, `d6` do; they found F6, F7, F8 and F10 in minutes. The maintainer wants the count flat: one file, about 10 `it.each` blocks over one row table, replacing nothing.
fix: add `dists/contract.test.ts` (or extend `dists.test.ts`): rows `[name, make, {support probes, skipChecks}]` for the 45 families probed in `d1`/`d3`/`d6`; checks, each one `it.each(rows)`: (1) `ppf(cdf(x)) ~ x` on a grid of quantiles in `[1e-10, 1 - 1e-10]` (discrete: the bracketing property); (2) `sf + cdf = 1`, `exp(logCdf) = cdf`, `exp(logSf) = sf`, `isf(q) = ppf(1 - q)`; (3) the density integrates to 1 (trapezoid between `ppf(1e-9)` and `ppf(1 - 1e-9)`; PMF sums to 1); (4) `mean`, `variance`, `entropy` against quadrature and `mode` against the grid argmax of `logProb` (with the supremum rule for singular densities); (5) `logProb`/`cdf`/`ppf` at NaN, +-Infinity and 1 step outside the support (F8 states the expectation); (6) sample mean/variance against `mean()`/`variance()` (exists; fold in); (7) `inSupport(support, mode())` and of every sample. About +10 cases, one table of ~45 rows.
size: M (a day for the table and the exclusions; the checks are the probes)
risk: low
preserves behaviour: yes
changes public surface: no

### F6: closed-form edge cases outside the support: `beta.mode`, `binomial.mode`, `vonMises`, `F` at 0, `categorical` with one class

kind: consistency
evidence: (a) `dists/continuous/bounded.ts:136` `beta` mode `(a - 1) / (a + b - 2)`: `beta(3, 0.5).mode()` = 1.333 with `logProb = -Infinity` there (probe `d6`; the support is [0, 1]; for `b <= 1 < a` the mode is 1, for `a <= 1 < b` it is 0, for `a, b < 1` it is bimodal); `dirichlet` mode (`simplex.ts:85`) is already `(alpha - 1) / (sum - k)` with the same shape of formula. (b) `discrete/binary.ts:156` `binomial` mode `floor((n + 1) p)`: `binomial(20, 1).mode()` = 21, outside {0..20} (probe `d7`; also `entropy` = -4.4e-16). (c) `multivariate/directional.ts:60,100-110`: `vonMises` declares `support: { kind: 'circular' }` = [-pi, pi] (`distribution.ts:331`) but draws and its CDF live on `[loc - pi, loc + pi]`: `vonMises(3, 0.5)` samples span [-0.14, 6.13], 46% fail `inSupport`, `cdf(-3) = -0.43` (negative) and `cdf(6) = 0.987` (probe `v2`); `logProb` is periodic and fine. (d) `variance()` of `vonMises` is the circular variance (documented, `directional.ts:60`), unlike every other family (the contract in `distribution.ts:73-82` does not say). (e) `continuous/sampling-distributions.ts` `fDistribution(1, 7)`: `logProb(0) = -Infinity`, while `gamma(0.5)` and `chi2(1)` return `+Infinity` at 0 for the same kind of singularity and scipy gives `inf` (probe `d6`); the mode is then 0 with `logProb(mode) = -Infinity`. (f) `continuous/extreme.ts:46-51` `shapeSupport` decides `greaterThan`/`lessThan`/`real` from the concrete sign of the shape at construction, so `generalisedExtremeValue([0,0],1,[0.2,-0.2])` is declared `real`; (g) `discrete/categorical.ts` with one category: `categorical({ probs: [1] }).sample(key, [3])` throws a `TypeError: Cannot read properties of undefined (reading 'length')` (probe `d7`).
what: each is a formula applied outside the parameter range it was derived for. They sit on lines no test runs (F5).
fix: (a) clip with the `where` on `a > 1`, `b > 1`; (b) `min(n, floor((n + 1) p))`; (c) either wrap `loc` into [-pi, pi) in construction and CDF (the torch convention is a CDF on the support) or declare `support` as an interval around `loc`; (d) one sentence in the contract; (e) return `+Infinity` for `d1 < 2` at 0; (f) document that `support` of a data-dependent family is `real`, or make the families interval-valued with the traced bounds; (g) guard `k = 1` in `drawIndices`.
size: S each (a to b and g are minutes; c is an hour)
risk: low (b, d, e, g) to medium (c changes `cdf` of `vonMises` with `loc` outside the window)
preserves behaviour: no (the cases above only)
changes public surface: no

### F7: zero-probability categories give NaN in `categorical` entropy and in discrete KL (the masked-logits case)

kind: consistency
evidence: `dists/discrete/categorical.ts:144` and `:186` `entropy: neg(sum(mul(probs, logp), -1))`: `0 * -Infinity = NaN`; probe `v1`: `categorical({ logits: [0, -Infinity, 0] }).entropy()` = NaN (should be log 2 = 0.693). `dists/divergences.ts:84` `klCategoricalLogits = sum(mul(exp(lp), sub(lp, lq)))` and `:138` (multinomial through `log(probs)`), `:80-81` (`klBernoulliLogits` through `logSigmoid` of infinite logits): probe `k1`: `kl(cat[1,0], cat[1,0])` = NaN (want 0), `kl(cat[1,0], cat[.5,.5])` = NaN (want 0.693), `kl(bernoulli(0), bernoulli(.5))` = NaN (want 0.693); the other direction (`q` has a zero where `p` has mass) correctly gives Infinity. The `logProb` of the same distributions treats `0 log 0 = 0` correctly (`categorical.ts:179`: "with 0 log 0 taken as 0").
what: masked action logits (`-Infinity` for illegal moves) are the standard use in the `rl`/`agents` code; their entropy bonus and KL penalty become NaN and poison a gradient step.
fix: `xlogy(probs, probs)` (exists in `numerics/special`, used in `poisson`'s `logPmf`) in the three entropies, and `where(p > 0, p * (lp - lq), 0)` in the discrete KLs; the gradient through the masked branch must be 0, not NaN, so use the double-`where` pattern the library already uses.
size: S (<1h)
risk: low
preserves behaviour: yes (except NaN becomes the right number)
changes public surface: no

### F8: non-finite arguments to `cdf`, `logProb` and `ppf` follow no rule: `cdf(NaN)` is 0, 0.578, 0.5, 1 or NaN; `cdf(Infinity)` is NaN for nine families

kind: consistency
evidence: probes `d1`, `d3`. `cdf(NaN)`: NaN for normal, studentT, cauchy, laplace, logistic, gumbel, uniform, beta, triangular, johnsonSU, skewNormal, noncentralT, vonMises, poisson, geometric, discreteUniform; **0** for halfNormal, logNormal, gamma, chi2, exponential, inverseGamma, weibull, pareto, inverseGaussian, F, noncentralChi2, halfCauchy, GEV, Frechet, negativeBinomial; **0.578** for `kumaraswamy`, **0.5** for `logitNormal`, **1** for `generalisedPareto` and `binomial`, **0.0029 / 0.066** (the value at 0) for `hypergeometric`/`betaBinomial`. `cdf(+Infinity)` is NaN for halfNormal, inverseGaussian, F, skewNormal, noncentralT, vonMises, poisson, negativeBinomial, zeroInflatedPoisson; `cdf(-Infinity)` NaN for skewNormal, noncentralT, vonMises. `logProb(+Infinity)` is NaN for gamma, chi2, weibull, inverseGaussian, F, noncentralChi2, poisson, negBin (it is `-Infinity` for normal, beta, exponential); `logProb(-Infinity)` NaN for logistic, gumbel, noncentralT, vonMises (`logistic`: `-z - 2 log1p(exp(-z))` at z = -Infinity is `Inf - Inf`; `heavy-tailed.ts:275-290`). `ppf(NaN)` is NaN everywhere (good). The cause is the idiom `where(inside(x), f(x), edge)` in `cdf`/`sf` (see `counts.ts:87`, `normal.ts:121,159-161`, `heavy-tailed.ts:188-190`), false for NaN, and unguarded `x * log(y)`-style terms at an infinite argument.
what: an upstream NaN is absorbed as a probability, which is how a log-likelihood or a p-value stays finite while its input is broken. The contract (`distribution.ts:83-105`) does not state any rule.
fix: the rule, stated on `Distribution`: **`NaN` in gives `NaN` out for `logProb`, `cdf`, `sf`, `logCdf`, `logSf`, `ppf`; `+-Infinity` gives the limit (`cdf` 0 or 1, `logProb` `-Infinity` for a bounded or light-tailed density).** Implement once in `defineDistribution` (`distribution.ts:187-197`): wrap each of the six methods as `where(isnan(x), NaN, spec.method(x))` (one extra op per call, 7% on `normal.logProb` by the proportions of probe `h1`) and fix the 12 families whose limit is wrong (F5's check 5 lists them). Prefer the wrapper to editing 40 `where` calls.
size: M (1-3h)
risk: low (wrapper) / medium (the families' limits)
preserves behaviour: no (NaN and infinite arguments only)
changes public surface: no

### F9: parameter validation: the vocabulary has no relational constraints, `positive` admits Infinity, and several families accept invalid parameters

kind: consistency
evidence: `dists/distribution.ts:396-418` `Constraint` and `CHECKS`: `positive: v > 0` passes Infinity and `openUnitInterval` (`:414`) is used by no family (0 occurrences; `positive` 57, `real` 15, `finite` 12, `unitInterval` 4). Probe `d2`: `normal(0, Infinity)` is accepted and samples NaN; `gamma(Infinity)` accepted, `logProb` NaN; `studentT(Infinity)` accepted, `logProb` NaN; `poisson(Infinity)` accepted; `uniform(0, 0)` and `uniform(0, -1)` accepted (`bounded.ts:68` checks `finite` only; the second samples from [-1, 0] and `logProb(0.5) = -Infinity`); `triangular(0, mode, 1)` with `mode = -1` or `2` accepted (`bounded.ts:222`, finite only, and `logProb` finite); `geometric(0)` and `negativeBinomial(3, 0)` / `(3, 1)` accepted (`counts.ts:116,169` use `unitInterval` where `(0, 1]` and `(0, 1)` are meant), `geometric(0).sample` then throws its own `p must be in (0,` message from the sampler instead of the constructor; `vonMises(0, 1e6)` accepted; `multivariateNormal` with a non-symmetric covariance `[[1, .5], [0, 1]]` accepted and evaluated with no error (probe `v1`; the lower triangle is used); `categorical({ probs: [2, 2] })` is normalised silently. `validateParams` skips traced values through a bare `catch {}` (`distribution.ts:438-442`; same pattern in `extreme.ts:50-54`, `elementwise.ts:153-157`, `families/extended.ts:179-183`, see F21).
what: the constraint set was written for single parameters; "low < high" and "mode in [low, high]" cannot be said, so two families skip them; infinities pass `positive` and make NaN at the first use. Foundations F9/F30 found the same for the samplers and `space` constructors.
fix: add `'positiveFinite'` (or make `positive` mean finite and add `'positiveInfinite'` for the few that allow it) and use `openUnitInterval` for `geometric`/`negativeBinomial`; add two relational checks next to `validateParams` (`validateOrdered(family, [['low', lo], ['high', hi]])`, `validateBetween`) for `uniform`, `triangular`, `truncatedNormal` (is `low < high` checked? not read), `hypergeometric`; check symmetry and finiteness in `multivariateNormal` (the Cholesky already rejects NaN).
size: S-M (1-2h)
risk: low (invalid parameters only; grep call sites that pass `scale = Infinity` deliberately)
preserves behaviour: no (invalid parameters now throw)
changes public surface: no

### F10: tail precision of `ppf`/`cdf` for families written with complements, and a floor in the incomplete-gamma inverse

kind: consistency
evidence: probes `d1`, `d5` (relative error of `ppf(p)` against the closed form). `cauchy`: `ppf` is `tan(pi (q - 0.5))` (`heavy-tailed.ts:138`), `cdf` is `0.5 + atan(z)/pi` (`:145`), `sf` is `0.5 - atan(z)/pi` (`:146`): `q - 0.5` carries an absolute error of 1e-16 so the relative error of the quantile is `1e-16 / p`: 1e-6 at p = 1e-10 (`-3.183097e9` vs `-3.183099e9`), 1.5e-3 at 1e-14, a factor 2 at 1e-17 (`-1.63e16` against `-3.18e16`), and `ppf(0) = -1.63e16` instead of `-Infinity`; `logCdf(ppf(1e-200))` = `-Infinity`. `laplace` (`:218-230`): `ppf(1e-14)` relative error 3e-5, `ppf(1e-17) = -Infinity` (true -38.45). `pareto` (`:321` `xm (1 - q)^(-1/alpha)`): `ppf(1e-14) - 1 = 3.3307e-15` against 3.3333e-15 (8e-4 relative error in `x - xm`) and `ppf(1e-17) = 1` exactly. `halfNormal` `ppf` (`normal.ts:105-124`, through `ndtri((1 + q)/2)`): `ppf(1e-14)` 1e-3 relative error, `ppf(1e-17) = 0`, while its `isf` uses `q/2` and is exact. `gamma`, `chi2`, `inverseGamma`, `inverseGaussian`: `ppf` through `gammaincinv` (`gamma.ts:94,199`) returns a floor: `gamma(2).ppf(1e-200) = 6.2e-64` where the true value is 1.4e-100 and `cdf(ppf(1e-300)) = 1.9e-127` (this is numerics F4's denormal-tail class); `truncatedNormal.cdf(ppf(1e-300))` = NaN and `logCdf(ppf(1e-200))` = NaN (`normal.ts:196-230`). The upper tails and `sf`/`logSf` are fine (probe `d4`).
what: quantile functions are used for sampling by inversion (`rsample = ppf(uniformOpen)` for cauchy, laplace, logistic, pareto), so the tail of those samplers is quantised and, for Cauchy, truncated at 1.6e16. Samples of `uniform` near 0 are rare (2^-54 floor), so this shows only in tails and in `logCdf`.
fix: write the quantile of the lower half through the smaller tail: Cauchy `-1/tan(pi q)` for `q < 0.5` (and the complement for `q > 0.5`) and `atan2`-based `cdf`; Laplace `loc + scale log(2q)` for `q < 0.5`; Pareto and Weibull `expm1(-log1p(-q)/alpha)`; halfNormal `sigma sqrt2 erfinv(q)` (needs an `erfinv`; `ndtri((1+q)/2)` is the cause); then `logCdf`/`logSf` through `log(cdf)` only where `cdf` is not subnormal. The incomplete-gamma floor belongs to numerics F4.
size: M (1-3h)
risk: low (tail digits only)
preserves behaviour: yes within tolerance (the fixtures sit at moderate `p`)
changes public surface: no

### F11: bijectors at extreme unconstrained values: `correlationCholesky` is NaN from `|u| = 20`, `stickBreaking` stops round-tripping at `u = 20`

kind: consistency
evidence: `dists/bijectors/structured.ts:160-215` (`correlationCholesky`: partial correlations `tanh(x)` and cumulative `log sqrt(1 - z^2)`), probe `b2` (`forward([u,u,u])` then `inverse`): `u` up to 15 round-trips, `u = 18` returns 18.02/17.68, `u = 19` returns 18.7 and NaN, `u >= 20` gives `forward = [[1,0,0],[NaN,0,0],[NaN,NaN,0]]` (probable cause: `tanh(20) = 1` exactly, so `1 - z^2 = 0` and a later `sqrt` of a rounded negative is NaN; not traced line by line) and `forwardLogDetJacobian = NaN` (probe `b1`: `[-30,30,0]` and `[800,0,0]` also NaN). `stickBreaking` (same file): `forward([20,20])` = `[1, 4.1e-9, 8.5e-18]`, `inverse` returns `[20, 18.15]`; at `u = 30` `[30, 8.70]`, at `u = 35` `[35.03, NaN]` (probably because the remainder is read from `1 - y1`, which has no digits left); `inverseLogDetJacobian(stickBreaking([-800,800])) = NaN`. `tanh` `inverseLogDetJacobian` at `y = -1` is NaN while at `y = +1` it is Infinity (`u = -20`, probe `b1`; asymmetric). `sigmoid`/`softplus`/`exp` saturate to 0/1/Infinity at `|u| >= 37, 745` which is inherent and consistent.
what: HMC/NUTS and optimisers on an LKJ or Dirichlet prior can step to `|u| = 20` in one leapfrog; the NaN then reads as a divergence with no cause. The existing bijector tests (`bijectors.test.ts`) round-trip near zero.
fix: in `correlationCholesky` compute the diagonal as `sqrt(prod(1 - z^2))` through `exp(0.5 * cumsum(log1p(-z^2)))` with `log1p(-z^2) = log(1 - z) + log(1 + z)` and `log(1 - tanh x) = log 2 - 2x - ... ` (softplus form, as `tanh`'s log-Jacobian already does, `elementwise.ts`), and in `stickBreaking` carry the remainder as a log (`log_remaining = cumsum(logSigmoid(-x))`) so `inverse` reads `log y_k - log remainder`. Add the round-trip at `|u| in {0, 5, 15, 25}` for every bijector to the existing bijector test (F5 style: one table).
size: M (1-3h)
risk: low (values agree within tolerance below `|u| = 15`; goldens at moderate `u`)
preserves behaviour: yes below the failure region
changes public surface: no

### F12: sampling costs 1.0 s per 1e5 draws for the gamma family and the count families (100 x `normal`), because every variate pays for its own key and uniform block

kind: hot-path
evidence: probes `h1`, `h2` (per 1e5 draws, after one warm-up): `normal.sample` 8 ms, `exponential` 8 ms, `weibull` 12 ms, `categorical(8)` 48 ms, `geometric` 35 ms, but `gamma` 1040 ms, `beta` 2080 ms, `studentT` 1053 ms, `dirichlet(5)` 1054 ms for 2e4 rows, `poisson` 1055 ms, `binomial` 1057 ms. `gammaVariates(key, Float64Array(1e5))` alone is 1045 ms (`random/continuous.ts:294-302`: `gammaOne(alpha, uniformStream(foldIn(k, i)))` per element: a Philox block per variate; this is foundations F2) and the random-level `poisson(key, 4, [1e5])` is 25 ms, `binomial` 42 ms: the dists side adds the 40x for the count families, because `discrete/counts.ts:90,145,199,259` and `binary.ts:151,198` call `elementwiseSample` (`_sampling.ts:99-117`), which derives `foldIn(key, i)` per element and calls the scalar `poissonDraw(kk, l, []).data[0]` (a tensor per element). `geometric` goes through the same path and costs 35 ms, so the per-element key and tensor are not the 40x; the cost is the uniform block that each scalar `poissonDraw`/`binomialDraw` call builds (the mechanism of foundations F2), which the batch call builds once. `negativeBinomial` (`counts.ts:199-202`) does a `gammaVariates(k, [rr])[0]` and a `poissonDraw` per element; `chi2`/`noncentral*`/`F` (`sampling-distributions.ts:193,291-292`), `families/dispersion.ts:395-396` (Tweedie) and `studentised-range.ts:167` the same per element. `_sampling.ts:36-60` `gammaSamplePrimitive` is the batch path (one call over `a.data`) and costs the 1 s because of foundations F2.
what: any Monte Carlo over a beta/gamma/Dirichlet prior, a Poisson or binomial likelihood simulator or a posterior-predictive check (`sampling`, `probprog`, `pgm`) is bounded by 10 us per draw regardless of vectorisation; `normal`-driven code is 100x faster, so tests choose normals.
fix: two steps. (1) Foundations F2 (a `gammaVariates` that streams several variates from one block): fixes `gamma`, `beta`, `studentT`, `dirichlet`, `chi2`, `F`. (2) In dists, replace the per-element loops of `elementwiseSample` by the `random` batch samplers over the broadcast parameter array (`poisson`, `binomial`, `geometric` already take shapes; give them array-valued rates, or a `Float64Array` kernel `poissonVariates(key, rates)` like `gammaVariates`), and in `negativeBinomial` draw all gammas then all Poissons in two batch calls. Keep the key derivation per element only where independence across batch indices matters (document which streams change).
size: M (1-3h for dists; foundations F2 is its own item)
risk: medium (the sample streams change for fixed keys: seed-pinned goldens and tests that pin a draw will move; the distributions do not)
preserves behaviour: no (draws under a fixed key change; distribution unchanged)
changes public surface: no

### F13: metrics have no convention for empty or degenerate input; several disagree with sklearn

kind: consistency
evidence: probe `m2` (156 metrics, each called with `([], [])`): NaN 60, a number 52 (0, 1, -1 or Infinity), `throw shape` 26, `throw value` 13, plus 5 TypeErrors (options required). Examples: `accuracyScore([],[])`, `meanSquaredError`, `r2Score`, `brierScoreLoss`, `rocAucScore` NaN; `precisionScore`, `recallScore`, `f1Score`, `jaccardScore`, `specificityScore` 0; `adjustedRandScore`, `normalizedMutualInfoScore`, `homogeneityScore`, `completenessScore`, `maskIou` 1; `informedness([],[])` and `markedness` **-1**; `maxError` Infinity; `expectedCalibrationError`, `brierDecomposition`, `precisionAtFixedRecall` 0; `kendallTau`, `spearmanCorrelation`, `silhouetteScore` throw `value` while `pearsonCorrelation` is NaN; `logLoss([],[])` throws `shape` ("needs [N, C] probabilities", `probabilistic.ts:456`) while a 1-D input of length 2 works. Single class: `rocAucScore([1,1,1], s)` NaN (sklearn raises, from memory), `cohenKappaScore` of identical single-class NaN, `specificityScore([1],[1])`, `negativePredictiveValue`, `informedness`, `markedness` of a perfect single-class prediction are 0 (the absent class gets `zeroDivision = 0`, `classification.ts:97,126,393`, `_labels.ts:132`). `averagePrecisionScore([0,0,0],[.1,.5,.9])` = **1** (`precision-recall.ts:56`: recall set to 1 when there are no positives, then the terminal point adds the full area); sklearn returns 0.0 with a warning (from memory, unverified). Zero total weight: `accuracyScore`, `meanSquaredError`, `rocAucScore` NaN, `adjustedRandScore` 1, `mutualInfoScore` 0. `zeroDivision` exists only on the classification rates (11 `@param`s, `classification.ts:38`); the other metrics hard-code NaN, 0 or 1.
what: a caller cannot write one guard; `informedness([])` = -1 reads as "worst possible" for no data, and the `zeroDivision = 0` for an absent class makes a perfect prediction score as chance. sklearn raises for empty input in the `check_array` of nearly every metric.
fix: one policy in `metrics/index.ts`'s summary, implemented in the readers: **fewer samples than the metric needs throws `value` naming the metric (`columns`, `encodeLabels`, `positives` check once: n = 0 for all, n = 1 where a correlation or a variance is involved); a metric undefined for the data (one class, no positives, zero weight) returns NaN with the reason in `breakdown`; `zeroDivision` is an option of every ratio metric, default NaN for the composite ones (informedness, markedness, NPV when the class is absent) and 0 for precision/recall as sklearn.** Fix `averagePrecisionScore` to 0 or NaN when there are no positives (confirm with the oracle). The table test (F26) pins the policy per metric.
size: M (1-3h for the readers, then one line per outlier)
risk: medium (callers that relied on the empty-input numbers; grep the `learn`/`ensembles` call sites of `accuracyScore`, `r2Score`)
preserves behaviour: no (degenerate inputs only)
changes public surface: no

### F14: `shortTimeObjectiveIntelligibility` never returns for a NaN or missing `fs`; a fractional `fs` takes 1.2 s then fails with the wrong message

kind: consistency
evidence: `metrics/audio.ts:351-357` `resampleOct` computes `gcd(p, q)` by `while (b !== 0) [a, b] = [b, a % b]`, and `:439-441` calls it with `options.fs`; with `fs = NaN` or `undefined` (a JS caller, or `options` omitted: probe `m4`, killed at 20 s; it also stalled the generic sweep `m2`) `b` is NaN, `NaN !== 0` is true forever. `fs = 16000.5` takes 1.2 s and throws `resamplePoly: down must be a positive integer` (probe `m5`); `fs = 0` and `-16000` throw promptly with the `firwin`/`resamplePoly` messages.
what: an infinite loop on invalid input is the worst failure mode in a test run (the runner times out with no stack). The type requires `fs`, so only an untyped caller or an options object built at run time reaches it.
fix: `if (!Number.isInteger(options.fs) || options.fs <= 0) throw new OmaiError('value', 'shortTimeObjectiveIntelligibility: fs must be a positive integer')` at the top of the metric, and the same guard in the other audio metrics that take `fs` (`dnsmos`, `segmentalSnr`, `melCepstralDistortion`, `logSpectralDistance`; not read) and in `resampleOct` (a `gcd` helper with an integer guard; `numerics` probably has one).
size: S (minutes)
risk: low
preserves behaviour: yes (the hang becomes an error)
changes public surface: no

### F15: `streaming()` copies all previous batches on every update (quadratic)

kind: hot-path
evidence: `metrics/metric.ts:295-330`: `update` does `trues = trues.concat(t)`, `preds = preds.concat(p)`, `weights = weights.concat(...)` (`:310-313`), each O(total). Probe `h1`: 500 updates of 100 samples 55 ms, 2000 updates 670 ms (4x the updates, 12x the time). Also each batch row of a 2-D input is a nested `reshapeNested` array (`:338`), and `compute` re-reads all rows through `asTensor` at the end.
what: `streaming` is the torchmetrics-style per-batch accumulator; a validation loop of a few thousand batches pays quadratic time before it computes anything. The docs say "batches are stored and the metric is computed on their concatenation", so storing is intended; the copying is not.
fix: `trues.push(...)` in a loop (or keep `unknown[][]` of batches and flatten once in `compute`).
size: S (minutes)
risk: low
preserves behaviour: yes
changes public surface: no

### F16: a `spec` builder is written in 18 metrics files

kind: duplicate
evidence: `metrics/classification.ts:41`, `cluster-validity.ts:13`, `clustering.ts:13`, `deviance.ts:12`, `regression.ts:17`, `probabilistic.ts:21`, `image-quality.ts:114`, `scoring-rules.ts:39` (and `audio.ts:63`, `classification-summary.ts:24`, `correlation-agreement.ts:26`, `distances.ts:11`, `forecasting.ts:25`, `generative.ts:13`, `operating-points.ts:14`, `ranking.ts:22`, `segmentation.ts:12`, `text.ts:21`, plus `_scores.ts:12` `scoreSpec`, `ranking.ts:362` `beyondSpec`, `foundations.ts:396` `decisionSpec`): each is 8 lines building a `MetricSpec` with `name`, `direction`, `range` and three fixed strings (`inputs`, `category`, `concept`); `deviance.ts:12-19` and `regression.ts:17` are identical (same category `regression`, concept `regression-metrics`, inputs `values`); the Phase 0 report lists `spec` x9 in the strict tier. `map` (`deviance.ts:10`, `regression.ts:10`), `median` (`regression.ts:14`), `mean` (`image-quality.ts:123`), `norm` (`audio.ts:408`), `sum` (`_ter.ts:137`), `zip` (`image-quality.ts:44`) are the other local helpers (see F18).
what: a missing field or a new field of `MetricSpec` is edited in 21 places.
fix: `specOf(category, inputs, concept)` in `metric.ts` returning `(name, direction, range) => MetricSpec`; each file keeps one line. About 150 lines go.
size: S (<1h, mechanical)
risk: low (the specs are compared by `evaluate`/docs; the surface digest is unchanged)
preserves behaviour: yes
changes public surface: no

### F17: statistics written again inside `metrics/foundations.ts`: BCa/percentile bootstrap, `percentile`, `phi`/`phiInv`, Wilson interval

kind: duplicate
evidence: `metrics/foundations.ts:76-125` `bootstrapInterval` re-implements percentile, basic and BCa intervals (bias correction from `reps.filter(r < value)`, jackknife acceleration over n metric evaluations) that `stats/resampling.ts:315-515` (`bootstrap`, method `percentile|basic|bca|studentized`, `jackknifeValues`) already provides; the two differ: the metrics version drops non-finite replicates silently (`:71` `if (Number.isFinite(v)) out.push(v)`), clamps `z0` at 1e-12, does not count ties in the bias correction, and is O(n) metric calls per interval for BCa. `:27-28` `phi = item(ndtr(x))`, `phiInv = item(ndtri(p))` are `stats/_util.ts:75,77` `normCdf`/`normPpf` (private there, so metrics could not import them); `:47-54` `percentile` is numpy's linear percentile that `stats/quantiles.ts:125` `quantile` implements, and so is `metrics/segmentation.ts:190`; `:121-127` `wilson` is `stats/tests/proportions.ts` `proportionConfint({ method: 'wilson' })`. `metrics/metric.ts:214-241` `weightedPercentile` (sklearn's) is used by `trees/criteria.ts` (own `weightedMedian` at `:93` as well), `ensembles/_boosting.ts:351`, `ensembles/adaboost.ts:256` (`weightedMedianPredict`): three weighted-median implementations beside the metrics one.
what: the library has two bootstrap CIs with different handling of failed resamples; a fix to one does not reach the other. Metrics may import `stats` (they already do, `correlation-agreement.ts:9`).
fix: `metrics/foundations.ts` calls `stats.bootstrap` with the metric as the statistic (paired resampling of rows, the stats version already supports `paired`), `normCdf`/`normPpf` move to a public `stats` helper (or `numerics/special` `ndtr`/`ndtri` scalar wrappers), `percentile` calls `quantile`, `wilson` calls `proportionConfint`, and `weightedPercentile` moves to `stats/quantiles.ts` (see F19).
size: M (1-3h, with the bootstrap interval behaviour decision on failed resamples)
risk: medium (the `ConfidenceInterval` of any metric moves within resampling noise; seed-pinned goldens move)
preserves behaviour: no (interval endpoints for a fixed key)
changes public surface: no

### F18: local copies of helpers inside the three folders, and four public aliases

kind: duplicate
evidence: `dists/families/extended.ts:207-225` `mean`/`variance` weighted (the same as `stats/descriptive.ts:64,85` with `weights`); the "try to read the concrete value of a possibly traced parameter" pattern `toFlat(primalOf(v))` inside a bare `catch` in `dists/distribution.ts:437-442`, `continuous/extreme.ts:46-62` (`shapeSupport`, `signOf`), `bijectors/elementwise.ts:153-157` (also `concreteTwin` in `_numeric.ts`): four copies of "is this value concrete"; `stats/resampling.ts:53,60` and `stats/smoothing/bases.ts:531` each define `factorial`/`binomialCoefficient` (`numerics/special` has `gamma`, `lgamma`, `binom`); `stats/histograms.ts:67` `linspace` (the tensor op exists); `stats/effect-size.ts:226-238` `oneWay` repeats the sums-of-squares loop of `stats/tests/means.ts:256-266` `fOneway` (and `etaSquared` recomputes what `fOneway` already has); `metrics/audio.ts:408` `norm` (`dnrm2`), `image-quality.ts:123` `mean`, `_ter.ts:137` `sum`; `dists/divergences.ts:66` `klGaussian` against `agents/_dreamer.ts:112`; `dists/families/links.ts:202` `logit` (`logitOf`) against `datasets/causal.ts:778` and `numerics/special#logit`, which `dists/bijectors/elementwise.ts:12` already imports (register F-005 covers the dataset copy); `dists/discrete/binary.ts:115` `binomial` (a distribution) and `dists/families/dispersion.ts:172` `binomial` (a GLM family) are two exports with one name, kept apart by the `families` namespace. `stats/design.ts` has four public aliases that call their twin and are never run by a test: `completelyRandomised` (`:118`) -> `simpleRandomisation`, `fullFactorial` (`:836`) -> `fullFactorialDesign`, `fractionalFactorial` (`:1086`), `matchedPairs` (`:752`) -> `matchedPairsDesign`.
what: small, but each is a place a fix does not reach. The aliases add 4 names to the surface and 4 never-run functions.
fix: `isConcrete(v)`/`concreteValues(v)` in `tensor` (or `dists/_numeric.ts`) replacing the four `try/catch`es (F21); weighted moments from `stats/descriptive`; `factorial`/`binom` from `numerics/special`; `etaSquared` takes the sums from one shared `oneWaySums`; decide whether the four design aliases are API (then test them in the F27 table) or remove them.
size: S-M (1-2h)
risk: low
preserves behaviour: yes
changes public surface: possibly (aliases)

### F19: layering: three metrics files import constructed domains, one stats function imports `information`, and a generic statistic is placed in `metrics`

kind: layering
evidence: `metrics/audio.ts:11` (`dtw, firwin, hann, resamplePoly, stft` from `signal`), `metrics/image-quality.ts:9` (`signal` image filters), `metrics/text.ts:12-17` (`lcsTable, ngramCounts, clippedOverlap, porterStem, countTotal` from `text`); no cycle: `signal`, `text` and `information` import neither `metrics` nor `stats`. `stats/tests/independence.ts:9` imports `mutualInformation` from `information` for one contingency-table statistic (`:184`). `metrics/metric.ts:214` `weightedPercentile` is imported by `trees/criteria.ts`, `ensembles/_boosting.ts` and (in `deviance.ts`) `metrics` itself: a statistic that two constructed domains reach through the metrics barrel. `stats` imports `optim` (`fit.ts:32` `familySolvers`, `SmoothProblem`), `trace`, `space`, `dists` (power, effect-size, extremes, transforms, fit): sound under `dists < kernels < stats`. `dists` imports only `base`, `tensor`, `numerics`, `random`, `autodiff`; metrics -> `stats` (correlation-agreement), `optim` (`linearSumAssignment`, audio), `dists` (forecasting, scoring rules: `normal` and a type), `exposition` (contingency).
what: this is register F-010 (core -> constructed) with specifics: the three metric files are the text, audio and image metrics, which are consumers of `text`/`signal` and could live beside them (`signal/metrics`, `text/metrics`) and be re-exported from `metrics` by an index in the constructed domain, or the five `signal`/`text` kernels they use are lower-level and could move down. `stats -> information` is one function.
fix: do not do this before the maintainer settles the layer order (register F-011/FD-17); then (1) `weightedPercentile` to `stats/quantiles.ts`, re-exported from `metrics` for compatibility; (2) the table-MI of `independence.ts` takes `numerics`/`stats` counts directly (it is `sum p log(p/(p_i p_j))`, 6 lines); (3) text/audio/image metrics move to their domain, or stay with the layering exception recorded.
size: M (1-3h for (1) and (2)); (3) is a decision
risk: low (1, 2) to medium (3: import paths)
preserves behaviour: yes
changes public surface: (1) no (re-export), (3) yes (entry points)

### F20: error codes: `domain` against `value`, `shape` for counts, inner-call names in messages

kind: consistency
evidence: `new OmaiError(...)` counts (non-test): dists `value` 33, `shape` 30, `not-implemented` 4, `convergence` 1; stats `value` 104, `domain` 65, `shape` 52, `singular` 4, `convergence` 3; metrics `shape` 52, `value` 32. `domain` is used in ten stats files for the same condition `value` means elsewhere ("threshold must be finite and > 0", "n must be a positive integer": `stats/concentration.ts` 28, `design.ts` 18, `shrinkage.ts` 5, `experiments.ts` 4, `missing.ts` 3, `descriptive.ts` 2, `effect-size.ts`, `extremes.ts`, `fit.ts`, `transforms.ts`); `shape` carries "need at least 2 treated and 2 control observations" (`experiments.ts:118`), "need at least 2 rows" (`missing.ts:277`), "strata must have at least 1 row" (`design.ts:250`), "jamesStein: shrinking towards a fixed target requires dimension p >= 3" (`shrinkage.ts`), "intraclassCorrelation: needs at least two targets and two raters" (`correlation-agreement.ts:283`), in metrics `logLoss([],[])` ("needs [N, C] probabilities", `probabilistic.ts:456`). `fOneway([[1],[2],[3]])` reports `levene: needs at least 2 observations` (F3c).
what: the code is part of the API (callers switch on it, see numerics F10/F12 for the same in `numerics`); three codes mean "bad argument value".
fix: `value` for a bad number or a too-small sample, `shape` only when ranks or dimensions disagree; rename the 65 `domain` throws (the code is in `base/errors.ts:12` and may be used by another folder; check before removing it from the union); wrap the inner calls of `fOneway`/`etaSquared` so the outer name appears.
size: S (<1h with a replace, plus the check of tests that assert the code)
risk: low-medium (tests that assert `domain`)
preserves behaviour: no (error code of invalid input)
changes public surface: yes (error codes)

### F21: bare `catch {}` swallows every exception in six places (extends optim F30)

kind: rule
evidence: `dists/distribution.ts:438-442` (`validateParams`), `dists/continuous/extreme.ts:50-54` and `:60-64`, `dists/bijectors/elementwise.ts:153-157`, `dists/families/extended.ts:179-183` (`valueAndGrad(f)(x)` failure becomes `logLik = -Infinity`, "the model is infinitely bad here"), `stats/extremes.ts:48` and `stats/fit.ts:605-609` (`invSpd` failure becomes a NaN covariance and NaN standard errors with no flag). The first four detect a traced value (the `toFlat(primalOf(v))` of a value with no concrete data throws); the last three turn a numerical failure or a bug in the user's log-likelihood into a plausible output.
what: a typo in a log-likelihood passed to the extended families is a flat `-Infinity` and the optimiser reports "converged at the start". The tracer test should not need an exception.
fix: `isConcrete(v: Value): boolean` in `tensor` (the data presence is checkable without throwing: the `Value` kinds are tensors or tracers) used by the four; for `extended.ts:179` catch only `OmaiError` with code `value`/`not-differentiable` and rethrow the rest, and report the failure on the result (`converged: false`); for `fit.ts:605` and `extremes.ts:48` return the covariance as NaN **and** a `warnings`/`method` note on the fit.
size: S-M (1-2h)
risk: low
preserves behaviour: yes for the tracer checks, no for the swallowed bugs (they surface)
changes public surface: no

### F22: the `Distribution` contract is "implemented" by throwing: 26 `unavailable(...)` sites and only `hasRsample` tells a caller

kind: rule
evidence: `dists/distribution.ts:83-105` (`Distribution` has `hasRsample` as its one capability flag), `:145-147` `unavailable`; call sites: `relaxed.ts:159-162` (the relaxed one-hot categorical has no `mean`, `variance`, `mode`, `entropy`), `multivariate/matrix.ts:298-301,344` (LKJ moments), `compose/structure.ts:152-162,399-400,463,552` (transformed, censored, empirical, deterministic), `compose/mixture.ts:189`, `families/dispersion.ts:402-403` (Tweedie), `multivariate/copulas.ts:176,233,322-323` (`cdf` for `d > 2`); and the `defineDistribution` defaults (`:171-179`) throw for `cdf`, `ppf`, `rsample` when the spec omits them. No test enumerates which families lack which methods (F5), the docs generated from `@example`s cannot say it, and `exposition`/`render` code that wants "the entropy if there is one" must `try`/`catch` a `OmaiError`.
what: the rule is that a `Distribution` implements the whole contract; 26 sites implement "throw" for one method. That is acceptable when it is queryable.
fix: either `capabilities: ReadonlySet<'cdf' | 'ppf' | 'rsample' | 'mean' | 'variance' | 'mode' | 'entropy'>` on `Distribution` (set by `defineDistribution` from which spec members exist and not `unavailable`: a spec can pass `entropy: undefined`), or return `NaN`-valued tensors for the moments (what `cauchy.mean` does, `heavy-tailed.ts:150`) and document the throwing ones. The first keeps `hasRsample` as a special case of the second.
size: M (1-3h, additive)
risk: low
preserves behaviour: yes
changes public surface: yes (an added member)

### F23: native arrays in public signatures of `dists` and `stats`, classified (6 + 28 functions)

kind: native-array
evidence: probe `sigs2` (`signatures.mjs` over dists/stats/metrics, 55 rows). **dists (6): shape, axis or index, 0 to migrate**: `distribution.ts` `batchShapeOf` (returns a shape), `extendedShape`, `broadcastBatch(batchShape)`; `bijectors/flows.ts` `madeParameters(hiddenSizes)`, `structured.ts` `permute(permutation)` (indices), `madeMasks(hiddenSizes)`. The `Distribution` contract itself is tensor-typed throughout (`TensorLike` in, `Value` out); `dists` is the cleanest of the three. **stats (28)**: _shape, level or index (9)_: `design.ts` `fullFactorialDesign`, `fullFactorial` (level counts), `aliasStructure`, `fractionalFactorialDesign`, `fractionalFactorial` (generator index sets), `tests/independence.ts` `partialCorrelationTest`, `gTestConditional`, `kernelCITest` (`z`: conditioning column indices), `tests/kernel.ts` `permutationNullAlgorithm` (`statistic(perm)`: a permutation). _Kernel, allowed (6)_: `tests/distribution.ts` `powerDivergenceStatistic(observed, expected)` (scalar of two arrays, used by loops), `tests/kernel.ts` `centreGram(K: Float64Array, n)`, `correlation.ts` `centredDistances(m: Rows)` and `distanceCorrelationOf(A, B, n, perm)` (flat matrices with an `n`), `smoothers.ts` `runningStatistic` (its `statistic(values: Float64Array)` callback) and `resampling.ts` `jackknife` (`statistic(x: Float64Array)`): all six are exported from public modules (`centreGram`, `centredDistances`, `distanceCorrelationOf` have no `@example`/`@param`/`@returns`, F29) and belong in `_` modules. _Public vector or matrix, migrate (13)_: `fit.ts` `newtonMle(eta0)`, `newtonMleAlgorithm(eta0)`, `observedInformation(theta)` (start vectors; the `logLikelihood` callbacks are already `Value`-typed, which is what lets `grad` reach them), `ranks.ts` `ranksOf(x): Float64Array` and `tieCounts(): number[]` (a public vector in and out: take `TensorLike`, return `Tensor`, keep the array kernel as `_ranks`), `smoothing/basis-derivatives.ts` `bsplineDerivativeDesign(knots, x)` returning `{ data: Float64Array, cols }` (a design matrix), `smoothing/penalties.ts` `constraintNullSpace(C: ArrayLike<number>, k)` (a flat matrix and its width; returns `Tensor`), `smoothers.ts` `smoothingSpline` option `weights: number[]`, `resampling.ts` `meanDifference(x: Float64Array, y)`, `tests/proportions.ts` `proportionsZtest(count, nobs: number | number[])`, `tests/result.ts` `checkExpectedCounts(expected)`, `tests/sequential.ts` `alphaSpending(times)`, `tests/two-sample.ts` `eppsSingleton` option `t`.
what: the stats module already works the way the rule asks for its main entry points (`TensorLike` data, `Tensor` outputs for matrices, `number` for scalars); the migrate list is 13 leaf functions with small signatures. Input records the script cannot see: `Rows` (`stats/_util.ts:36`, private), `OlsFit` (`_linear.ts`, private).
fix: migrate the 13 in one step each (widening parameters is source compatible; return types are the break: `ranksOf`, `tieCounts`, `bsplineDerivativeDesign`); move the 6 kernels to `_` modules or document them as kernels; no change to the 9 shape/index ones.
size: S-M (1-2h: most are one parameter)
risk: low-medium (return-type changes of `ranksOf`/`tieCounts`, used by `sampling`, `learn`, `cluster`)
preserves behaviour: yes (values), no (return types of three)
changes public surface: yes (the deliberate break, D-009)

### F24: native arrays in public signatures of `metrics` (21 functions) and the right input type of a metric

kind: native-array
evidence: probe `sigs2`. **Kernel, allowed (5)**: `metric.ts` `sampleWeights(...) : Float64Array`, `flat(x): Float64Array`, `perOutputMetric(score: (yTrue: Float64Array, ...) => number)` (callback), `weightedMean(values, weight: ArrayLike<number>)`, `audio.ts` `permutationInvariant(metric: (target: Float64Array, estimate: Float64Array) => number)` (callback); they are the readers every metric uses and belong in a `_` module (they are exported through the barrel and have no `@example`, F29). **Index, label or id (5)**: `metric.ts` `selectRows(x, rows: readonly number[])`, `foundations.ts` `bayesDecision(...) : number[]` (decided class indices), `interleaving.ts` `teamDraftInterleave(a, b: readonly number[])` and `interleavingCredit(teams, clicks)` (document ids), `sequence-labelling.ts` `bioViolations(tags): number[]` (positions). **Public vector or matrix, migrate (11)**: `metric.ts` `weightedPercentile(values, weights: ArrayLike<number>)` (moves to stats, F19), `cluster-validity.ts` `silhouetteSamples(...) : number[]`, `clustering.ts` `pairConfusionMatrix(...) : number[][]`, `classification.ts` `multilabelConfusionMatrix(...) : number[][][]`, `contingency.ts` `tableCost(table, costs: number[][])` and `thresholdCosts(..., costs: number[][])`, `curves.ts` `auc(x: readonly number[], y: readonly number[])`, `detection.ts` `boxIou`, `generalizedBoxIou`, `distanceBoxIou`, `completeBoxIou` (`Boxes` in, `number[][]` out). Record fields the script cannot see: `ConfusionMatrix.matrix: readonly (readonly number[])[]` (`classification.ts:525-528`), the curve records (`rocCurve`, `precisionRecallCurve`, `calibrationCurve`; fields not read), `Recommendations` in the ranking metrics.
what: a metric consumes labels and scores and produces a number; under the rule the input type is `MetricInput = TensorLike | readonly string[]` (as it is: labels are numbers or strings, scores `TensorLike`, weights `TensorLike`) and the output a `MetricResult` (a plain number inside a record: metrics are not differentiated, so no `Value` is needed, and that is the right call). What the rule does ask is that **vector- and matrix-valued outputs are `Tensor`s**: `silhouetteSamples`, the four box IoUs, `pairConfusionMatrix`, `multilabelConfusionMatrix`, `ConfusionMatrix.matrix`, the curve records; and that **matrix inputs** (`costs`, `Boxes`, `x`, `y` of `auc`) are `TensorLike`. Label lists and fixed-size boxes are not arrays in the rule's sense.
fix: widen the 5 vector inputs to `TensorLike`; return `Tensor` from the 6 vector/matrix outputs and the record fields (`ConfusionMatrix.matrix: Tensor` is the one with callers: `classification-summary.ts`, and probably `learn` report code and `render` views; callers not checked); move `weightedPercentile` to stats; the 5 kernels to `_`.
size: M (1-3h; the `ConfusionMatrix` record has the callers)
risk: medium (the return-type changes; `matrix` is read as nested arrays by its callers)
preserves behaviour: yes (values), no (types)
changes public surface: yes (deliberate)

### F25: test gap: the hypothesis tests have no table of degenerate inputs; branch coverage 69.5% with 1,793 branches missed

kind: test-gap
evidence: coverage (probe `unrun2`): stats 93.8% statements, 96.5% functions, **69.5% branches** (1,793 missed): `design.ts` 61% (182 missed), `smoothing/bases.ts` 67% (144), `experiments.ts` 60% (123), `smoothing/smoothers.ts` 60% (79), `missing.ts` 60% (74), `correlation.ts` 70% (73), `shrinkage.ts` 51% (70), `tests/independence.ts` 66% (62), `tests/two-sample.ts` 68% (62), `smoothing/penalties.ts` 58% (52). The degenerate branches of the tests (constant data, ties, n = 1, all-equal, zero row) are the missed ones; F3 and F4 found 8 wrong answers there in an hour of probing.
what: one table covers them: rows `[test, call, expectation]` for the 60 or so tests with a branch on degenerate input, each cell one of `throws value`, `statistic NaN and p NaN`, or `a documented limit`, plus two generic properties that need no oracle: (1) for every test taking `alternative`, `p_less + p_greater >= 1` (continuous statistics) and `p_two-sided ~ min(1, 2 min(p_less, p_greater))`; (2) `0 <= p <= 1` or NaN, and `statistic` NaN iff `p` NaN.
fix: `stats/tests/degenerate.test.ts`: about 5 `it.each` blocks over the table (empty, n = 1, constant, ties/all-equal, NaN) plus the two properties; +5 test cases, ~60 rows; it fails today on F2, F3, F4 and is the regression guard for them.
size: M (a day)
risk: low
preserves behaviour: yes
changes public surface: no

### F26: test gap: no generic test of the metrics' weights, empty input and NaN policy

kind: test-gap
evidence: metrics coverage is 95.9% statements, 98.1% functions, **68.1% branches** (1,293 missed: `text.ts` 71% (94 missed), `image-quality.ts` 62% (83), `segmentation.ts` 63% (75), `detection.ts` 70% (72), `audio.ts` 65% (68), `probabilistic.ts` 67% (67), `curves.ts` 66% (66), `operating-points.ts` 65% (57), `cluster-validity.ts` 58% (52), `classification-summary.ts` 74% (51)); the probes `m2` and `m7` are generic sweeps over `Object.entries(metrics)` that took seconds and found F1, F2, F13, F14.
fix: `metrics/policy.test.ts`: (1) `it.each(numericMetrics)` "integer sample weights equal repeated rows" (`m(a, b, { sampleWeight: w })` equals `m(repeat(a, w), repeat(b, w))`) with an explicit exclusion list for metrics that declare `weights: false` (F1); (2) "n = 0 throws `value` or is NaN, never a number" (F13); (3) "a NaN in the scores or labels gives NaN or throws" (F2); (4) "unweighted equals weights of ones"; (5) the zero-positive and single-class cases against a small expected table. One file, +5 cases over ~100 metrics (the sweep is the registry: `Object.values(metrics).filter(m => m.spec)`), and new metrics are covered at once.
size: M (half a day)
risk: low
preserves behaviour: yes
changes public surface: no

### F27: test gap: 31 named functions never run, among them the multiple-testing wrappers and the conditional distributions

kind: test-gap
evidence: probe `unrun2` (named functions with zero calls in the vitest run; some may run in `@example`s): **stats (14)**: `tests/multiple.ts:151,181,196,212` `bonferroni`, `hochberg`, `benjaminiHochberg`, `benjaminiYekutieli` (thin wrappers over `multipleTests`, which is tested), `tests/result.ts:189,221` `formatTest`, `rejects`, `tests/kernel.ts:352` `hsic`, `robust.ts:69` `trimmedVariance`, `smoothing/penalties.ts:142` `shrinkagePenalty`, `_util.ts:47` `column`, `design.ts:118,752,836,1086` the four aliases (F18); **dists (9)**: `distribution.ts:236,271` `extendedShape`, `supportRank`, `divergences.ts:303` `sameFixed`, `compose/conditional.ts:271,287,303,340,358` `bernoulliFromOutput`, `categoricalFromOutput`, `poissonFromOutput`, `betaFromOutput`, `gammaFromOutput`, `multivariate/copula-fitting.ts:116` `repairCorrelation`; **metrics (2)**: `image-quality.ts:126` `rangeOf`, `text.ts:557` `normaliseAsian`. Lowest statement coverage in a sizeable file: `stats/shrinkage.ts` 76.2% (51% of its branches), `stats/smoothing/penalties.ts` 85%, `stats/concentration.ts` 84%, `dists/distribution.ts` 73.4%.
fix: a single table per folder, one `it.each` each: stats `[bonferroni, hochberg, benjaminiHochberg, benjaminiYekutieli]` against `multipleTests(method)` on one p-vector, `rejects`/`formatTest` on the F4 shapes, the four design aliases equal to their twins; dists the five `*FromOutput` mappers on one output tensor each (a conditional-distribution table), `supportRank` and `extendedShape` in the F5 table; metrics two cases. About +4 test cases, ~25 rows. Not worth tests: `column`, `sameFixed` if they are only dead helpers (check callers; delete if unused).
size: S-M (2-3h)
risk: low
preserves behaviour: yes
changes public surface: no

### F28: hot-path note: stats and metrics are healthy; measurements for the record

kind: hot-path
evidence: probes `h1`, `h3` (ms, after one warm-up): `ttestInd` 2000 9.0, `mannWhitneyU` 2000 1.5, `ks2samp` 2000 31 (the exact distribution), `shapiro` 2000 3.3, `wilcoxon` 2000 1.2, `kruskal` 3 x 2000 2.3, `fOneway` 3 x 2000 **17.6** (the Shapiro-Wilk on each of 3 groups alone is about 10 ms by the `shapiro` timing, then Levene and the eta-squared interval it computes on every call), `fisherExact` big 1.9, `spearman` 5000 2.7, `kendallTau` 20000 12, `distanceCorrelation` 1500 38, `distanceCovarianceTest` 300 with its permutations 224, `permutationTest` 2000 x n = 200 68, `bootstrap` 2000 x 200 78 (BCa as fast as percentile), `normPpf` scalar 2.7 us per call (`ndtri` itself; the scalar wrappers `stats/_util.ts:104-115` cost the same per element as one tensor call of 2e4). Metrics at n = 1e5: `accuracyScore` 29, `f1Score` macro 25, `confusionMatrix` 22, `rocAucScore` 54, `averagePrecisionScore` 45, `meanSquaredError` 11, `r2Score` 15, `logLoss` 17, `adjustedRandScore` 16, `spearmanCorrelation` 57, `silhouetteScore` n = 1000 236 (O(n^2), as expected), `ndcgScore` 1000 x 50 61. Dists at 1e5 evaluations: `normal.logProb` 9.5, `studentT` 10.6, `gamma` 15.7, `poisson` 30, `beta` 49, `binomial` 66, `normal.cdf` 70, `gamma.cdf` 26, `poisson.cdf` 24; `ppf` of 1e4 quantiles: `gamma` 17, `beta` 38, `studentT` 71. The only hot paths found are F12 (sampling) and F15 (`streaming`).
what: no work needed beyond F12 and F15; the one avoidable cost is `fOneway`'s unconditional assumption checks and effect-size interval (an `options.assumptions: false` would let loops skip them).
fix: none (`fOneway`, `ttest*`, `welchAnova` could take `assumptions: false`; with F3c this also removes the failure path).
size: S
risk: low
preserves behaviour: yes
changes public surface: maybe (one option)

### F29: TSDoc gaps: 11 exported functions of the three folders lack `@example`, `@param` or `@returns`; `quantile` overloads have no doc

kind: rule
evidence: `tsdoc.mjs` over each folder (non-private modules): dists 203 functions, **0 gaps**; stats 303 functions: `quantiles.ts:126,127` (`quantile` overload signatures: the doc is on the implementation only), `correlation.ts:478,505` `centredDistances`, `distanceCorrelationOf`, `tests/kernel.ts:304` `centreGram` (no `@example`, `@param`, `@returns`); metrics 64: `deviance.ts:28` `tweedieUnitDeviance`, `metric.ts:146` `columns`, `:160` `perOutputMetric`, `:188` `weightedMean`, `:244` `rowCount`, `:250` `selectRows`, `:342` `isStringInput` (a one-line summary only). All of the metrics ones are the helpers every metric uses, exported through the barrel.
what: the rule is `@param`/`@returns`/`@example` for every core export; `dists` shows it can be met.
fix: add the three tags to the 11, or move the helpers to `_` modules where the rule does not apply (F23/F24 propose that for six of them anyway); the `quantile` overloads need the doc on the first signature.
size: S (<1h)
risk: low
preserves behaviour: yes
changes public surface: no

### F30: smaller consistency notes (confirmed, each a line or two)

kind: consistency
evidence: (a) `dists/distribution.ts:190-197`: the defaults `logCdf = log(cdf)`, `sf = 1 - cdf`, `isf = ppf(1 - q)` lose the tail; the families that matter override them (normal, heavy-tailed, gamma family, discrete: probe `d4` shows every tested family accurate in the upper tail), but a user-defined family via `defineDistribution` silently gets `sf(x) = 0` where the tail is 1e-17; the TSDoc of `defineDistribution` lists the defaults without that warning. (b) `metrics/metric.ts:245-247` `rowCount` accepts an empty array as strings (`every` on `[]` is true) while `isStringInput` (`:343`) requires length > 0. (c) `stats/shrinkage.ts` `jamesStein` with `p < 3` throws `shape` (F20) although a positive-part version exists; `ledoitWolf` of one row returns an all-zero covariance with shrinkage 0 and no flag (sklearn warns); `oas` with `p > n` returns shrinkage 1. (d) `stats/tests/result.ts:384` `checkEqualVariances` maps a NaN p to 1 ("variances equal"); `checkNormality` skips constant samples (`:359-361`) and reports `normal: true` for them.
fix: (a) one sentence on `defineDistribution` and `sf = exp(logSf)` when `logSf` is given; (b) `isStringInput` as the single definition; (c, d) NaN for undefined, per F4.
size: S
risk: low
preserves behaviour: yes except (e)
changes public surface: no
