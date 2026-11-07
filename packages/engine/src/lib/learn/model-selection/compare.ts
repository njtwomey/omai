/**
 * Comparing learning algorithms from cross-validation or resampling scores: the corrected resampled t-test of Nadeau
 * and Bengio (the plain resampled t-test is anti-conservative because the folds overlap), Dietterich's 5x2 cv paired
 * t-test and Alpaydin's 5x2 cv F-test, the Bayesian correlated t-test with a region of practical equivalence (Benavoli
 * et al.), and the Demsar procedures for many algorithms over many data sets (Iman-Davenport corrected Friedman test,
 * Nemenyi critical difference with its cliques, comparisons with a control by Bonferroni-Dunn and Holm). McNemar and
 * Wilcoxon are in `stats/tests`; this file supplies the ones that depend on how the scores were resampled.
 */

import { type Alternative, MdmlError, meanOf, type TestResult, varianceOf } from '../../base/index.ts'
import { studentisedRange } from '../../dists/index.ts'
import { fdtrc, ndtr, stdtr, stdtrit } from '../../numerics/special/index.ts'
import { type Key, key as makeKey } from '../../random/index.ts'
import { ranksOf } from '../../stats/index.ts'
import { friedman, holm, type TestOptions, testResult } from '../../stats/tests/index.ts'
import { asTensor, item, type TensorLike, toFlat } from '../../tensor/index.ts'
import type { Estimator, LearnData } from '../estimator.ts'
import { type Scorer, type ScorerName, scorerFor } from '../scoring.ts'
import { takeData } from '../validation.ts'
import { splitInputOf } from './cross-validation.ts'
import { repeatedKFold, type Splitter, splitIndices } from './splitters.ts'

const tCdf = (df: number, t: number): number => item(stdtr(df, t))
const tPpf = (df: number, p: number): number => item(stdtrit(df, p))
const normCdf = (x: number): number => item(ndtr(x))

function vector(who: string, x: TensorLike, min: number): Float64Array {
  const v = Float64Array.from(toFlat(asTensor(x)))
  if (v.length < min) throw new MdmlError('value', `${who}: needs at least ${min} values`)
  return v
}

function pFromT(alternative: Alternative, t: number, df: number): number {
  const lo = tCdf(df, t)
  if (alternative === 'less') return lo
  if (alternative === 'greater') return 1 - lo
  return Math.min(1, 2 * Math.min(lo, 1 - lo))
}

/** Options of {@link correctedResampledTTest}. */
export interface CorrectedResampledOptions extends TestOptions {
  /** Size of each training set (n1). */
  readonly nTrain: number
  /** Size of each test set (n2). */
  readonly nTest: number
}

/**
 * The (uncorrected) resampled t-test: the one-sample t-test of the differences between two algorithms' scores over k
 * resampled train/test splits, treated as independent. It ignores that the training sets overlap, so its variance
 * estimate is too small and its type I error far above the nominal level; it is here as the thing to compare the
 * corrected test with.
 *
 * @param differences - The score difference (A minus B) on each of the k splits.
 * @param options - The alternative (default two-sided).
 * @returns The t statistic with k - 1 degrees of freedom and its p-value; the estimate is the mean difference.
 * @concept model-comparison
 * @example
 * ```ts
 * import { resampledTTest } from 'mdml-engine/learn/model-selection'
 * resampledTTest([0.1, 0.2, 0.15, 0.05, 0.1]).df // => 4
 * ```
 */
export function resampledTTest(differences: TensorLike, options: TestOptions = {}): TestResult {
  const d = vector('resampledTTest', differences, 2)
  const k = d.length
  const se = Math.sqrt(varianceOf(d, 1) / k)
  const t = meanOf(d) / se
  const alternative = options.alternative ?? 'two-sided'
  return testResult({
    test: 'resampled t-test',
    statistic: t,
    pValue: pFromT(alternative, t, k - 1),
    alternative,
    df: k - 1,
    estimate: { value: meanOf(d), standardError: se, method: 'mean difference' },
  })
}

/**
 * The corrected resampled t-test (Nadeau and Bengio 2003): the variance of the mean difference over k random splits
 * is (1/k + n2/n1) s^2 rather than s^2 / k, because the scores share training data (their correlation is about
 * n2 / (n1 + n2)); the statistic is referred to Student's t with k - 1 degrees of freedom. Valid for k random
 * train/test splits or k-fold cross-validation (n2 / n1 = 1 / (k - 1)) repeated; the correction holds the type I error
 * near its level where the plain test rejects far too often.
 *
 * @param differences - The score difference (A minus B) on each of the k splits.
 * @param options - Training and test set sizes (required) and the alternative.
 * @returns The corrected t statistic, its p-value and k - 1 degrees of freedom.
 * @throws MdmlError with code `value` for fewer than two differences or non-positive sizes.
 * @concept model-comparison
 * @example
 * ```ts
 * import { correctedResampledTTest } from 'mdml-engine/learn/model-selection'
 * const r = correctedResampledTTest([0.1, 0.2, 0.15, 0.05, 0.1], { nTrain: 80, nTest: 20 })
 * r.statistic // => 3.1379
 * ```
 */
export function correctedResampledTTest(differences: TensorLike, options: CorrectedResampledOptions): TestResult {
  const d = vector('correctedResampledTTest', differences, 2)
  if (!(options.nTrain > 0 && options.nTest > 0))
    throw new MdmlError('value', 'correctedResampledTTest: nTrain and nTest must be positive')
  const k = d.length
  const se = Math.sqrt((1 / k + options.nTest / options.nTrain) * varianceOf(d, 1))
  const t = meanOf(d) / se
  const alternative = options.alternative ?? 'two-sided'
  return testResult({
    test: 'corrected resampled t-test',
    statistic: t,
    pValue: pFromT(alternative, t, k - 1),
    alternative,
    df: k - 1,
    estimate: { value: meanOf(d), standardError: se, method: 'mean difference' },
    method: 'Nadeau-Bengio',
  })
}

/** Five replications of two-fold cross-validation: row i holds the score differences on the two folds of replication i. */
function fiveByTwo(who: string, differences: TensorLike): readonly (readonly [number, number])[] {
  const v = vector(who, differences, 10)
  if (v.length !== 10) throw new MdmlError('shape', `${who}: expects 5 x 2 differences, got ${v.length} values`)
  return Array.from({ length: 5 }, (_, i): readonly [number, number] => [v[2 * i] ?? 0, v[2 * i + 1] ?? 0])
}

/**
 * Dietterich's 5x2 cv paired t-test (1998): five replications of two-fold cross-validation; with p_i^(j) the score
 * difference on fold j of replication i, p-bar_i their mean and s_i^2 = (p_i^(1) - p-bar_i)^2 + (p_i^(2) - p-bar_i)^2
 * the replication's variance, t = p_1^(1) / sqrt(sum s_i^2 / 5) on 5 degrees of freedom. Its type I error is close to
 * its level and it is cheap, though it uses only one of the ten differences in the numerator.
 *
 * @param differences - Ten score differences (A minus B), replication by replication: [r1f1, r1f2, r2f1, ...].
 * @param options - The alternative (default two-sided).
 * @returns The t statistic with 5 degrees of freedom and its p-value.
 * @throws MdmlError with code `shape` unless there are exactly ten differences.
 * @concept model-comparison
 * @example
 * ```ts
 * import { fiveByTwoCvTTest } from 'mdml-engine/learn/model-selection'
 * fiveByTwoCvTTest([0.1, 0.12, 0.08, 0.1, 0.11, 0.09, 0.12, 0.1, 0.1, 0.1]).df // => 5
 * ```
 */
export function fiveByTwoCvTTest(differences: TensorLike, options: TestOptions = {}): TestResult {
  const p = fiveByTwo('fiveByTwoCvTTest', differences)
  let sum = 0
  for (const [a, b] of p) {
    const m = (a + b) / 2
    sum += (a - m) ** 2 + (b - m) ** 2
  }
  const t = (p[0]?.[0] ?? 0) / Math.sqrt(sum / 5)
  const alternative = options.alternative ?? 'two-sided'
  return testResult({
    test: '5x2 cv paired t-test',
    statistic: t,
    pValue: pFromT(alternative, t, 5),
    alternative,
    df: 5,
    method: 'Dietterich',
  })
}

/**
 * Alpaydin's combined 5x2 cv F-test (1999): f = sum_i sum_j (p_i^(j))^2 / (2 sum_i s_i^2), approximately F with 10 and
 * 5 degrees of freedom under the null of equal error; it uses all ten differences and is more powerful and more
 * robust than Dietterich's t-test.
 *
 * @param differences - Ten score differences (A minus B), replication by replication.
 * @returns The F statistic with (10, 5) degrees of freedom and its p-value.
 * @throws MdmlError with code `shape` unless there are exactly ten differences.
 * @concept model-comparison
 * @example
 * ```ts
 * import { fiveByTwoCvFTest } from 'mdml-engine/learn/model-selection'
 * fiveByTwoCvFTest([0.1, 0.12, 0.08, 0.1, 0.11, 0.09, 0.12, 0.1, 0.1, 0.1]).df // => [10, 5]
 * ```
 */
export function fiveByTwoCvFTest(differences: TensorLike): TestResult {
  const p = fiveByTwo('fiveByTwoCvFTest', differences)
  let num = 0
  let den = 0
  for (const [a, b] of p) {
    const m = (a + b) / 2
    num += a * a + b * b
    den += (a - m) ** 2 + (b - m) ** 2
  }
  const f = num / (2 * den)
  return testResult({
    test: '5x2 cv F-test',
    statistic: f,
    pValue: item(fdtrc(10, 5, f)),
    alternative: 'greater',
    df: [10, 5],
    method: 'Alpaydin',
  })
}

/** Options of {@link bayesianCorrelatedTTest}. */
export interface BayesianCorrelatedOptions {
  /** The correlation between the scores of different splits: n_test / n_total, which is 1 / k for k-fold (default 1 / n). */
  readonly rho?: number
  /** The region of practical equivalence |difference| <= rope (default 0: no region). */
  readonly rope?: number
  /** The credible level of the interval (default 0.95). */
  readonly level?: number
}

/** The posterior of the Bayesian correlated t-test of the mean difference between two algorithms. */
export interface BayesianCorrelatedResult {
  /** The posterior is Student's t with `df` degrees of freedom, centred at `mean`, with this scale. */
  readonly mean: number
  readonly scale: number
  readonly df: number
  /** The equal-tailed credible interval of the mean difference. */
  readonly interval: readonly [number, number]
  /** P(mean difference < -rope): the second algorithm is practically better. */
  readonly left: number
  /** P(|mean difference| <= rope): practically equivalent. */
  readonly rope: number
  /** P(mean difference > rope): the first algorithm is practically better. */
  readonly right: number
}

/**
 * The Bayesian correlated t-test (Corani and Benavoli 2015; Benavoli et al. 2017): with a flat prior the posterior of
 * the mean difference mu is a Student t with n - 1 degrees of freedom, location the sample mean and scale
 * sqrt((1/n + rho / (1 - rho)) s^2), where rho accounts for the correlation between overlapping resamples. The posterior
 * probabilities that mu lies below, inside and above the region of practical equivalence answer the question the
 * p-value does not: how probable is it that one algorithm is practically better.
 *
 * @param differences - The score difference (A minus B) on each of the n splits.
 * @param options - The correlation rho (default 1 / n), the ROPE half-width (default 0) and the credible level.
 * @returns The posterior (location, scale, degrees of freedom), the credible interval and the three probabilities.
 * @concept model-comparison
 * @example
 * ```ts
 * import { bayesianCorrelatedTTest } from 'mdml-engine/learn/model-selection'
 * const r = bayesianCorrelatedTTest([0.02, 0.03, 0.01, 0.04, 0.02, 0.03, 0.02, 0.01, 0.03, 0.02], { rho: 0.1, rope: 0.01 })
 * r.right > 0.9 // => true
 * ```
 */
export function bayesianCorrelatedTTest(
  differences: TensorLike,
  options: BayesianCorrelatedOptions = {},
): BayesianCorrelatedResult {
  const d = vector('bayesianCorrelatedTTest', differences, 2)
  const n = d.length
  const rho = options.rho ?? 1 / n
  if (!(rho >= 0 && rho < 1)) throw new MdmlError('value', 'bayesianCorrelatedTTest: rho must be in [0, 1)')
  const rope = options.rope ?? 0
  const level = options.level ?? 0.95
  const mean = meanOf(d)
  const df = n - 1
  const scale = Math.sqrt((1 / n + rho / (1 - rho)) * varianceOf(d, 1))
  const cdf = (x: number): number => (scale === 0 ? (x >= mean ? 1 : 0) : tCdf(df, (x - mean) / scale))
  const q = tPpf(df, 0.5 + level / 2)
  return {
    mean,
    scale,
    df,
    interval: [mean - q * scale, mean + q * scale],
    left: cdf(-rope),
    rope: cdf(rope) - cdf(-rope),
    right: 1 - cdf(rope),
  }
}

/** The result of a paired cross-validated comparison of two estimators on the same folds. */
export interface PairedScores {
  readonly scoresA: readonly number[]
  readonly scoresB: readonly number[]
  /** `scoresA - scoresB`, fold by fold. */
  readonly differences: readonly number[]
  /** Training and test set sizes of the first fold (the n1 and n2 of the corrected test). */
  readonly nTrain: number
  readonly nTest: number
}

/** Options of {@link pairedCvScores}. */
export interface PairedScoresOptions {
  /** The splitter (default 5x2: `repeatedKFold(2, 5)`, so the scores feed {@link fiveByTwoCvFTest} directly). */
  readonly splitter?: Splitter
  /** The scorer (default each estimator's own `score`). */
  readonly scoring?: ScorerName | Scorer
  /** Key for the splits and the fits (default `key(0)`). */
  readonly key?: Key
}

/**
 * Score two estimators on exactly the same train/test splits, so that the difference is the algorithm and not the
 * split. The result feeds {@link correctedResampledTTest} (with its sizes), {@link fiveByTwoCvTTest},
 * {@link fiveByTwoCvFTest} and {@link bayesianCorrelatedTTest}.
 *
 * @param a - The first estimator.
 * @param b - The second estimator.
 * @param data - The data.
 * @param options - Splitter (default 5x2), scorer and key.
 * @returns Both score vectors, their differences and the set sizes.
 * @concept model-comparison
 * @example
 * ```ts
 * import { pairedCvScores } from 'mdml-engine/learn/model-selection'
 * import { defineEstimator } from 'mdml-engine/learn'
 * import { full, mean, item } from 'mdml-engine/tensor'
 * const meanModel = defineEstimator(
 *   { name: 'mean', kind: 'regressor', fit: (d) => ({ m: item(mean(d.y)) }), methods: (p) => ({ predict: (x) => full([x.shape[0]], p.m) }) },
 *   {},
 * )
 * const data = { x: Array.from({ length: 20 }, (_, i) => [i]), y: Array.from({ length: 20 }, (_, i) => i % 3) }
 * pairedCvScores(meanModel, meanModel, data, { scoring: 'neg_mean_absolute_error' }).differences.every((v) => v === 0) // => true
 * ```
 */
export function pairedCvScores(
  a: Estimator,
  b: Estimator,
  data: LearnData,
  options: PairedScoresOptions = {},
): PairedScores {
  const key0 = options.key ?? makeKey(0)
  const splitter = options.splitter ?? repeatedKFold(2, 5)
  const input = splitInputOf(data)
  const folds = splitIndices(splitter, input, key0)
  const scorer = scorerFor(options.scoring)
  const sa: number[] = []
  const sb: number[] = []
  folds.forEach((fold, i) => {
    const train = takeData(data, fold.train)
    const test = takeData(data, fold.test)
    const fitOptions = { key: makeKey(1000 + i), explain: false }
    sa.push(scorer.score(a.fit(train, fitOptions), test))
    sb.push(scorer.score(b.fit(train, fitOptions), test))
  })
  return {
    scoresA: sa,
    scoresB: sb,
    differences: sa.map((v, i) => v - (sb[i] ?? 0)),
    nTrain: folds[0]?.train.length ?? 0,
    nTest: folds[0]?.test.length ?? 0,
  }
}

/**
 * The critical value q_alpha of the Nemenyi test for k algorithms: the (1 - alpha) quantile of the studentised range
 * distribution with infinite degrees of freedom, divided by sqrt(2) (Demsar 2006, table 5: 2.343 for k = 3, 2.569 for
 * k = 4, 2.728 for k = 5 at alpha = 0.05).
 *
 * @param k - The number of algorithms (at least 2).
 * @param alpha - The significance level (default 0.05).
 * @returns q_alpha.
 * @concept model-comparison
 * @example
 * ```ts
 * import { nemenyiQ } from 'mdml-engine/learn/model-selection'
 * nemenyiQ(3) // => 2.3437
 * ```
 */
export function nemenyiQ(k: number, alpha = 0.05): number {
  return item(studentisedRange(k, Number.POSITIVE_INFINITY).ppf(1 - alpha)) / Math.SQRT2
}

/**
 * The Nemenyi critical difference CD = q_alpha sqrt(k (k + 1) / (6 N)) for k algorithms over N data sets: two algorithms
 * whose average ranks differ by more than CD are significantly different at level alpha.
 *
 * @param k - The number of algorithms.
 * @param nDatasets - The number of data sets (blocks) N.
 * @param alpha - The significance level (default 0.05).
 * @returns The critical difference in rank units.
 * @concept model-comparison
 * @example
 * ```ts
 * import { criticalDifference } from 'mdml-engine/learn/model-selection'
 * criticalDifference(4, 14) // => 1.2536
 * ```
 */
export function criticalDifference(k: number, nDatasets: number, alpha = 0.05): number {
  if (!(Number.isInteger(k) && k >= 2 && nDatasets >= 1))
    throw new MdmlError('value', 'criticalDifference: needs k >= 2 algorithms and at least one data set')
  return nemenyiQ(k, alpha) * Math.sqrt((k * (k + 1)) / (6 * nDatasets))
}

/** Options of {@link compareAlgorithms}. */
export interface CompareAlgorithmsOptions {
  /** Whether larger scores are better (default true); ranks are 1 for the best. */
  readonly higherIsBetter?: boolean
  /** The significance level (default 0.05). */
  readonly alpha?: number
  /** Algorithm names (default `A0`, `A1`, ...). */
  readonly names?: readonly string[]
  /** The index of the control algorithm, to compare the others against it (default the best-ranked). */
  readonly control?: number
}

/** A comparison of the others with a control algorithm. */
export interface ControlComparison {
  readonly control: number
  /** z statistic of each algorithm against the control (0 for the control). */
  readonly z: readonly number[]
  /** Unadjusted two-sided p-values. */
  readonly pValue: readonly number[]
  /** Bonferroni-Dunn adjusted p-values (min(1, (k - 1) p)). */
  readonly bonferroniDunn: readonly number[]
  /** Holm step-down adjusted p-values. */
  readonly holm: readonly number[]
}

/** The result of {@link compareAlgorithms}. */
export interface AlgorithmComparison {
  readonly names: readonly string[]
  readonly nDatasets: number
  /** Average rank of each algorithm (1 is best). */
  readonly averageRanks: readonly number[]
  /** The Friedman chi-square test of equal ranks. */
  readonly friedman: TestResult
  /** The Iman-Davenport F test (Demsar's recommendation: less conservative than the chi-square). */
  readonly imanDavenport: TestResult
  /** The Nemenyi critical difference at `alpha`. */
  readonly criticalDifference: number
  /** Maximal groups of algorithms (indices, best-ranked first) whose ranks all lie within the critical difference. */
  readonly cliques: readonly (readonly number[])[]
  readonly control: ControlComparison
}

/**
 * Demsar's procedure for comparing k algorithms over N data sets (2006): rank the algorithms on each data set
 * (ties share the average rank), test equal average ranks with the Friedman test and the Iman-Davenport F correction,
 * and follow with the Nemenyi critical difference (every pair) and the Bonferroni-Dunn and Holm comparisons against a
 * control (the more powerful way to compare with one baseline). The cliques are the groups of algorithms that the
 * critical-difference diagram joins with a bar.
 *
 * @param scores - An N x k matrix: one row per data set, one column per algorithm.
 * @param options - Direction, alpha, names and the control.
 * @returns Ranks, both omnibus tests, the critical difference, the cliques and the control comparison.
 * @concept model-comparison
 * @example
 * ```ts
 * import { compareAlgorithms } from 'mdml-engine/learn/model-selection'
 * const scores = [[0.9, 0.8, 0.7], [0.85, 0.8, 0.6], [0.95, 0.7, 0.75], [0.9, 0.85, 0.7], [0.8, 0.75, 0.6]]
 * compareAlgorithms(scores).averageRanks // => [1, 2.2, 2.8]
 * ```
 */
export function compareAlgorithms(scores: TensorLike, options: CompareAlgorithmsOptions = {}): AlgorithmComparison {
  const m = asTensor(scores)
  const flat = Float64Array.from(toFlat(m))
  const n = m.shape[0] ?? 0
  const k = m.shape.length === 2 ? (m.shape[1] ?? 0) : 0
  if (!(Number.isInteger(k) && k >= 2 && n >= 2))
    throw new MdmlError(
      'shape',
      'compareAlgorithms: expects an N x k matrix with N >= 2 data sets and k >= 2 algorithms',
    )
  const sign = options.higherIsBetter === false ? 1 : -1
  const alpha = options.alpha ?? 0.05
  const mean = new Float64Array(k)
  for (let i = 0; i < n; i++) {
    const row = Float64Array.from({ length: k }, (_, j) => sign * (flat[i * k + j] ?? 0))
    const r = ranksOf(row).data
    for (let j = 0; j < k; j++) mean[j] = (mean[j] ?? 0) + (r[j] ?? 0) / n
  }
  const fr = friedman(scores)
  const chi2 = fr.statistic
  const ff = ((n - 1) * chi2) / (n * (k - 1) - chi2)
  const imanDavenport = testResult({
    test: 'Iman-Davenport test',
    statistic: ff,
    pValue: Number.isFinite(ff) ? item(fdtrc(k - 1, (k - 1) * (n - 1), ff)) : 0,
    alternative: 'greater',
    df: [k - 1, (k - 1) * (n - 1)],
  })
  const cd = criticalDifference(k, n, alpha)
  const order = Array.from(mean.keys()).sort((a, b) => (mean[a] ?? 0) - (mean[b] ?? 0))
  const cliques: number[][] = []
  for (let s = 0; s < k; s++) {
    let e = s
    while (e + 1 < k && (mean[order[e + 1] as number] ?? 0) - (mean[order[s] as number] ?? 0) <= cd) e++
    const last = cliques[cliques.length - 1]
    const group = order.slice(s, e + 1)
    if (e > s && !(last !== undefined && group.every((g) => last.includes(g)))) cliques.push(group)
  }
  const control = options.control ?? (order[0] as number)
  const se = Math.sqrt((k * (k + 1)) / (6 * n))
  const z = Array.from(mean, (r) => (r - (mean[control] ?? 0)) / se)
  const p = z.map((v) => 2 * (1 - normCdf(Math.abs(v))))
  const others = Array.from(mean.keys()).filter((j) => j !== control)
  const holmAdj = Array.from(toFlat(holm(others.map((j) => p[j] ?? 1))))
  const bonf = others.map((j) => Math.min(1, (k - 1) * (p[j] ?? 1)))
  const spread = (vals: readonly number[]): number[] => {
    const out = new Array<number>(k).fill(1)
    others.forEach((j, i) => {
      out[j] = vals[i] ?? 1
    })
    out[control] = 1
    return out
  }
  return {
    names: options.names ?? Array.from({ length: k }, (_, j) => `A${j}`),
    nDatasets: n,
    averageRanks: Array.from(mean),
    friedman: fr,
    imanDavenport,
    criticalDifference: cd,
    cliques,
    control: { control, z, pValue: p, bonferroniDunn: spread(bonf), holm: spread(holmAdj) },
  }
}
