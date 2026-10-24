import { type Estimate, type MetricResult, type MetricSpec, type TestResult, WhiteboxError } from '../base/index.ts'
import { betainc, chdtrc, ndtr, ndtri } from '../numerics/special/index.ts'
import { bootstrapIndices, type Key, key as makeKey, split } from '../random/index.ts'
import { item, type TensorLike, toFlat } from '../tensor/index.ts'
import { encodeLabels, type Label, labelValues } from './_labels.ts'
import { positives } from './_scores.ts'
import {
  defineMetric,
  flat,
  type Metric,
  type MetricFunction,
  type MetricInput,
  type MetricOptions,
  matrix,
  metricResult,
  rowCount,
  selectRows,
} from './metric.ts'

const phi = (x: number): number => item(ndtr(x))
const phiInv = (p: number): number => item(ndtri(p))

/** How {@link withInterval} builds its interval. */
export type IntervalMethod = 'bootstrap' | 'analytic' | 'wilson' | 'delong'

/** Options of {@link withInterval}. */
export interface IntervalOptions {
  readonly method?: IntervalMethod
  /** Coverage (default 0.95). */
  readonly level?: number
  /** The key for bootstrap resampling (default key(0)). */
  readonly key?: Key
  /** Bootstrap replicates (default 1000). */
  readonly nResamples?: number
  /** Bootstrap interval: 'percentile' (default), 'basic' or 'bca' (bias-corrected and accelerated). */
  readonly bootstrap?: 'percentile' | 'basic' | 'bca'
}

/** numpy.percentile with linear interpolation of sorted values. */
function percentile(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return Number.NaN
  const pos = q * (sorted.length - 1)
  const lo = Math.floor(pos)
  const hi = Math.min(lo + 1, sorted.length - 1)
  return (sorted[lo] ?? 0) + (pos - lo) * ((sorted[hi] ?? 0) - (sorted[lo] ?? 0))
}

/** Bootstrap replicates of a metric over rows (targets, predictions and weights resampled together). */
function replicates<O extends MetricOptions, I extends MetricInput>(
  metric: Metric<O, I>,
  yTrue: I,
  yPred: I,
  options: O,
  k: Key,
  b: number,
): number[] {
  const n = rowCount(yTrue)
  const w = options.sampleWeight === undefined ? undefined : Array.from(flat(options.sampleWeight))
  const out: number[] = []
  for (const kb of split(k, b)) {
    const idx = toFlat(bootstrapIndices(kb, n))
    const opts = (w === undefined ? options : { ...options, sampleWeight: idx.map((i) => w[i] ?? 0) }) as O
    const v = metric(selectRows(yTrue, idx), selectRows(yPred, idx), opts).estimate.value
    if (Number.isFinite(v)) out.push(v)
  }
  return out
}

/** The bootstrap interval of a statistic from its replicates (percentile, basic or BCa). */
function bootstrapInterval<O extends MetricOptions, I extends MetricInput>(
  metric: Metric<O, I>,
  yTrue: I,
  yPred: I,
  options: O,
  value: number,
  o: IntervalOptions,
): Estimate {
  const level = o.level ?? 0.95
  const reps = replicates(metric, yTrue, yPred, options, o.key ?? makeKey(0), o.nResamples ?? 1000).sort(
    (a, b) => a - b,
  )
  const mean = reps.reduce((a, b) => a + b, 0) / reps.length
  const se = Math.sqrt(reps.reduce((a, b) => a + (b - mean) ** 2, 0) / (reps.length - 1))
  const alpha = (1 - level) / 2
  const kind = o.bootstrap ?? 'percentile'
  let lo = percentile(reps, alpha)
  let hi = percentile(reps, 1 - alpha)
  if (kind === 'basic') {
    ;[lo, hi] = [2 * value - hi, 2 * value - lo]
  } else if (kind === 'bca') {
    // Bias correction from the share of replicates below the estimate; acceleration from the jackknife.
    const below = reps.filter((r) => r < value).length / reps.length
    const z0 = phiInv(Math.min(Math.max(below, 1e-12), 1 - 1e-12))
    const n = rowCount(yTrue)
    const w = options.sampleWeight === undefined ? undefined : Array.from(flat(options.sampleWeight))
    const jack: number[] = []
    for (let i = 0; i < n; i++) {
      const idx = Array.from({ length: n - 1 }, (_, j) => (j < i ? j : j + 1))
      const opts = (w === undefined ? options : { ...options, sampleWeight: idx.map((r) => w[r] ?? 0) }) as O
      jack.push(metric(selectRows(yTrue, idx), selectRows(yPred, idx), opts).estimate.value)
    }
    const jm = jack.reduce((a, b) => a + b, 0) / n
    const num = jack.reduce((a, t) => a + (jm - t) ** 3, 0)
    const den = 6 * jack.reduce((a, t) => a + (jm - t) ** 2, 0) ** 1.5
    const acc = den === 0 ? 0 : num / den
    const adjust = (z: number): number => phi(z0 + (z0 + z) / (1 - acc * (z0 + z)))
    lo = percentile(reps, adjust(phiInv(alpha)))
    hi = percentile(reps, adjust(phiInv(1 - alpha)))
  }
  return { value, standardError: se, interval: [lo, hi], level, method: `bootstrap (${kind}, ${reps.length})` }
}

/** The Wilson score interval of a proportion of n trials. */
function wilson(p: number, n: number, level: number): readonly [number, number] {
  const z = phiInv(0.5 + level / 2)
  const centre = (p + (z * z) / (2 * n)) / (1 + (z * z) / n)
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / (1 + (z * z) / n)
  return [centre - half, centre + half]
}

/**
 * Attach a confidence interval to any metric: 'bootstrap' resamples the samples (percentile, basic or BCa intervals,
 * with the bootstrap standard error), 'wilson' and 'analytic' (Wald) treat a proportion metric (one reporting
 * successes and trials, like accuracy) as binomial, and 'delong' uses DeLong's variance for the binary ROC AUC.
 *
 * @param metric - The metric.
 * @param options - method (default 'bootstrap'), level, key, nResamples and bootstrap kind.
 * @returns The same metric, whose estimate carries an interval and standard error.
 * @throws WhiteboxError with code `value` for an interval the metric cannot have (Wilson without counts, DeLong
 *   for a metric other than rocAucScore).
 * @concept confidence-intervals
 * @example
 * ```ts
 * import { accuracyScore, withInterval } from 'whitebox-engine/metrics'
 * const acc = withInterval(accuracyScore, { method: 'wilson' })
 * acc([0, 1, 1, 0], [0, 1, 0, 0]).estimate.interval?.map((v) => Number(v.toFixed(4))) // => [0.3006, 0.9544]
 * ```
 */
export function withInterval<O extends MetricOptions, I extends MetricInput>(
  metric: Metric<O, I>,
  options: IntervalOptions = {},
): Metric<O, I> {
  const method = options.method ?? 'bootstrap'
  const level = options.level ?? 0.95
  const fn = (yTrue: I, yPred: I, o: O = {} as O): MetricResult => {
    const r = metric(yTrue, yPred, o)
    const value = r.estimate.value
    if (method === 'bootstrap') return { ...r, estimate: bootstrapInterval(metric, yTrue, yPred, o, value, options) }
    if (method === 'delong') {
      if (metric.spec.name !== 'rocAucScore') throw new WhiteboxError('value', 'DeLong intervals are for rocAucScore')
      const se = Math.sqrt(delongVariance(yTrue, yPred as TensorLike).variance)
      const z = phiInv(0.5 + level / 2)
      return {
        ...r,
        estimate: { value, standardError: se, interval: [value - z * se, value + z * se], level, method: 'DeLong' },
      }
    }
    const trials = r.breakdown?.['trials']
    const successes = r.breakdown?.['successes']
    if (trials === undefined || successes === undefined)
      throw new WhiteboxError('value', `${metric.spec.name}: binomial intervals need a proportion metric`)
    const p = successes / trials
    const se = Math.sqrt((p * (1 - p)) / trials)
    const z = phiInv(0.5 + level / 2)
    const interval = method === 'wilson' ? wilson(p, trials, level) : ([p - z * se, p + z * se] as const)
    return {
      ...r,
      estimate: { value, standardError: se, interval, level, method: method === 'wilson' ? 'Wilson' : 'Wald' },
    }
  }
  return Object.assign(fn, { spec: metric.spec })
}

/** DeLong's structural components of a binary AUC: placements of positives (V10) and negatives (V01). */
function placements(y: Float64Array, s: Float64Array): { v10: number[]; v01: number[]; auc: number } {
  const pos: number[] = []
  const neg: number[] = []
  for (const [i, v] of y.entries()) (v > 0 ? pos : neg).push(s[i] ?? 0)
  const psi = (a: number, b: number): number => (a > b ? 1 : a === b ? 0.5 : 0)
  const v10 = pos.map((x) => neg.reduce((acc, z) => acc + psi(x, z), 0) / neg.length)
  const v01 = neg.map((z) => pos.reduce((acc, x) => acc + psi(x, z), 0) / pos.length)
  return { v10, v01, auc: v10.reduce((a, b) => a + b, 0) / v10.length }
}

const covariance = (a: readonly number[], b: readonly number[]): number => {
  const ma = a.reduce((x, y) => x + y, 0) / a.length
  const mb = b.reduce((x, y) => x + y, 0) / b.length
  return a.reduce((acc, v, i) => acc + (v - ma) * ((b[i] ?? 0) - mb), 0) / (a.length - 1)
}

/**
 * DeLong's estimate of the variance of an empirical binary ROC AUC (DeLong, DeLong and Clarke-Pearson, 1988) from its
 * structural components: var(V10) / m + var(V01) / n for m positives and n negatives.
 *
 * @param yTrue - True binary labels `[N]`.
 * @param yScore - Scores `[N]`.
 * @returns The AUC and its variance.
 * @example
 * ```ts
 * import { delongVariance } from 'whitebox-engine/metrics'
 * delongVariance([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]).auc // => 0.75
 * ```
 */
export function delongVariance(
  yTrue: MetricInput,
  yScore: TensorLike,
): { readonly auc: number; readonly variance: number } {
  const p = placements(positives(yTrue, undefined), flat(yScore))
  return { auc: p.auc, variance: covariance(p.v10, p.v10) / p.v10.length + covariance(p.v01, p.v01) / p.v01.length }
}

const test = (name: string, statistic: number, pValue: number, extra: Partial<TestResult> = {}): TestResult => ({
  test: name,
  statistic,
  pValue,
  alternative: 'two-sided',
  ...extra,
})

/**
 * DeLong's test for two correlated ROC AUCs on the same samples (DeLong et al., 1988): z = (AUC_A - AUC_B) / sd with
 * the covariance of the two AUCs from their structural components; two-sided normal p-value.
 *
 * @param yTrue - True binary labels `[N]`.
 * @param scoresA - Scores of model A `[N]`.
 * @param scoresB - Scores of model B `[N]`.
 * @returns The test result, with the AUC difference and its interval as the estimate.
 * @concept model-comparison
 * @example
 * ```ts
 * import { delongTest } from 'whitebox-engine/metrics'
 * delongTest([0, 0, 1, 1, 0, 1], [0.1, 0.4, 0.35, 0.8, 0.2, 0.9], [0.1, 0.4, 0.35, 0.8, 0.2, 0.9]).statistic // => 0
 * ```
 */
export function delongTest(yTrue: MetricInput, scoresA: TensorLike, scoresB: TensorLike): TestResult {
  const y = positives(yTrue, undefined)
  const a = placements(y, flat(scoresA))
  const b = placements(y, flat(scoresB))
  const m = a.v10.length
  const n = a.v01.length
  const varA = covariance(a.v10, a.v10) / m + covariance(a.v01, a.v01) / n
  const varB = covariance(b.v10, b.v10) / m + covariance(b.v01, b.v01) / n
  const cov = covariance(a.v10, b.v10) / m + covariance(a.v01, b.v01) / n
  const diff = a.auc - b.auc
  const sd = Math.sqrt(varA + varB - 2 * cov)
  const z = sd === 0 ? 0 : diff / sd
  const zc = phiInv(0.975)
  return test('DeLong test', z, sd === 0 ? 1 : 2 * phi(-Math.abs(z)), {
    estimate: { value: diff, standardError: sd, interval: [diff - zc * sd, diff + zc * sd], level: 0.95 },
    method: 'DeLong',
  })
}

/**
 * McNemar's test for two classifiers on the same samples (statsmodels.stats.contingency_tables.mcnemar): from the
 * discordant counts b (only A right) and c (only B right), the exact binomial test (default) or the chi-square test
 * with optional continuity correction.
 *
 * @param yTrue - True labels `[N]`.
 * @param predA - Predictions of model A.
 * @param predB - Predictions of model B.
 * @param options - exact (default true) and correction (default true, for the chi-square version).
 * @returns The test result (statistic min(b, c) for the exact test, chi-square otherwise).
 * @example
 * ```ts
 * import { mcnemarTest } from 'whitebox-engine/metrics'
 * mcnemarTest([1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]).pValue // => 1
 * ```
 */
export function mcnemarTest(
  yTrue: MetricInput,
  predA: MetricInput,
  predB: MetricInput,
  options: { readonly exact?: boolean; readonly correction?: boolean } = {},
): TestResult {
  const t = labelValues(yTrue)
  const a = labelValues(predA)
  const bp = labelValues(predB)
  let b = 0
  let c = 0
  for (const [i, v] of t.entries()) {
    const ra = a[i] === v
    const rb = bp[i] === v
    if (ra && !rb) b++
    else if (rb && !ra) c++
  }
  if (options.exact ?? true) {
    const k = Math.min(b, c)
    const m = b + c
    // Two-sided binomial p-value: 2 P(X <= min(b, c)), X ~ Binomial(b + c, 1/2), via the regularised beta function.
    const p = m === 0 ? 1 : Math.min(1, 2 * (k >= m ? 1 : item(betainc(m - k, k + 1, 0.5))))
    return test('McNemar test', k, p, { method: 'exact binomial' })
  }
  const corr = (options.correction ?? true) ? 1 : 0
  const stat = b + c === 0 ? 0 : (Math.abs(b - c) - corr) ** 2 / (b + c)
  return test('McNemar test', stat, item(chdtrc(1, stat)), {
    df: 1,
    method: corr ? 'chi-square, corrected' : 'chi-square',
  })
}

/**
 * Paired bootstrap comparison of two models on one test set (Koehn, 2004; Berg-Kirkpatrick et al., 2012): resample
 * the samples, recompute the metric for both models on each resample, and report the observed difference A - B with a
 * percentile interval and the two-sided p-value of the difference's sign.
 *
 * @param metric - Any metric of targets and predictions.
 * @param yTrue - True targets.
 * @param predA - Predictions of model A.
 * @param predB - Predictions of model B.
 * @param options - key, nResamples (default 1000), level and the metric's options.
 * @returns The test result (statistic: the observed difference).
 * @example
 * ```ts
 * import { accuracyScore, pairedBootstrap } from 'whitebox-engine/metrics'
 * pairedBootstrap(accuracyScore, [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 0, 0], { nResamples: 50 }).statistic // => 0.25
 * ```
 */
export function pairedBootstrap<O extends MetricOptions, I extends MetricInput>(
  metric: Metric<O, I>,
  yTrue: I,
  predA: I,
  predB: I,
  options: IntervalOptions & { readonly metricOptions?: O } = {},
): TestResult {
  const o = options.metricOptions ?? ({} as O)
  const observed = metric(yTrue, predA, o).estimate.value - metric(yTrue, predB, o).estimate.value
  const n = rowCount(yTrue)
  const diffs: number[] = []
  for (const kb of split(options.key ?? makeKey(0), options.nResamples ?? 1000)) {
    const idx = toFlat(bootstrapIndices(kb, n))
    const t = selectRows(yTrue, idx)
    const d = metric(t, selectRows(predA, idx), o).estimate.value - metric(t, selectRows(predB, idx), o).estimate.value
    if (Number.isFinite(d)) diffs.push(d)
  }
  diffs.sort((x, y) => x - y)
  const level = options.level ?? 0.95
  const alpha = (1 - level) / 2
  // The sign test of the bootstrap distribution: twice the smaller tail beyond zero.
  const below = diffs.filter((d) => d <= 0).length / diffs.length
  const above = diffs.filter((d) => d >= 0).length / diffs.length
  const p = Math.min(1, 2 * Math.min(below, above))
  const mean = diffs.reduce((a, b) => a + b, 0) / diffs.length
  const se = Math.sqrt(diffs.reduce((a, d) => a + (d - mean) ** 2, 0) / (diffs.length - 1))
  return test('paired bootstrap', observed, p, {
    estimate: {
      value: observed,
      standardError: se,
      interval: [percentile(diffs, alpha), percentile(diffs, 1 - alpha)],
      level,
    },
    method: `paired bootstrap (${diffs.length})`,
  })
}

/**
 * Compute many metrics of the same targets and predictions in one pass, optionally each with a bootstrap interval
 * from the same resamples (so that intervals are comparable across metrics).
 *
 * @param yTrue - True targets.
 * @param yPred - Predictions (or scores, for score metrics).
 * @param metrics - The metrics, keyed by their spec names in the result.
 * @param options - sampleWeight and an interval specification.
 * @returns The results, keyed by metric name.
 * @concept evaluation-metrics
 * @example
 * ```ts
 * import { accuracyScore, evaluate, f1Score } from 'whitebox-engine/metrics'
 * const r = evaluate([0, 1, 1, 0], [0, 1, 0, 0], [accuracyScore, f1Score])
 * [r.accuracyScore?.estimate.value, r.f1Score?.estimate.value] // => [0.75, 0.6666666666666666]
 * ```
 */
export function evaluate<I extends MetricInput>(
  yTrue: I,
  yPred: I,
  metrics: readonly Metric<MetricOptions, I>[],
  options: MetricOptions & { readonly interval?: IntervalOptions } = {},
): Record<string, MetricResult> {
  const base = options.sampleWeight === undefined ? {} : { sampleWeight: options.sampleWeight }
  const out: Record<string, MetricResult> = {}
  for (const m of metrics) {
    out[m.spec.name] =
      options.interval === undefined ? m(yTrue, yPred, base) : withInterval(m, options.interval)(yTrue, yPred, base)
  }
  return out
}

const decisionSpec = (name: string, direction: 'higher' | 'lower'): MetricSpec => ({
  name,
  direction,
  range: [-Infinity, Infinity],
  inputs: 'labels',
  category: 'decision',
  concept: 'decision-theory',
})

/** The mean of matrix[true][pred] over samples (classes in label order). */
function meanMatrix(
  name: string,
  yTrue: MetricInput,
  yPred: MetricInput,
  m: TensorLike,
  options: MetricOptions & { readonly labels?: readonly Label[] },
): MetricResult {
  const e = encodeLabels(name, yTrue, yPred, options)
  const c = matrix(m)
  const k = c.shape[1] ?? 0
  let total = 0
  let weight = 0
  for (let i = 0; i < e.n; i++) {
    const a = e.t[i] ?? -1
    const b = e.p[i] ?? -1
    if (a < 0 || b < 0) continue
    total += (e.w[i] ?? 0) * (c.data[a * k + b] ?? 0)
    weight += e.w[i] ?? 0
  }
  return metricResult(name, total / weight, e.n)
}

/** Options of the decision metrics. */
export interface DecisionOptions extends MetricOptions {
  /** The classes of the matrix rows and columns, in order (default the sorted labels). */
  readonly labels?: readonly Label[]
}

/**
 * The expected (mean) cost of decisions under a cost matrix C[true][predicted], the metric to use when errors have
 * different consequences.
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Decisions `[N]`.
 * @param costs - The cost matrix `[C, C]`.
 * @param options - labels and sample weights.
 * @returns The metric result.
 * @concept decision-theory
 * @example
 * ```ts
 * import { expectedCost } from 'whitebox-engine/metrics'
 * expectedCost([0, 1, 1], [0, 0, 1], [[0, 1], [10, 0]]).estimate.value // => 3.3333333333333335
 * ```
 */
export const expectedCost: MetricFunction<[MetricInput, MetricInput, TensorLike, DecisionOptions?]> = defineMetric(
  decisionSpec('expectedCost', 'lower'),
  (yTrue: MetricInput, yPred: MetricInput, costs: TensorLike, options: DecisionOptions = {}): MetricResult =>
    meanMatrix('expectedCost', yTrue, yPred, costs, options),
)

/**
 * The expected (mean) utility of decisions under a utility matrix U[true][predicted] (the negative of a cost).
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Decisions `[N]`.
 * @param utilities - The utility matrix `[C, C]`.
 * @param options - labels and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { expectedUtility } from 'whitebox-engine/metrics'
 * expectedUtility([0, 1], [0, 1], [[1, 0], [0, 2]]).estimate.value // => 1.5
 * ```
 */
export const expectedUtility: MetricFunction<[MetricInput, MetricInput, TensorLike, DecisionOptions?]> = defineMetric(
  decisionSpec('expectedUtility', 'higher'),
  (yTrue: MetricInput, yPred: MetricInput, utilities: TensorLike, options: DecisionOptions = {}): MetricResult =>
    meanMatrix('expectedUtility', yTrue, yPred, utilities, options),
)

/**
 * Bayes decisions under a cost matrix: for each row of class probabilities, the action minimising the expected cost
 * sum_i p_i C[i][a] (for 0-1 costs, the most probable class).
 *
 * @param probs - Class probabilities `[N, C]`.
 * @param costs - The cost matrix C[true][action] `[C, A]`.
 * @returns The chosen action index per sample.
 * @example
 * ```ts
 * import { bayesDecision } from 'whitebox-engine/metrics'
 * bayesDecision([[0.8, 0.2], [0.95, 0.05]], [[0, 1], [10, 0]]) // => [1, 0]
 * ```
 */
export function bayesDecision(probs: TensorLike, costs: TensorLike): number[] {
  const p = matrix(probs)
  const c = matrix(costs)
  const n = p.shape[0] ?? 0
  const k = p.shape[1] ?? 0
  const actions = c.shape[1] ?? 0
  return Array.from({ length: n }, (_, i) => {
    let best = 0
    let bestCost = Number.POSITIVE_INFINITY
    for (let a = 0; a < actions; a++) {
      let cost = 0
      for (let j = 0; j < k; j++) cost += (p.data[i * k + j] ?? 0) * (c.data[j * actions + a] ?? 0)
      if (cost < bestCost) {
        bestCost = cost
        best = a
      }
    }
    return best
  })
}

/**
 * The cost-minimising probability threshold of a calibrated binary classifier: predict positive when
 * p >= c_fp / (c_fp + c_fn) (Elkan, 2001).
 *
 * @param costFp - The cost of a false positive.
 * @param costFn - The cost of a false negative.
 * @returns The threshold.
 * @example
 * ```ts
 * import { costThreshold } from 'whitebox-engine/metrics'
 * costThreshold(1, 9) // => 0.1
 * ```
 */
export function costThreshold(costFp: number, costFn: number): number {
  return costFp / (costFp + costFn)
}
