/**
 * The uncertainty of fairness metrics: bootstrap intervals for any gap between groups (fairlearn's MetricFrame with
 * `n_boot`), permutation tests of whether a gap could be chance, Wilson intervals for each group's rates, the minimum
 * difference a comparison of two groups can detect, and warnings about groups too small to measure.
 */

import { type Estimate, MdmlError, type MetricResult } from '../base/index.ts'
import { labelValues, sortLabels } from '../metrics/_labels.ts'
import type { MetricInput } from '../metrics/metric.ts'
import { brentq } from '../numerics/roots/index.ts'
import { foldIn, type Key, key as makeKey, permutation, randint } from '../random/index.ts'
import { type BootstrapMethod, bootstrap, powerProportions, quantile } from '../stats/index.ts'
import { type ProportionInterval, proportionConfint } from '../stats/tests/index.ts'

/** A statistic of labels, predictions and groups: a fairness metric (its value, or its result) on some rows. */
export type GroupStatistic = (yTrue: number[], yPred: number[], groups: string[]) => number | MetricResult

const numberOf = (v: number | MetricResult): number => (typeof v === 'number' ? v : v.estimate.value)

/** Labels, predictions and groups as columns. */
function columns(
  name: string,
  yTrue: MetricInput,
  yPred: MetricInput,
  groups: MetricInput,
): { t: number[]; p: number[]; g: string[]; names: string[] } {
  const t = labelValues(yTrue).map(Number)
  const p = labelValues(yPred).map(Number)
  const raw = labelValues(groups)
  if (t.length !== p.length || t.length !== raw.length) throw new MdmlError('shape', `${name}: inputs differ in length`)
  return { t, p, g: raw.map(String), names: sortLabels(raw).map(String) }
}

/** Options of {@link bootstrapDisparity}. */
export interface BootstrapDisparityOptions {
  /** Resamples (default 1000). */
  readonly nResamples?: number
  readonly level?: number
  /** Interval method (default `'percentile'`, as fairlearn's MetricFrame quantiles). */
  readonly method?: BootstrapMethod
  readonly key?: Key
  readonly seed?: number
}

/** The result of {@link bootstrapDisparity}. */
export interface DisparityInterval {
  /** The statistic on all the rows, with its bootstrap standard error and interval. */
  readonly estimate: Estimate
  /** The statistic on every resample. */
  readonly distribution: readonly number[]
}

/**
 * A bootstrap confidence interval for any fairness statistic (a gap between groups, a group's rate): rows are resampled
 * with replacement, labels, predictions and groups together, and the statistic recomputed on each resample.
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Predicted labels (or scores) `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param statistic - The statistic, e.g. `(t, p, g) => demographicParityDifference(t, p, g)`.
 * @param options - nResamples, level, method and the key or seed.
 * @returns The estimate with its interval and the bootstrap distribution.
 * @example
 * ```ts
 * import { bootstrapDisparity, demographicParityDifference } from 'mdml-engine/fairness'
 * const yPred = [1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]
 * const groups = ['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b']
 * const r = bootstrapDisparity(yPred, yPred, groups, (t, p, g) => demographicParityDifference(t, p, g), { seed: 1 })
 * r.estimate.value.toFixed(3) // => '0.333'
 * r.estimate.interval?.[0] !== undefined && r.estimate.interval[0] <= r.estimate.interval[1] // => true
 * ```
 */
export function bootstrapDisparity(
  yTrue: MetricInput,
  yPred: MetricInput,
  groups: MetricInput,
  statistic: GroupStatistic,
  options: BootstrapDisparityOptions = {},
): DisparityInterval {
  const { t, p, g, names } = columns('bootstrapDisparity', yTrue, yPred, groups)
  const code = new Map(names.map((n, i) => [n, i]))
  const gi = g.map((v) => code.get(v) ?? 0)
  const stat = (a: Float64Array, b: Float64Array, c: Float64Array): number =>
    numberOf(
      statistic(
        Array.from(a),
        Array.from(b),
        Array.from(c, (v) => names[v] ?? ''),
      ),
    )
  const r = bootstrap([t, p, gi], stat, {
    paired: true,
    method: options.method ?? 'percentile',
    nResamples: options.nResamples ?? 1000,
    ...(options.level === undefined ? {} : { level: options.level }),
    ...(options.key === undefined ? { seed: options.seed ?? 0 } : { key: options.key }),
  })
  return { estimate: r.estimate, distribution: Array.from(r.distribution.data as ArrayLike<number>) }
}

/** Options of {@link bootstrapStatistics}. */
export interface BootstrapStatisticsOptions {
  /** Resamples (default 500). */
  readonly nResamples?: number
  readonly level?: number
  readonly key?: Key
  readonly seed?: number
}

/** The result of {@link bootstrapStatistics}. */
export interface StatisticIntervals {
  /** Each output's percentile interval (NaN ends where the output was never finite). */
  readonly intervals: readonly (readonly [number, number])[]
  /** distribution[b][j]: output j on resample b. */
  readonly distribution: readonly (readonly number[])[]
  readonly level: number
}

/**
 * Percentile bootstrap intervals for several statistics of the same rows at once (each group's rate and every
 * disparity of a report, from one set of resamples): each resample draws `n` row indices with replacement and the
 * statistic returns one value per output.
 *
 * @param n - The number of rows.
 * @param statistic - The outputs on a resample of row indices.
 * @param options - nResamples, level and the key or seed.
 * @returns Each output's interval and the bootstrap distribution.
 * @example
 * ```ts
 * import { bootstrapStatistics } from 'mdml-engine/fairness'
 * const x = [1, 2, 3, 4, 5, 6, 7, 8]
 * const mean = (rows: Int32Array) => [rows.reduce((s, i) => s + (x[i] ?? 0), 0) / rows.length]
 * const [lo, hi] = bootstrapStatistics(x.length, mean, { seed: 0 }).intervals[0] ?? [0, 0]
 * lo < 4.5 && hi > 4.5 // => true
 * ```
 */
export function bootstrapStatistics(
  n: number,
  statistic: (rows: Int32Array) => readonly number[],
  options: BootstrapStatisticsOptions = {},
): StatisticIntervals {
  if (!(Number.isInteger(n) && n > 0)) throw new MdmlError('value', 'bootstrapStatistics: needs at least one row')
  const total = options.nResamples ?? 500
  const level = options.level ?? 0.95
  const k = options.key ?? makeKey(options.seed ?? 0)
  const distribution: number[][] = []
  for (let b = 0; b < total; b++) distribution.push([...statistic(randint(foldIn(k, b), [n], 0, n).data as Int32Array)])
  const outputs = distribution[0]?.length ?? 0
  const alpha = (1 - level) / 2
  const intervals = Array.from({ length: outputs }, (_, j): [number, number] => {
    const values = distribution.map((d) => d[j] ?? Number.NaN).filter(Number.isFinite)
    return values.length === 0 ? [Number.NaN, Number.NaN] : [quantile(values, alpha), quantile(values, 1 - alpha)]
  })
  return { intervals, distribution, level }
}

/** Options of {@link permutationTestDisparity}. */
export interface PermutationDisparityOptions {
  /** Permutations (default 999). */
  readonly nResamples?: number
  /** `'two-sided'` compares |statistic| (default); `'greater'` and `'less'` the signed statistic. */
  readonly alternative?: 'two-sided' | 'greater' | 'less'
  readonly key?: Key
  readonly seed?: number
}

/** The result of {@link permutationTestDisparity}. */
export interface DisparityTest {
  readonly statistic: number
  /** (1 + permutations at least as extreme) / (1 + permutations). */
  readonly pValue: number
  /** The statistic under each permutation of the groups. */
  readonly distribution: readonly number[]
}

/**
 * A permutation test of a fairness gap: under the null hypothesis that the group carries no information, permuting
 * the group labels across rows leaves the statistic's distribution unchanged; the p-value is the share of permutations
 * giving a gap at least as extreme as the observed one.
 *
 * @param yTrue - True labels `[N]`.
 * @param yPred - Predicted labels (or scores) `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param statistic - The gap, e.g. `(t, p, g) => equalizedOddsDifference(t, p, g)`.
 * @param options - nResamples, alternative and the key or seed.
 * @returns The observed statistic, the p-value and the permutation distribution.
 * @example
 * ```ts
 * import { demographicParityDifference, permutationTestDisparity } from 'mdml-engine/fairness'
 * const groups = Array.from({ length: 40 }, (_, i) => (i < 20 ? 'a' : 'b'))
 * const yPred = groups.map((g, i) => (g === 'a' ? 1 : i % 5 === 0 ? 1 : 0))
 * const r = permutationTestDisparity(yPred, yPred, groups, (t, p, g) => demographicParityDifference(t, p, g), { seed: 0 })
 * r.pValue < 0.01 // => true
 * ```
 */
export function permutationTestDisparity(
  yTrue: MetricInput,
  yPred: MetricInput,
  groups: MetricInput,
  statistic: GroupStatistic,
  options: PermutationDisparityOptions = {},
): DisparityTest {
  const { t, p, g } = columns('permutationTestDisparity', yTrue, yPred, groups)
  const n = options.nResamples ?? 999
  const k = options.key ?? makeKey(options.seed ?? 0)
  const alternative = options.alternative ?? 'two-sided'
  const observed = numberOf(statistic(t, p, g))
  const extreme = (v: number): boolean =>
    alternative === 'greater'
      ? v >= observed - 1e-12
      : alternative === 'less'
        ? v <= observed + 1e-12
        : Math.abs(v) >= Math.abs(observed) - 1e-12
  const distribution: number[] = []
  let count = 0
  for (let b = 0; b < n; b++) {
    const order = permutation(foldIn(k, b), g.length).data
    const v = numberOf(
      statistic(
        t,
        p,
        Array.from(order, (j) => g[j] ?? ''),
      ),
    )
    distribution.push(v)
    if (extreme(v)) count++
  }
  return { statistic: observed, pValue: (1 + count) / (1 + n), distribution }
}

/** The rates {@link groupRateIntervals} gives intervals for. */
export type IntervalRate = 'selectionRate' | 'truePositiveRate' | 'falsePositiveRate' | 'precision' | 'accuracy'

/** One group's rates with their intervals. */
export interface GroupRateIntervals {
  readonly group: string
  readonly count: number
  readonly rates: Readonly<Record<IntervalRate, Estimate>>
}

/**
 * Each group's rates (selection, true- and false-positive, precision, accuracy) with binomial confidence intervals
 * (Wilson's score interval by default, statsmodels' proportion_confint): the uncertainty a small group carries. A rate
 * with an empty denominator is NaN with no interval.
 *
 * @param yTrue - True binary labels `[N]` (1 positive).
 * @param yPred - Predicted binary labels `[N]`.
 * @param groups - Sensitive feature `[N]`.
 * @param options - level (default 0.95) and the interval method.
 * @returns Each group's rates with intervals, in sorted group order.
 * @example
 * ```ts
 * import { groupRateIntervals } from 'mdml-engine/fairness'
 * const r = groupRateIntervals([1, 0, 1, 0], [1, 1, 1, 0], ['a', 'a', 'b', 'b'])
 * r[0]?.rates.selectionRate.value // => 1
 * r[0]?.rates.selectionRate.interval?.[1] // => 1
 * ```
 */
export function groupRateIntervals(
  yTrue: MetricInput,
  yPred: MetricInput,
  groups: MetricInput,
  options: { readonly level?: number; readonly method?: ProportionInterval } = {},
): GroupRateIntervals[] {
  const { t, p, g, names } = columns('groupRateIntervals', yTrue, yPred, groups)
  const ci = (k: number, n: number): Estimate =>
    n === 0
      ? { value: Number.NaN }
      : proportionConfint(k, n, {
          level: options.level ?? 0.95,
          ...(options.method === undefined ? {} : { method: options.method }),
        })
  return names.map((name) => {
    let tp = 0
    let fp = 0
    let tn = 0
    let fn = 0
    for (const [i, v] of g.entries()) {
      if (v !== name) continue
      const y = t[i] === 1
      const yh = p[i] === 1
      if (y && yh) tp++
      else if (!y && yh) fp++
      else if (!y && !yh) tn++
      else fn++
    }
    const n = tp + fp + tn + fn
    return {
      group: name,
      count: n,
      rates: {
        selectionRate: ci(tp + fp, n),
        truePositiveRate: ci(tp, tp + fn),
        falsePositiveRate: ci(fp, fp + tn),
        precision: ci(tp, tp + fp),
        accuracy: ci(tp + tn, n),
      },
    }
  })
}

/** Options of {@link minimumDetectableDifference}. */
export interface DetectableDifferenceOptions {
  /** Rows in the first group. */
  readonly n1: number
  /** Rows in the second group. */
  readonly n2: number
  /** The first group's rate (default 0.5, the hardest case). */
  readonly baseRate?: number
  readonly alpha?: number
  readonly power?: number
}

/** The result of {@link minimumDetectableDifference}. */
export interface DetectableDifference {
  /** Cohen's h a two-sided two-proportion test detects with the power. */
  readonly effectSize: number
  /** The smallest rate difference above the base rate it detects (on the probability scale). */
  readonly difference: number
}

/**
 * The smallest gap in a rate between two groups that a two-sided test of proportions detects with the given power
 * (statsmodels' NormalIndPower solved for the effect size, Cohen's h), and the rate difference it means above the
 * base rate. A gap smaller than this is not evidence of fairness: the groups may be too small to see it.
 *
 * @param options - The two group sizes, the base rate, alpha (default 0.05) and power (default 0.8).
 * @returns Cohen's h and the rate difference.
 * @throws MdmlError `value` for a group without rows or a base rate outside (0, 1).
 * @example
 * ```ts
 * import { minimumDetectableDifference } from 'mdml-engine/fairness'
 * minimumDetectableDifference({ n1: 100, n2: 100 }).effectSize.toFixed(4) // => '0.3962'
 * ```
 */
export function minimumDetectableDifference(options: DetectableDifferenceOptions): DetectableDifference {
  const { n1, n2 } = options
  const base = options.baseRate ?? 0.5
  if (!(n1 > 0 && n2 > 0)) throw new MdmlError('value', 'minimumDetectableDifference: both groups need rows')
  if (!(base > 0 && base < 1)) throw new MdmlError('value', 'minimumDetectableDifference: baseRate must be in (0, 1)')
  const alpha = options.alpha ?? 0.05
  const target = options.power ?? 0.8
  const power = (h: number): number => powerProportions({ effectSize: h, nobs: n1, ratio: n2 / n1, alpha }) - target
  const h = brentq(power, 1e-10, 40, { xtol: 1e-12 }).root
  // Cohen's h = 2 asin(sqrt(p2)) - 2 asin(sqrt(p1)); move up from the base rate, capped at 1.
  const phi = Math.min(Math.PI / 2, Math.asin(Math.sqrt(base)) + h / 2)
  return { effectSize: h, difference: Math.sin(phi) ** 2 - base }
}

/** Options of {@link smallGroupWarnings}. */
export interface SmallGroupOptions {
  /** Fewest rows a group should have (default 30). */
  readonly minSize?: number
  /** Fewest positives and negatives, for the error rates (default 10). */
  readonly minPerClass?: number
}

/** A warning about one group. */
export interface GroupWarning {
  readonly group: string
  readonly count: number
  readonly positives: number
  readonly negatives: number
  readonly message: string
}

/**
 * Warnings about groups too small for their rates to mean much (Aequitas and fairlearn both warn of this): fewer rows
 * than `minSize`, or fewer positives or negatives than `minPerClass` (whose true- and false-positive rates rest on them).
 *
 * @param yTrue - True binary labels `[N]` (1 positive).
 * @param groups - Sensitive feature `[N]`.
 * @param options - minSize and minPerClass.
 * @returns One warning per small group (none when every group is large enough).
 * @example
 * ```ts
 * import { smallGroupWarnings } from 'mdml-engine/fairness'
 * smallGroupWarnings([1, 0, 1], ['a', 'a', 'b'], { minSize: 2, minPerClass: 1 }).map((w) => w.group) // => ['b']
 * ```
 */
export function smallGroupWarnings(
  yTrue: MetricInput,
  groups: MetricInput,
  options: SmallGroupOptions = {},
): GroupWarning[] {
  const t = labelValues(yTrue)
  const g = labelValues(groups)
  if (t.length !== g.length) throw new MdmlError('shape', 'smallGroupWarnings: inputs differ in length')
  const minSize = options.minSize ?? 30
  const minPerClass = options.minPerClass ?? 10
  const out: GroupWarning[] = []
  for (const name of sortLabels(g)) {
    let positives = 0
    let negatives = 0
    for (const [i, v] of g.entries())
      if (v === name) {
        if (t[i] === 1) positives++
        else negatives++
      }
    const count = positives + negatives
    const reasons: string[] = []
    if (count < minSize) reasons.push(`${count} rows (fewer than ${minSize})`)
    if (positives < minPerClass) reasons.push(`${positives} positives`)
    if (negatives < minPerClass) reasons.push(`${negatives} negatives`)
    if (reasons.length > 0)
      out.push({
        group: String(name),
        count,
        positives,
        negatives,
        message: `group ${String(name)} is small: ${reasons.join(', ')}; its rates are uncertain`,
      })
  }
  return out
}
