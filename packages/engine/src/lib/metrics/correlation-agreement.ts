/**
 * Correlation and agreement between two variables or raters: Pearson, Spearman and Kendall correlations, Lin's
 * concordance correlation, Bland-Altman limits of agreement, Cramer's V and Theil's U for categorical variables,
 * intraclass correlations, weighted Cohen's kappa, Fleiss' kappa for many raters and Krippendorff's alpha.
 */

import { MdmlError, type MetricResult, type MetricSpec } from '../base/index.ts'
import { ndtri } from '../numerics/special/index.ts'
import { cov, kendallTau as kendallTest, spearman } from '../stats/index.ts'
import { item, type TensorLike, toFlat } from '../tensor/index.ts'
import { rejectSampleWeight, sortLabels } from './_labels.ts'
import { cohenKappaScore, type KappaOptions } from './classification-summary.ts'
import { contingencyMatrix } from './clustering.ts'
import {
  defineMetric,
  flat,
  type Metric,
  type MetricFunction,
  type MetricInput,
  type MetricOptions,
  metricResult,
  sampleWeights,
  shapedArray,
} from './metric.ts'

const spec = (
  name: string,
  range: readonly [number, number] = [-1, 1],
  inputs: MetricSpec['inputs'] = 'values',
): MetricSpec => ({
  name,
  direction: 'higher',
  range,
  inputs,
  category: 'agreement',
  concept: 'correlation-agreement',
})

const pair = (name: string, x: TensorLike, y: TensorLike): [Float64Array, Float64Array] => {
  const a = flat(x)
  const b = flat(y)
  if (a.length !== b.length) throw new MdmlError('shape', `${name}: inputs differ in length`)
  return [a, b]
}

/**
 * Pearson's correlation coefficient (scipy.stats.pearsonr), with optional sample weights (the weighted correlation,
 * from stats' weighted covariance).
 *
 * @param x - First variable `[N]`.
 * @param y - Second variable `[N]`.
 * @param options - Sample weights.
 * @returns The metric result.
 * @concept correlation
 * @example
 * ```ts
 * import { pearsonCorrelation } from 'mdml-engine/metrics'
 * pearsonCorrelation([1, 2, 3], [2, 4, 6]).estimate.value // => 1
 * ```
 */
export const pearsonCorrelation: Metric = Object.assign(
  (x: TensorLike, y: TensorLike, options: MetricOptions = {}): MetricResult => {
    const [a, b] = pair('pearsonCorrelation', x, y)
    const w =
      options.sampleWeight === undefined
        ? undefined
        : sampleWeights('pearsonCorrelation', a.length, options.sampleWeight)
    // The weighted correlation from stats' weighted covariance (analytic weights).
    const c = toFlat(cov(a, b, w === undefined ? {} : { aweights: w }))
    return metricResult('pearsonCorrelation', (c[1] ?? 0) / Math.sqrt((c[0] ?? 0) * (c[3] ?? 0)), a.length)
  },
  { spec: spec('pearsonCorrelation') },
)

/**
 * Spearman's rank correlation (scipy.stats.spearmanr, through stats' spearman): Pearson's correlation of the average
 * ranks.
 *
 * @param x - First variable `[N]`.
 * @param y - Second variable `[N]`.
 * @returns The metric result.
 * @example
 * ```ts
 * import { spearmanCorrelation } from 'mdml-engine/metrics'
 * spearmanCorrelation([1, 2, 3, 4], [1, 4, 9, 16]).estimate.value // => 1
 * ```
 */
export const spearmanCorrelation: Metric = Object.assign(
  (x: TensorLike, y: TensorLike, options?: MetricOptions): MetricResult => {
    rejectSampleWeight('spearmanCorrelation', options)
    const [a, b] = pair('spearmanCorrelation', x, y)
    return metricResult('spearmanCorrelation', spearman(a, b).statistic, a.length)
  },
  { spec: spec('spearmanCorrelation') },
)

/**
 * Kendall's tau (scipy.stats.kendalltau, through stats' kendallTau): tau-b (default), which corrects for ties in either variable, or tau-c
 * (Stuart's, for rectangular tables).
 *
 * @param x - First variable `[N]`.
 * @param y - Second variable `[N]`.
 * @param options - variant ('b' or 'c').
 * @returns The metric result.
 * @example
 * ```ts
 * import { kendallTau } from 'mdml-engine/metrics'
 * kendallTau([1, 2, 3, 4], [1, 3, 2, 4]).estimate.value // => 0.6666666666666666
 * ```
 */
export const kendallTau: Metric<MetricOptions & { readonly variant?: 'b' | 'c' }> = Object.assign(
  (x: TensorLike, y: TensorLike, options: MetricOptions & { readonly variant?: 'b' | 'c' } = {}): MetricResult => {
    rejectSampleWeight('kendallTau', options)
    const [a, b] = pair('kendallTau', x, y)
    const n = a.length
    const value = kendallTest(a, b, { variant: options.variant ?? 'b', method: 'asymptotic' }).statistic
    return metricResult('kendallTau', value, n)
  },
  { spec: spec('kendallTau') },
)

/**
 * Lin's concordance correlation coefficient (Lin, 1989): 2 cov(x, y) / (var x + var y + (mean x - mean y)^2), the
 * agreement of two measurements with the identity line; ddof 1 matches torchmetrics' concordance_corrcoef, ddof 0
 * Lin's original.
 *
 * @param x - First measurement `[N]`.
 * @param y - Second measurement `[N]`.
 * @param options - ddof (default 1).
 * @returns The metric result.
 * @example
 * ```ts
 * import { concordanceCorrelation } from 'mdml-engine/metrics'
 * concordanceCorrelation([1, 2, 3], [1, 2, 3]).estimate.value // => 1
 * ```
 */
export const concordanceCorrelation: Metric<MetricOptions & { readonly ddof?: number }> = Object.assign(
  (x: TensorLike, y: TensorLike, options: MetricOptions & { readonly ddof?: number } = {}): MetricResult => {
    const [a, b] = pair('concordanceCorrelation', x, y)
    const n = a.length
    // Weights count repeated samples: the total weight takes the place of n.
    const w = sampleWeights('concordanceCorrelation', n, options.sampleWeight)
    const total = w.reduce((s, v) => s + v, 0)
    const ma = a.reduce((s, v, i) => s + v * (w[i] ?? 0), 0) / total
    const mb = b.reduce((s, v, i) => s + v * (w[i] ?? 0), 0) / total
    let sab = 0
    let saa = 0
    let sbb = 0
    for (let i = 0; i < n; i++) {
      const wi = w[i] ?? 0
      sab += wi * ((a[i] ?? 0) - ma) * ((b[i] ?? 0) - mb)
      saa += wi * ((a[i] ?? 0) - ma) ** 2
      sbb += wi * ((b[i] ?? 0) - mb) ** 2
    }
    const d = total - (options.ddof ?? 1)
    const value = (2 * (sab / d)) / (saa / d + sbb / d + (ma - mb) ** 2)
    return metricResult('concordanceCorrelation', value, n)
  },
  { spec: spec('concordanceCorrelation') },
)

/** Bland-Altman limits of agreement. */
export interface BlandAltman {
  /** The mean difference x - y. */
  readonly bias: number
  /** The standard deviation of the differences (ddof 1). */
  readonly sd: number
  readonly lower: number
  readonly upper: number
  readonly level: number
}

/**
 * Bland-Altman analysis of two measurement methods (Bland and Altman, 1986): the mean difference and the limits of
 * agreement bias -+ z sd of the differences.
 *
 * @param x - First method `[N]`.
 * @param y - Second method `[N]`.
 * @param options - level (default 0.95, z = 1.96 to the usual two decimals).
 * @returns The bias, the standard deviation and the limits.
 * @example
 * ```ts
 * import { blandAltman } from 'mdml-engine/metrics'
 * blandAltman([2, 4, 6], [1, 3, 5]).bias // => 1
 * ```
 */
export function blandAltman(x: TensorLike, y: TensorLike, options: { readonly level?: number } = {}): BlandAltman {
  const [a, b] = pair('blandAltman', x, y)
  const d = Array.from(a, (v, i) => v - (b[i] ?? 0))
  const n = d.length
  const bias = d.reduce((s, v) => s + v, 0) / n
  const sd = Math.sqrt(d.reduce((s, v) => s + (v - bias) ** 2, 0) / (n - 1))
  const level = options.level ?? 0.95
  const z = item(ndtri(0.5 + level / 2))
  return { bias, sd, lower: bias - z * sd, upper: bias + z * sd, level }
}

/**
 * Cramer's V (scipy.stats.contingency.association with method 'cramer'): sqrt(chi^2 / (n (min(r, c) - 1))) for the
 * contingency table of two categorical variables, in [0, 1].
 *
 * @param x - First categorical variable `[N]`.
 * @param y - Second categorical variable `[N]`.
 * @returns The metric result.
 * @example
 * ```ts
 * import { cramersV } from 'mdml-engine/metrics'
 * cramersV([0, 0, 1, 1], ['a', 'a', 'b', 'b']).estimate.value // => 1
 * ```
 */
export const cramersV: Metric<MetricOptions, MetricInput> = Object.assign(
  (x: MetricInput, y: MetricInput, options: MetricOptions = {}): MetricResult => {
    const m = contingencyMatrix(x, y, options).matrix
    const rows = m.map((r) => r.reduce((a, b) => a + b, 0))
    const cols = Array.from({ length: m[0]?.length ?? 0 }, (_, j) => m.reduce((a, r) => a + (r[j] ?? 0), 0))
    const n = rows.reduce((a, b) => a + b, 0)
    let chi2 = 0
    for (const [i, r] of m.entries()) {
      for (const [j, o] of r.entries()) {
        const e = ((rows[i] ?? 0) * (cols[j] ?? 0)) / n
        chi2 += (o - e) ** 2 / e
      }
    }
    const k = Math.min(rows.length, cols.length)
    return metricResult('cramersV', Math.sqrt(chi2 / (n * (k - 1))), n)
  },
  { spec: spec('cramersV', [0, 1], 'labels') },
)

const entropyOf = (counts: readonly number[]): number => {
  const n = counts.reduce((a, b) => a + b, 0)
  return counts.reduce((h, c) => (c > 0 ? h - (c / n) * Math.log(c / n) : h), 0)
}

/**
 * Theil's U, the uncertainty coefficient U(x | y) = (H(x) - H(x | y)) / H(x): the fraction of the uncertainty about
 * x that knowing y removes (asymmetric; 1 when y determines x).
 *
 * @param x - The predicted categorical variable `[N]`.
 * @param y - The given categorical variable `[N]`.
 * @returns The metric result, in [0, 1].
 * @example
 * ```ts
 * import { theilsU } from 'mdml-engine/metrics'
 * theilsU([0, 0, 1, 1], [0, 1, 2, 3]).estimate.value // => 1
 * ```
 */
export const theilsU: Metric<MetricOptions, MetricInput> = Object.assign(
  (x: MetricInput, y: MetricInput, options: MetricOptions = {}): MetricResult => {
    const m = contingencyMatrix(x, y, options).matrix
    const rows = m.map((r) => r.reduce((a, b) => a + b, 0))
    const n = rows.reduce((a, b) => a + b, 0)
    const hx = entropyOf(rows)
    let hxGivenY = 0
    const cols = m[0]?.length ?? 0
    for (let j = 0; j < cols; j++) {
      const column = m.map((r) => r[j] ?? 0)
      const nj = column.reduce((a, b) => a + b, 0)
      hxGivenY += (nj / n) * entropyOf(column)
    }
    return metricResult('theilsU', hx === 0 ? 1 : (hx - hxGivenY) / hx, n)
  },
  { spec: spec('theilsU', [0, 1], 'labels') },
)

/** The forms of the intraclass correlation (Shrout and Fleiss, 1979; McGraw and Wong, 1996). */
export type IccType = 'ICC1' | 'ICC2' | 'ICC3' | 'ICC1k' | 'ICC2k' | 'ICC3k'

/**
 * The intraclass correlation of ratings (pingouin.intraclass_corr): ICC1 (one-way random, ICC(1,1)), ICC2 (two-way random,
 * absolute agreement, ICC(A,1)) and ICC3 (two-way mixed, consistency, ICC(C,1)), for single raters or the mean of k raters (ICC1k, ICC2k, ICC3k), from the
 * two-way ANOVA mean squares.
 *
 * @param ratings - Ratings `[targets, raters]` (complete).
 * @param options - type (default 'ICC2').
 * @returns The metric result, with the mean squares in the breakdown.
 * @example
 * ```ts
 * import { intraclassCorrelation } from 'mdml-engine/metrics'
 * intraclassCorrelation([[1, 1], [2, 2], [3, 3]]).estimate.value // => 1
 * ```
 */
export const intraclassCorrelation: MetricFunction<[TensorLike, { readonly type?: IccType }?]> = defineMetric(
  spec('intraclassCorrelation'),
  (ratings: TensorLike, options: { readonly type?: IccType } = {}): MetricResult => {
    const r = shapedArray(ratings)
    const n = r.shape[0] ?? 0
    const k = r.shape[1] ?? 0
    if (n < 2 || k < 2) throw new MdmlError('shape', 'intraclassCorrelation: needs at least two targets and two raters')
    let grand = 0
    for (const v of r.data) grand += v
    grand /= n * k
    const rowMeans = Array.from(
      { length: n },
      (_, i) => r.data.subarray(i * k, (i + 1) * k).reduce((a, b) => a + b, 0) / k,
    )
    const colMeans = Array.from({ length: k }, (_, j) => {
      let s = 0
      for (let i = 0; i < n; i++) s += r.data[i * k + j] ?? 0
      return s / n
    })
    const ssRows = k * rowMeans.reduce((a, m) => a + (m - grand) ** 2, 0)
    const ssCols = n * colMeans.reduce((a, m) => a + (m - grand) ** 2, 0)
    let ssTotal = 0
    for (const v of r.data) ssTotal += (v - grand) ** 2
    const ssError = ssTotal - ssRows - ssCols
    const msr = ssRows / (n - 1)
    const msc = ssCols / (k - 1)
    const mse = ssError / ((n - 1) * (k - 1))
    const msw = (ssCols + ssError) / (n * (k - 1))
    const type = options.type ?? 'ICC2'
    const value = {
      ICC1: (msr - msw) / (msr + (k - 1) * msw),
      ICC2: (msr - mse) / (msr + (k - 1) * mse + (k * (msc - mse)) / n),
      ICC3: (msr - mse) / (msr + (k - 1) * mse),
      ICC1k: (msr - msw) / msr,
      ICC2k: (msr - mse) / (msr + (msc - mse) / n),
      ICC3k: (msr - mse) / msr,
    }[type]
    return metricResult('intraclassCorrelation', value, n, { msr, msc, mse, msw })
  },
)

/**
 * Weighted Cohen's kappa for ordinal ratings: {@link cohenKappaScore} with linear (default) or quadratic
 * disagreement weights.
 *
 * @param y1 - First rater.
 * @param y2 - Second rater.
 * @param options - weights (default 'linear'), labels and sample weights.
 * @returns The metric result.
 * @example
 * ```ts
 * import { weightedCohenKappa } from 'mdml-engine/metrics'
 * weightedCohenKappa([0, 1, 2], [0, 1, 2]).estimate.value // => 1
 * ```
 */
export const weightedCohenKappa: Metric<KappaOptions, MetricInput> = Object.assign(
  (y1: MetricInput, y2: MetricInput, options: KappaOptions = {}): MetricResult => ({
    ...cohenKappaScore(y1, y2, { ...options, weights: options.weights ?? 'linear' }),
    metric: 'weightedCohenKappa',
  }),
  { spec: spec('weightedCohenKappa', [-1, 1], 'labels') },
)

/**
 * Fleiss' kappa for many raters (statsmodels.stats.inter_rater.fleiss_kappa): agreement on categorical ratings of
 * subjects by a fixed number of raters, from the table of counts; method 'randolph' uses uniform chance agreement
 * (free-marginal kappa).
 *
 * @param table - Counts `[subjects, categories]`: how many raters put each subject in each category.
 * @param options - method ('fleiss', default, or 'randolph').
 * @returns The metric result.
 * @example
 * ```ts
 * import { fleissKappa } from 'mdml-engine/metrics'
 * fleissKappa([[3, 0], [0, 3], [3, 0]]).estimate.value // => 1
 * ```
 */
export const fleissKappa: MetricFunction<[TensorLike, { readonly method?: 'fleiss' | 'randolph' }?]> = defineMetric(
  spec('fleissKappa', [-1, 1], 'labels'),
  (table: TensorLike, options: { readonly method?: 'fleiss' | 'randolph' } = {}): MetricResult => {
    const t = shapedArray(table)
    const n = t.shape[0] ?? 0
    const c = t.shape[1] ?? 0
    const raters = t.data.subarray(0, c).reduce((a, b) => a + b, 0)
    const total = n * raters
    const pj = Array.from({ length: c }, (_, j) => {
      let s = 0
      for (let i = 0; i < n; i++) s += t.data[i * c + j] ?? 0
      return s / total
    })
    let pBar = 0
    for (let i = 0; i < n; i++) {
      let s = 0
      for (let j = 0; j < c; j++) s += (t.data[i * c + j] ?? 0) ** 2
      pBar += (s - raters) / (raters * (raters - 1))
    }
    pBar /= n
    const pe = options.method === 'randolph' ? 1 / c : pj.reduce((a, p) => a + p * p, 0)
    return metricResult('fleissKappa', (pBar - pe) / (1 - pe), n)
  },
)

/** The level of measurement of Krippendorff's alpha. */
export type MeasurementLevel = 'nominal' | 'ordinal' | 'interval' | 'ratio'

/**
 * Krippendorff's alpha (Krippendorff, 2011; the krippendorff package): 1 - D_o / D_e, observed over expected
 * disagreement of the pairable values within units, for any number of raters, missing data (NaN) and nominal,
 * ordinal, interval or ratio data.
 *
 * @param reliabilityData - Ratings `[raters, units]`, NaN where missing.
 * @param options - level (default 'interval').
 * @returns The metric result.
 * @example
 * ```ts
 * import { krippendorffAlpha } from 'mdml-engine/metrics'
 * krippendorffAlpha([[1, 2, 3], [1, 2, 3]], { level: 'nominal' }).estimate.value // => 1
 * ```
 */
export const krippendorffAlpha: MetricFunction<[TensorLike, { readonly level?: MeasurementLevel }?]> = defineMetric(
  spec('krippendorffAlpha', [-Infinity, 1], 'labels'),
  (reliabilityData: TensorLike, options: { readonly level?: MeasurementLevel } = {}): MetricResult => {
    const r = shapedArray(reliabilityData)
    const raters = r.shape[0] ?? 0
    const units = r.shape[1] ?? 0
    const values = sortLabels(Array.from(r.data).filter((v) => !Number.isNaN(v))) as number[]
    const v = values.length
    const index = new Map(values.map((x, i) => [x, i]))
    // The coincidence matrix o[c][k]: pairs of values within units, each unit weighted by 1 / (m_u - 1).
    const o = Array.from({ length: v }, () => new Float64Array(v))
    for (let u = 0; u < units; u++) {
      const present: number[] = []
      for (let i = 0; i < raters; i++) {
        const x = r.data[i * units + u] ?? Number.NaN
        if (!Number.isNaN(x)) present.push(index.get(x) ?? 0)
      }
      const m = present.length
      if (m < 2) continue
      for (const [a, ca] of present.entries()) {
        for (const [b, cb] of present.entries())
          if (a !== b) (o[ca] as Float64Array)[cb] = ((o[ca] as Float64Array)[cb] ?? 0) + 1 / (m - 1)
      }
    }
    const nc = o.map((row) => row.reduce((a, b) => a + b, 0))
    const n = nc.reduce((a, b) => a + b, 0)
    const level = options.level ?? 'interval'
    const delta = (c: number, k: number): number => {
      if (level === 'nominal') return c === k ? 0 : 1
      if (level === 'interval') return ((values[c] ?? 0) - (values[k] ?? 0)) ** 2
      if (level === 'ratio') {
        const a = values[c] ?? 0
        const b = values[k] ?? 0
        return a + b === 0 ? 0 : ((a - b) / (a + b)) ** 2
      }
      // Ordinal: (sum of the marginals from c to k, minus half of the two ends)^2.
      const lo = Math.min(c, k)
      const hi = Math.max(c, k)
      let s = 0
      for (let g = lo; g <= hi; g++) s += nc[g] ?? 0
      return (s - ((nc[c] ?? 0) + (nc[k] ?? 0)) / 2) ** 2
    }
    let observed = 0
    let expected = 0
    for (let c = 0; c < v; c++) {
      for (let k = 0; k < v; k++) {
        const d = delta(c, k)
        observed += ((o[c] as Float64Array)[k] ?? 0) * d
        expected += (nc[c] ?? 0) * (nc[k] ?? 0) * d
      }
    }
    const value = expected === 0 ? 1 : 1 - ((n - 1) * observed) / expected
    return metricResult('krippendorffAlpha', value, units)
  },
)
