/**
 * Nonparametric density estimates: the Gaussian kernel density estimate (as in scipy's `gaussian_kde` and
 * scikit-learn's `KernelDensity`) with the bandwidth by rule, given or chosen by cross-validated log likelihood, and
 * histogram densities for one or two features (as numpy's `histogram` with `density`).
 */

import { WhiteboxError } from '../../lib/base/index.ts'
import { crossValidationCurve, type Exposition } from '../../lib/exposition/index.ts'
import {
  decisionSurfaceOf,
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  type Fitted,
  fitRunOptions,
  type LearnData,
  matrixView,
} from '../../lib/learn/index.ts'
import { type Key, permutation } from '../../lib/random/index.ts'
import { choice, dict, int } from '../../lib/space/index.ts'
import { gaussianKde, histogram, histogram2d, scottFactor, silvermanFactor } from '../../lib/stats/index.ts'
import { exp, fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'

/** The verbs of a density estimate: log densities per row and their mean. */
export interface DensityVerbs {
  /** Per-row log densities `[m]`. */
  readonly scoreSamples: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
  readonly nFeaturesIn: number
}

const mean = (xs: readonly number[]): number => xs.reduce((a, v) => a + v, 0) / xs.length

/**
 * A density surface over the data (the density, not its log, so the modes read clearly), for one or two features;
 * piecewise constant for a histogram.
 */
function densitySurface(fitted: Fitted & DensityVerbs, x: TensorLike, d: number, smooth = true): Exposition[] {
  if (d > 2) return []
  const density = (at: TensorLike): Tensor => primalOf(exp(fitted.scoreSamples(at)))
  return [
    decisionSurfaceOf({ ...fitted, kind: 'density', predict: density }, x, {
      resolution: smooth ? 40 : 60,
      field: { name: 'density', smooth },
    }),
  ]
}

// ---------------------------------------------------------------------------------------------------------------
// Kernel density

/** Hyperparameters of the kernel density estimate. */
export type KernelDensityHyper = {
  /**
   * The bandwidth factor multiplying the data covariance's square root: Scott's or Silverman's rule, a number, or
   * `'cv'`, chosen by k-fold held-out log likelihood over a grid of factors (default Scott).
   */
  readonly bandwidth?: 'scott' | 'silverman' | 'cv' | number
  /** The factors tried by `'cv'` (default 12 from 0.1 to 3 times Scott's, log spaced). */
  readonly factors?: readonly number[]
  /** Folds of the bandwidth search (default 5). */
  readonly folds?: number
}

/** The parameters of a kernel density estimate: the data and the bandwidth factor. */
export type KernelDensityParams = { readonly data: number[][]; readonly factor: number }

/** A fitted kernel density estimate. */
export type FittedKernelDensity = ExplainedFitted<KernelDensityParams, DensityVerbs>

function kdeVerbs(p: KernelDensityParams): DensityVerbs {
  const d = p.data[0]?.length ?? 0
  const kde = gaussianKde(d === 1 ? p.data.map((r) => r[0] ?? 0) : p.data, { bandwidth: p.factor })
  const logpdf = (x: TensorLike): number[] => {
    const m = matrixView(x, 'kernelDensity', { nFeatures: d })
    const rows = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * d, (i + 1) * d)))
    return toFlat(kde.logpdf(d === 1 ? rows.map((r) => r[0] ?? 0) : rows))
  }
  return {
    scoreSamples: (x) => {
      const v = logpdf(x)
      return fromStorage(Float64Array.from(v), [v.length], 'float64')
    },
    score: (data) => mean(logpdf(data.x)),
    nFeaturesIn: d,
  }
}

/**
 * The k-fold held-out mean log density of a Gaussian KDE at each bandwidth factor.
 *
 * @param rows - Points `[n][d]`.
 * @param factors - The factors.
 * @param folds - The number of folds.
 * @param key - Randomness for the fold assignment.
 * @returns Per factor, the score on each fold.
 */
export function kdeFoldScores(
  rows: readonly (readonly number[])[],
  factors: readonly number[],
  folds: number,
  key: Key,
): number[][] {
  const n = rows.length
  if (folds < 2 || folds > n) throw new WhiteboxError('value', `kdeFoldScores: folds must be in 2..${n}`)
  const order = Array.from(permutation(key, n).data)
  const fold = new Array<number>(n)
  for (const [r, i] of order.entries()) fold[i] = r % folds
  return factors.map((factor) =>
    Array.from({ length: folds }, (_, f) => {
      const train = rows.filter((_, i) => fold[i] !== f).map((r) => [...r])
      const test = rows.filter((_, i) => fold[i] === f).map((r) => [...r])
      return kdeVerbs({ data: train, factor }).score({ x: test })
    }),
  )
}

/**
 * A Gaussian kernel density estimate (scipy.stats.gaussian_kde, sklearn.neighbors.KernelDensity) over `stats`'
 * `gaussianKde`, with the bandwidth by rule, given, or chosen by cross-validated log likelihood. Its exposition is a
 * `crossValidationCurve` over bandwidth factors (for `'cv'`) and, for one or two features, a `decisionSurface` of the
 * log density.
 *
 * @param hyper - The bandwidth rule or factor, the factors searched and the folds.
 * @returns The estimator.
 */
export function kernelDensity(hyper: KernelDensityHyper = {}): Estimator<KernelDensityHyper, FittedKernelDensity> {
  return defineExplainedEstimator<KernelDensityHyper, KernelDensityParams, DensityVerbs>(
    {
      name: 'kernel-density',
      kind: 'density',
      hyperSpace: (h) =>
        dict({
          bandwidth: choice(['scott', 'silverman', 'cv'], {
            label: 'bandwidth rule',
            description: 'Scott’s or Silverman’s rule of thumb, or cross-validated likelihood.',
            default: typeof h.bandwidth === 'string' ? h.bandwidth : 'scott',
          }),
          folds: int(2, 20, {
            label: 'folds',
            description: 'Cross-validation folds for the cv bandwidth.',
            default: h.folds ?? 5,
          }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'kernelDensity', { minSamples: 2 })
        const rows = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const bw = h.bandwidth ?? 'scott'
        const exposition: Exposition[] = []
        let factor: number
        if (typeof bw === 'number') factor = bw
        else if (bw === 'scott') factor = scottFactor(m.n, m.d)
        else if (bw === 'silverman') factor = silvermanFactor(m.n, m.d)
        else {
          const base = scottFactor(m.n, m.d)
          const factors = h.factors ?? Array.from({ length: 12 }, (_, i) => base * 0.1 * 30 ** (i / 11))
          const folds = h.folds ?? 5
          const scores = kdeFoldScores(rows, factors, folds, fitRunOptions(options, 'kernel-density').key as Key)
          const means = scores.map(mean)
          const ses = scores.map((s, i) => {
            const mu = means[i] ?? 0
            return Math.sqrt(s.reduce((a, v) => a + (v - mu) ** 2, 0) / Math.max(1, s.length - 1) / s.length)
          })
          const best = means.indexOf(Math.max(...means))
          factor = factors[best] ?? base
          exposition.push(
            crossValidationCurve({
              parameter: 'bandwidth factor',
              values: [...factors],
              meanScores: means,
              standardErrors: ses,
              foldScores: scores,
              higherIsBetter: true,
              best: factor,
              scoreName: 'held-out log likelihood',
            }),
          )
        }
        const params = { data: rows, factor }
        const fitted = { estimator: 'kernel-density', kind: 'density' as const, params, ...kdeVerbs(params) }
        return { params, exposition: [...exposition, ...densitySurface(fitted, data.x, m.d)] }
      },
      methods: kdeVerbs,
    },
    hyper,
  )
}

/** The kernel density estimate as a class. */
export class KernelDensity extends EstimatorClass<KernelDensityHyper, FittedKernelDensity> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link kernelDensity}.
   */
  constructor(hyper: KernelDensityHyper = {}) {
    super(kernelDensity(hyper))
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Histograms

/** Hyperparameters of the histogram density. */
export type HistogramDensityHyper = {
  /** Bins per feature (default 10), or (one feature) a numpy rule. */
  readonly bins?: number | 'auto' | 'fd' | 'scott' | 'sturges' | 'doane' | 'rice' | 'sqrt'
}

/** The parameters of a histogram density: edges per feature and densities per bin (row-major for two features). */
export type HistogramDensityParams = { readonly edges: number[][]; readonly density: number[] }

/** A fitted histogram density. */
export type FittedHistogramDensity = ExplainedFitted<HistogramDensityParams, DensityVerbs>

function binOf(edges: readonly number[], v: number): number {
  const last = edges.length - 1
  if (!(v >= (edges[0] ?? 0) && v <= (edges[last] ?? 0))) return -1
  let lo = 0
  let hi = last
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (v >= (edges[mid] ?? 0)) lo = mid
    else hi = mid
  }
  return lo
}

function histogramVerbs(p: HistogramDensityParams): DensityVerbs {
  const d = p.edges.length
  const ny = d === 2 ? (p.edges[1]?.length ?? 1) - 1 : 1
  const logpdf = (x: TensorLike): number[] => {
    const m = matrixView(x, 'histogramDensity', { nFeatures: d })
    return Array.from({ length: m.n }, (_, i) => {
      const a = binOf(p.edges[0] ?? [], m.data[i * d] ?? 0)
      const b = d === 2 ? binOf(p.edges[1] ?? [], m.data[i * d + 1] ?? 0) : 0
      return a < 0 || b < 0 ? Number.NEGATIVE_INFINITY : Math.log(p.density[a * ny + b] ?? 0)
    })
  }
  return {
    scoreSamples: (x) => {
      const v = logpdf(x)
      return fromStorage(Float64Array.from(v), [v.length], 'float64')
    },
    score: (data) => mean(logpdf(data.x)),
    nFeaturesIn: d,
  }
}

/**
 * A histogram density for one or two features (numpy.histogram and histogram2d with `density`), over `stats`'
 * histograms. Its exposition is a `decisionSurface` of the log density.
 *
 * @param hyper - The bins.
 * @returns The estimator.
 */
export function histogramDensity(
  hyper: HistogramDensityHyper = {},
): Estimator<HistogramDensityHyper, FittedHistogramDensity> {
  return defineExplainedEstimator<HistogramDensityHyper, HistogramDensityParams, DensityVerbs>(
    {
      name: 'histogram-density',
      kind: 'density',
      hyperSpace: (h) =>
        dict({
          bins: int(2, 100, {
            label: 'bins',
            description: 'The number of histogram bins along each feature.',
            default: typeof h.bins === 'number' ? h.bins : 10,
          }),
        }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'histogramDensity')
        if (m.d > 2) throw new WhiteboxError('shape', 'histogramDensity: one or two features')
        const col = (j: number): number[] => Array.from({ length: m.n }, (_, i) => m.data[i * m.d + j] ?? 0)
        let params: HistogramDensityParams
        if (m.d === 1) {
          const hist = histogram(col(0), { bins: h.bins ?? 10, density: true })
          params = { edges: [toFlat(hist.edges)], density: toFlat(hist.counts) }
        } else {
          const bins = typeof h.bins === 'number' ? h.bins : 10
          const hist = histogram2d(col(0), col(1), { bins: [bins, bins], density: true })
          params = { edges: [toFlat(hist.xEdges), toFlat(hist.yEdges)], density: toFlat(hist.counts) }
        }
        const fitted = { estimator: 'histogram-density', kind: 'density' as const, params, ...histogramVerbs(params) }
        return { params, exposition: densitySurface(fitted, data.x, m.d, false) }
      },
      methods: histogramVerbs,
    },
    hyper,
  )
}

/** The histogram density as a class. */
export class HistogramDensity extends EstimatorClass<HistogramDensityHyper, FittedHistogramDensity> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link histogramDensity}.
   */
  constructor(hyper: HistogramDensityHyper = {}) {
    super(histogramDensity(hyper))
  }
}
