import { WhiteboxError } from '../../lib/base/index.ts'
import { componentsRecord } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  featureNamesOf,
  fitRunOptions,
  type LearnData,
  matrixView,
} from '../../lib/learn/index.ts'
import { linalg } from '../../lib/numerics/index.ts'
import type { Key } from '../../lib/random/index.ts'
import { bool, choice, dict, int } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'
import { centred, columnMeans, flipSigns, project, rankErrors, reconstruct, rowsOf, thinSvd } from './_linear.ts'

/** The parameters of a fitted PCA. */
export type PcaParams = {
  readonly mean: number[]
  /** components[k][f]: the principal axes, by decreasing variance. */
  readonly components: number[][]
  readonly explainedVariance: number[]
  readonly explainedVarianceRatio: number[]
  readonly singularValues: number[]
  /** The mean variance of the discarded directions (the probabilistic PCA noise variance). */
  readonly noiseVariance: number
  readonly whiten: boolean
}

/** The verbs of a linear projection: scores, reconstructions and (for PCA) the probabilistic log likelihood. */
export interface ProjectionVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly inverseTransform: (z: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
  readonly nFeaturesIn: number
  readonly featureNamesOut: (input?: readonly string[]) => string[]
}

/** A fitted PCA. */
export type FittedPca = ExplainedFitted<PcaParams, ProjectionVerbs>

/**
 * The average log likelihood of rows under probabilistic PCA with the fitted axes and noise variance (sklearn's
 * `PCA.score`): a Gaussian with covariance W^T diag(ev - sigma^2) W + sigma^2 I.
 *
 * @param p - The PCA parameters.
 * @param x - Points `[n, d]`.
 * @returns The mean log likelihood.
 */
export function pcaLogLikelihood(p: PcaParams, x: TensorLike): number {
  const d = p.mean.length
  const s2 = Math.max(p.noiseVariance, 1e-300)
  const cov = Array.from({ length: d }, (_, a) =>
    Array.from({ length: d }, (_, b) => {
      let v = a === b ? s2 : 0
      for (const [c, w] of p.components.entries()) v += ((p.explainedVariance[c] ?? 0) - s2) * (w[a] ?? 0) * (w[b] ?? 0)
      return v
    }),
  )
  const precision = toFlat(linalg.inv(cov))
  const logdet = toFlat(linalg.slogdet(cov).logabsdet)[0] ?? 0
  const m = matrixView(x, 'pca score', { nFeatures: d })
  let total = 0
  for (let i = 0; i < m.n; i++) {
    let q = 0
    for (let a = 0; a < d; a++)
      for (let b = 0; b < d; b++)
        q +=
          ((m.data[i * d + a] ?? 0) - (p.mean[a] ?? 0)) *
          (precision[a * d + b] ?? 0) *
          ((m.data[i * d + b] ?? 0) - (p.mean[b] ?? 0))
    total += -0.5 * (q + d * Math.log(2 * Math.PI) + logdet)
  }
  return total / m.n
}

/**
 * The verbs of a PCA-like model.
 *
 * @param p - The parameters.
 * @param prefix - The output feature prefix.
 * @returns Projection, reconstruction and the probabilistic score.
 */
export function pcaVerbs(p: PcaParams, prefix = 'pca'): ProjectionVerbs {
  const scale = p.whiten ? p.explainedVariance.map(Math.sqrt) : undefined
  return {
    transform: (x) => project(x, p.mean, p.components, scale),
    inverseTransform: (z) => reconstruct(z, p.mean, p.components, scale),
    score: (data) => pcaLogLikelihood(p, data.x),
    nFeaturesIn: p.mean.length,
    featureNamesOut: () => p.components.map((_, c) => `${prefix}${c}`),
  }
}

/** Hyperparameters of PCA. */
export type PcaHyper = {
  /** Components kept (default all, min(n, d)). */
  readonly nComponents?: number
  /** Scale scores to unit variance (default false). */
  readonly whiten?: boolean
  /** A full SVD, or the randomised range finder of `numerics/linalg` (default full). */
  readonly svdSolver?: 'full' | 'randomized'
}

/**
 * Principal component analysis (sklearn.decomposition.PCA) by the SVD of the centred data (`numerics/linalg`), full or
 * randomised, with whitening and the inverse transform. Its exposition is a `components` record (axes, explained
 * variance and ratios, the training scores and the reconstruction error with 1, 2, ... components).
 *
 * @param hyper - Components, whitening and the solver.
 * @returns The estimator.
 */
export function pca(hyper: PcaHyper = {}): Estimator<PcaHyper, FittedPca> {
  return defineExplainedEstimator<PcaHyper, PcaParams, ProjectionVerbs>(
    {
      name: 'pca',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 50, { default: h.nComponents ?? 2, label: 'components' }),
          whiten: bool({ default: h.whiten ?? false }),
          svdSolver: choice(['full', 'randomized'], { default: h.svdSolver ?? 'full' }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'pca', { minSamples: 2 })
        const full = Math.min(m.n, m.d)
        const k = h.nComponents ?? full
        if (!(Number.isInteger(k) && k >= 1 && k <= full))
          throw new WhiteboxError('value', `pca: nComponents must be in 1..${full}, got ${k}`)
        const mean = columnMeans(m)
        const xc = centred(m, mean)
        let totalSquares = 0
        for (const v of xc) totalSquares += v * v
        let s: number[]
        let vt: number[][]
        if (h.svdSolver === 'randomized') {
          const r = linalg.randomizedSvd(fromStorage(xc, [m.n, m.d], 'float64'), k, {
            key: fitRunOptions(options, 'pca').key as Key,
          })
          s = toFlat(r.s)
          const flat = toFlat(r.vh)
          vt = Array.from({ length: k }, (_, c) => flat.slice(c * m.d, (c + 1) * m.d))
          flipSigns(undefined, vt)
        } else ({ s, vt } = thinSvd(xc, m.n, m.d))
        const ev = s.map((v) => (v * v) / (m.n - 1))
        const total = totalSquares / (m.n - 1)
        const kept = ev.slice(0, k)
        const rest = ev.slice(k)
        const noiseVariance =
          h.svdSolver === 'randomized'
            ? Math.max(0, total - kept.reduce((a, v) => a + v, 0)) / Math.max(1, Math.min(m.n, m.d) - k)
            : rest.length > 0
              ? rest.reduce((a, v) => a + v, 0) / rest.length
              : 0
        const params: PcaParams = {
          mean,
          components: vt.slice(0, k),
          explainedVariance: kept,
          explainedVarianceRatio: kept.map((v) => v / total),
          singularValues: s.slice(0, k),
          noiseVariance,
          whiten: h.whiten ?? false,
        }
        const scores = rowsOf(pcaVerbs(params).transform(data.x))
        return {
          params,
          exposition: [
            componentsRecord({
              method: 'pca',
              features: featureNamesOf(data, m.d),
              components: params.components,
              explainedVariance: params.explainedVariance,
              explainedVarianceRatio: params.explainedVarianceRatio,
              scores,
              reconstructionError: rankErrors(s, totalSquares, m.n, m.d, k),
            }),
          ],
        }
      },
      methods: (p) => pcaVerbs(p),
    },
    hyper,
  )
}

/** PCA as a class. */
export class PCA extends EstimatorClass<PcaHyper, FittedPca> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link pca}.
   */
  constructor(hyper: PcaHyper = {}) {
    super(pca(hyper))
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Incremental PCA

/** The state of incremental PCA after some batches. */
export interface IncrementalPcaState {
  readonly seen: number
  readonly mean: number[]
  readonly variance: number[]
  readonly components: number[][]
  readonly singularValues: number[]
  readonly explainedVariance: number[]
  readonly explainedVarianceRatio: number[]
  /** The mean explained variance of the directions beyond the kept ones (the noise variance). */
  readonly restVariance: number
}

/**
 * The batches of sklearn's `gen_batches`: consecutive slices of `size`, a short last one merged into its predecessor
 * when it has fewer than `min` rows.
 *
 * @param n - Rows.
 * @param size - Batch size.
 * @param min - The smallest batch.
 * @returns [start, end) per batch.
 */
export function batchRanges(n: number, size: number, min = 0): [number, number][] {
  const out: [number, number][] = []
  let start = 0
  for (let b = 0; b < Math.floor(n / size); b++) {
    const end = start + size
    if (end + min > n) continue
    out.push([start, end])
    start = end
  }
  if (start < n) out.push([start, n])
  return out
}

/**
 * Incremental PCA (Ross et al. 2008; sklearn.decomposition.IncrementalPCA) as a traced algorithm: each step folds one
 * batch into the running mean, variance and SVD (the previous axes scaled by their singular values, the centred
 * batch and a mean correction row).
 *
 * @param x - Points `[n, d]`.
 * @param k - Components kept.
 * @param batches - The batches, as [start, end) row ranges.
 * @returns The algorithm; one step per batch.
 */
export function incrementalPcaAlgorithm(
  x: TensorLike,
  k: number,
  batches: readonly (readonly [number, number])[],
): Algorithm<IncrementalPcaState> {
  const m = matrixView(x, 'incrementalPca')
  const d = m.d
  return {
    name: 'incremental-pca',
    config: { k, batches: batches.length },
    init: () => ({
      seen: 0,
      mean: new Array<number>(d).fill(0),
      variance: new Array<number>(d).fill(0),
      components: [],
      singularValues: [],
      explainedVariance: [],
      explainedVarianceRatio: [],
      restVariance: 0,
    }),
    step: (s, ctx) => {
      const [a, b] = batches[ctx.t] as [number, number]
      const nb = b - a
      const total = s.seen + nb
      const view = { n: nb, d, data: m.data.subarray(a * d, b * d) }
      const batchMean = columnMeans(view)
      const batchVar = Array.from({ length: d }, (_, j) => {
        let v = 0
        for (let i = 0; i < nb; i++) v += ((view.data[i * d + j] ?? 0) - (batchMean[j] ?? 0)) ** 2
        return v / nb
      })
      // Chan's parallel update of the mean and (population) variance.
      const mean = batchMean.map((mb, j) => ((s.mean[j] ?? 0) * s.seen + mb * nb) / total)
      const variance = batchVar.map((vb, j) => {
        const delta = (batchMean[j] ?? 0) - (s.mean[j] ?? 0)
        return ((s.variance[j] ?? 0) * s.seen + vb * nb + (delta * delta * s.seen * nb) / total) / total
      })
      const rows: number[][] = []
      for (const [c, w] of s.components.entries()) rows.push(w.map((v) => v * (s.singularValues[c] ?? 0)))
      for (let i = 0; i < nb; i++)
        rows.push(Array.from({ length: d }, (_, j) => (view.data[i * d + j] ?? 0) - (batchMean[j] ?? 0)))
      if (s.seen > 0) {
        const f = Math.sqrt((s.seen / total) * nb)
        rows.push(s.mean.map((v, j) => f * (v - (batchMean[j] ?? 0))))
      }
      const { s: sv, vt } = thinSvd(Float64Array.from(rows.flat()), rows.length, d)
      const totalVar = variance.reduce((acc, v) => acc + v * total, 0)
      return {
        seen: total,
        mean,
        variance,
        components: vt.slice(0, k),
        singularValues: sv.slice(0, k),
        explainedVariance: sv.slice(0, k).map((v) => (v * v) / (total - 1)),
        explainedVarianceRatio: sv.slice(0, k).map((v) => (v * v) / totalVar),
        restVariance:
          sv.length > k ? sv.slice(k).reduce((acc, v) => acc + (v * v) / (total - 1), 0) / (sv.length - k) : 0,
      }
    },
    done: (_s, t) => (t >= batches.length ? 'completed' : undefined),
    observe: (s) => ({ metrics: { seen: s.seen, explained: s.explainedVarianceRatio.reduce((a, v) => a + v, 0) } }),
  }
}

/** Hyperparameters of incremental PCA. */
export type IncrementalPcaHyper = {
  readonly nComponents?: number
  /** Rows per batch (default 5 times the features). */
  readonly batchSize?: number
  readonly whiten?: boolean
}

/**
 * Incremental PCA (sklearn.decomposition.IncrementalPCA): PCA over batches in constant memory, traced batch by
 * batch. Its exposition is a `components` record of the final axes with the training scores.
 *
 * @param hyper - Components, batch size and whitening.
 * @returns The estimator.
 */
export function incrementalPca(hyper: IncrementalPcaHyper = {}): Estimator<IncrementalPcaHyper, FittedPca> {
  return defineExplainedEstimator<IncrementalPcaHyper, PcaParams, ProjectionVerbs>(
    {
      name: 'incremental-pca',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 50, { default: h.nComponents ?? 2, label: 'components' }),
          batchSize: int(2, 1000, { default: h.batchSize ?? 20, label: 'batch size' }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'incrementalPca', { minSamples: 2 })
        const k = h.nComponents ?? Math.min(m.n, m.d)
        const batches = batchRanges(m.n, h.batchSize ?? 5 * m.d, k)
        if (batches.some(([a, b]) => b - a < k))
          throw new WhiteboxError('value', 'incrementalPca: every batch needs at least nComponents rows')
        const out = runAlgorithm(incrementalPcaAlgorithm(data.x, k, batches), {
          ...fitRunOptions(options, 'incremental-pca'),
          maxSteps: batches.length,
        })
        const s = out.state
        const params: PcaParams = {
          mean: s.mean,
          components: s.components,
          explainedVariance: s.explainedVariance,
          explainedVarianceRatio: s.explainedVarianceRatio,
          singularValues: s.singularValues,
          noiseVariance: k < m.d ? s.restVariance : 0,
          whiten: h.whiten ?? false,
        }
        return {
          params,
          exposition: [
            componentsRecord({
              method: 'incrementalPca',
              features: featureNamesOf(data, m.d),
              components: params.components,
              explainedVariance: params.explainedVariance,
              explainedVarianceRatio: params.explainedVarianceRatio,
              scores: rowsOf(pcaVerbs(params).transform(data.x)),
            }),
          ],
        }
      },
      methods: (p) => pcaVerbs(p, 'incrementalpca'),
    },
    hyper,
  )
}

/** Incremental PCA as a class. */
export class IncrementalPCA extends EstimatorClass<IncrementalPcaHyper, FittedPca> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link incrementalPca}.
   */
  constructor(hyper: IncrementalPcaHyper = {}) {
    super(incrementalPca(hyper))
  }
}
