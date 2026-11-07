/**
 * Principal component analysis as a transformer (scikit-learn's `PCA`): the orthogonal directions of greatest variance,
 * found by the SVD of the centred data (full or randomised), or by power iteration one component per step, with
 * whitening, the inverse transform and the probabilistic PCA log likelihood. Incremental PCA folds in one batch per step
 * in constant memory, and sparse PCA finds one sparse component per step by a thresholded power iteration.
 */

import { MdmlError, maxOf } from '../base/index.ts'
import { type Fitted, type IterativeEstimator, iterative, type LearnData, matrixView } from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { normal } from '../random/index.ts'
import { bool, choice, dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { centred, columnMeans, flipSigns, mm, project, reconstruct, thinSvd, tr } from './_linear.ts'

/** The state of the PCA algorithms: the mean and the components found so far, with their variances. */
export interface PcaState {
  readonly mean: number[]
  /** components[c][f]: the principal axes found so far, by decreasing variance. */
  readonly components: number[][]
  readonly singularValues: number[]
  readonly explainedVariance: number[]
  readonly explainedVarianceRatio: number[]
  /** The mean variance of the directions not found (the probabilistic PCA noise variance). */
  readonly noiseVariance: number
  /** The total variance of the data. */
  readonly totalVariance: number
}

/** The fitted parameters of a PCA-like model. */
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
export type FittedPca = Fitted<PcaParams> & ProjectionVerbs

const pcaSchema = channels({
  metrics: {
    explained: { description: 'the share of the variance explained by the components found so far' },
    found: { description: 'the number of components found' },
  },
  tensors: { components: { role: 'state', description: 'the principal axes found so far', shape: ['k', 'd'] } },
})

const observePca = (s: PcaState, d: number) => ({
  metrics: {
    explained: s.explainedVarianceRatio.reduce((a, v) => a + v, 0),
    found: s.components.length,
  },
  tensors: { components: { role: 'state' as const, value: s.components.flat(), shape: [s.components.length, d] } },
})

/**
 * The average log likelihood of rows under probabilistic PCA with the fitted axes and noise variance (sklearn's
 * `PCA.score`): a Gaussian with covariance W^T diag(ev - sigma^2) W + sigma^2 I.
 *
 * @param p - The PCA parameters.
 * @param x - Points `[n, d]`.
 * @returns The mean log likelihood.
 * @example
 * ```ts
 * import { pcaLogLikelihood } from 'mdml-engine/decomposition'
 * const p = { mean: [0, 0], components: [[1, 0]], explainedVariance: [4], explainedVarianceRatio: [0.8], singularValues: [2], noiseVariance: 1, whiten: false }
 * pcaLogLikelihood(p, [[0, 0]]) // => -2.531024
 * ```
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
 * The verbs of a PCA-like model: projection (whitened if asked), reconstruction and the probabilistic score.
 *
 * @param p - The parameters.
 * @param prefix - The output feature prefix.
 * @returns Projection, reconstruction and the probabilistic score.
 * @example
 * ```ts
 * import { pcaVerbs } from 'mdml-engine/decomposition'
 * import { toArray } from 'mdml-engine/tensor'
 * const p = { mean: [1, 1], components: [[1, 0]], explainedVariance: [4], explainedVarianceRatio: [1], singularValues: [2], noiseVariance: 0, whiten: false }
 * toArray(pcaVerbs(p).transform([[3, 5]])) // => [[2]]
 * ```
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

/** The state after `s` and `vt` (singular values and right vectors, k of them) are known. */
function stateFrom(
  mean: number[],
  s: readonly number[],
  vt: readonly (readonly number[])[],
  k: number,
  n: number,
  totalVariance: number,
  noiseVariance: number,
): PcaState {
  const ev = s.slice(0, k).map((v) => (v * v) / (n - 1))
  return {
    mean,
    components: vt.slice(0, k).map((r) => [...r]),
    singularValues: s.slice(0, k),
    explainedVariance: ev,
    explainedVarianceRatio: ev.map((v) => v / totalVariance),
    noiseVariance,
    totalVariance,
  }
}

const emptyState = (mean: number[], totalVariance: number): PcaState => ({
  mean,
  components: [],
  singularValues: [],
  explainedVariance: [],
  explainedVarianceRatio: [],
  noiseVariance: 0,
  totalVariance,
})

/** Options of the PCA algorithms. */
export interface PcaAlgorithmOptions {
  /** `'full'`: an exact SVD; `'randomized'`: the randomised range finder of `numerics/linalg`. */
  readonly solver?: 'full' | 'randomized'
}

/**
 * PCA by the SVD of the centred data as a traced {@link Algorithm}: one step finds every component (exactly, or by the
 * randomised range finder of `numerics/linalg`).
 *
 * @param x - Points `[n, d]`.
 * @param k - Components kept.
 * @param options - The solver.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { pcaSvdAlgorithm } from 'mdml-engine/decomposition'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[-2, 0], [-1, 0.1], [1, -0.1], [2, 0]]
 * runAlgorithm(pcaSvdAlgorithm(x, 1), { maxSteps: 2 }).state.explainedVarianceRatio[0] // => 0.998404
 * ```
 */
export function pcaSvdAlgorithm(x: TensorLike, k: number, options: PcaAlgorithmOptions = {}): Algorithm<PcaState> {
  const m = matrixView(x, 'pca', { minSamples: 2 })
  const full = Math.min(m.n, m.d)
  if (!(Number.isInteger(k) && k >= 1 && k <= full))
    throw new MdmlError('value', `pca: nComponents must be in 1..${full}, got ${k}`)
  const solver = options.solver ?? 'full'
  const mean = columnMeans(m)
  const xc = centred(m, mean)
  let totalSquares = 0
  for (const v of xc) totalSquares += v * v
  const total = totalSquares / (m.n - 1)
  return {
    name: 'pca-svd',
    config: { k, solver },
    schema: pcaSchema,
    init: () => emptyState(mean, total),
    step: (_s, ctx) => {
      if (solver === 'randomized') {
        const r = linalg.randomizedSvd(fromStorage(xc, [m.n, m.d], 'float64'), k, { key: ctx.key })
        const flat = toFlat(r.vh)
        const vt = Array.from({ length: k }, (_, c) => flat.slice(c * m.d, (c + 1) * m.d))
        flipSigns(undefined, vt)
        const sv = toFlat(r.s)
        const kept = sv.slice(0, k).reduce((a, v) => a + (v * v) / (m.n - 1), 0)
        return stateFrom(mean, sv, vt, k, m.n, total, Math.max(0, total - kept) / Math.max(1, full - k))
      }
      const { s: sv, vt } = thinSvd(xc, m.n, m.d)
      const rest = sv.slice(k).map((v) => (v * v) / (m.n - 1))
      return stateFrom(mean, sv, vt, k, m.n, total, rest.length > 0 ? rest.reduce((a, v) => a + v, 0) / rest.length : 0)
    },
    done: (s) => (s.components.length >= k ? 'completed' : undefined),
    observe: (s) => observePca(s, m.d),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PcaState,
  }
}

/** Remove from `w` its components along the unit rows `against` (Gram-Schmidt), in place. */
function orthogonalise(w: Float64Array, against: readonly (readonly number[])[]): void {
  for (const u of against) {
    let dot = 0
    for (let j = 0; j < w.length; j++) dot += (w[j] ?? 0) * (u[j] ?? 0)
    for (let j = 0; j < w.length; j++) w[j] = (w[j] ?? 0) - dot * (u[j] ?? 0)
  }
}

/** X^T (X v) for a flat `[n, d]` matrix. */
function normalMatvec(xc: Float64Array, n: number, d: number, v: ArrayLike<number>): Float64Array {
  const out = new Float64Array(d)
  for (let i = 0; i < n; i++) {
    let t = 0
    for (let j = 0; j < d; j++) t += (xc[i * d + j] ?? 0) * (v[j] ?? 0)
    for (let j = 0; j < d; j++) out[j] = (out[j] ?? 0) + (xc[i * d + j] ?? 0) * t
  }
  return out
}

/**
 * PCA by power iteration as a traced {@link Algorithm}, one component per step: the power method on the covariance
 * (applied as X^T X v, never formed), kept orthogonal to the components already found, run to convergence each step.
 * It needs no SVD and shows the components emerging in order of variance.
 *
 * @param x - Points `[n, d]`.
 * @param k - Components to find.
 * @param options - Tolerance on the change of the iterate (default 1e-12) and the iteration cap per component (default 2000).
 * @returns The algorithm; k steps.
 * @example
 * ```ts
 * import { powerPcaAlgorithm } from 'mdml-engine/decomposition'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[-2, -1], [-1, -0.5], [1, 0.5], [2, 1]]
 * runAlgorithm(powerPcaAlgorithm(x, 1), { maxSteps: 1, key: key(0) }).state.components[0]?.map((v) => Math.round(v * 100) / 100) // => [0.89, 0.45]
 * ```
 */
export function powerPcaAlgorithm(
  x: TensorLike,
  k: number,
  options: { readonly tol?: number; readonly maxIter?: number } = {},
): Algorithm<PcaState> {
  const m = matrixView(x, 'pca', { minSamples: 2 })
  const full = Math.min(m.n, m.d)
  if (!(Number.isInteger(k) && k >= 1 && k <= full))
    throw new MdmlError('value', `pca: nComponents must be in 1..${full}, got ${k}`)
  const tol = options.tol ?? 1e-12
  const maxIter = options.maxIter ?? 2000
  const mean = columnMeans(m)
  const xc = centred(m, mean)
  let totalSquares = 0
  for (const v of xc) totalSquares += v * v
  const total = totalSquares / (m.n - 1)
  return {
    name: 'pca-power',
    config: { k, tol, maxIter },
    schema: pcaSchema,
    init: () => emptyState(mean, total),
    step: (s, ctx) => {
      let v = Float64Array.from(normal(ctx.key, [m.d]).data)
      orthogonalise(v, s.components)
      let norm = Math.hypot(...v) || 1
      v = v.map((e) => e / norm)
      let lambda = 0
      for (let it = 0; it < maxIter; it++) {
        const w = normalMatvec(xc, m.n, m.d, v)
        orthogonalise(w, s.components)
        norm = Math.hypot(...w)
        lambda = norm
        if (norm < 1e-300) break
        let change = 0
        const next = w.map((e) => e / norm)
        for (let j = 0; j < m.d; j++) change += ((next[j] ?? 0) - (v[j] ?? 0)) ** 2
        v = next
        if (Math.sqrt(change) < tol) break
      }
      const row = [Array.from(v)]
      flipSigns(undefined, row)
      const found = s.components.length + 1
      const singularValues = [...s.singularValues, Math.sqrt(Math.max(lambda, 0))]
      const kept = singularValues.reduce((a, sv) => a + (sv * sv) / (m.n - 1), 0)
      return stateFrom(
        mean,
        singularValues,
        [...s.components, row[0] as number[]],
        found,
        m.n,
        total,
        Math.max(0, total - kept) / Math.max(1, full - found),
      )
    },
    done: (s) => (s.components.length >= k ? 'completed' : undefined),
    observe: (s) => observePca(s, m.d),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PcaState,
  }
}

/** Hyperparameters of {@link pca}. */
export interface PcaHyper {
  /** Components kept (`null`: all, min(n, d)). */
  readonly nComponents: number | null
  /** Scale scores to unit variance. */
  readonly whiten: boolean
  /** An exact SVD, the randomised SVD of `numerics/linalg`, or power iteration (one component per step). */
  readonly solver: 'full' | 'randomized' | 'power'
}

const kOf = (h: { readonly nComponents: number | null }, data: LearnData, who: string): number => {
  const m = matrixView(data.x, who, { minSamples: 2 })
  return h.nComponents ?? Math.min(m.n, m.d)
}

const paramsOf = (s: PcaState, whiten: boolean): PcaParams => ({
  mean: s.mean,
  components: s.components,
  explainedVariance: s.explainedVariance,
  explainedVarianceRatio: s.explainedVarianceRatio,
  singularValues: s.singularValues,
  noiseVariance: s.noiseVariance,
  whiten,
})

/**
 * Principal component analysis (sklearn.decomposition.PCA) on the iterative base. The `'full'` and `'randomized'`
 * solvers find every component in one step; `'power'` finds one per step, so a session shows the axes appearing in
 * order of variance.
 *
 * @index estimator PCA, exact or by power iteration, with whitening and inverse transform.
 * @example
 * ```ts
 * import { pca } from 'mdml-engine/decomposition'
 * const fit = pca.with({ nComponents: 1 }).fit({ x: [[-2, 0], [-1, 0.1], [1, -0.1], [2, 0]] })
 * fit.params.explainedVarianceRatio[0] // => 0.998404
 * ```
 */
export const pca: IterativeEstimator<PcaHyper, PcaState, PcaParams, ProjectionVerbs> = iterative<
  PcaHyper,
  PcaState,
  PcaParams,
  ProjectionVerbs
>({
  name: 'decomposition/pca',
  kind: 'transformer',
  defaults: { nComponents: null, whiten: false, solver: 'full' },
  hyperSpace: (h) =>
    dict({
      nComponents: int(1, 50, { default: h.nComponents ?? 2, label: 'components' }),
      whiten: bool({ default: h.whiten, label: 'whiten' }),
      solver: choice(['full', 'randomized', 'power'] as const, { default: h.solver, label: 'solver' }),
    }),
  algorithm: (data, h) => {
    const k = kOf(h, data, 'pca')
    return h.solver === 'power' ? powerPcaAlgorithm(data.x, k) : pcaSvdAlgorithm(data.x, k, { solver: h.solver })
  },
  maxSteps: (h, data) => (h.solver === 'power' ? kOf(h, data, 'pca') : 1),
  params: (s, _data, h) => paramsOf(s, h.whiten),
  verbs: (p) => pcaVerbs(p),
})

// ---------------------------------------------------------------------------------------------------------------
// Incremental PCA

/** The state of incremental PCA after some batches. */
export interface IncrementalPcaState extends PcaState {
  readonly seen: number
  readonly batch: number
  readonly variance: number[]
}

/**
 * The batches of sklearn's `gen_batches`: consecutive slices of `size`, a short last one merged into its predecessor
 * when it has fewer than `min` rows.
 *
 * @param n - Rows.
 * @param size - Batch size.
 * @param min - The smallest batch.
 * @returns [start, end) per batch.
 * @example
 * ```ts
 * import { batchRanges } from 'mdml-engine/decomposition'
 * batchRanges(10, 4) // => [[0, 4], [4, 8], [8, 10]]
 * batchRanges(10, 4, 3) // => [[0, 4], [4, 10]]
 * ```
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
 * Incremental PCA (Ross et al. 2008; sklearn.decomposition.IncrementalPCA) as a traced {@link Algorithm}: each step
 * folds one batch into the running mean, variance and SVD (the previous axes scaled by their singular values, the
 * centred batch and a mean correction row).
 *
 * @param x - Points `[n, d]`.
 * @param k - Components kept.
 * @param batches - The batches, as [start, end) row ranges.
 * @returns The algorithm; one step per batch.
 * @example
 * ```ts
 * import { incrementalPcaAlgorithm } from 'mdml-engine/decomposition'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[-2, 0], [-1, 0.1], [1, -0.1], [2, 0]]
 * runAlgorithm(incrementalPcaAlgorithm(x, 1, [[0, 2], [2, 4]]), { maxSteps: 2 }).state.seen // => 4
 * ```
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
    schema: channels({
      metrics: {
        seen: { description: 'the rows folded in so far' },
        explained: { description: 'the share of the variance explained by the components' },
      },
      tensors: { components: { role: 'state', description: 'the principal axes so far', shape: ['k', 'd'] } },
    }),
    init: () => ({
      ...emptyState(new Array<number>(d).fill(0), 1),
      seen: 0,
      batch: 0,
      variance: new Array<number>(d).fill(0),
    }),
    step: (s) => {
      const [a, b] = batches[s.batch] as [number, number]
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
        mean,
        components: vt.slice(0, k),
        singularValues: sv.slice(0, k),
        explainedVariance: sv.slice(0, k).map((v) => (v * v) / (total - 1)),
        explainedVarianceRatio: sv.slice(0, k).map((v) => (v * v) / totalVar),
        noiseVariance:
          sv.length > k ? sv.slice(k).reduce((acc, v) => acc + (v * v) / (total - 1), 0) / (sv.length - k) : 0,
        totalVariance: totalVar / (total - 1),
        seen: total,
        batch: s.batch + 1,
        variance,
      }
    },
    done: (s) => (s.batch >= batches.length ? 'completed' : undefined),
    observe: (s) => ({
      metrics: { seen: s.seen, explained: s.explainedVarianceRatio.reduce((acc, v) => acc + v, 0) },
      tensors: { components: { role: 'state', value: s.components.flat(), shape: [s.components.length, d] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as IncrementalPcaState,
  }
}

/** Hyperparameters of {@link incrementalPca}. */
export interface IncrementalPcaHyper {
  readonly nComponents: number
  /** Rows per batch (`null`: 5 times the features). */
  readonly batchSize: number | null
  readonly whiten: boolean
}

/**
 * Incremental PCA (sklearn.decomposition.IncrementalPCA) on the iterative base: PCA over batches in constant memory,
 * one batch folded in per step.
 *
 * @example
 * ```ts
 * import { incrementalPca } from 'mdml-engine/decomposition'
 * const x = [[-2, 0], [-1, 0.1], [1, -0.1], [2, 0], [3, 0.2], [-3, -0.2]]
 * incrementalPca.with({ nComponents: 1, batchSize: 3 }).fit({ x }).params.components.length // => 1
 * ```
 */
export const incrementalPca: IterativeEstimator<IncrementalPcaHyper, IncrementalPcaState, PcaParams, ProjectionVerbs> =
  iterative<IncrementalPcaHyper, IncrementalPcaState, PcaParams, ProjectionVerbs>({
    name: 'decomposition/incremental-pca',
    kind: 'transformer',
    defaults: { nComponents: 2, batchSize: null, whiten: false },
    hyperSpace: (h) =>
      dict({
        nComponents: int(1, 50, { default: h.nComponents, label: 'components' }),
        batchSize: int(2, 1000, { default: h.batchSize ?? 20, label: 'batch size' }),
      }),
    algorithm: (data, h) => {
      const m = matrixView(data.x, 'incrementalPca', { minSamples: 2 })
      const batches = batchRanges(m.n, h.batchSize ?? 5 * m.d, h.nComponents)
      if (batches.some(([a, b]) => b - a < h.nComponents))
        throw new MdmlError('value', 'incrementalPca: every batch needs at least nComponents rows')
      return incrementalPcaAlgorithm(data.x, h.nComponents, batches)
    },
    maxSteps: (h, data) => {
      const m = matrixView(data.x, 'incrementalPca', { minSamples: 2 })
      return batchRanges(m.n, h.batchSize ?? 5 * m.d, h.nComponents).length
    },
    params: (s, _data, h) => paramsOf(s, h.whiten),
    verbs: (p) => pcaVerbs(p, 'incrementalpca'),
  })

// ---------------------------------------------------------------------------------------------------------------
// Sparse PCA

/**
 * Sparse PCA by the penalised rank-one decomposition (Shen and Huang 2008) as a traced {@link Algorithm}, one component
 * per step: alternate u = R v / |R v| and v = soft-threshold(R^T u) (renormalised) on the data deflated by the
 * components found, where the threshold is the fraction `alpha` of the largest loading. With `alpha = 0` this is the
 * power method for the principal axis; larger values zero more loadings.
 *
 * @param x - Points `[n, d]`.
 * @param k - Components to find.
 * @param options - Sparsity `alpha` in `[0, 1)` (default 0.3), tolerance (1e-9) and iteration cap per component (500).
 * @returns The algorithm; k steps.
 * @example
 * ```ts
 * import { sparsePcaAlgorithm } from 'mdml-engine/decomposition'
 * import { key } from 'mdml-engine/random'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[-2, -2, 0.1], [-1, -1, -0.1], [1, 1, 0.1], [2, 2, -0.1]]
 * runAlgorithm(sparsePcaAlgorithm(x, 1, { alpha: 0.6 }), { maxSteps: 1, key: key(0) }).state.components[0]?.map((v) => Math.round(v * 100) / 100) // => [0.71, 0.71, 0]
 * ```
 */
export function sparsePcaAlgorithm(
  x: TensorLike,
  k: number,
  options: { readonly alpha?: number; readonly tol?: number; readonly maxIter?: number } = {},
): Algorithm<PcaState> {
  const m = matrixView(x, 'sparsePca', { minSamples: 2 })
  const alpha = options.alpha ?? 0.3
  if (!(alpha >= 0 && alpha < 1)) throw new MdmlError('value', `sparsePca: alpha must be in [0, 1), got ${alpha}`)
  const tol = options.tol ?? 1e-9
  const maxIter = options.maxIter ?? 500
  const mean = columnMeans(m)
  const xc = centred(m, mean)
  let totalSquares = 0
  for (const v of xc) totalSquares += v * v
  const total = totalSquares / (m.n - 1)
  return {
    name: 'sparse-pca',
    config: { k, alpha },
    schema: pcaSchema,
    init: () => emptyState(mean, total),
    step: (s, ctx) => {
      // The data deflated by the sparse components found: R = X Π (I - v v^T).
      const r = Float64Array.from(xc)
      for (const v of s.components)
        for (let i = 0; i < m.n; i++) {
          let t = 0
          for (let j = 0; j < m.d; j++) t += (r[i * m.d + j] ?? 0) * (v[j] ?? 0)
          for (let j = 0; j < m.d; j++) r[i * m.d + j] = (r[i * m.d + j] ?? 0) - t * (v[j] ?? 0)
        }
      let v = Float64Array.from(normal(ctx.key, [m.d]).data)
      v = v.map((e) => e / (Math.hypot(...v) || 1))
      for (let it = 0; it < maxIter; it++) {
        const z = normalMatvec(r, m.n, m.d, v)
        const top = maxOf(Array.from(z, Math.abs))
        const cut = alpha * top
        const soft = z.map((e) => Math.sign(e) * Math.max(Math.abs(e) - cut, 0))
        const norm = Math.hypot(...soft)
        if (norm < 1e-300) break
        const next = soft.map((e) => e / norm)
        let change = 0
        for (let j = 0; j < m.d; j++) change += ((next[j] ?? 0) - (v[j] ?? 0)) ** 2
        v = next
        if (Math.sqrt(change) < tol) break
      }
      const row = [Array.from(v)]
      flipSigns(undefined, row)
      const w = row[0] as number[]
      let sq = 0
      for (let i = 0; i < m.n; i++) {
        let t = 0
        for (let j = 0; j < m.d; j++) t += (r[i * m.d + j] ?? 0) * (w[j] ?? 0)
        sq += t * t
      }
      const found = s.components.length + 1
      const singularValues = [...s.singularValues, Math.sqrt(sq)]
      const ev = singularValues.map((sv) => (sv * sv) / (m.n - 1))
      return {
        mean,
        components: [...s.components, w],
        singularValues,
        explainedVariance: ev,
        explainedVarianceRatio: ev.map((e) => e / total),
        noiseVariance: Math.max(0, total - ev.reduce((a, e) => a + e, 0)) / Math.max(1, Math.min(m.n, m.d) - found),
        totalVariance: total,
      }
    },
    done: (s) => (s.components.length >= k ? 'completed' : undefined),
    observe: (s) => observePca(s, m.d),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as PcaState,
  }
}

/** Hyperparameters of {@link sparsePca}. */
export interface SparsePcaHyper {
  readonly nComponents: number
  /** The sparsity in `[0, 1)`: the fraction of the largest loading below which loadings are zeroed. */
  readonly alpha: number
  /** The ridge of the least-squares projection of {@link SparsePcaVerbs.transform} (scikit-learn's `ridge_alpha`). */
  readonly ridgeAlpha: number
}

/** The verbs of a fitted sparse PCA: ridge-regularised least-squares scores (as sklearn) and reconstructions. */
export interface SparsePcaVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly inverseTransform: (z: TensorLike) => Tensor
  readonly nFeaturesIn: number
  readonly featureNamesOut: (input?: readonly string[]) => string[]
}

/**
 * Sparse PCA on the iterative base: one sparse component per step. Scores are the ridge least-squares coefficients
 * of the centred rows on the (non-orthogonal) components, as scikit-learn's `SparsePCA.transform`.
 *
 * @example
 * ```ts
 * import { sparsePca } from 'mdml-engine/decomposition'
 * const x = [[-2, -2, 0.1], [-1, -1, -0.1], [1, 1, 0.1], [2, 2, -0.1]]
 * sparsePca.with({ nComponents: 1, alpha: 0.6 }).fit({ x }).params.components.length // => 1
 * ```
 */
export const sparsePca: IterativeEstimator<SparsePcaHyper, PcaState, PcaParams, SparsePcaVerbs> = iterative<
  SparsePcaHyper,
  PcaState,
  PcaParams,
  SparsePcaVerbs
>({
  name: 'decomposition/sparse-pca',
  kind: 'transformer',
  defaults: { nComponents: 2, alpha: 0.3, ridgeAlpha: 0.01 },
  hyperSpace: (h) =>
    dict({
      nComponents: int(1, 50, { default: h.nComponents, label: 'components' }),
      alpha: real(0, 0.95, { default: h.alpha, label: 'sparsity' }),
    }),
  algorithm: (data, h) => sparsePcaAlgorithm(data.x, h.nComponents, { alpha: h.alpha }),
  maxSteps: (h) => h.nComponents,
  params: (s) => paramsOf(s, false),
  verbs: (p, h) => {
    const c = p.components
    const d = p.mean.length
    const k = c.length
    // Scores: (C C^T + ridge I)^-1 C (x - mean)^T.
    const gram = mm(c, tr(c)).map((row, i) => row.map((v, j) => v + (i === j ? h.ridgeAlpha : 0)))
    const solve = k === 0 ? [] : (toFlat(linalg.inv(gram)) as number[])
    const gramInv = Array.from({ length: k }, (_, i) => solve.slice(i * k, (i + 1) * k))
    return {
      transform: (x) => {
        const raw = project(x, p.mean, c)
        const n = raw.shape[0] ?? 0
        const rows = Array.from({ length: n }, (_, i) => toFlat(raw).slice(i * k, (i + 1) * k))
        return fromStorage(Float64Array.from(mm(rows, gramInv).flat()), [n, k], 'float64')
      },
      inverseTransform: (z) => reconstruct(z, p.mean, c),
      nFeaturesIn: d,
      featureNamesOut: () => c.map((_, i) => `sparsepca${i}`),
    }
  },
})
