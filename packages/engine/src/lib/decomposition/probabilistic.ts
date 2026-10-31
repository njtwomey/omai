/**
 * Linear-Gaussian latent variable models: probabilistic PCA (Tipping and Bishop 1999) in closed form from PCA, and
 * factor analysis (as in scikit-learn), where each feature has its own noise variance, fitted by an SVD-based
 * maximum-likelihood iteration (one per step) and optionally rotated by varimax or quartimax. Both give posterior factor
 * means and log likelihoods.
 */

import { MdmlError } from '../base/index.ts'
import { type IterativeEstimator, iterative, type LearnData, matrixView } from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { choice, dict, int } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { centred, columnMeans, thinSvd } from './_linear.ts'
import { type PcaState, pcaSvdAlgorithm } from './pca.ts'

/** A linear-Gaussian latent model x = W^T z + mean + noise, z ~ N(0, I), noise ~ N(0, diag(psi)). */
export type LatentGaussianParams = {
  readonly mean: number[]
  /** components[k][f]: the loadings W. */
  readonly components: number[][]
  /** Per-feature noise variances psi. */
  readonly noiseVariance: number[]
}

/** The verbs of a latent-Gaussian model: posterior means of the factors and the mean log likelihood. */
export interface LatentVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
  readonly scoreSamples: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/**
 * The model covariance W^T W + diag(psi).
 *
 * @param p - The parameters.
 * @returns The covariance `[d, d]`.
 * @example
 * ```ts
 * import { latentCovariance } from 'mdml-engine/decomposition'
 * latentCovariance({ mean: [0, 0], components: [[1, 1]], noiseVariance: [0.5, 0.5] }) // => [[1.5, 1], [1, 1.5]]
 * ```
 */
export function latentCovariance(p: LatentGaussianParams): number[][] {
  const d = p.mean.length
  return Array.from({ length: d }, (_, a) =>
    Array.from({ length: d }, (_, b) => {
      let v = a === b ? (p.noiseVariance[a] ?? 0) : 0
      for (const w of p.components) v += (w[a] ?? 0) * (w[b] ?? 0)
      return v
    }),
  )
}

/**
 * The verbs of a latent-Gaussian model: E[z | x] = (I + W Psi^-1 W^T)^-1 W Psi^-1 (x - mean) and Gaussian log
 * densities under the model covariance.
 *
 * @param p - The parameters.
 * @returns The verbs.
 * @example
 * ```ts
 * import { latentVerbs } from 'mdml-engine/decomposition'
 * latentVerbs({ mean: [0], components: [[1]], noiseVariance: [1] }).nFeaturesIn // => 1
 * ```
 */
export function latentVerbs(p: LatentGaussianParams): LatentVerbs {
  const d = p.mean.length
  const k = p.components.length
  const wpsi = p.components.map((w) => w.map((v, j) => v / (p.noiseVariance[j] ?? 1)))
  const inner = Array.from({ length: k }, (_, a) =>
    Array.from({ length: k }, (_, b) => {
      let v = a === b ? 1 : 0
      for (let j = 0; j < d; j++) v += (wpsi[a]?.[j] ?? 0) * (p.components[b]?.[j] ?? 0)
      return v
    }),
  )
  const covZ = k === 0 ? [] : toFlat(linalg.inv(inner))
  const cov = latentCovariance(p)
  const precision = toFlat(linalg.inv(cov))
  const logdet = toFlat(linalg.slogdet(cov).logabsdet)[0] ?? 0
  const logpdf = (x: TensorLike): number[] => {
    const m = matrixView(x, 'latent score', { nFeatures: d })
    return Array.from({ length: m.n }, (_, i) => {
      let q = 0
      for (let a = 0; a < d; a++)
        for (let b = 0; b < d; b++)
          q +=
            ((m.data[i * d + a] ?? 0) - (p.mean[a] ?? 0)) *
            (precision[a * d + b] ?? 0) *
            ((m.data[i * d + b] ?? 0) - (p.mean[b] ?? 0))
      return -0.5 * (q + d * Math.log(2 * Math.PI) + logdet)
    })
  }
  return {
    transform: (x) => {
      const m = matrixView(x, 'transform', { nFeatures: d })
      const out = new Float64Array(m.n * k)
      for (let i = 0; i < m.n; i++) {
        const proj = wpsi.map((w) => w.reduce((s, v, j) => s + v * ((m.data[i * d + j] ?? 0) - (p.mean[j] ?? 0)), 0))
        for (let a = 0; a < k; a++) {
          let s = 0
          for (let b = 0; b < k; b++) s += (proj[b] ?? 0) * (covZ[b * k + a] ?? 0)
          out[i * k + a] = s
        }
      }
      return fromStorage(out, [m.n, k], 'float64')
    },
    scoreSamples: (x) => {
      const v = logpdf(x)
      return fromStorage(Float64Array.from(v), [v.length], 'float64')
    },
    score: (data) => {
      const v = logpdf(data.x)
      return v.reduce((a, b) => a + b, 0) / v.length
    },
    nFeaturesIn: d,
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Probabilistic PCA

/** Hyperparameters of {@link probabilisticPca}. */
export interface ProbabilisticPcaHyper {
  /** Components (`null`: min(n, d) - 1). */
  readonly nComponents: number | null
}

/**
 * Probabilistic PCA (Tipping and Bishop 1999) on the iterative base, in closed form from the SVD: loadings
 * sqrt(ev - sigma^2) along the principal axes and isotropic noise sigma^2, the mean of the discarded variances;
 * transform gives the posterior factor means.
 *
 * @example
 * ```ts
 * import { probabilisticPca } from 'mdml-engine/decomposition'
 * const x = [[-2, 0.1, 0], [-1, -0.1, 0.1], [1, 0.1, -0.1], [2, -0.1, 0]]
 * probabilisticPca.with({ nComponents: 1 }).fit({ x }).params.components.length // => 1
 * ```
 */
export const probabilisticPca: IterativeEstimator<ProbabilisticPcaHyper, PcaState, LatentGaussianParams, LatentVerbs> =
  iterative<ProbabilisticPcaHyper, PcaState, LatentGaussianParams, LatentVerbs>({
    name: 'decomposition/probabilistic-pca',
    kind: 'transformer',
    defaults: { nComponents: null },
    hyperSpace: (h) => dict({ nComponents: int(1, 50, { default: h.nComponents ?? 2, label: 'components' }) }),
    algorithm: (data, h) => {
      const m = matrixView(data.x, 'probabilisticPca', { minSamples: 2 })
      return pcaSvdAlgorithm(data.x, h.nComponents ?? Math.max(1, Math.min(m.n, m.d) - 1))
    },
    maxSteps: () => 1,
    params: (s) => {
      const s2 = s.noiseVariance
      return {
        mean: s.mean,
        components: s.components.map((w, c) =>
          w.map((v) => v * Math.sqrt(Math.max(0, (s.explainedVariance[c] ?? 0) - s2))),
        ),
        noiseVariance: s.mean.map(() => s2),
      }
    },
    verbs: latentVerbs,
  })

// ---------------------------------------------------------------------------------------------------------------
// Rotations

/**
 * An orthogonal rotation of loadings (sklearn's `_ortho_rotation`): varimax (Kaiser 1958) maximises the variance of
 * the squared loadings per factor, quartimax per feature.
 *
 * @param loadings - Loadings `[d, k]` (features by factors).
 * @param method - 'varimax' or 'quartimax'.
 * @param options - Tolerance (1e-6) and iteration cap (100).
 * @returns The rotated loadings `[d, k]`.
 * @example
 * ```ts
 * import { rotateLoadings } from 'mdml-engine/decomposition'
 * rotateLoadings([[1, 0], [0, 1]], 'varimax').length // => 2
 * ```
 */
export function rotateLoadings(
  loadings: readonly (readonly number[])[],
  method: 'varimax' | 'quartimax' = 'varimax',
  options: { readonly tol?: number; readonly maxIter?: number } = {},
): number[][] {
  const d = loadings.length
  const k = loadings[0]?.length ?? 0
  const a = loadings.map((r) => [...r])
  let r: number[][] = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => (i === j ? 1 : 0)))
  const mul = (x: number[][], y: number[][]): number[][] =>
    x.map((row) => (y[0] ?? []).map((_, j) => row.reduce((s, v, l) => s + v * (y[l]?.[j] ?? 0), 0)))
  let variance = 0
  for (let it = 0; it < (options.maxIter ?? 100); it++) {
    const rot = mul(a, r)
    const colSq = Array.from({ length: k }, (_, j) => rot.reduce((s, row) => s + (row[j] ?? 0) ** 2, 0) / d)
    const target = rot.map((row) => row.map((v, j) => v ** 3 - (method === 'varimax' ? v * (colSq[j] ?? 0) : 0)))
    const g = Array.from({ length: k }, (_, i) =>
      Array.from({ length: k }, (_, j) => a.reduce((s, row, l) => s + (row[i] ?? 0) * (target[l]?.[j] ?? 0), 0)),
    )
    const svd = linalg.svd(g)
    const u = toFlat(svd.u)
    const vh = toFlat(svd.vh)
    r = Array.from({ length: k }, (_, i) =>
      Array.from({ length: k }, (_, j) => {
        let s = 0
        for (let l = 0; l < k; l++) s += (u[i * k + l] ?? 0) * (vh[l * k + j] ?? 0)
        return s
      }),
    )
    const next = toFlat(svd.s).reduce((s, v) => s + v, 0)
    if (variance !== 0 && next < variance * (1 + (options.tol ?? 1e-6))) break
    variance = next
  }
  return mul(a, r)
}

// ---------------------------------------------------------------------------------------------------------------
// Factor analysis

/** The state of factor analysis's iterations. */
export interface FactorAnalysisState {
  readonly psi: number[]
  readonly components: number[][]
  /** The log likelihood of the last step (a very negative number before the first). */
  readonly logLikelihood: number
  /** The change of the log likelihood (a large number before the first step). */
  readonly change: number
}

/**
 * Factor analysis by sklearn's SVD-based maximum likelihood iteration (Barber 2012, 21.2.2) as a traced {@link Algorithm}:
 * each step takes the SVD of the noise-scaled data, sets the loadings from its leading singular values and updates the
 * noise variances, observing the log likelihood and the loadings.
 *
 * @param x - Points `[n, d]`.
 * @param k - The number of factors.
 * @param tol - The log-likelihood change that stops the iteration (default 1e-2, as sklearn).
 * @returns The algorithm.
 * @example
 * ```ts
 * import { factorAnalysisAlgorithm } from 'mdml-engine/decomposition'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[-2, -1.8, 0.3], [-1, -1.1, -0.2], [1, 0.9, 0.1], [2, 2.2, -0.3], [0, 0.1, 0.2]]
 * runAlgorithm(factorAnalysisAlgorithm(x, 1), { maxSteps: 50 }).state.components.length // => 1
 * ```
 */
export function factorAnalysisAlgorithm(x: TensorLike, k: number, tol = 1e-2): Algorithm<FactorAnalysisState> {
  const m = matrixView(x, 'factorAnalysis', { minSamples: 2 })
  const { n, d } = m
  if (!(Number.isInteger(k) && k >= 1 && k <= d))
    throw new MdmlError('value', `factorAnalysis: nComponents must be in 1..${d}`)
  const xc = centred(m, columnMeans(m))
  const variance = Array.from({ length: d }, (_, j) => {
    let s = 0
    for (let i = 0; i < n; i++) s += (xc[i * d + j] ?? 0) ** 2
    return s / n
  })
  const small = 1e-12
  const llconst = d * Math.log(2 * Math.PI) + k
  return {
    name: 'factor-analysis',
    config: { k, tol },
    schema: channels({
      metrics: { logLikelihood: { description: 'the log likelihood of the data', direction: 'higher' } },
      tensors: { components: { role: 'state', description: 'the factor loadings', shape: ['k', 'd'] } },
    }),
    init: () => ({
      psi: new Array<number>(d).fill(1),
      components: [],
      logLikelihood: -Number.MAX_VALUE,
      change: Number.MAX_VALUE,
    }),
    step: (s) => {
      const sqrtPsi = s.psi.map((v) => Math.sqrt(v) + small)
      const scaled = new Float64Array(n * d)
      for (let i = 0; i < n; i++)
        for (let j = 0; j < d; j++) scaled[i * d + j] = (xc[i * d + j] ?? 0) / ((sqrtPsi[j] ?? 1) * Math.sqrt(n))
      const { s: sv, vt } = thinSvd(scaled, n, d)
      const unexplained = sv.slice(k).reduce((a, v) => a + v * v, 0)
      const s2 = sv.slice(0, k).map((v) => v * v)
      const components = s2.map((v, c) =>
        (vt[c] ?? []).map((u, j) => Math.sqrt(Math.max(v - 1, 0)) * u * (sqrtPsi[j] ?? 1)),
      )
      const ll =
        (-n / 2) *
        (llconst + s2.reduce((a, v) => a + Math.log(v), 0) + unexplained + s.psi.reduce((a, v) => a + Math.log(v), 0))
      // As sklearn: on convergence the noise variances are not updated after the last loadings.
      const change = s.logLikelihood === -Number.MAX_VALUE ? Number.MAX_VALUE : ll - s.logLikelihood
      const psi =
        change < tol
          ? s.psi
          : variance.map((v, j) => Math.max(v - components.reduce((a, w) => a + (w[j] ?? 0) ** 2, 0), small))
      return { psi, components, logLikelihood: ll, change }
    },
    done: (s) => (s.change < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: s.components.length === 0 ? {} : { logLikelihood: s.logLikelihood },
      tensors: { components: { role: 'state', value: s.components.flat(), shape: [s.components.length, d] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as FactorAnalysisState,
  }
}

/** Hyperparameters of {@link factorAnalysis}. */
export interface FactorAnalysisHyper {
  readonly nComponents: number
  readonly tol: number
  readonly maxIter: number
  /** Rotate the loadings for interpretability: none, varimax or quartimax. */
  readonly rotation: 'none' | 'varimax' | 'quartimax'
}

/**
 * Factor analysis (sklearn.decomposition.FactorAnalysis) on the iterative base: a linear-Gaussian latent model with
 * per-feature noise, one maximum-likelihood iteration per step, optionally rotated by varimax or quartimax.
 *
 * @example
 * ```ts
 * import { factorAnalysis } from 'mdml-engine/decomposition'
 * const x = [[-2, -1.8, 0.3], [-1, -1.1, -0.2], [1, 0.9, 0.1], [2, 2.2, -0.3], [0, 0.1, 0.2]]
 * factorAnalysis.with({ nComponents: 1 }).fit({ x }).params.components.length // => 1
 * ```
 */
export const factorAnalysis: IterativeEstimator<
  FactorAnalysisHyper,
  FactorAnalysisState,
  LatentGaussianParams,
  LatentVerbs
> = iterative<FactorAnalysisHyper, FactorAnalysisState, LatentGaussianParams, LatentVerbs>({
  name: 'decomposition/factor-analysis',
  kind: 'transformer',
  defaults: { nComponents: 2, tol: 1e-2, maxIter: 1000, rotation: 'none' },
  hyperSpace: (h) =>
    dict({
      nComponents: int(1, 50, { default: h.nComponents, label: 'factors' }),
      rotation: choice(['none', 'varimax', 'quartimax'] as const, { default: h.rotation, label: 'rotation' }),
    }),
  algorithm: (data, h) => factorAnalysisAlgorithm(data.x, h.nComponents, h.tol),
  maxSteps: (h) => h.maxIter,
  params: (s, data, h) => {
    const m = matrixView(data.x, 'factorAnalysis', { minSamples: 2 })
    let components = s.components
    if (h.rotation !== 'none' && components.length > 0) {
      const t = components[0]?.map((_, j) => components.map((w) => w[j] ?? 0)) ?? []
      const rotated = rotateLoadings(t, h.rotation)
      components = Array.from({ length: components.length }, (_, c) => rotated.map((row) => row[c] ?? 0))
    }
    return { mean: columnMeans(m), components, noiseVariance: s.psi }
  },
  verbs: latentVerbs,
})
