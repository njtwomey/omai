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
import { choice, dict, int } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'
import { centred, columnMeans, rowsOf, thinSvd } from './_linear.ts'
import { pca } from './pca.ts'

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

const matrix = (rows: readonly (readonly number[])[]): number[][] => rows.map((r) => [...r])

/**
 * The model covariance W^T W + diag(psi).
 *
 * @param p - The parameters.
 * @returns The covariance `[d][d]`.
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
  const covZ = toFlat(linalg.inv(inner))
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

/** Hyperparameters of probabilistic PCA. */
export type ProbabilisticPcaHyper = { readonly nComponents?: number }

/** A fitted latent-Gaussian model. */
export type FittedLatentGaussian = ExplainedFitted<LatentGaussianParams, LatentVerbs>

/**
 * Probabilistic PCA (Tipping and Bishop 1999) in closed form from PCA: loadings sqrt(ev - sigma^2) along the principal
 * axes and isotropic noise sigma^2, the mean of the discarded variances; transform gives the posterior factor means.
 * Its exposition is a `components` record. TODO: the EM route through the probprog EM engine.
 *
 * @param hyper - The number of components.
 * @returns The estimator.
 */
export function probabilisticPca(
  hyper: ProbabilisticPcaHyper = {},
): Estimator<ProbabilisticPcaHyper, FittedLatentGaussian> {
  return defineExplainedEstimator<ProbabilisticPcaHyper, LatentGaussianParams, LatentVerbs>(
    {
      name: 'probabilistic-pca',
      kind: 'transformer',
      hyperSpace: (h) => dict({ nComponents: int(1, 50, { default: h.nComponents ?? 2, label: 'components' }) }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'probabilisticPca', { minSamples: 2 })
        const k = h.nComponents ?? Math.max(1, Math.min(m.n, m.d) - 1)
        const p = pca({ nComponents: k }).fit(data).params
        const s2 = p.noiseVariance
        const components = p.components.map((w, c) =>
          w.map((v) => v * Math.sqrt(Math.max(0, (p.explainedVariance[c] ?? 0) - s2))),
        )
        const params: LatentGaussianParams = { mean: p.mean, components, noiseVariance: p.mean.map(() => s2) }
        return {
          params,
          exposition: [
            componentsRecord({
              method: 'probabilisticPca',
              features: featureNamesOf(data, m.d),
              components,
              explainedVariance: p.explainedVariance,
              explainedVarianceRatio: p.explainedVarianceRatio,
              scores: rowsOf(latentVerbs(params).transform(data.x)),
            }),
          ],
        }
      },
      methods: latentVerbs,
    },
    hyper,
  )
}

/** Probabilistic PCA as a class. */
export class ProbabilisticPCA extends EstimatorClass<ProbabilisticPcaHyper, FittedLatentGaussian> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link probabilisticPca}.
   */
  constructor(hyper: ProbabilisticPcaHyper = {}) {
    super(probabilisticPca(hyper))
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Rotations

/**
 * An orthogonal rotation of loadings (sklearn's `_ortho_rotation`): varimax (Kaiser 1958) maximises the variance of
 * the squared loadings per factor, quartimax per feature.
 *
 * @param loadings - Loadings `[d][k]` (features by factors).
 * @param method - 'varimax' or 'quartimax'.
 * @param options - Tolerance (1e-6) and iteration cap (100).
 * @returns The rotated loadings `[d][k]`.
 */
export function rotateLoadings(
  loadings: readonly (readonly number[])[],
  method: 'varimax' | 'quartimax' = 'varimax',
  options: { readonly tol?: number; readonly maxIter?: number } = {},
): number[][] {
  const d = loadings.length
  const k = loadings[0]?.length ?? 0
  const a = matrix(loadings)
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
  readonly logLikelihood: number
  readonly change: number
}

/**
 * Factor analysis by sklearn's SVD-based maximum likelihood iteration (Barber 2012, 21.2.2) as a traced algorithm:
 * each step takes the SVD of the noise-scaled data, sets the loadings from its leading singular values and updates the
 * noise variances, recording the log likelihood.
 *
 * @param x - Points `[n, d]`.
 * @param k - The number of factors.
 * @param tol - The log-likelihood change that stops the iteration (default 1e-2, as sklearn).
 * @returns The algorithm.
 */
export function factorAnalysisAlgorithm(x: TensorLike, k: number, tol = 1e-2): Algorithm<FactorAnalysisState> {
  const m = matrixView(x, 'factorAnalysis', { minSamples: 2 })
  const { n, d } = m
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
    init: () => ({
      psi: new Array<number>(d).fill(1),
      components: [],
      logLikelihood: Number.NEGATIVE_INFINITY,
      change: Number.POSITIVE_INFINITY,
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
      const change = ll - s.logLikelihood
      const psi =
        change < tol
          ? s.psi
          : variance.map((v, j) => Math.max(v - components.reduce((a, w) => a + (w[j] ?? 0) ** 2, 0), small))
      return { psi, components, logLikelihood: ll, change }
    },
    done: (s) => (s.change < tol ? 'converged' : undefined),
    observe: (s) => ({ metrics: Number.isFinite(s.logLikelihood) ? { logLikelihood: s.logLikelihood } : {} }),
  }
}

/** Hyperparameters of factor analysis. */
export type FactorAnalysisHyper = {
  readonly nComponents?: number
  readonly tol?: number
  readonly maxIter?: number
  /** Rotate the loadings (default none). */
  readonly rotation?: 'none' | 'varimax' | 'quartimax'
}

/**
 * Factor analysis (sklearn.decomposition.FactorAnalysis): a linear-Gaussian latent model with per-feature noise,
 * fitted by the traced SVD iteration, optionally rotated by varimax or quartimax. Its exposition is a `components`
 * record (loadings and posterior factor scores).
 *
 * @param hyper - Factors, tolerance, iteration cap and rotation.
 * @returns The estimator.
 */
export function factorAnalysis(hyper: FactorAnalysisHyper = {}): Estimator<FactorAnalysisHyper, FittedLatentGaussian> {
  return defineExplainedEstimator<FactorAnalysisHyper, LatentGaussianParams, LatentVerbs>(
    {
      name: 'factor-analysis',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 50, { default: h.nComponents ?? 2, label: 'factors' }),
          rotation: choice(['none', 'varimax', 'quartimax'], {
            label: 'rotation',
            description: 'Rotate the loadings for interpretability: none, varimax or quartimax.',
            default: h.rotation ?? 'none',
          }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'factorAnalysis', { minSamples: 2 })
        const k = h.nComponents ?? m.d
        if (!(Number.isInteger(k) && k >= 1 && k <= m.d))
          throw new WhiteboxError('value', `factorAnalysis: nComponents must be in 1..${m.d}`)
        const out = runAlgorithm(factorAnalysisAlgorithm(data.x, k, h.tol ?? 1e-2), {
          ...fitRunOptions(options, 'factor-analysis'),
          maxSteps: h.maxIter ?? 1000,
        })
        let components = out.state.components
        if (h.rotation !== undefined && h.rotation !== 'none') {
          const t = components[0]?.map((_, j) => components.map((w) => w[j] ?? 0)) ?? []
          const rotated = rotateLoadings(t, h.rotation)
          components = Array.from({ length: k }, (_, c) => rotated.map((row) => row[c] ?? 0))
        }
        const params: LatentGaussianParams = { mean: columnMeans(m), components, noiseVariance: out.state.psi }
        return {
          params,
          exposition: [
            componentsRecord({
              method: 'factorAnalysis',
              features: featureNamesOf(data, m.d),
              components,
              scores: rowsOf(latentVerbs(params).transform(data.x)),
            }),
          ],
        }
      },
      methods: latentVerbs,
    },
    hyper,
  )
}

/** Factor analysis as a class. */
export class FactorAnalysis extends EstimatorClass<FactorAnalysisHyper, FittedLatentGaussian> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link factorAnalysis}.
   */
  constructor(hyper: FactorAnalysisHyper = {}) {
    super(factorAnalysis(hyper))
  }
}
