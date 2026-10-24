import { WhiteboxError } from '../../lib/base/index.ts'
import { multivariateNormal } from '../../lib/dists/index.ts'
import { mixtureSteps } from '../../lib/exposition/index.ts'
import {
  decisionSurfaceOf,
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  type FitOptions,
  fitRunOptions,
  type LearnData,
  type MatrixView,
  matrixView,
} from '../../lib/learn/index.ts'
import { foldIn, type Key, uniform } from '../../lib/random/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { fromStorage, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'
import { kmeans } from '../cluster/index.ts'

/** How component covariances are shared and shaped (sklearn's `covariance_type`). */
export type CovarianceType = 'full' | 'tied' | 'diag' | 'spherical'

/** A Gaussian mixture's parameters: weights, means and full covariance matrices (whatever the type). */
export type GaussianMixtureParams = {
  readonly weights: number[]
  readonly means: number[][]
  /** covariances[k]: component k's covariance `[d, d]` (shared for tied, diagonal for diag and spherical). */
  readonly covariances: number[][][]
  readonly covarianceType: CovarianceType
  /** The mean log likelihood per row after the last step. */
  readonly logLikelihood: number
  readonly nIter: number
  readonly converged: boolean
}

/** The verbs of a fitted mixture: component labels, responsibilities and the mean log likelihood. */
export interface MixtureVerbs {
  readonly predict: (x: TensorLike) => Tensor
  readonly predictProba: (x: TensorLike) => Tensor
  readonly score: (data: LearnData) => number
  /** Per-row log densities `[m]`. */
  readonly scoreSamples: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/** A fitted Gaussian mixture. */
export type FittedGaussianMixture = ExplainedFitted<GaussianMixtureParams, MixtureVerbs>

/**
 * Per-row, per-component log joint densities log w_k + log N(x_i | mu_k, Sigma_k), `[n][k]`, through the
 * multivariate normal of `dists`.
 *
 * @param x - Points `[n, d]`.
 * @param weights - Mixing weights.
 * @param means - Component means.
 * @param covariances - Component covariances `[k][d][d]`.
 * @returns The log joint densities.
 */
export function componentLogDensities(
  x: TensorLike,
  weights: readonly number[],
  means: readonly (readonly number[])[],
  covariances: readonly (readonly (readonly number[])[])[],
): number[][] {
  const cols = means.map((mu, k) =>
    toFlat(multivariateNormal(mu as number[], { covariance: covariances[k] as number[][] }).logProb(x)).map(
      (v) => v + Math.log(weights[k] ?? 0),
    ),
  )
  const n = cols[0]?.length ?? 0
  return Array.from({ length: n }, (_, i) => cols.map((c) => c[i] ?? 0))
}

/** Normalise log joint densities into responsibilities and the per-row log normalisers. */
function normalise(logJoint: readonly (readonly number[])[]): { resp: number[][]; logNorm: number[] } {
  const logNorm = logJoint.map((row) => {
    const m = Math.max(...row)
    return m + Math.log(row.reduce((a, v) => a + Math.exp(v - m), 0))
  })
  return { resp: logJoint.map((row, i) => row.map((v) => Math.exp(v - (logNorm[i] ?? 0)))), logNorm }
}

/**
 * The M-step of a Gaussian mixture (sklearn's `_estimate_gaussian_parameters`): weights, means and covariances of the
 * given type from responsibilities, with `reg` added to every diagonal.
 *
 * @param m - Points.
 * @param resp - Responsibilities `[n][k]`.
 * @param type - The covariance type.
 * @param reg - Diagonal regularisation.
 * @returns Weights, means and full covariance matrices.
 */
export function gaussianMStep(
  m: MatrixView,
  resp: readonly (readonly number[])[],
  type: CovarianceType,
  reg: number,
): { weights: number[]; means: number[][]; covariances: number[][][] } {
  const k = resp[0]?.length ?? 0
  const { n, d, data } = m
  const nk = Array.from({ length: k }, (_, j) => resp.reduce((a, r) => a + (r[j] ?? 0), 0) + 10 * Number.EPSILON)
  const means = Array.from({ length: k }, (_, j) =>
    Array.from({ length: d }, (_, q) => {
      let s = 0
      for (let i = 0; i < n; i++) s += (resp[i]?.[j] ?? 0) * (data[i * d + q] ?? 0)
      return s / (nk[j] ?? 1)
    }),
  )
  const scatter = (j: number): number[][] => {
    const c = Array.from({ length: d }, () => new Array<number>(d).fill(0))
    const mu = means[j] as number[]
    for (let i = 0; i < n; i++) {
      const r = resp[i]?.[j] ?? 0
      if (r === 0) continue
      for (let a = 0; a < d; a++) {
        const da = (data[i * d + a] ?? 0) - (mu[a] ?? 0)
        const row = c[a] as number[]
        for (let b = 0; b < d; b++) row[b] = (row[b] ?? 0) + r * da * ((data[i * d + b] ?? 0) - (mu[b] ?? 0))
      }
    }
    return c
  }
  const withReg = (c: number[][]): number[][] => c.map((row, a) => row.map((v, b) => (a === b ? v + reg : v)))
  let covariances: number[][][]
  if (type === 'tied') {
    const total = Array.from({ length: d }, () => new Array<number>(d).fill(0))
    for (let j = 0; j < k; j++) {
      const s = scatter(j)
      for (let a = 0; a < d; a++)
        for (let b = 0; b < d; b++) (total[a] as number[])[b] = (total[a]?.[b] ?? 0) + (s[a]?.[b] ?? 0)
    }
    const sumNk = nk.reduce((a, v) => a + v, 0)
    const shared = withReg(total.map((row) => row.map((v) => v / sumNk)))
    covariances = means.map(() => shared)
  } else {
    covariances = means.map((_, j) => {
      const s = scatter(j).map((row) => row.map((v) => v / (nk[j] ?? 1)))
      if (type === 'full') return withReg(s)
      const diag = s.map((row, a) => (row[a] ?? 0) + reg)
      const values = type === 'diag' ? diag : diag.map(() => diag.reduce((a, v) => a + v, 0) / d)
      return values.map((v, a) => values.map((_, b) => (a === b ? v : 0)))
    })
  }
  return { weights: nk.map((v) => v / n), means, covariances }
}

/** The state of EM for a Gaussian mixture. */
export interface GaussianEmState {
  readonly weights: number[]
  readonly means: number[][]
  readonly covariances: number[][][]
  readonly logLikelihood: number
  readonly change: number
  readonly resp: number[][]
  readonly history: {
    readonly weights: number[][]
    readonly means: number[][][]
    readonly covariances: number[][][][]
    readonly objective: number[]
  }
}

/**
 * EM for a Gaussian mixture as a traced algorithm (sklearn's loop): each step computes responsibilities and the mean
 * log likelihood under the current parameters (E), then re-estimates the parameters (M). Converged when the mean log
 * likelihood changes by less than tol.
 *
 * @param x - Points `[n, d]`.
 * @param start - Initial weights, means and covariances.
 * @param options - Covariance type, regularisation (1e-6) and tolerance (1e-3).
 * @returns The algorithm.
 */
export function gaussianEmAlgorithm(
  x: TensorLike,
  start: { readonly weights: number[]; readonly means: number[][]; readonly covariances: number[][][] },
  options: { readonly covarianceType?: CovarianceType; readonly regCovar?: number; readonly tol?: number } = {},
): Algorithm<GaussianEmState> {
  const m = matrixView(x, 'gaussianEm')
  const type = options.covarianceType ?? 'full'
  const reg = options.regCovar ?? 1e-6
  const tol = options.tol ?? 1e-3
  return {
    name: 'gaussian-mixture-em',
    config: { covarianceType: type, regCovar: reg, tol },
    init: () => ({
      ...start,
      logLikelihood: Number.NEGATIVE_INFINITY,
      change: Number.POSITIVE_INFINITY,
      resp: [],
      history: { weights: [], means: [], covariances: [], objective: [] },
    }),
    step: (s) => {
      const { resp, logNorm } = normalise(componentLogDensities(x, s.weights, s.means, s.covariances))
      const ll = logNorm.reduce((a, v) => a + v, 0) / m.n
      const next = gaussianMStep(m, resp, type, reg)
      return {
        ...next,
        logLikelihood: ll,
        change: Math.abs(ll - s.logLikelihood),
        resp,
        history: {
          weights: [...s.history.weights, next.weights],
          means: [...s.history.means, next.means],
          covariances: [...s.history.covariances, next.covariances],
          objective: [...s.history.objective, ll],
        },
      }
    },
    done: (s) => (s.change < tol ? 'converged' : undefined),
    observe: (s) => ({ metrics: Number.isFinite(s.logLikelihood) ? { logLikelihood: s.logLikelihood } : {} }),
  }
}

/** Hyperparameters of a Gaussian mixture. */
export type GaussianMixtureHyper = {
  /** Components (default 2). */
  readonly nComponents?: number
  readonly covarianceType?: CovarianceType
  /** Mean log-likelihood change that stops EM (default 1e-3). */
  readonly tol?: number
  /** Added to every covariance diagonal (default 1e-6). */
  readonly regCovar?: number
  readonly maxIter?: number
  /** Initial responsibilities from k-means (default) or random; or given parameters. */
  readonly init?:
    | 'kmeans'
    | 'random'
    | {
        readonly weights: readonly number[]
        readonly means: readonly (readonly number[])[]
        readonly covariances: readonly (readonly (readonly number[])[])[]
      }
}

/**
 * Initial parameters: the M-step on one-hot k-means labels (sklearn's default) or on random responsibilities, or the
 * given parameters.
 *
 * @param x - Points `[n, d]`.
 * @param hyper - Components, covariance type, regularisation and the initialisation.
 * @param key - Randomness.
 * @returns Initial weights, means and covariances.
 */
export function initialMixture(
  x: TensorLike,
  hyper: GaussianMixtureHyper,
  key: Key,
): { weights: number[]; means: number[][]; covariances: number[][][] } {
  const m = matrixView(x, 'gaussianMixture')
  const k = hyper.nComponents ?? 2
  if (!(Number.isInteger(k) && k >= 1 && k <= m.n))
    throw new WhiteboxError('value', `gaussianMixture: nComponents must be in 1..${m.n}`)
  const init = hyper.init ?? 'kmeans'
  if (typeof init !== 'string')
    return {
      weights: [...init.weights],
      means: init.means.map((r) => [...r]),
      covariances: init.covariances.map((c) => c.map((r) => [...r])),
    }
  let resp: number[][]
  if (init === 'kmeans') {
    const labels = kmeans({ nClusters: k }).fit({ x }, { key }).params.labels
    resp = labels.map((l) => Array.from({ length: k }, (_, j) => (j === l ? 1 : 0)))
  } else {
    const r = toFlat(uniform(key, [m.n, k]))
    resp = Array.from({ length: m.n }, (_, i) => {
      const row = r.slice(i * k, (i + 1) * k)
      const s = row.reduce((a, v) => a + v, 0)
      return row.map((v) => v / s)
    })
  }
  return gaussianMStep(m, resp, hyper.covarianceType ?? 'full', hyper.regCovar ?? 1e-6)
}

/** The verbs of a fitted Gaussian mixture. */
function mixtureVerbs(p: GaussianMixtureParams): MixtureVerbs {
  const d = p.means[0]?.length ?? 0
  const posterior = (x: TensorLike) => {
    matrixView(x, 'gaussianMixture', { nFeatures: d })
    return normalise(componentLogDensities(x, p.weights, p.means, p.covariances))
  }
  return {
    predict: (x) => {
      const { resp } = posterior(x)
      const labels = resp.map((r) => r.indexOf(Math.max(...r)))
      return fromStorage(Int32Array.from(labels), [labels.length], 'int32')
    },
    predictProba: (x) => {
      const { resp } = posterior(x)
      return fromStorage(Float64Array.from(resp.flat()), [resp.length, p.weights.length], 'float64')
    },
    scoreSamples: (x) => {
      const { logNorm } = posterior(x)
      return fromStorage(Float64Array.from(logNorm), [logNorm.length], 'float64')
    },
    score: (data) => {
      const { logNorm } = posterior(data.x)
      return logNorm.reduce((a, v) => a + v, 0) / logNorm.length
    },
    nFeaturesIn: d,
  }
}

/**
 * A Gaussian mixture fitted by EM (sklearn.mixture.GaussianMixture), with full, tied, diagonal or spherical
 * covariances, initialised from k-means, traced step by step. Its exposition is a `mixtureSteps` record (weights,
 * means, covariances and mean log likelihood per EM step; responsibilities of the last step).
 *
 * @param hyper - Components, covariance type, tolerance, regularisation, iteration cap and initialisation.
 * @returns The estimator.
 */
export function gaussianMixture(
  hyper: GaussianMixtureHyper = {},
): Estimator<GaussianMixtureHyper, FittedGaussianMixture> {
  return defineExplainedEstimator<GaussianMixtureHyper, GaussianMixtureParams, MixtureVerbs>(
    {
      name: 'gaussian-mixture',
      kind: 'density',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 20, { default: h.nComponents ?? 2, label: 'components' }),
          covarianceType: choice(['full', 'tied', 'diag', 'spherical'], {
            label: 'covariance type',
            description: 'Each component’s covariance: full, diagonal, one shared (tied) or spherical.',
            default: h.covarianceType ?? 'full',
          }),
          maxIter: int(1, 500, { default: h.maxIter ?? 100, label: 'EM steps' }),
          tol: real(1e-8, 1e-1, {
            label: 'tolerance',
            description: 'Stop when the log-likelihood gains less than this.',
            log: true,
            default: h.tol ?? 1e-3,
          }),
        }),
      fit: (data, h, options) => fitGaussianMixture(data, h, options),
      methods: mixtureVerbs,
    },
    hyper,
  )
}

function fitGaussianMixture(data: LearnData, h: GaussianMixtureHyper, options: FitOptions) {
  const run = fitRunOptions(options, 'gaussian-mixture')
  const key = run.key as Key
  const start = initialMixture(data.x, h, foldIn(key, 0))
  const type = h.covarianceType ?? 'full'
  const out = runAlgorithm(
    gaussianEmAlgorithm(data.x, start, { covarianceType: type, regCovar: h.regCovar ?? 1e-6, tol: h.tol ?? 1e-3 }),
    { ...run, key: foldIn(key, 1), maxSteps: h.maxIter ?? 100 },
  )
  const s = out.state
  const final = normalise(componentLogDensities(data.x, s.weights, s.means, s.covariances))
  const flat = (cs: number[][][]): number[][] => cs.map((c) => c.flat())
  const params: GaussianMixtureParams = {
    weights: s.weights,
    means: s.means,
    covariances: s.covariances,
    covarianceType: type,
    logLikelihood: s.logLikelihood,
    nIter: out.steps,
    converged: out.status === 'converged',
  }
  const verbs = mixtureVerbs(params)
  // The log density over the data's range, for one or two features.
  const surface =
    (s.means[0]?.length ?? 0) <= 2
      ? [
          decisionSurfaceOf(
            { estimator: 'gaussian-mixture', kind: 'density', params, ...verbs, predict: verbs.scoreSamples },
            data.x,
            { resolution: 40 },
          ),
        ]
      : []
  return {
    params,
    exposition: [
      mixtureSteps({
        family: `gaussian-${type}`,
        weights: s.history.weights,
        means: s.history.means,
        covariances: s.history.covariances.map(flat),
        responsibilities: [final.resp],
        objective: s.history.objective,
      }),
      ...surface,
    ],
  }
}

/** A Gaussian mixture as a class. */
export class GaussianMixture extends EstimatorClass<GaussianMixtureHyper, FittedGaussianMixture> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link gaussianMixture}.
   */
  constructor(hyper: GaussianMixtureHyper = {}) {
    super(gaussianMixture(hyper))
  }
}

/**
 * The number of free parameters of a Gaussian mixture (sklearn's `_n_parameters`).
 *
 * @param k - Components.
 * @param d - Features.
 * @param type - The covariance type.
 * @returns The parameter count.
 */
export function mixtureParameterCount(k: number, d: number, type: CovarianceType): number {
  const cov =
    type === 'full' ? (k * d * (d + 1)) / 2 : type === 'tied' ? (d * (d + 1)) / 2 : type === 'diag' ? k * d : k
  return cov + k * d + k - 1
}

/**
 * The Bayesian and Akaike information criteria of a fitted mixture on data (lower is better).
 *
 * @param fitted - The fitted mixture.
 * @param x - Points `[n, d]`.
 * @returns BIC and AIC.
 */
export function mixtureCriteria(
  fitted: FittedGaussianMixture,
  x: TensorLike,
): { readonly bic: number; readonly aic: number } {
  const n = matrixView(x, 'mixtureCriteria').n
  const p = fitted.params
  const params = mixtureParameterCount(p.weights.length, p.means[0]?.length ?? 0, p.covarianceType)
  const ll = fitted.score({ x }) * n
  return { bic: -2 * ll + params * Math.log(n), aic: -2 * ll + 2 * params }
}
