/**
 * The Gaussian mixture model fitted by expectation-maximisation (as in scikit-learn's `GaussianMixture`): each step
 * computes every row's responsibilities under the components (E), then re-estimates weights, means and full, tied,
 * diagonal or spherical covariances (M), from a k-means start. The state keeps no past (the tape records it); `observe`
 * exposes the log likelihood, how far it moved, and the responsibilities, means and weights.
 */

import { MdmlError, maxOf } from '../base/index.ts'
import { initialCentres, lloydAlgorithm } from '../cluster/index.ts'
import { multivariateNormal } from '../dists/index.ts'
import { type MatrixView, matrixView } from '../learn/index.ts'
import { type Key, uniform } from '../random/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'

/** How component covariances are shared and shaped (sklearn's `covariance_type`). */
export type CovarianceType = 'full' | 'tied' | 'diag' | 'spherical'

/** A Gaussian mixture's parameters: weights, means and covariance matrices (full, whatever the type). */
export interface MixtureParameters {
  readonly weights: number[]
  readonly means: number[][]
  /** covariances[k]: component k's covariance `[d, d]` (shared for tied, diagonal for diag and spherical). */
  readonly covariances: number[][][]
}

/**
 * Per-row, per-component log joint densities log w_k + log N(x_i | mu_k, Sigma_k), `[n][k]`, through the
 * multivariate normal of `dists`.
 *
 * @param x - Points `[n, d]`.
 * @param weights - Mixing weights.
 * @param means - Component means.
 * @param covariances - Component covariances `[k][d][d]`.
 * @returns The log joint densities.
 * @example
 * ```ts
 * import { componentLogDensities } from 'mdml-engine/mixture'
 * componentLogDensities([[0]], [1], [[0]], [[[1]]])[0][0].toFixed(4) // => '-0.9189'
 * ```
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

/**
 * Normalise log joint densities into responsibilities and the per-row log normalisers (each row's log density).
 *
 * @param logJoint - Log joint densities `[n][k]`, as from {@link componentLogDensities}.
 * @returns Responsibilities `[n][k]` and the log density of each row.
 * @example
 * ```ts
 * import { normaliseLogDensities } from 'mdml-engine/mixture'
 * normaliseLogDensities([[0, 0]]).resp // => [[0.5, 0.5]]
 * ```
 */
export function normaliseLogDensities(logJoint: readonly (readonly number[])[]): {
  readonly resp: number[][]
  readonly logNorm: number[]
} {
  const logNorm = logJoint.map((row) => {
    const m = maxOf(row)
    return m + Math.log(row.reduce((a, v) => a + Math.exp(v - m), 0))
  })
  return { resp: logJoint.map((row, i) => row.map((v) => Math.exp(v - (logNorm[i] ?? 0)))), logNorm }
}

/**
 * The responsibilities and log density of rows under a mixture.
 *
 * @param x - Points `[m, d]`.
 * @param params - Weights, means and covariances.
 * @returns Responsibilities `[m][k]` and the log density of each row.
 * @example
 * ```ts
 * import { mixturePosterior } from 'mdml-engine/mixture'
 * const p = { weights: [0.5, 0.5], means: [[0], [10]], covariances: [[[1]], [[1]]] }
 * mixturePosterior([[0.1]], p).resp[0][0].toFixed(3) // => '1.000'
 * ```
 */
export function mixturePosterior(
  x: TensorLike,
  params: MixtureParameters,
): { readonly resp: number[][]; readonly logNorm: number[] } {
  return normaliseLogDensities(componentLogDensities(x, params.weights, params.means, params.covariances))
}

/**
 * The most responsible component of each row.
 *
 * @param resp - Responsibilities `[n][k]`.
 * @returns The component index per row.
 * @example
 * ```ts
 * import { componentLabels } from 'mdml-engine/mixture'
 * componentLabels([[0.2, 0.8], [0.9, 0.1]]) // => [1, 0]
 * ```
 */
export function componentLabels(resp: readonly (readonly number[])[]): number[] {
  return resp.map((r) => r.indexOf(maxOf(r)))
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
): MixtureParameters {
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

/**
 * The state of EM for a Gaussian mixture: the parameters, the mean log likelihood of the step before them and how far
 * it moved, and the responsibilities that gave them. Before the first step the likelihood is `-Number.MAX_VALUE` and
 * the change `Number.MAX_VALUE` (finite, so the state is plain JSON).
 */
export interface GaussianEmState extends MixtureParameters {
  readonly logLikelihood: number
  readonly change: number
  readonly resp: number[][]
}

const emSchema = channels({
  metrics: {
    logLikelihood: {
      description: 'mean log likelihood per row under the parameters before this step',
      direction: 'higher',
    },
    change: { description: 'how much the mean log likelihood moved this step' },
  },
  tensors: {
    responsibilities: { role: 'state', description: 'the posterior of each component for each row', shape: ['n', 'k'] },
    means: { role: 'param', description: 'the component means', shape: ['k', 'd'] },
    weights: { role: 'param', description: 'the mixing weights', shape: ['k'] },
    covariances: { role: 'param', description: 'the component covariances, row-major', shape: ['k', 'd', 'd'] },
  },
})

/**
 * EM for a Gaussian mixture as a traced algorithm (sklearn's loop): each step computes responsibilities and the mean
 * log likelihood under the current parameters (E), then re-estimates the parameters (M). Converged when the mean log
 * likelihood changes by less than tol. The state keeps no past: the tape records the log likelihood, change,
 * responsibilities, means and weights.
 *
 * @param x - Points `[n, d]`.
 * @param start - Initial weights, means and covariances.
 * @param options - Covariance type, regularisation (1e-6) and tolerance (1e-3).
 * @returns The algorithm.
 * @index algorithm Gaussian mixture by EM, stepwise; no estimator wraps it.
 * @example
 * ```ts
 * import { gaussianEmAlgorithm } from 'mdml-engine/mixture'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0], [0.2], [10], [10.2]]
 * const start = { weights: [0.5, 0.5], means: [[0], [10]], covariances: [[[1]], [[1]]] }
 * runAlgorithm(gaussianEmAlgorithm(x, start), { maxSteps: 20 }).state.means[1][0].toFixed(1) // => '10.1'
 * ```
 */
export function gaussianEmAlgorithm(
  x: TensorLike,
  start: MixtureParameters,
  options: { readonly covarianceType?: CovarianceType; readonly regCovar?: number; readonly tol?: number } = {},
): Algorithm<GaussianEmState> {
  const m = matrixView(x, 'gaussianEm')
  const type = options.covarianceType ?? 'full'
  const reg = options.regCovar ?? 1e-6
  const tol = options.tol ?? 1e-3
  const k = start.weights.length
  return {
    name: 'gaussian-mixture-em',
    schema: emSchema,
    config: { covarianceType: type, regCovar: reg, tol, k },
    init: () => ({
      ...start,
      logLikelihood: -Number.MAX_VALUE,
      change: Number.MAX_VALUE,
      resp: mixturePosterior(x, start).resp,
    }),
    step: (s) => {
      const { resp, logNorm } = mixturePosterior(x, s)
      const ll = logNorm.reduce((a, v) => a + v, 0) / m.n
      const next = gaussianMStep(m, resp, type, reg)
      return {
        ...next,
        logLikelihood: ll,
        change: s.logLikelihood === -Number.MAX_VALUE ? Number.MAX_VALUE : Math.abs(ll - s.logLikelihood),
        resp,
      }
    },
    done: (s) => (s.change < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics:
        s.logLikelihood === -Number.MAX_VALUE
          ? {}
          : { logLikelihood: s.logLikelihood, ...(s.change === Number.MAX_VALUE ? {} : { change: s.change }) },
      tensors: {
        responsibilities: { role: 'state', value: s.resp.flat(), shape: [m.n, k] },
        means: { role: 'param', value: s.means.flat(), shape: [k, m.d] },
        weights: { role: 'param', value: s.weights, shape: [k] },
        covariances: { role: 'param', value: s.covariances.flat(2), shape: [k, m.d, m.d] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as GaussianEmState,
  }
}

/** The initial parameters of EM: from k-means (default), random responsibilities, or given parameters. */
export type MixtureInit =
  | 'kmeans'
  | 'random'
  | {
      readonly weights: readonly number[]
      readonly means: readonly (readonly number[])[]
      readonly covariances: readonly (readonly (readonly number[])[])[]
    }

/** What {@link initialMixture} reads: components, covariance type, regularisation and the initialisation. */
export interface MixtureInitOptions {
  /** Components (default 2). */
  readonly nComponents?: number
  readonly covarianceType?: CovarianceType
  /** Added to every covariance diagonal (default 1e-6). */
  readonly regCovar?: number
  readonly init?: MixtureInit
}

/**
 * Initial responsibilities for a mixture: one-hot labels of a converged k-means (k-means++ start, scikit-learn's
 * default), or uniform random weights normalised over the components.
 *
 * @param x - Points `[n, d]`.
 * @param k - The number of components (at most the number of points).
 * @param init - `kmeans` or `random`.
 * @param key - Randomness.
 * @returns Responsibilities `[n][k]`, each row summing to one.
 * @throws MdmlError with code `value` unless `k` is an integer in `1..n`.
 * @example
 * ```ts
 * import { initialResponsibilities } from 'mdml-engine/mixture'
 * import { key } from 'mdml-engine/random'
 * initialResponsibilities([[0], [0.1], [9], [9.1]], 2, 'kmeans', key(0)).map((r) => r[0] + r[1]) // => [1, 1, 1, 1]
 * ```
 */
export function initialResponsibilities(x: TensorLike, k: number, init: 'kmeans' | 'random', key: Key): number[][] {
  const m = matrixView(x, 'gaussianMixture')
  if (!(Number.isInteger(k) && k >= 1 && k <= m.n))
    throw new MdmlError('value', `gaussianMixture: nComponents must be in 1..${m.n}`)
  if (init === 'kmeans') {
    const lloyd = lloydAlgorithm(x, initialCentres(x, k, 'k-means++', key))
    const labels = runAlgorithm(lloyd, { maxSteps: 300 }).state.labels
    return labels.map((l) => Array.from({ length: k }, (_, j) => (j === l ? 1 : 0)))
  }
  const r = toFlat(uniform(key, [m.n, k]))
  return Array.from({ length: m.n }, (_, i) => {
    const row = r.slice(i * k, (i + 1) * k)
    const s = row.reduce((a, v) => a + v, 0)
    return row.map((v) => v / s)
  })
}

/**
 * Initial parameters: the M-step on one-hot k-means labels (sklearn's default) or on random responsibilities, or the
 * given parameters.
 *
 * @param x - Points `[n, d]`.
 * @param options - Components, covariance type, regularisation and the initialisation.
 * @param key - Randomness.
 * @returns Initial weights, means and covariances.
 * @example
 * ```ts
 * import { initialMixture } from 'mdml-engine/mixture'
 * import { key } from 'mdml-engine/random'
 * initialMixture([[0], [0.1], [9], [9.1]], { nComponents: 2 }, key(0)).weights // => [0.5, 0.5]
 * ```
 */
export function initialMixture(x: TensorLike, options: MixtureInitOptions, key: Key): MixtureParameters {
  const m = matrixView(x, 'gaussianMixture')
  const k = options.nComponents ?? 2
  if (!(Number.isInteger(k) && k >= 1 && k <= m.n))
    throw new MdmlError('value', `gaussianMixture: nComponents must be in 1..${m.n}`)
  const init = options.init ?? 'kmeans'
  if (typeof init !== 'string')
    return {
      weights: [...init.weights],
      means: init.means.map((r) => [...r]),
      covariances: init.covariances.map((c) => c.map((r) => [...r])),
    }
  const resp = initialResponsibilities(x, k, init, key)
  return gaussianMStep(m, resp, options.covarianceType ?? 'full', options.regCovar ?? 1e-6)
}

/**
 * The number of free parameters of a Gaussian mixture (sklearn's `_n_parameters`).
 *
 * @param k - Components.
 * @param d - Features.
 * @param type - The covariance type.
 * @returns The parameter count.
 * @example
 * ```ts
 * import { mixtureParameterCount } from 'mdml-engine/mixture'
 * mixtureParameterCount(2, 2, 'full') // => 11
 * ```
 */
export function mixtureParameterCount(k: number, d: number, type: CovarianceType): number {
  const cov =
    type === 'full' ? (k * d * (d + 1)) / 2 : type === 'tied' ? (d * (d + 1)) / 2 : type === 'diag' ? k * d : k
  return cov + k * d + k - 1
}
