/**
 * The Bayesian Gaussian mixture by variational inference (as in scikit-learn's `BayesianGaussianMixture`): a Dirichlet
 * or stick-breaking Dirichlet-process prior on the mixing weights lets the data switch off the components it does not
 * need, with a Normal-Wishart posterior on each component, fitted by coordinate-ascent updates (responsibilities, then
 * the weight and Normal-Wishart factors) as a traced `Algorithm`. Full, tied, diagonal and spherical covariances, as
 * full matrices throughout, like `gaussian.ts`, whose M-step it builds on.
 */

import { OmaiError } from '../base/index.ts'
import { type MatrixView, matrixView } from '../learn/index.ts'
import { slogdet } from '../numerics/linalg/index.ts'
import { digamma, lgamma } from '../numerics/special/index.ts'
import { item, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { type CovarianceType, componentLogDensities, gaussianMStep, normaliseLogDensities } from './gaussian.ts'

/** The prior on the mixing weights: a finite Dirichlet, or the stick-breaking Dirichlet process. */
export type WeightPrior = 'dirichlet_distribution' | 'dirichlet_process'

/** The priors of the variational Gaussian mixture: on the weights and on each component's Normal-Wishart. */
export interface VariationalPriors {
  /** The concentration of the weight prior (the Dirichlet's, or the Dirichlet process's). */
  readonly weightConcentration: number
  /** The precision of the prior on the component means. */
  readonly meanPrecision: number
  /** The prior mean of the component means. */
  readonly mean: number[]
  /** The prior degrees of freedom of the Wishart on the precisions. */
  readonly degreesOfFreedom: number
  /** The prior covariance `[d][d]` (a diagonal matrix for diag and spherical types). */
  readonly covariance: number[][]
}

/** The variational posterior: Dirichlet (or Beta sticks) on the weights, Normal-Wishart per component. */
export interface VariationalPosterior {
  /** Dirichlet concentrations `[k]`, or the two Beta parameters `[[k], [k]]` of each stick. */
  readonly concentration: number[] | [number[], number[]]
  readonly meanPrecision: number[]
  readonly means: number[][]
  readonly degreesOfFreedom: number[]
  /** The posterior Wishart's inverse scale divided by its degrees of freedom (sklearn's `covariances_`), `[k][d][d]`. */
  readonly covariances: number[][][]
}

/** What {@link variationalPriors} reads beyond the data. */
export interface VariationalPriorOptions {
  /** The weight concentration (default `1 / k`, scikit-learn's). */
  readonly weightConcentration?: number
  /** The mean precision (default 1). */
  readonly meanPrecision?: number
  /** The Wishart degrees of freedom (default the number of features). */
  readonly degreesOfFreedom?: number
  /** Shapes the prior covariance like the posterior's (default `full`). */
  readonly covarianceType?: CovarianceType
}

const digammaOf = (xs: readonly number[]): number[] => toFlat(digamma(xs as number[]))
const lgammaOf = (xs: readonly number[]): number[] => toFlat(lgamma(xs as number[]))
const sum = (xs: readonly number[]): number => xs.reduce((a, v) => a + v, 0)
const isSticks = (c: VariationalPosterior['concentration']): c is [number[], number[]] => Array.isArray(c[0])

/**
 * Data-dependent priors, as scikit-learn chooses them: the prior mean is the data mean and the prior covariance is the
 * data's covariance (its diagonal for `diag`, the mean variance for `spherical`).
 *
 * @param x - Points `[n, d]`, at least two.
 * @param k - The number of components (sets the default weight concentration `1 / k`).
 * @param options - Overrides, and the covariance type.
 * @returns The priors.
 * @example
 * ```ts
 * import { variationalPriors } from 'omai-engine/mixture'
 * variationalPriors([[0], [2]], 4).covariance // => [[2]]
 * variationalPriors([[0], [2]], 4).weightConcentration // => 0.25
 * ```
 */
export function variationalPriors(x: TensorLike, k: number, options: VariationalPriorOptions = {}): VariationalPriors {
  const { n, d, data } = matrixView(x, 'variationalMixture', { minSamples: 2 })
  const mean = Array.from({ length: d }, (_, q) => {
    let s = 0
    for (let i = 0; i < n; i++) s += data[i * d + q] ?? 0
    return s / n
  })
  const cov = mean.map((ma, a) =>
    mean.map((mb, b) => {
      let s = 0
      for (let i = 0; i < n; i++) s += ((data[i * d + a] ?? 0) - ma) * ((data[i * d + b] ?? 0) - mb)
      return s / (n - 1)
    }),
  )
  const type = options.covarianceType ?? 'full'
  const meanVariance = sum(cov.map((row, a) => row[a] ?? 0)) / d
  const covariance =
    type === 'full' || type === 'tied'
      ? cov
      : cov.map((row, a) => row.map((_, b) => (a !== b ? 0 : type === 'diag' ? (cov[a]?.[a] ?? 0) : meanVariance)))
  return {
    weightConcentration: options.weightConcentration ?? 1 / k,
    meanPrecision: options.meanPrecision ?? 1,
    mean,
    degreesOfFreedom: options.degreesOfFreedom ?? d,
    covariance,
  }
}

/**
 * The variational M-step (sklearn's `_estimate_weights`, `_estimate_means` and `_estimate_precisions`): the posterior
 * from responsibilities and priors.
 */
function variationalMStep(
  m: MatrixView,
  resp: readonly (readonly number[])[],
  priors: VariationalPriors,
  prior: WeightPrior,
  type: CovarianceType,
  reg: number,
): VariationalPosterior {
  const { weights, means: xk, covariances: sk } = gaussianMStep(m, resp, type, reg)
  const k = xk.length
  const d = m.d
  const nk = weights.map((w) => w * m.n)
  const concentration: VariationalPosterior['concentration'] =
    prior === 'dirichlet_distribution'
      ? nk.map((v) => v + priors.weightConcentration)
      : [nk.map((v) => 1 + v), nk.map((_, j) => priors.weightConcentration + sum(nk.slice(j + 1)))]
  const meanPrecision = nk.map((v) => priors.meanPrecision + v)
  const means = xk.map((x, j) =>
    x.map((v, q) => (priors.meanPrecision * (priors.mean[q] ?? 0) + (nk[j] ?? 0) * v) / (meanPrecision[j] ?? 1)),
  )
  const diffs = xk.map((x) => x.map((v, q) => v - (priors.mean[q] ?? 0)))
  const eyeOf = (v: number): number[][] =>
    Array.from({ length: d }, (_, a) => Array.from({ length: d }, (_, b) => (a === b ? v : 0)))
  if (type === 'tied') {
    // One shared covariance and degrees of freedom: the components pool their counts (sklearn's `_estimate_wishart_tied`).
    const total = sum(nk)
    const dof = priors.degreesOfFreedom + total / k
    const shared = priors.covariance.map((row, a) =>
      row.map((v, b) => {
        let cross = 0
        for (let j = 0; j < k; j++)
          cross += ((nk[j] ?? 0) / (meanPrecision[j] ?? 1)) * (diffs[j]?.[a] ?? 0) * (diffs[j]?.[b] ?? 0)
        return (v + ((sk[0]?.[a]?.[b] ?? 0) * total) / k + (priors.meanPrecision / k) * cross) / dof
      }),
    )
    return {
      concentration,
      meanPrecision,
      means,
      degreesOfFreedom: new Array<number>(k).fill(dof),
      covariances: Array.from({ length: k }, () => shared),
    }
  }
  const degreesOfFreedom = nk.map((v) => priors.degreesOfFreedom + v)
  const covariances = xk.map((_, j) => {
    const diff = diffs[j] as number[]
    const dof = degreesOfFreedom[j] ?? 1
    const c = ((nk[j] ?? 0) * priors.meanPrecision) / (meanPrecision[j] ?? 1)
    if (type === 'spherical') {
      const meanSquare = sum(diff.map((v) => v * v)) / d
      return eyeOf(((priors.covariance[0]?.[0] ?? 0) + (nk[j] ?? 0) * (sk[j]?.[0]?.[0] ?? 0) + c * meanSquare) / dof)
    }
    return priors.covariance.map((row, a) =>
      row.map((v, b) =>
        type === 'diag' && a !== b
          ? 0
          : (v + (nk[j] ?? 0) * (sk[j]?.[a]?.[b] ?? 0) + c * (diff[a] ?? 0) * (diff[b] ?? 0)) / dof,
      ),
    )
  })
  return { concentration, meanPrecision, means, degreesOfFreedom, covariances }
}

/** Expected log weights E[log pi_k]. */
function expectedLogWeights(post: VariationalPosterior): number[] {
  const c = post.concentration
  if (!isSticks(c)) {
    const total = digammaOf([sum(c)])[0] ?? 0
    return digammaOf(c).map((v) => v - total)
  }
  const [a, b] = c
  const dSum = digammaOf(a.map((v, k) => v + (b[k] ?? 0)))
  const dA = digammaOf(a)
  const dB = digammaOf(b)
  let acc = 0
  return a.map((_, k) => {
    const out = (dA[k] ?? 0) - (dSum[k] ?? 0) + acc
    acc += (dB[k] ?? 0) - (dSum[k] ?? 0)
    return out
  })
}

/**
 * The expected mixing weights of a variational posterior: the Dirichlet's mean, or the stick-breaking weights.
 *
 * @param post - The posterior.
 * @returns The weights, summing to one.
 * @example
 * ```ts
 * import { expectedWeights } from 'omai-engine/mixture'
 * const post = { concentration: [1, 3], meanPrecision: [1, 1], means: [[0], [1]], degreesOfFreedom: [1, 1], covariances: [[[1]], [[1]]] }
 * expectedWeights(post) // => [0.25, 0.75]
 * ```
 */
export function expectedWeights(post: Pick<VariationalPosterior, 'concentration'>): number[] {
  const c = post.concentration
  if (!isSticks(c)) return c.map((v) => v / sum(c))
  const [a, b] = c
  let stick = 1
  const w = a.map((v, k) => {
    const total = v + (b[k] ?? 0)
    const out = (v / total) * stick
    stick *= (b[k] ?? 0) / total
    return out
  })
  return w.map((v) => v / sum(w))
}

/** Per-row, per-component expected log joint (sklearn's `_estimate_weighted_log_prob`). */
function weightedLogProb(x: TensorLike, d: number, post: VariationalPosterior): number[][] {
  const k = post.means.length
  const logGauss = componentLogDensities(x, new Array<number>(k).fill(1), post.means, post.covariances)
  const logW = expectedLogWeights(post)
  const extra = post.degreesOfFreedom.map((nu, j) => {
    const logLambda = d * Math.LN2 + sum(digammaOf(Array.from({ length: d }, (_, q) => 0.5 * (nu - q))))
    return -0.5 * d * Math.log(nu) + 0.5 * (logLambda - d / (post.meanPrecision[j] ?? 1)) + (logW[j] ?? 0)
  })
  return logGauss.map((row) => row.map((v, j) => v + (extra[j] ?? 0)))
}

/**
 * The responsibilities of rows under a variational posterior (the expectation of the log joint under the posterior, as
 * scikit-learn's `predict_proba`), and each row's log normaliser.
 *
 * @param x - Points `[m, d]`.
 * @param post - The posterior.
 * @returns Responsibilities `[m][k]` and the log normaliser of each row.
 * @example
 * ```ts
 * import { variationalResponsibilities } from 'omai-engine/mixture'
 * const post = { concentration: [5, 5], meanPrecision: [5, 5], means: [[0], [10]], degreesOfFreedom: [6, 6], covariances: [[[1]], [[1]]] }
 * variationalResponsibilities([[0.1]], post).resp[0].map((r) => Math.round(r)) // => [1, 0]
 * ```
 */
export function variationalResponsibilities(
  x: TensorLike,
  post: VariationalPosterior,
): { readonly resp: number[][]; readonly logNorm: number[] } {
  return normaliseLogDensities(weightedLogProb(x, post.means[0]?.length ?? 0, post))
}

/** sklearn's variational lower bound (up to a constant). */
function lowerBoundOf(resp: readonly (readonly number[])[], post: VariationalPosterior, d: number): number {
  let entropy = 0
  for (const row of resp) for (const r of row) if (r > 0) entropy -= r * Math.log(r)
  let wishart = 0
  for (const [k, nu] of post.degreesOfFreedom.entries()) {
    const logDetChol = -0.5 * item(slogdet(post.covariances[k] as number[][]).logabsdet) - 0.5 * d * Math.log(nu)
    const lg = sum(lgammaOf(Array.from({ length: d }, (_, q) => 0.5 * (nu - q))))
    wishart += -(nu * logDetChol + nu * d * 0.5 * Math.LN2 + lg)
  }
  const c = post.concentration
  let logNormWeight: number
  if (!isSticks(c)) {
    logNormWeight = (lgammaOf([sum(c)])[0] ?? 0) - sum(lgammaOf(c))
  } else {
    const [a, b] = c
    const ab = lgammaOf(a.map((v, k) => v + (b[k] ?? 0)))
    logNormWeight = -sum(lgammaOf(a).map((v, k) => v + (lgammaOf([b[k] ?? 0])[0] ?? 0) - (ab[k] ?? 0)))
  }
  return entropy - wishart - logNormWeight - 0.5 * d * sum(post.meanPrecision.map(Math.log))
}

/**
 * The state of variational inference for a Bayesian Gaussian mixture: the posterior, its lower bound and how far the
 * bound moved, and the responsibilities that gave the posterior. Before the first step the bound is
 * `-Number.MAX_VALUE` and the change `Number.MAX_VALUE` (finite, so the state is plain JSON).
 */
export interface VariationalState {
  readonly posterior: VariationalPosterior
  readonly lowerBound: number
  readonly change: number
  readonly resp: number[][]
}

/** What {@link variationalMixtureAlgorithm} reads: the weight prior, covariance type, regularisation and tolerance. */
export interface VariationalOptions {
  /** Default `dirichlet_process`. */
  readonly weightPrior?: WeightPrior
  /** Default `full`. */
  readonly covarianceType?: CovarianceType
  /** Added to every covariance diagonal in the M-step (default 1e-6). */
  readonly regCovar?: number
  /** The change of the lower bound that stops the run (default 1e-3). */
  readonly tol?: number
}

const variationalSchema = (k: number, d: number, n: number): ReturnType<typeof channels> =>
  channels({
    metrics: {
      lowerBound: { description: 'the variational lower bound (up to a constant)', direction: 'higher' },
      change: { description: 'how much the lower bound moved this step' },
    },
    tensors: {
      responsibilities: { role: 'state', description: 'the posterior of each component for each row', shape: [n, k] },
      means: { role: 'param', description: 'the posterior means of the components', shape: [k, d] },
      weights: { role: 'param', description: 'the expected mixing weights', shape: [k] },
      covariances: { role: 'param', description: 'the component covariances, row-major', shape: [k, d, d] },
    },
  })

/**
 * Coordinate-ascent variational inference for a Bayesian Gaussian mixture as a traced algorithm: each step computes the
 * responsibilities under the current posterior (E), then the weight and Normal-Wishart factors (M), and records the
 * lower bound, which never falls. Converged when the bound moves by less than `tol`. The state keeps no past: the tape
 * records the bound, its change, the responsibilities, and the expected weights, means and covariances.
 *
 * @param x - Points `[n, d]`.
 * @param start - The initial responsibilities `[n][k]`.
 * @param priors - The priors (see {@link variationalPriors}).
 * @param options - The weight prior, covariance type, regularisation and tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { variationalMixtureAlgorithm, variationalPriors } from 'omai-engine/mixture'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[0], [0.2], [-0.1], [10], [10.2], [9.9]]
 * const start = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
 * const alg = variationalMixtureAlgorithm(x, start, variationalPriors(x, 3, { weightConcentration: 0.01 }))
 * runAlgorithm(alg, { maxSteps: 100 }).state.posterior.means.length // => 3
 * ```
 */
export function variationalMixtureAlgorithm(
  x: TensorLike,
  start: readonly (readonly number[])[],
  priors: VariationalPriors,
  options: VariationalOptions = {},
): Algorithm<VariationalState> {
  const m = matrixView(x, 'variationalMixture')
  if (start.length !== m.n)
    throw new OmaiError('shape', `variationalMixture: ${start.length} responsibilities for ${m.n} rows`)
  const prior = options.weightPrior ?? 'dirichlet_process'
  const type = options.covarianceType ?? 'full'
  const reg = options.regCovar ?? 1e-6
  const tol = options.tol ?? 1e-3
  const k = start[0]?.length ?? 0
  return {
    name: 'bayesian-gaussian-mixture',
    schema: variationalSchema(k, m.d, m.n),
    config: { weightPrior: prior, covarianceType: type, regCovar: reg, tol, k },
    init: () => ({
      posterior: variationalMStep(m, start, priors, prior, type, reg),
      lowerBound: -Number.MAX_VALUE,
      change: Number.MAX_VALUE,
      resp: start.map((r) => [...r]),
    }),
    step: (s) => {
      const { resp } = variationalResponsibilities(x, s.posterior)
      const posterior = variationalMStep(m, resp, priors, prior, type, reg)
      const lowerBound = lowerBoundOf(resp, posterior, m.d)
      const change = s.lowerBound === -Number.MAX_VALUE ? Number.MAX_VALUE : Math.abs(lowerBound - s.lowerBound)
      return { posterior, lowerBound, change, resp }
    },
    done: (s) => (s.change < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics:
        s.lowerBound === -Number.MAX_VALUE
          ? {}
          : { lowerBound: s.lowerBound, ...(s.change === Number.MAX_VALUE ? {} : { change: s.change }) },
      tensors: {
        responsibilities: { role: 'state', value: s.resp.flat(), shape: [m.n, k] },
        means: { role: 'param', value: s.posterior.means.flat(), shape: [k, m.d] },
        weights: { role: 'param', value: expectedWeights(s.posterior), shape: [k] },
        covariances: { role: 'param', value: s.posterior.covariances.flat(2), shape: [k, m.d, m.d] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as VariationalState,
  }
}
