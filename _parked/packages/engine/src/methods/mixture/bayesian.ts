/**
 * The Bayesian Gaussian mixture by variational inference (as in scikit-learn's `BayesianGaussianMixture`): a Dirichlet
 * or stick-breaking Dirichlet-process prior on the weights lets the data switch off components it does not need, with
 * Normal-Wishart posteriors on each component, fitted by coordinate-ascent updates traced step by step.
 */

import { mixtureSteps } from '../../lib/exposition/index.ts'
import {
  defineExplainedEstimator,
  type Estimator,
  EstimatorClass,
  type ExplainedFitted,
  fitRunOptions,
  type MatrixView,
  matrixView,
} from '../../lib/learn/index.ts'
import { linalg, special } from '../../lib/numerics/index.ts'
import { foldIn, type Key } from '../../lib/random/index.ts'
import { choice, dict, int, real } from '../../lib/space/index.ts'
import { fromStorage, item, type Tensor, type TensorLike, toFlat } from '../../lib/tensor/index.ts'
import { type Algorithm, runAlgorithm } from '../../lib/trace/index.ts'
import { componentLogDensities, gaussianMStep, initialMixture, type MixtureVerbs } from './gaussian.ts'

/** The prior on the mixing weights: a finite Dirichlet, or the stick-breaking Dirichlet process. */
export type WeightPrior = 'dirichlet_distribution' | 'dirichlet_process'

/** The priors of the variational Gaussian mixture (sklearn's defaults when omitted). */
export interface BayesianPriors {
  readonly weightConcentration: number
  readonly meanPrecision: number
  readonly mean: number[]
  readonly degreesOfFreedom: number
  readonly covariance: number[][]
}

/** The variational posterior: Dirichlet (or Beta sticks), Normal-Wishart per component. */
export interface VariationalPosterior {
  /** Dirichlet concentrations, or the two Beta parameters per stick. */
  readonly concentration: number[] | [number[], number[]]
  readonly meanPrecision: number[]
  readonly means: number[][]
  readonly degreesOfFreedom: number[]
  /** The posterior Wishart's inverse scale divided by its degrees of freedom (sklearn's covariances_). */
  readonly covariances: number[][][]
}

const digamma = (xs: readonly number[]): number[] => toFlat(special.digamma(xs as number[]))
const lgamma = (xs: readonly number[]): number[] => toFlat(special.lgamma(xs as number[]))
const sum = (xs: readonly number[]): number => xs.reduce((a, v) => a + v, 0)

/** The variational M-step for full covariances (sklearn's `_estimate_weights`, `_means` and `_precisions`). */
function variationalMStep(
  m: MatrixView,
  resp: readonly (readonly number[])[],
  priors: BayesianPriors,
  prior: WeightPrior,
  reg: number,
): VariationalPosterior {
  const { weights, means: xk, covariances: sk } = gaussianMStep(m, resp, 'full', reg)
  const nk = weights.map((w) => w * m.n)
  const concentration: VariationalPosterior['concentration'] =
    prior === 'dirichlet_distribution'
      ? nk.map((v) => v + priors.weightConcentration)
      : [nk.map((v) => 1 + v), nk.map((_, k) => priors.weightConcentration + sum(nk.slice(k + 1)))]
  const meanPrecision = nk.map((v) => priors.meanPrecision + v)
  const means = xk.map((x, k) =>
    x.map((v, q) => (priors.meanPrecision * (priors.mean[q] ?? 0) + (nk[k] ?? 0) * v) / (meanPrecision[k] ?? 1)),
  )
  const degreesOfFreedom = nk.map((v) => priors.degreesOfFreedom + v)
  const covariances = xk.map((x, k) => {
    const diff = x.map((v, q) => v - (priors.mean[q] ?? 0))
    const c = ((nk[k] ?? 0) * priors.meanPrecision) / (meanPrecision[k] ?? 1)
    return priors.covariance.map((row, a) =>
      row.map(
        (v, b) =>
          (v + (nk[k] ?? 0) * (sk[k]?.[a]?.[b] ?? 0) + c * (diff[a] ?? 0) * (diff[b] ?? 0)) /
          (degreesOfFreedom[k] ?? 1),
      ),
    )
  })
  return { concentration, meanPrecision, means, degreesOfFreedom, covariances }
}

/** Expected log weights E[log pi_k]. */
function expectedLogWeights(post: VariationalPosterior): number[] {
  const c = post.concentration
  if (!Array.isArray(c[0])) {
    const a = c as number[]
    const total = digamma([sum(a)])[0] ?? 0
    return digamma(a).map((v) => v - total)
  }
  const [a, b] = c as [number[], number[]]
  const dSum = digamma(a.map((v, k) => v + (b[k] ?? 0)))
  const dA = digamma(a)
  const dB = digamma(b)
  let acc = 0
  return a.map((_, k) => {
    const out = (dA[k] ?? 0) - (dSum[k] ?? 0) + acc
    acc += (dB[k] ?? 0) - (dSum[k] ?? 0)
    return out
  })
}

/** The expected mixing weights. */
function expectedWeights(post: VariationalPosterior): number[] {
  const c = post.concentration
  if (!Array.isArray(c[0])) {
    const a = c as number[]
    return a.map((v) => v / sum(a))
  }
  const [a, b] = c as [number[], number[]]
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
    const logLambda = d * Math.LN2 + sum(digamma(Array.from({ length: d }, (_, q) => 0.5 * (nu - q))))
    return -0.5 * d * Math.log(nu) + 0.5 * (logLambda - d / (post.meanPrecision[j] ?? 1)) + (logW[j] ?? 0)
  })
  return logGauss.map((row) => row.map((v, j) => v + (extra[j] ?? 0)))
}

/** sklearn's variational lower bound (up to a constant) for full covariances. */
function lowerBound(resp: readonly (readonly number[])[], post: VariationalPosterior, d: number): number {
  let entropy = 0
  for (const row of resp) for (const r of row) if (r > 0) entropy -= r * Math.log(r)
  let wishart = 0
  for (const [k, nu] of post.degreesOfFreedom.entries()) {
    const logDetChol = -0.5 * item(linalg.slogdet(post.covariances[k] as number[][]).logabsdet) - 0.5 * d * Math.log(nu)
    const lg = sum(lgamma(Array.from({ length: d }, (_, q) => 0.5 * (nu - q))))
    wishart += -(nu * logDetChol + nu * d * 0.5 * Math.LN2 + lg)
  }
  const c = post.concentration
  let logNormWeight: number
  if (!Array.isArray(c[0])) {
    const a = c as number[]
    logNormWeight = (lgamma([sum(a)])[0] ?? 0) - sum(lgamma(a))
  } else {
    const [a, b] = c as [number[], number[]]
    const ab = lgamma(a.map((v, k) => v + (b[k] ?? 0)))
    logNormWeight = -sum(lgamma(a).map((v, k) => v + (lgamma([b[k] ?? 0])[0] ?? 0) - (ab[k] ?? 0)))
  }
  return entropy - wishart - logNormWeight - 0.5 * d * sum(post.meanPrecision.map(Math.log))
}

/** The state of variational inference for a Bayesian Gaussian mixture. */
export interface VariationalState {
  readonly posterior: VariationalPosterior
  readonly elbo: number
  readonly change: number
  readonly resp: number[][]
  readonly history: { readonly weights: number[][]; readonly means: number[][][]; readonly objective: number[] }
}

const normaliseRows = (logJoint: readonly (readonly number[])[]): { resp: number[][]; logNorm: number[] } => {
  const logNorm = logJoint.map((row) => {
    const m = Math.max(...row)
    return m + Math.log(row.reduce((a, v) => a + Math.exp(v - m), 0))
  })
  return { resp: logJoint.map((row, i) => row.map((v) => Math.exp(v - (logNorm[i] ?? 0)))), logNorm }
}

/**
 * Coordinate-ascent variational inference for a Bayesian Gaussian mixture (full covariances) as a traced algorithm:
 * each step updates the responsibilities under the current posterior, then the Dirichlet (or stick-breaking) and
 * Normal-Wishart factors, recording the lower bound.
 *
 * @param x - Points `[n, d]`.
 * @param start - The initial responsibilities `[n][k]`.
 * @param priors - The priors.
 * @param options - The weight prior, regularisation (1e-6) and tolerance (1e-3).
 * @returns The algorithm.
 */
export function variationalMixtureAlgorithm(
  x: TensorLike,
  start: readonly (readonly number[])[],
  priors: BayesianPriors,
  options: { readonly weightPrior?: WeightPrior; readonly regCovar?: number; readonly tol?: number } = {},
): Algorithm<VariationalState> {
  const m = matrixView(x, 'bayesianGaussianMixture')
  const prior = options.weightPrior ?? 'dirichlet_process'
  const reg = options.regCovar ?? 1e-6
  const tol = options.tol ?? 1e-3
  return {
    name: 'bayesian-gaussian-mixture',
    config: { weightPrior: prior, regCovar: reg, tol },
    init: () => ({
      posterior: variationalMStep(m, start, priors, prior, reg),
      elbo: Number.NEGATIVE_INFINITY,
      change: Number.POSITIVE_INFINITY,
      resp: start.map((r) => [...r]),
      history: { weights: [], means: [], objective: [] },
    }),
    step: (s) => {
      const { resp } = normaliseRows(weightedLogProb(x, m.d, s.posterior))
      const posterior = variationalMStep(m, resp, priors, prior, reg)
      const elbo = lowerBound(resp, posterior, m.d)
      return {
        posterior,
        elbo,
        change: Math.abs(elbo - s.elbo),
        resp,
        history: {
          weights: [...s.history.weights, expectedWeights(posterior)],
          means: [...s.history.means, posterior.means],
          objective: [...s.history.objective, elbo],
        },
      }
    },
    done: (s) => (s.change < tol ? 'converged' : undefined),
    observe: (s) => ({ metrics: Number.isFinite(s.elbo) ? { elbo: s.elbo } : {} }),
  }
}

/** Hyperparameters of the Bayesian Gaussian mixture. */
export type BayesianGaussianMixtureHyper = {
  /** The most components used (default 5); unneeded ones get vanishing weight. */
  readonly nComponents?: number
  readonly weightConcentrationPriorType?: WeightPrior
  /** The Dirichlet concentration (default 1 / nComponents). */
  readonly weightConcentrationPrior?: number
  readonly meanPrecisionPrior?: number
  readonly tol?: number
  readonly regCovar?: number
  readonly maxIter?: number
}

/** The parameters of a fitted Bayesian Gaussian mixture. */
export type BayesianGaussianMixtureParams = {
  readonly weights: number[]
  /** The posterior Dirichlet concentrations, or the Beta parameters of each stick. */
  readonly concentration: number[] | number[][]
  readonly means: number[][]
  readonly covariances: number[][][]
  readonly degreesOfFreedom: number[]
  readonly meanPrecision: number[]
  readonly lowerBound: number
  readonly nIter: number
}

/** A fitted Bayesian Gaussian mixture. */
export type FittedBayesianGaussianMixture = ExplainedFitted<BayesianGaussianMixtureParams, MixtureVerbs>

function bayesianVerbs(p: BayesianGaussianMixtureParams): MixtureVerbs {
  const d = p.means[0]?.length ?? 0
  const post: VariationalPosterior = {
    concentration: p.concentration as VariationalPosterior['concentration'],
    meanPrecision: p.meanPrecision,
    means: p.means,
    degreesOfFreedom: p.degreesOfFreedom,
    covariances: p.covariances,
  }
  const rows = (x: TensorLike) => {
    matrixView(x, 'bayesianGaussianMixture', { nFeatures: d })
    return normaliseRows(weightedLogProb(x, d, post))
  }
  const logDensity = (x: TensorLike): number[] =>
    componentLogDensities(x, p.weights, p.means, p.covariances).map((row) => {
      const m = Math.max(...row)
      return m + Math.log(row.reduce((a, v) => a + Math.exp(v - m), 0))
    })
  return {
    predict: (x) => {
      const labels = rows(x).resp.map((r) => r.indexOf(Math.max(...r)))
      return fromStorage(Int32Array.from(labels), [labels.length], 'int32')
    },
    predictProba: (x) => {
      const { resp } = rows(x)
      return fromStorage(Float64Array.from(resp.flat()), [resp.length, p.weights.length], 'float64')
    },
    scoreSamples: (x): Tensor => {
      const v = logDensity(x)
      return fromStorage(Float64Array.from(v), [v.length], 'float64')
    },
    score: (data) => {
      const v = logDensity(data.x)
      return sum(v) / v.length
    },
    nFeaturesIn: d,
  }
}

/**
 * A Bayesian Gaussian mixture by variational inference (sklearn.mixture.BayesianGaussianMixture, full covariances):
 * a Dirichlet or Dirichlet-process prior on the weights switches off unneeded components. Its exposition is a
 * `mixtureSteps` record of expected weights, means and the lower bound per step. TODO: tied, diagonal and spherical
 * covariances; collapsed Gibbs.
 *
 * @param hyper - The most components, the weight prior and its concentration, tolerance and iteration cap.
 * @returns The estimator.
 */
export function bayesianGaussianMixture(
  hyper: BayesianGaussianMixtureHyper = {},
): Estimator<BayesianGaussianMixtureHyper, FittedBayesianGaussianMixture> {
  return defineExplainedEstimator<BayesianGaussianMixtureHyper, BayesianGaussianMixtureParams, MixtureVerbs>(
    {
      name: 'bayesian-gaussian-mixture',
      kind: 'density',
      hyperSpace: (h) =>
        dict({
          nComponents: int(1, 20, { default: h.nComponents ?? 5, label: 'components' }),
          weightConcentrationPriorType: choice(['dirichlet_process', 'dirichlet_distribution'], {
            label: 'weight prior',
            description: 'A Dirichlet process (stick-breaking) or a finite Dirichlet distribution on the weights.',
            default: h.weightConcentrationPriorType ?? 'dirichlet_process',
          }),
          weightConcentrationPrior: real(1e-3, 1e3, {
            label: 'weight concentration γ₀',
            description: 'The prior concentration of the mixture weights: small values switch components off.',
            log: true,
            default: h.weightConcentrationPrior ?? 0.2,
          }),
        }),
      fit: (data, h, options) => {
        const m = matrixView(data.x, 'bayesianGaussianMixture', { minSamples: 2 })
        const k = h.nComponents ?? 5
        const run = fitRunOptions(options, 'bayesian-gaussian-mixture')
        const key = run.key as Key
        const mean = Array.from({ length: m.d }, (_, q) => {
          let s = 0
          for (let i = 0; i < m.n; i++) s += m.data[i * m.d + q] ?? 0
          return s / m.n
        })
        const covariance = mean.map((ma, a) =>
          mean.map((mb, b) => {
            let s = 0
            for (let i = 0; i < m.n; i++) s += ((m.data[i * m.d + a] ?? 0) - ma) * ((m.data[i * m.d + b] ?? 0) - mb)
            return s / (m.n - 1)
          }),
        )
        const priors: BayesianPriors = {
          weightConcentration: h.weightConcentrationPrior ?? 1 / k,
          meanPrecision: h.meanPrecisionPrior ?? 1,
          mean,
          degreesOfFreedom: m.d,
          covariance,
        }
        const start = initialMixture(data.x, { nComponents: k }, foldIn(key, 0))
        const resp = componentLogDensities(data.x, start.weights, start.means, start.covariances).map((row) => {
          const best = row.indexOf(Math.max(...row))
          return row.map((_, j) => (j === best ? 1 : 0))
        })
        const out = runAlgorithm(
          variationalMixtureAlgorithm(data.x, resp, priors, {
            weightPrior: h.weightConcentrationPriorType ?? 'dirichlet_process',
            regCovar: h.regCovar ?? 1e-6,
            tol: h.tol ?? 1e-3,
          }),
          { ...run, key: foldIn(key, 1), maxSteps: h.maxIter ?? 100 },
        )
        const s = out.state
        const post = s.posterior
        return {
          params: {
            weights: expectedWeights(post),
            concentration: post.concentration,
            means: post.means,
            covariances: post.covariances,
            degreesOfFreedom: post.degreesOfFreedom,
            meanPrecision: post.meanPrecision,
            lowerBound: s.elbo,
            nIter: out.steps,
          },
          exposition: [
            mixtureSteps({
              family: 'bayesian-gaussian-full',
              weights: s.history.weights,
              means: s.history.means,
              responsibilities: [s.resp],
              objective: s.history.objective,
            }),
          ],
        }
      },
      methods: bayesianVerbs,
    },
    hyper,
  )
}

/** The Bayesian Gaussian mixture as a class. */
export class BayesianGaussianMixture extends EstimatorClass<
  BayesianGaussianMixtureHyper,
  FittedBayesianGaussianMixture
> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link bayesianGaussianMixture}.
   */
  constructor(hyper: BayesianGaussianMixtureHyper = {}) {
    super(bayesianGaussianMixture(hyper))
  }
}
