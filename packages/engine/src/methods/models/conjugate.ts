/**
 * Conjugate Bayesian models, whose posterior has the same form as the prior and is known exactly: beta-Bernoulli and
 * beta-binomial, gamma-Poisson, gamma-exponential and normal-normal with a known scale, each updated observation by
 * observation from the core's `conjugatePosterior` so the prior-to-posterior story can be shown.
 */

// TODO: normal-gamma (unknown mean and variance), Dirichlet-multinomial and normal-inverse-Wishart studies, the
// predictive distributions, and CAVI on the same models compared with the exact posterior.
import { WhiteboxError } from '../../lib/base/index.ts'
import { beta, conjugatePosterior, type Distribution, gamma, normal } from '../../lib/dists/index.ts'
import {
  coefficientTable,
  type Explained,
  type Exposition,
  learningCurve,
  posteriorGrid,
} from '../../lib/exposition/index.ts'
import { asTensor, item, type TensorLike, toFlat, type Value } from '../../lib/tensor/index.ts'

/** The likelihoods with a conjugate prior on one parameter that {@link conjugateModel} studies. */
export type ConjugateLikelihood = 'bernoulli' | 'binomial' | 'poisson' | 'exponential' | 'normal'

/** Options of {@link conjugateModel}. */
export interface ConjugateModelOptions {
  readonly likelihood: ConjugateLikelihood
  /** The prior on the parameter: beta for bernoulli and binomial, gamma for poisson and exponential, normal for normal. */
  readonly prior: Distribution
  /** Known likelihood parameters: `n` (trials) for binomial, `scale` for normal. */
  readonly fixed?: Readonly<Record<string, number>>
  /** The credible level of the intervals (default 0.95). */
  readonly level?: number
  /** Points of the density grid (default 200). */
  readonly gridSize?: number
  /** Most prefix posteriors in the learning curve (default 50). */
  readonly steps?: number
}

/** The result of {@link conjugateModel}. */
export interface ConjugateModel {
  readonly prior: Distribution
  readonly posterior: Distribution
  /** The parameter the prior is on (`probs`, `rate` or `loc`). */
  readonly parameter: string
  readonly posteriorMean: number
  readonly interval: readonly [number, number]
  /** After each prefix of the observations (the first n): the posterior mean and sd, and the maximum likelihood. */
  readonly sequence: {
    readonly n: readonly number[]
    readonly mean: readonly number[]
    readonly sd: readonly number[]
    readonly mle: readonly number[]
  }
}

const PARAMETER: Readonly<Record<ConjugateLikelihood, string>> = {
  bernoulli: 'probs',
  binomial: 'probs',
  poisson: 'rate',
  exponential: 'rate',
  normal: 'loc',
}

/** The maximum-likelihood estimate of the parameter from observations (the prior mean with none). */
function mle(
  likelihood: ConjugateLikelihood,
  xs: readonly number[],
  fixed: Readonly<Record<string, number>>,
  fallback: number,
): number {
  if (xs.length === 0) return fallback
  const mean = xs.reduce((a, v) => a + v, 0) / xs.length
  switch (likelihood) {
    case 'binomial':
      return mean / (fixed['n'] ?? 1)
    case 'exponential':
      return 1 / mean
    default:
      return mean
  }
}

/** Evenly spread prefix lengths 0..n, at most max of them, keeping both ends. */
function prefixes(n: number, max: number): number[] {
  if (n + 1 <= max) return Array.from({ length: n + 1 }, (_, t) => t)
  const out = new Set<number>()
  for (let j = 0; j < max; j++) out.add(Math.round((j * n) / (max - 1)))
  return [...out].sort((a, b) => a - b)
}

/**
 * A conjugate Bayesian model with its exact posterior (dists' `conjugatePosterior`): beta-Bernoulli and
 * beta-binomial, gamma-Poisson, gamma-exponential and normal-normal (known scale). Exposition: a `learningCurve` of
 * the posterior mean (with its standard deviation as the spread) and the maximum-likelihood estimate as the
 * observations arrive, a `posteriorGrid` of the posterior density over the parameter with the prior density as its
 * one sample line, and a coefficient table of the prior and posterior means with credible intervals.
 *
 * @param observations - The data `[n]`.
 * @param options - Likelihood, prior, known parameters, level, grid size and curve steps.
 * @returns The prior and posterior, the posterior mean and interval and the sequence, with the exposition records.
 * @throws WhiteboxError `value` when the prior's family does not suit the likelihood or a known parameter is missing.
 */
export function conjugateModel(observations: TensorLike, options: ConjugateModelOptions): Explained<ConjugateModel> {
  const xs = Array.from(toFlat(asTensor(observations)))
  const { likelihood, prior } = options
  const fixed = options.fixed ?? {}
  const parameter = PARAMETER[likelihood]
  if (likelihood === 'binomial' && fixed['n'] === undefined)
    throw new WhiteboxError('value', 'conjugateModel: the binomial likelihood needs the trials n in fixed')
  if (likelihood === 'normal' && fixed['scale'] === undefined)
    throw new WhiteboxError('value', 'conjugateModel: the normal likelihood needs its known scale in fixed')
  const level = options.level ?? 0.95
  const posteriorOf = (data: readonly number[]): Distribution =>
    data.length === 0 ? prior : conjugatePosterior(prior, likelihood, parameter, data as number[], fixed)
  const posterior = posteriorOf(xs)
  const scalar = (v: Value): number => item(v)
  const ns = prefixes(xs.length, options.steps ?? 50)
  const fits = ns.map((n) => posteriorOf(xs.slice(0, n)))
  const priorMean = scalar(prior.mean())
  const sequence = {
    n: ns,
    mean: fits.map((d) => scalar(d.mean())),
    sd: fits.map((d) => scalar(d.stddev())),
    mle: ns.map((n) => mle(likelihood, xs.slice(0, n), fixed, priorMean)),
  }
  const tail = (1 - level) / 2
  const interval = (d: Distribution): [number, number] => [scalar(d.ppf(tail)), scalar(d.ppf(1 - tail))]
  // The density grid spans both the prior's and the posterior's central 99.8%.
  const lo = Math.min(scalar(prior.ppf(0.001)), scalar(posterior.ppf(0.001)))
  const hi = Math.max(scalar(prior.ppf(0.999)), scalar(posterior.ppf(0.999)))
  const m = options.gridSize ?? 200
  const grid = Array.from({ length: m }, (_, i) => lo + ((hi - lo) * i) / (m - 1))
  const density = (d: Distribution): number[] => Array.from(toFlat(d.prob(grid)))
  const records: Exposition[] = [
    learningCurve({
      sizes: sequence.n,
      train: sequence.mean,
      validation: sequence.mle,
      trainSpread: sequence.sd,
      resource: 'observations',
      labels: { train: `posterior mean of ${parameter}`, validation: 'maximum likelihood' },
    }),
    posteriorGrid({
      inputs: grid.map((g) => [g]),
      mean: density(posterior),
      variance: grid.map(() => 0),
      labels: { input: parameter, output: 'density' },
      densities: [
        { name: `prior of ${parameter}`, role: 'prior', values: density(prior) },
        { name: `posterior of ${parameter}`, role: 'posterior', values: density(posterior) },
      ],
    }),
    coefficientTable({
      names: [`prior ${parameter}`, `posterior ${parameter}`],
      estimates: [priorMean, scalar(posterior.mean())],
      standardErrors: [scalar(prior.stddev()), scalar(posterior.stddev())],
      intervals: [interval(prior), interval(posterior)],
      level,
    }),
  ]
  return {
    prior,
    posterior,
    parameter,
    posteriorMean: scalar(posterior.mean()),
    interval: interval(posterior),
    sequence,
    exposition: records,
  }
}

/**
 * The beta-binomial model: successes out of `trials` per observation (1 for coin flips) under a Beta(a, b) prior on
 * the success probability. See {@link conjugateModel}.
 *
 * @param observations - Successes per observation.
 * @param options - The prior's a and b (default 1, 1), trials (default 1) and the level.
 * @returns As {@link conjugateModel}.
 */
export function betaBinomialModel(
  observations: TensorLike,
  options: { readonly a?: number; readonly b?: number; readonly trials?: number; readonly level?: number } = {},
): Explained<ConjugateModel> {
  const trials = options.trials ?? 1
  return conjugateModel(observations, {
    likelihood: trials === 1 ? 'bernoulli' : 'binomial',
    prior: beta(options.a ?? 1, options.b ?? 1),
    ...(trials === 1 ? {} : { fixed: { n: trials } }),
    ...(options.level === undefined ? {} : { level: options.level }),
  })
}

/**
 * The gamma-Poisson model: counts under a Gamma(shape, rate) prior on the Poisson rate. See {@link conjugateModel}.
 *
 * @param observations - Counts.
 * @param options - The prior's shape and rate (default 1, 1) and the level.
 * @returns As {@link conjugateModel}.
 */
export function gammaPoissonModel(
  observations: TensorLike,
  options: { readonly shape?: number; readonly rate?: number; readonly level?: number } = {},
): Explained<ConjugateModel> {
  return conjugateModel(observations, {
    likelihood: 'poisson',
    prior: gamma(options.shape ?? 1, options.rate ?? 1),
    ...(options.level === undefined ? {} : { level: options.level }),
  })
}

/**
 * The normal-normal model: observations with a known standard deviation under a normal prior on their mean. See
 * {@link conjugateModel}.
 *
 * @param observations - Values.
 * @param options - The prior's mean and sd (default 0, 1), the known sd (default 1) and the level.
 * @returns As {@link conjugateModel}.
 */
export function normalNormalModel(
  observations: TensorLike,
  options: { readonly mean?: number; readonly sd?: number; readonly sigma?: number; readonly level?: number } = {},
): Explained<ConjugateModel> {
  return conjugateModel(observations, {
    likelihood: 'normal',
    prior: normal(options.mean ?? 0, options.sd ?? 1),
    fixed: { scale: options.sigma ?? 1 },
    ...(options.level === undefined ? {} : { level: options.level }),
  })
}
