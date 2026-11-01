/**
 * Bayesian online linear regression as an estimator on the iterative base: a Gaussian prior on the weights, then one
 * observation at a time: the likelihood of the new point (a ridge in weight space, a line of weights that explain it
 * equally well) multiplies the current belief and the product, the posterior, is the prior for the next point. The
 * state is the posterior (mean, covariance, evidence) with the belief before the last point, so a step can show the
 * three things the update is made of: prior, likelihood, posterior. The update is the rank-one one of
 * `bayesianUpdate` (Bishop, PRML 3.3), so after `t` steps the state is the batch posterior of the first `t` rows.
 */

import type { Json } from '../base/index.ts'
import {
  type FittedFacts,
  type IterativeEstimator,
  iterative,
  type LearnData,
  matrixView,
  type Verbs,
} from '../learn/index.ts'
import { r2Score } from '../metrics/index.ts'
import { ddot, dgemv } from '../numerics/linalg/index.ts'
import { bool, dict, real } from '../space/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  derived,
  Log,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../trace/index.ts'
import { type BayesianPosterior, bayesianUpdate, priorPosterior } from './bayesian.ts'

/** The hyperparameters of {@link bayesianOnlineLinear}. */
export interface BayesianOnlineHyper {
  /** The precision of the zero-mean isotropic Gaussian prior on the weights (the prior's standard deviation is `1 / sqrt(alpha)`). */
  readonly alpha: number
  /** The precision (inverse variance) of the Gaussian noise on the targets, taken as known. */
  readonly beta: number
  /** Prepend a constant 1 to the features, so the first weight is the intercept (default true). */
  readonly intercept: boolean
}

/** The state: the posterior after `n` observations, the belief before the last one, and the last one. */
export type BayesianOnlineState = BayesianPosterior & {
  /** The posterior before the last observation (the prior of the last step). */
  readonly before: { readonly mean: number[]; readonly covariance: number[][] }
  /** The basis functions of the last observation (empty before any). */
  readonly phi: number[]
  /** The last target. */
  readonly y: number
}

/** The paths of the algorithm: the belief, the evidence, how surprising each point was, and the point. */
export type BayesianOnlinePath =
  | 'blr/mean'
  | 'blr/covariance'
  | 'blr/log-evidence'
  | 'blr/surprise'
  | 'blr/predictive-std'
  | 'blr/observation'

/** The contract of {@link bayesianOnlineAlgorithm}. */
export const bayesianOnlineContract: Contract<BayesianOnlineState, BayesianOnlinePath> = defineContract({
  'blr/mean': tensor({ role: 'param', shape: ['weights'], description: 'the posterior mean of the weights' }),
  'blr/covariance': tensor({
    role: 'param',
    shape: ['weights', 'weights'],
    description: 'the posterior covariance of the weights, row-major',
  }),
  'blr/log-evidence': scalar({ description: 'log p(y_1..y_t | x_1..x_t): the evidence so far', direction: 'higher' }),
  'blr/surprise': scalar({
    description: '-log p(y_t | y_1..y_{t-1}): how unlikely the new target was under the belief before it',
    direction: 'lower',
  }),
  'blr/predictive-std': scalar({ description: 'the standard deviation of the prediction of y_t, before seeing it' }),
  'blr/observation': tensor({
    role: 'output',
    shape: ['basis functions + 1'],
    description: 'the basis functions of the new point followed by its target',
  }),
})

function rowsOf(data: LearnData, intercept: boolean, who: string): { phi: number[][]; y: number[] } {
  const m = matrixView(data.x, who)
  const y = Array.from(toFlat(asTensor((data.y ?? []) as TensorLike)))
  if (y.length !== m.n) throw new Error(`${who}: ${m.n} rows of x but ${y.length} targets`)
  const phi = Array.from({ length: m.n }, (_, i) => {
    const row = Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))
    return intercept ? [1, ...row] : row
  })
  return { phi, y }
}

/**
 * Bayesian online linear regression as a steppable algorithm over the rows of `data` in order: each step multiplies the
 * current Gaussian belief on the weights by the likelihood of one observation and keeps the product.
 *
 * @param data - Features and targets.
 * @param hyper - The prior precision, the noise precision and whether there is an intercept.
 * @returns The algorithm.
 * @concept bayesian-linear-regression
 * @example
 * ```ts
 * import { bayesianOnlineAlgorithm } from 'mdml-engine/linear'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = bayesianOnlineAlgorithm({ x: [[0], [1], [2]], y: [1, 3, 5] }, { alpha: 1, beta: 100, intercept: true })
 * runAlgorithm(alg, { maxSteps: 3 }).state.mean.map((w) => Math.round(w * 10) / 10) // => [1, 2]
 * ```
 */
export function bayesianOnlineAlgorithm(
  data: LearnData,
  hyper: BayesianOnlineHyper,
): Algorithm<BayesianOnlineState, BayesianOnlinePath> {
  const { phi, y } = rowsOf(data, hyper.intercept, 'bayesianOnlineAlgorithm')
  const p = phi[0]?.length ?? 0
  const prior = { alpha: hyper.alpha, beta: hyper.beta }
  return {
    name: 'bayesian-online-linear',
    config: { alpha: hyper.alpha, beta: hyper.beta, intercept: hyper.intercept, rows: y.length },
    init: () => {
      const start = priorPosterior(p, prior)
      return { ...start, before: { mean: start.mean, covariance: start.covariance }, phi: [], y: 0 }
    },
    step: (s, ctx) => {
      const x = phi[s.n] as number[]
      const target = y[s.n] as number
      const next = bayesianUpdate(s, x, target, hyper.beta)
      const std = Math.sqrt(1 / hyper.beta + ddot(x, dgemv(Float64Array.from(s.covariance.flat()), p, p, x)))
      if (ctx.wants('blr/mean')) ctx.log('blr/mean', Float64Array.from(next.mean))
      if (ctx.wants('blr/covariance')) ctx.log('blr/covariance', Float64Array.from(next.covariance.flat()))
      if (ctx.wants('blr/log-evidence')) ctx.log('blr/log-evidence', next.logEvidence)
      if (ctx.wants('blr/surprise')) ctx.log('blr/surprise', s.logEvidence - next.logEvidence)
      if (ctx.wants('blr/predictive-std')) ctx.log('blr/predictive-std', std)
      if (ctx.wants('blr/observation')) ctx.log('blr/observation', Float64Array.from([...x, target]))
      return { ...next, before: { mean: s.mean, covariance: s.covariance }, phi: x, y: target }
    },
    done: (s) => (s.n >= y.length ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as BayesianOnlineState,
  }
}

/** The fitted parameters: the posterior after all rows. */
export type BayesianOnlineParams = {
  readonly n: number
  readonly mean: number[]
  readonly covariance: number[][]
  readonly logEvidence: number
  readonly alpha: number
  readonly beta: number
  readonly intercept: boolean
}

/** What a fitted online Bayesian linear regression offers. */
export interface BayesianOnlineVerbs extends Verbs, FittedFacts {
  /** The predictive mean at new inputs `[m]`. */
  readonly predict: (x: TensorLike) => Tensor
  /** The predictive mean and variance (noise included) at new inputs. */
  readonly predictive: (x: TensorLike) => { readonly mean: number[]; readonly variance: number[] }
  /** The posterior over the weights. */
  readonly weights: () => { readonly mean: number[]; readonly covariance: number[][] }
}

const withIntercept = (x: TensorLike, intercept: boolean, who: string): number[][] => {
  const m = matrixView(x, who)
  return Array.from({ length: m.n }, (_, i) => {
    const row = Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))
    return intercept ? [1, ...row] : row
  })
}

function predictiveAt(
  mean: readonly number[],
  covariance: readonly (readonly number[])[],
  beta: number,
  rows: readonly (readonly number[])[],
): { mean: number[]; variance: number[] } {
  const p = mean.length
  const flat = Float64Array.from(covariance.flat())
  return {
    mean: rows.map((r) => ddot(r, mean)),
    variance: rows.map((r) => 1 / beta + ddot(r, dgemv(flat, p, p, r))),
  }
}

// The lower Cholesky factor of a small symmetric positive-definite matrix.
function choleskyOf(a: readonly (readonly number[])[]): number[][] {
  const p = a.length
  const l = Array.from({ length: p }, () => new Array<number>(p).fill(0))
  for (let i = 0; i < p; i++)
    for (let j = 0; j <= i; j++) {
      let sum = a[i]?.[j] ?? 0
      for (let k = 0; k < j; k++) sum -= (l[i]?.[k] ?? 0) * (l[j]?.[k] ?? 0)
      ;(l[i] as number[])[j] = i === j ? Math.sqrt(Math.max(sum, 1e-300)) : sum / (l[j]?.[j] ?? 1)
    }
  return l
}

const gaussianDensity = (
  w: readonly number[],
  mean: readonly number[],
  cov: readonly (readonly number[])[],
): number => {
  const [a = 1, b = 0, c = 0, d = 1] = [cov[0]?.[0], cov[0]?.[1], cov[1]?.[0], cov[1]?.[1]]
  const det = a * d - b * c
  const dx = (w[0] ?? 0) - (mean[0] ?? 0)
  const dy = (w[1] ?? 0) - (mean[1] ?? 0)
  const q = (d * dx * dx - (b + c) * dx * dy + a * dy * dy) / det
  return Math.exp(-0.5 * q) / (2 * Math.PI * Math.sqrt(det))
}

/**
 * Bayesian online linear regression: a zero-mean Gaussian prior on the weights, then the data one observation at a time,
 * each update multiplying the belief by the likelihood of the point. `fit` returns the posterior after all rows;
 * `session` steps live and logs the paths of {@link bayesianOnlineContract}; the derived paths `model/weights` (the
 * prior, the likelihood of the last point and the posterior over two weights on a grid, which are the three panels of
 * the picture of an update) and `model/lines` (the predictive mean and band over a grid of inputs, and lines drawn from
 * the posterior) can be read at any step.
 *
 * @concept bayesian-linear-regression
 * @reference C. M. Bishop, Pattern Recognition and Machine Learning (2006), section 3.3.
 * @example
 * ```ts
 * import { bayesianOnlineLinear } from 'mdml-engine/linear'
 * const fitted = bayesianOnlineLinear.with({ beta: 100 }).fit({ x: [[0], [1], [2]], y: [1, 3, 5] })
 * fitted.weights().mean.map((w) => Math.round(w * 10) / 10) // => [1, 2]
 * ```
 */
export const bayesianOnlineLinear: IterativeEstimator<
  BayesianOnlineHyper,
  BayesianOnlineState,
  BayesianOnlineParams,
  BayesianOnlineVerbs,
  BayesianOnlinePath | 'model/weights' | 'model/lines'
> = iterative<
  BayesianOnlineHyper,
  BayesianOnlineState,
  BayesianOnlineParams,
  BayesianOnlineVerbs,
  BayesianOnlinePath | 'model/weights' | 'model/lines'
>({
  name: 'linear/bayesian-online',
  kind: 'regressor',
  defaults: { alpha: 1, beta: 4, intercept: true },
  hyperSpace: (h) =>
    dict({
      alpha: real(1e-3, 1e3, { log: true, default: h.alpha, label: 'prior precision of the weights' }),
      beta: real(1e-2, 1e4, { log: true, default: h.beta, label: 'noise precision' }),
      intercept: bool({ default: h.intercept, label: 'intercept' }),
    }),
  algorithm: (data, h) => bayesianOnlineAlgorithm(data, h),
  maxSteps: (_, data) => matrixView(data.x, 'bayesianOnlineLinear').n,
  params: (s, _data, h) => ({
    n: s.n,
    mean: s.mean,
    covariance: s.covariance,
    logEvidence: s.logEvidence,
    alpha: h.alpha,
    beta: h.beta,
    intercept: h.intercept,
  }),
  verbs: (p) => {
    const predictive = (x: TensorLike): { mean: number[]; variance: number[] } =>
      predictiveAt(p.mean, p.covariance, p.beta, withIntercept(x, p.intercept, 'bayesianOnlineLinear'))
    const predict = (x: TensorLike): Tensor => {
      const mean = predictive(x).mean
      return fromStorage(Float64Array.from(mean), [mean.length], 'float64')
    }
    return {
      predictive,
      predict,
      weights: () => ({ mean: p.mean, covariance: p.covariance }),
      nFeaturesIn: p.mean.length - (p.intercept ? 1 : 0),
      score: (data) => r2Score(data.y ?? [], predict(data.x)).estimate.value,
    }
  },
  contract: (_data, h) =>
    defineContract({
      ...bayesianOnlineContract,
      'model/weights': derived<BayesianOnlineState>({
        description:
          'for two weights: the prior (the belief before the last point), the likelihood of the last point as a function of the weights, and the posterior, each on a grid; arguments `{ range: [lo0, hi0, lo1, hi1], points }`',
        cost: 'dear',
        compute: (s, args) => {
          if (s.mean.length !== 2) throw new Error('model/weights: two weights are needed to draw them in the plane')
          const a = args as { range?: number[]; points?: number } | null | undefined
          const sd = 1 / Math.sqrt(h.alpha)
          const [lo0, hi0, lo1, hi1] = a?.range ?? [-3 * sd, 3 * sd, -3 * sd, 3 * sd]
          const n = a?.points ?? 61
          const axis = (lo: number, hi: number): number[] =>
            Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
          const w0 = axis(lo0 as number, hi0 as number)
          const w1 = axis(lo1 as number, hi1 as number)
          const on = (f: (w: number[]) => number): number[][] => w1.map((b) => w0.map((a0) => f([a0, b])))
          const observed = s.n > 0
          const norm = Math.sqrt(h.beta / (2 * Math.PI))
          return {
            w0,
            w1,
            n: s.n,
            prior: on((w) => gaussianDensity(w, s.before.mean, s.before.covariance)),
            likelihood: observed ? on((w) => norm * Math.exp(-0.5 * h.beta * (s.y - ddot(s.phi, w)) ** 2)) : null,
            posterior: on((w) => gaussianDensity(w, s.mean, s.covariance)),
            priorMean: s.before.mean,
            posteriorMean: s.mean,
            observation: observed ? { phi: s.phi, y: s.y } : null,
          } as unknown as Json
        },
      }),
      'model/lines': derived<BayesianOnlineState>({
        description:
          'for one input: the predictive mean and band over `x`, and `samples` lines drawn from the posterior (fixed draws, so the lines move smoothly as the belief tightens); arguments `{ x, samples }`',
        compute: (s, args) => {
          const a = args as { x?: number[]; samples?: number } | null | undefined
          const grid = a?.x ?? []
          const rows = grid.map((v) => (h.intercept ? [1, v] : [v]))
          const { mean, variance } = predictiveAt(s.mean, s.covariance, h.beta, rows)
          const draws = a?.samples ?? 12
          const l = choleskyOf(s.covariance)
          const p = s.mean.length
          const z = toFlat(asTensor(normalDraws(draws, p)))
          const samples = Array.from({ length: draws }, (_, k) => {
            const w = s.mean.map((m, i) => m + (l[i] ?? []).reduce((sum, lij, j) => sum + lij * (z[k * p + j] ?? 0), 0))
            return rows.map((r) => ddot(r, w))
          })
          return {
            x: grid,
            mean,
            std: variance.map(Math.sqrt),
            samples,
            n: s.n,
          } as unknown as Json
        },
      }),
    }),
  logs: {
    'blr/mean': Log(),
    'blr/covariance': Log(),
    'blr/log-evidence': Log(),
    'blr/surprise': Log(),
    'blr/predictive-std': Log(),
    'blr/observation': Log(),
  },
})

// Fixed standard normal draws `[count, p]`: the same every call, so a drawn line moves only as the belief does.
function normalDraws(count: number, p: number): Tensor {
  let state = 0x9e3779b9
  const next = (): number => {
    state = (Math.imul(state ^ (state >>> 15), 0x2c1b3c6d) + 0x297a2d39) | 0
    return ((state >>> 0) + 0.5) / 4294967296
  }
  const out = new Float64Array(count * p)
  for (let i = 0; i < out.length; i += 2) {
    const r = Math.sqrt(-2 * Math.log(next()))
    const t = 2 * Math.PI * next()
    out[i] = r * Math.cos(t)
    if (i + 1 < out.length) out[i + 1] = r * Math.sin(t)
  }
  return fromStorage(out, [count, p], 'float64')
}
