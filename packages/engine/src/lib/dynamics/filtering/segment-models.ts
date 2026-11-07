/**
 * The segment models of Bayesian online changepoint detection: a conjugate model of the observations within one
 * segment, with the knobs of its prior, that gives an exact posterior predictive. The models are the kinds of shift a
 * series can show: a shift of the mean (Gaussian with known noise), of the variance (Gaussian with known mean), of both
 * (normal-gamma), of the rate of counts (Poisson), of binary frequencies (Bernoulli), of the coefficient of an AR(1)
 * process and of the intercept and slope of a linear trend. A run's posterior is kept as plain numbers (its sufficient
 * statistics), so the state of an algorithm that holds many runs serialises; `priorRun` wraps the same arithmetic as a
 * closure for one-off use.
 */

import { LOG_2PI, MdmlError } from '../../base/index.ts'
import { lgamma } from '../../numerics/special/index.ts'
import { item } from '../../tensor/index.ts'
import { positive } from './_check.ts'

/**
 * A conjugate model of the observations within a segment (with its prior), which gives an exact posterior predictive
 * for Bayesian online changepoint detection: Gaussian with known variance (normal prior on the mean), Gaussian with
 * known mean (inverse-gamma prior on the variance), Gaussian with both unknown (normal-gamma, Student-t predictive),
 * Poisson (gamma prior, negative binomial predictive), Bernoulli (beta prior), and two regressions whose predictive
 * reads the previous value or the time: an AR(1) coefficient with a normal prior and known noise, and a linear trend
 * (intercept and slope, normal priors, known noise) in the time since the segment began.
 */
export type ConjugateModel =
  | {
      readonly kind: 'normal-known-variance'
      readonly sigma: number
      readonly priorMean: number
      readonly priorStd: number
    }
  | { readonly kind: 'normal-known-mean'; readonly mean: number; readonly alpha: number; readonly beta: number }
  | {
      readonly kind: 'normal-gamma'
      readonly mu0: number
      readonly kappa0: number
      readonly alpha0: number
      readonly beta0: number
    }
  | { readonly kind: 'poisson-gamma'; readonly shape: number; readonly rate: number }
  | { readonly kind: 'bernoulli-beta'; readonly alpha: number; readonly beta: number }
  | { readonly kind: 'ar1-known-noise'; readonly sigma: number; readonly priorStd: number }
  | {
      readonly kind: 'trend-known-noise'
      readonly sigma: number
      readonly priorMean: number
      readonly interceptStd: number
      readonly slopeStd: number
    }

/** The kinds of {@link ConjugateModel}. */
export type ConjugateKind = ConjugateModel['kind']

/** What the regression models read besides the value: the step and the value before it (0 at the first step). */
export interface RunInput {
  /** The index of the observation in the series. */
  readonly t: number
  /** The observation before it (0 for the first). */
  readonly prev: number
}

/** The input a run is asked about when the caller gives none: the first step, so only the regressions care. */
const FIRST: RunInput = { t: 0, prev: 0 }

/**
 * The posterior of one run as plain numbers: how many observations it holds, the index of its first observation (the
 * origin of the trend model's time) and the model's sufficient statistics, in an order {@link segmentOps} knows.
 */
export interface RunStats {
  readonly length: number
  readonly start: number
  readonly stats: readonly number[]
}

/** The operations of a segment model over {@link RunStats}: pure functions of the numbers, nothing hidden. */
export interface SegmentOps {
  /** The run of a segment that starts at step `start`: the prior. */
  fresh(start: number): RunStats
  /** log p(y | run) for the next observation. */
  logPredictive(run: RunStats, y: number, at: RunInput): number
  /** E[y | run]. */
  predictiveMean(run: RunStats, at: RunInput): number
  /**
   * The variance of the predictive. A Student-t predictive with at most two degrees of freedom has none; its squared
   * scale stands in, so the spread of a mixture of runs stays finite.
   */
  predictiveVariance(run: RunStats, at: RunInput): number
  /** The run after one more observation. */
  update(run: RunStats, y: number, at: RunInput): RunStats
}

const lg = (x: number): number => item(lgamma(x))

function studentLog(y: number, nu: number, loc: number, scale2: number): number {
  return (
    lg((nu + 1) / 2) -
    lg(nu / 2) -
    0.5 * Math.log(nu * Math.PI * scale2) -
    ((nu + 1) / 2) * Math.log1p((y - loc) ** 2 / (nu * scale2))
  )
}

/** The variance of a Student-t with `nu` degrees of freedom and squared scale `scale2`, the scale where it is infinite. */
function studentVariance(nu: number, scale2: number): number {
  return nu > 2 ? (scale2 * nu) / (nu - 2) : scale2
}

function gaussianLog(y: number, mean: number, variance: number): number {
  return -0.5 * (LOG_2PI + Math.log(variance) + (y - mean) ** 2 / variance)
}

function finite(who: string, name: string, v: number): void {
  if (!Number.isFinite(v)) throw new MdmlError('value', `${who}: ${name} must be finite`)
}

const make = (start: number, length: number, stats: readonly number[]): RunStats => ({ start, length, stats })

/**
 * The operations of a conjugate segment model: the prior run, the predictive log density, mean and variance of the next
 * observation, and the update by one observation, all over the plain numbers of {@link RunStats}.
 *
 * @param model - The model and its prior.
 * @returns Its operations.
 * @throws MdmlError with code `value` for a scale, shape or rate that is not positive, or a prior that is not finite.
 * @concept bayesian-changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * const ops = filtering.segmentOps({ kind: 'poisson-gamma', shape: 2, rate: 1 })
 * const run = ops.update(ops.update(ops.fresh(0), 3, { t: 0, prev: 0 }), 5, { t: 1, prev: 3 })
 * [run.length, ops.predictiveMean(run, { t: 2, prev: 5 })] // => [2, 3.3333333333333335]
 * ```
 */
export function segmentOps(model: ConjugateModel): SegmentOps {
  const who = `segmentOps(${model.kind})`
  switch (model.kind) {
    case 'normal-known-variance': {
      positive(who, 'sigma', model.sigma)
      positive(who, 'priorStd', model.priorStd)
      finite(who, 'priorMean', model.priorMean)
      const s2 = model.sigma ** 2
      // stats: [precision, precision-weighted sum]
      const moments = (r: RunStats): { m: number; v: number } => {
        const [precision = 1, weighted = 0] = r.stats
        return { m: weighted / precision, v: 1 / precision + s2 }
      }
      return {
        fresh: (start) => make(start, 0, [1 / model.priorStd ** 2, model.priorMean / model.priorStd ** 2]),
        logPredictive: (r, y) => {
          const { m, v } = moments(r)
          return -0.5 * (LOG_2PI + Math.log(v) + (y - m) ** 2 / v)
        },
        predictiveMean: (r) => moments(r).m,
        predictiveVariance: (r) => moments(r).v,
        update: (r, y) => {
          const [precision = 1, weighted = 0] = r.stats
          return make(r.start, r.length + 1, [precision + 1 / s2, weighted + y / s2])
        },
      }
    }
    case 'normal-known-mean': {
      positive(who, 'alpha', model.alpha)
      positive(who, 'beta', model.beta)
      finite(who, 'mean', model.mean)
      const mu = model.mean
      // stats: [shape a, scale b] of the inverse-gamma on the variance
      return {
        fresh: (start) => make(start, 0, [model.alpha, model.beta]),
        logPredictive: (r, y) => {
          const [a = 1, b = 1] = r.stats
          return studentLog(y, 2 * a, mu, b / a)
        },
        predictiveMean: () => mu,
        predictiveVariance: (r) => {
          const [a = 1, b = 1] = r.stats
          return studentVariance(2 * a, b / a)
        },
        update: (r, y) => {
          const [a = 1, b = 1] = r.stats
          return make(r.start, r.length + 1, [a + 0.5, b + 0.5 * (y - mu) ** 2])
        },
      }
    }
    case 'normal-gamma': {
      positive(who, 'kappa0', model.kappa0)
      positive(who, 'alpha0', model.alpha0)
      positive(who, 'beta0', model.beta0)
      finite(who, 'mu0', model.mu0)
      // stats: [mu, kappa, a, b]
      const scale2 = (r: RunStats): number => {
        const [, kappa = 1, a = 1, b = 1] = r.stats
        return (b * (kappa + 1)) / (a * kappa)
      }
      return {
        fresh: (start) => make(start, 0, [model.mu0, model.kappa0, model.alpha0, model.beta0]),
        logPredictive: (r, y) => {
          const [mu = 0, , a = 1] = r.stats
          return studentLog(y, 2 * a, mu, scale2(r))
        },
        predictiveMean: (r) => r.stats[0] ?? 0,
        predictiveVariance: (r) => studentVariance(2 * (r.stats[2] ?? 1), scale2(r)),
        update: (r, y) => {
          const [mu = 0, kappa = 1, a = 1, b = 1] = r.stats
          return make(r.start, r.length + 1, [
            (kappa * mu + y) / (kappa + 1),
            kappa + 1,
            a + 0.5,
            b + (kappa * (y - mu) ** 2) / (2 * (kappa + 1)),
          ])
        },
      }
    }
    case 'poisson-gamma': {
      positive(who, 'shape', model.shape)
      positive(who, 'rate', model.rate)
      // stats: [shape a, rate b] of the gamma on the Poisson rate
      return {
        fresh: (start) => make(start, 0, [model.shape, model.rate]),
        logPredictive: (r, y) => {
          const [a = 1, b = 1] = r.stats
          return lg(a + y) - lg(a) - lg(y + 1) + a * Math.log(b / (b + 1)) - y * Math.log(b + 1)
        },
        predictiveMean: (r) => (r.stats[0] ?? 1) / (r.stats[1] ?? 1),
        predictiveVariance: (r) => {
          const [a = 1, b = 1] = r.stats
          return (a * (b + 1)) / (b * b)
        },
        update: (r, y) => {
          const [a = 1, b = 1] = r.stats
          return make(r.start, r.length + 1, [a + y, b + 1])
        },
      }
    }
    case 'bernoulli-beta': {
      positive(who, 'alpha', model.alpha)
      positive(who, 'beta', model.beta)
      // stats: [alpha a, beta b] of the beta on the probability of a one
      return {
        fresh: (start) => make(start, 0, [model.alpha, model.beta]),
        logPredictive: (r, y) => {
          const [a = 1, b = 1] = r.stats
          return Math.log(y > 0.5 ? a / (a + b) : b / (a + b))
        },
        predictiveMean: (r) => (r.stats[0] ?? 1) / ((r.stats[0] ?? 1) + (r.stats[1] ?? 1)),
        predictiveVariance: (r) => {
          const [a = 1, b = 1] = r.stats
          const p = a / (a + b)
          return p * (1 - p)
        },
        update: (r, y) => {
          const [a = 1, b = 1] = r.stats
          return make(r.start, r.length + 1, [a + (y > 0.5 ? 1 : 0), b + (y > 0.5 ? 0 : 1)])
        },
      }
    }
    case 'ar1-known-noise': {
      positive(who, 'sigma', model.sigma)
      positive(who, 'priorStd', model.priorStd)
      const s2 = model.sigma ** 2
      // Bayesian regression of y_t on y_(t-1) with a N(0, priorStd^2) coefficient. stats: [precision, sum of prev * y / s2]
      return {
        fresh: (start) => make(start, 0, [1 / model.priorStd ** 2, 0]),
        logPredictive: (r, y, at) => {
          const [precision = 1, xy = 0] = r.stats
          return gaussianLog(y, (xy / precision) * at.prev, s2 + (at.prev * at.prev) / precision)
        },
        predictiveMean: (r, at) => ((r.stats[1] ?? 0) / (r.stats[0] ?? 1)) * at.prev,
        predictiveVariance: (r, at) => s2 + (at.prev * at.prev) / (r.stats[0] ?? 1),
        update: (r, y, at) => {
          const [precision = 1, xy = 0] = r.stats
          return make(r.start, r.length + 1, [precision + (at.prev * at.prev) / s2, xy + (at.prev * y) / s2])
        },
      }
    }
    case 'trend-known-noise': {
      positive(who, 'sigma', model.sigma)
      positive(who, 'interceptStd', model.interceptStd)
      positive(who, 'slopeStd', model.slopeStd)
      finite(who, 'priorMean', model.priorMean)
      const s2 = model.sigma ** 2
      // Bayesian regression on [1, t - start] with independent normal priors on the intercept and the slope.
      // stats: the precision matrix [a, b, c] (symmetric 2x2) and the vector q = precision * mean.
      const solved = (r: RunStats): { m0: number; m1: number; ia: number; ib: number; ic: number } => {
        const [a = 1, b = 0, c = 1, q0 = 0, q1 = 0] = r.stats
        const det = a * c - b * b
        const [ia, ib, ic] = [c / det, -b / det, a / det]
        return { m0: ia * q0 + ib * q1, m1: ib * q0 + ic * q1, ia, ib, ic }
      }
      return {
        fresh: (start) =>
          make(start, 0, [
            1 / model.interceptStd ** 2,
            0,
            1 / model.slopeStd ** 2,
            model.priorMean / model.interceptStd ** 2,
            0,
          ]),
        logPredictive: (r, y, at) => {
          const { m0, m1, ia, ib, ic } = solved(r)
          const u = at.t - r.start
          return gaussianLog(y, m0 + m1 * u, s2 + ia + 2 * ib * u + ic * u * u)
        },
        predictiveMean: (r, at) => {
          const { m0, m1 } = solved(r)
          return m0 + m1 * (at.t - r.start)
        },
        predictiveVariance: (r, at) => {
          const { ia, ib, ic } = solved(r)
          const u = at.t - r.start
          return s2 + ia + 2 * ib * u + ic * u * u
        },
        update: (r, y, at) => {
          const [a = 1, b = 0, c = 1, q0 = 0, q1 = 0] = r.stats
          const u = at.t - r.start
          return make(r.start, r.length + 1, [a + 1 / s2, b + u / s2, c + (u * u) / s2, q0 + y / s2, q1 + (u * y) / s2])
        },
      }
    }
  }
}

/** The posterior of one run: the predictive log density and mean of the next value, and the update by one value. */
export interface RunPosterior {
  readonly length: number
  /** log p(y | run); the regression models read `at` (default: the first step). */
  logPredictive(y: number, at?: RunInput): number
  predictiveMean(at?: RunInput): number
  /** The standard deviation of the predictive (see {@link SegmentOps.predictiveVariance}). */
  predictiveStd(at?: RunInput): number
  update(y: number, at?: RunInput): RunPosterior
}

function posterior(ops: SegmentOps, run: RunStats): RunPosterior {
  return {
    length: run.length,
    logPredictive: (y, at = FIRST) => ops.logPredictive(run, y, at),
    predictiveMean: (at = FIRST) => ops.predictiveMean(run, at),
    predictiveStd: (at = FIRST) => Math.sqrt(ops.predictiveVariance(run, at)),
    update: (y, at = FIRST) => posterior(ops, ops.update(run, y, at)),
  }
}

/**
 * The prior run of a conjugate model (no observations yet), as a closure over {@link segmentOps}.
 *
 * @param model - The model.
 * @returns The run posterior before any observation.
 * @concept bayesian-changepoint-detection
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * filtering.priorRun({ kind: 'bernoulli-beta', alpha: 1, beta: 1 }).predictiveMean() // => 0.5
 * ```
 */
export function priorRun(model: ConjugateModel): RunPosterior {
  const ops = segmentOps(model)
  return posterior(ops, ops.fresh(0))
}
