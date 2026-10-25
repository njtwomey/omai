/**
 * Differential privacy, which bounds how much any single person's data can change a released result, diffprivlib and
 * opacus style. The Laplace, Gaussian (classic and analytic calibration) and exponential mechanisms, randomised
 * response, basic, advanced and heterogeneous composition of privacy budgets, and the Renyi-DP accountant for DP-SGD
 * with the subsampled Gaussian mechanism.
 */

import { MdmlError } from '../base/index.ts'
import { erf, logNdtr } from '../numerics/special/index.ts'
import { bernoulli, categorical, type Key, laplace, normal } from '../random/index.ts'
import { add, asTensor, mul, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import type { FitOptions } from './estimator.ts'

function positive(v: number, name: string, who: string): number {
  if (!(v > 0 && Number.isFinite(v)))
    throw new MdmlError('value', `${who}: ${name} must be positive and finite, got ${v}`)
  return v
}

function probability(v: number, name: string, who: string): number {
  if (!(v > 0 && v < 1)) throw new MdmlError('value', `${who}: ${name} must lie in (0, 1), got ${v}`)
  return v
}

/** Options of the additive-noise mechanisms. */
export interface NoiseOptions {
  /** The query's sensitivity (L1 for Laplace, L2 for Gaussian). */
  readonly sensitivity: number
  readonly epsilon: number
}

/**
 * The Laplace mechanism (Dwork, McSherry, Nissim and Smith 2006): add Laplace(0, sensitivity / epsilon) noise to each
 * value, which is epsilon-differentially private for a query of that L1 sensitivity.
 *
 * @param key - The key.
 * @param value - The query's answer (any shape).
 * @param options - sensitivity and epsilon.
 * @returns The noisy answer.
 * @concept differential-privacy
 * @example
 * ```ts
 * import { laplaceMechanism } from 'mdml-engine/learn'
 * import { key } from 'mdml-engine/random'
 * laplaceMechanism(key(0), [10, 20], { sensitivity: 1, epsilon: 1 }).shape // => [2]
 * ```
 */
export function laplaceMechanism(key: Key, value: TensorLike, options: NoiseOptions): Tensor {
  const scale =
    positive(options.sensitivity, 'sensitivity', 'laplaceMechanism') /
    positive(options.epsilon, 'epsilon', 'laplaceMechanism')
  const v = asTensor(value) as Tensor
  return add(v, laplace(key, v.shape, 0, scale)) as Tensor
}

/**
 * The classic Gaussian mechanism's noise scale (Dwork and Roth 2014, theorem A.1): sigma = sqrt(2 ln(1.25 / delta))
 * sensitivity / epsilon, valid for epsilon < 1 (diffprivlib's Gaussian).
 *
 * @param options - sensitivity (L2), epsilon in (0, 1) and delta in (0, 1).
 * @returns sigma.
 * @throws MdmlError with code `domain` for epsilon >= 1 (use {@link analyticGaussianScale}).
 * @example
 * ```ts
 * import { gaussianScale } from 'mdml-engine/learn'
 * Math.round(gaussianScale({ sensitivity: 1, epsilon: 0.5, delta: 1e-5 }) * 1e6) / 1e6 // => 9.689611
 * ```
 */
export function gaussianScale(options: NoiseOptions & { readonly delta: number }): number {
  const eps = positive(options.epsilon, 'epsilon', 'gaussianScale')
  if (eps >= 1)
    throw new MdmlError('domain', 'gaussianScale: the classic bound needs epsilon < 1; use analyticGaussianScale')
  return (
    (Math.sqrt(2 * Math.log(1.25 / probability(options.delta, 'delta', 'gaussianScale'))) *
      positive(options.sensitivity, 'sensitivity', 'gaussianScale')) /
    eps
  )
}

const phi = (v: number): number => (1 + (toFlat(erf(v / Math.SQRT2))[0] ?? 0)) / 2

/**
 * The analytic Gaussian mechanism's noise scale (Balle and Wang 2018, algorithm 1; diffprivlib's GaussianAnalytic):
 * the smallest sigma that is (epsilon, delta)-private by the exact privacy profile of the Gaussian, found by
 * bisection; valid for any epsilon > 0 and tighter than the classic bound.
 *
 * @param options - sensitivity (L2), epsilon and delta.
 * @returns sigma.
 * @example
 * ```ts
 * import { analyticGaussianScale } from 'mdml-engine/learn'
 * Math.round(analyticGaussianScale({ sensitivity: 1, epsilon: 1, delta: 1e-5 }) * 1e6) / 1e6 // => 3.730632
 * ```
 */
export function analyticGaussianScale(options: NoiseOptions & { readonly delta: number }): number {
  const eps = positive(options.epsilon, 'epsilon', 'analyticGaussianScale')
  const delta = probability(options.delta, 'delta', 'analyticGaussianScale')
  const bPlus = (v: number): number => phi(Math.sqrt(eps * v)) - Math.exp(eps) * phi(-Math.sqrt(eps * (v + 2))) - delta
  const bMinus = (v: number): number =>
    phi(-Math.sqrt(eps * v)) - Math.exp(eps) * phi(-Math.sqrt(eps * (v + 2))) - delta
  const delta0 = bPlus(0)
  let alpha = 1
  if (delta0 !== 0) {
    const target = delta0 < 0 ? bPlus : bMinus
    let left = 0
    let right = 1
    for (let guard = 0; target(left) * target(right) > 0 && guard < 2000; guard++) {
      left = right
      right *= 2
    }
    let oldSize = (right - left) * 2
    for (let guard = 0; oldSize > right - left && guard < 2000; guard++) {
      oldSize = right - left
      const middle = (right + left) / 2
      if (target(middle) * target(left) <= 0) right = middle
      if (target(middle) * target(right) <= 0) left = middle
    }
    alpha = Math.sqrt(1 + (left + right) / 4) + (delta0 < 0 ? -1 : 1) * Math.sqrt((left + right) / 4)
  }
  return (alpha * positive(options.sensitivity, 'sensitivity', 'analyticGaussianScale')) / Math.sqrt(2 * eps)
}

/**
 * The Gaussian mechanism: add N(0, sigma^2) noise with the analytic (default) or classic calibration.
 *
 * @param key - The key.
 * @param value - The query's answer.
 * @param options - sensitivity (L2), epsilon, delta and the calibration.
 * @returns The noisy answer.
 * @example
 * ```ts
 * import { gaussianMechanism } from 'mdml-engine/learn'
 * import { key } from 'mdml-engine/random'
 * gaussianMechanism(key(0), [1, 2, 3], { sensitivity: 1, epsilon: 1, delta: 1e-5 }).shape // => [3]
 * ```
 */
export function gaussianMechanism(
  key: Key,
  value: TensorLike,
  options: NoiseOptions & { readonly delta: number; readonly calibration?: 'analytic' | 'classic' },
): Tensor {
  const sigma = options.calibration === 'classic' ? gaussianScale(options) : analyticGaussianScale(options)
  const v = asTensor(value) as Tensor
  return add(v, mul(normal(key, v.shape), sigma)) as Tensor
}

/**
 * The selection probabilities of the exponential mechanism (McSherry and Talwar 2007; diffprivlib's Exponential):
 * proportional to exp(epsilon u / (2 sensitivity)), or exp(epsilon u / sensitivity) for monotonic utilities, times an
 * optional base measure.
 *
 * @param utilities - The utility of each candidate.
 * @param options - epsilon, sensitivity, monotonic (default false) and measure.
 * @returns The probabilities.
 * @example
 * ```ts
 * import { exponentialProbabilities } from 'mdml-engine/learn'
 * exponentialProbabilities([0, Math.LN2 * 2], { epsilon: 1, sensitivity: 1 }).map((p) => Math.round(p * 1e12) / 1e12) // => [0.333333333333, 0.666666666667]
 * ```
 */
export function exponentialProbabilities(
  utilities: readonly number[],
  options: {
    readonly epsilon: number
    readonly sensitivity: number
    readonly monotonic?: boolean
    readonly measure?: readonly number[]
  },
): number[] {
  if (utilities.length === 0) throw new MdmlError('value', 'exponentialProbabilities: no candidates')
  const scale =
    positive(options.epsilon, 'epsilon', 'exponentialProbabilities') /
    positive(options.sensitivity, 'sensitivity', 'exponentialProbabilities') /
    (options.monotonic === true ? 1 : 2)
  const top = utilities.reduce((a, b) => Math.max(a, b))
  const w = utilities.map((u, i) => Math.exp(scale * (u - top)) * (options.measure?.[i] ?? 1))
  const total = w.reduce((a, b) => a + b, 0)
  return w.map((v) => v / total)
}

/**
 * The exponential mechanism: privately select a candidate, favouring high utility.
 *
 * @param key - The key.
 * @param utilities - The utility of each candidate.
 * @param options - epsilon, sensitivity, monotonic and measure.
 * @returns The selected index.
 * @example
 * ```ts
 * import { exponentialMechanism } from 'mdml-engine/learn'
 * import { key } from 'mdml-engine/random'
 * // With a huge epsilon the best candidate is (almost) always chosen.
 * exponentialMechanism(key(1), [1, 5, 2], { epsilon: 100, sensitivity: 1 }) // => 1
 * ```
 */
export function exponentialMechanism(
  key: Key,
  utilities: readonly number[],
  options: {
    readonly epsilon: number
    readonly sensitivity: number
    readonly monotonic?: boolean
    readonly measure?: readonly number[]
  },
): number {
  const p = exponentialProbabilities(utilities, options)
  return toFlat(categorical(key, p, [1]))[0] ?? 0
}

/**
 * Randomised response (Warner 1965): report each private bit truthfully with probability e^epsilon / (1 + e^epsilon)
 * and flipped otherwise, which is epsilon-locally differentially private.
 *
 * @param key - The key.
 * @param bits - The private bits (0 or 1).
 * @param epsilon - The privacy level.
 * @returns The reported bits.
 * @example
 * ```ts
 * import { randomisedResponse } from 'mdml-engine/learn'
 * import { key } from 'mdml-engine/random'
 * randomisedResponse(key(0), [1, 0, 1], 50) // => [1, 0, 1]
 * ```
 */
export function randomisedResponse(key: Key, bits: readonly number[], epsilon: number): number[] {
  const keep = Math.exp(positive(epsilon, 'epsilon', 'randomisedResponse')) / (1 + Math.exp(epsilon))
  const flips = toFlat(bernoulli(key, 1 - keep, [bits.length]))
  return bits.map((b, i) => {
    if (b !== 0 && b !== 1) throw new MdmlError('value', 'randomisedResponse: bits must be 0 or 1')
    return flips[i] === 1 ? 1 - b : b
  })
}

/**
 * The unbiased estimate of the true proportion of ones from randomised responses: (mean - q) / (p - q) with p the
 * probability of a truthful report and q = 1 - p.
 *
 * @param reports - The reported bits.
 * @param epsilon - The privacy level they were reported with.
 * @returns The estimated proportion (it can fall outside [0, 1] by chance).
 * @example
 * ```ts
 * import { randomisedResponseEstimate } from 'mdml-engine/learn'
 * // With p = 0.75 a proportion of 0.5 is reported as 0.5 whatever the truth... and 0.75 reports mean all ones.
 * randomisedResponseEstimate([1, 1, 1, 0], Math.log(3)) // => 1
 * ```
 */
export function randomisedResponseEstimate(reports: readonly number[], epsilon: number): number {
  const p = Math.exp(positive(epsilon, 'epsilon', 'randomisedResponseEstimate')) / (1 + Math.exp(epsilon))
  const mean = reports.reduce((a, b) => a + b, 0) / reports.length
  return (mean - (1 - p)) / (2 * p - 1)
}

/** A privacy cost (epsilon, delta). */
export interface PrivacyBudget {
  readonly epsilon: number
  readonly delta: number
}

/**
 * Basic sequential composition: epsilons and deltas add.
 *
 * @param budgets - The mechanisms' costs.
 * @returns The total.
 * @example
 * ```ts
 * import { basicComposition } from 'mdml-engine/learn'
 * basicComposition([{ epsilon: 0.5, delta: 0 }, { epsilon: 1, delta: 1e-6 }]) // => { epsilon: 1.5, delta: 0.000001 }
 * ```
 */
export function basicComposition(budgets: readonly PrivacyBudget[]): PrivacyBudget {
  return budgets.reduce((t, b) => ({ epsilon: t.epsilon + b.epsilon, delta: t.delta + b.delta }), {
    epsilon: 0,
    delta: 0,
  })
}

/**
 * Advanced composition of k (epsilon, delta) mechanisms (Dwork, Rothblum and Vadhan 2010): for any slack delta',
 * the composition is (sqrt(2k ln(1/delta')) epsilon + k epsilon (e^epsilon - 1), k delta + delta')-private.
 *
 * @param budget - One mechanism's cost.
 * @param k - The number of compositions.
 * @param slack - delta'.
 * @returns The total.
 * @example
 * ```ts
 * import { advancedComposition } from 'mdml-engine/learn'
 * const t = advancedComposition({ epsilon: 0.1, delta: 0 }, 100, 1e-5)
 * t.epsilon < 10 // => true
 * ```
 */
export function advancedComposition(budget: PrivacyBudget, k: number, slack: number): PrivacyBudget {
  probability(slack, 'slack', 'advancedComposition')
  const e = budget.epsilon
  return {
    epsilon: Math.sqrt(2 * k * Math.log(1 / slack)) * e + k * e * Math.expm1(e),
    delta: k * budget.delta + slack,
  }
}

/**
 * The tightest of three composition bounds for heterogeneous mechanisms with a delta slack, as diffprivlib's
 * BudgetAccountant (Kairouz, Oh and Viswanath 2015, theorem 3.5, and Dwork-Rothblum-Vadhan): the naive sum, and
 * sum eps_i tanh(eps_i / 2) + sqrt(2 sum eps_i^2 ln(1 / slack)) and its ln(e + sqrt(sum eps^2) / slack) variant; the
 * deltas compose as 1 - (1 - slack) prod (1 - delta_i).
 *
 * @param budgets - The mechanisms' costs.
 * @param slack - The delta slack (0 for the naive bound).
 * @returns The total.
 * @example
 * ```ts
 * import { heterogeneousComposition } from 'mdml-engine/learn'
 * heterogeneousComposition([{ epsilon: 1, delta: 0 }, { epsilon: 1, delta: 0 }], 0).epsilon // => 2
 * ```
 */
export function heterogeneousComposition(budgets: readonly PrivacyBudget[], slack: number): PrivacyBudget {
  if (!(slack >= 0 && slack < 1)) throw new MdmlError('value', 'heterogeneousComposition: slack must lie in [0, 1)')
  let sum = 0
  let expSum = 0
  let sqSum = 0
  for (const { epsilon } of budgets) {
    sum += epsilon
    expSum += ((1 - Math.exp(-epsilon)) * epsilon) / (1 + Math.exp(-epsilon))
    sqSum += epsilon * epsilon
  }
  const deltas = [slack, ...budgets.map((b) => b.delta)].sort((a, b) => a - b)
  let delta = 0
  for (const d of deltas) delta += d - delta * d
  if (slack === 0) return { epsilon: sum, delta }
  const drv = expSum + Math.sqrt(2 * sqSum * Math.log(1 / slack))
  const kov = expSum + Math.sqrt(2 * sqSum * Math.log(Math.E + Math.sqrt(sqSum) / slack))
  return { epsilon: Math.min(sum, drv, kov), delta }
}

function logAdd(a: number, b: number): number {
  const lo = Math.min(a, b)
  const hi = Math.max(a, b)
  if (lo === Number.NEGATIVE_INFINITY) return hi
  return Math.log1p(Math.exp(lo - hi)) + hi
}

function logSub(a: number, b: number): number {
  if (a < b) throw new MdmlError('value', 'rdp: log subtraction would be negative')
  if (b === Number.NEGATIVE_INFINITY) return a
  if (a === b) return Number.NEGATIVE_INFINITY
  const d = Math.expm1(a - b)
  return Number.isFinite(d) ? Math.log(d) + b : a
}

/** The binomial coefficient C(alpha, i) for real alpha, by the falling product (negative for i > alpha). */
function binomialReal(alpha: number, i: number): number {
  let c = 1
  for (let t = 0; t < i; t++) c *= (alpha - t) / (t + 1)
  return c
}

const logErfc = (x: number): number => Math.LN2 + (toFlat(logNdtr(-x * Math.SQRT2))[0] ?? 0)

/** log A_alpha of the sampled Gaussian mechanism (Mironov, Talwar and Zhang 2019), integer or fractional alpha. */
function logA(q: number, sigma: number, alpha: number): number {
  if (Number.isInteger(alpha)) {
    let la = Number.NEGATIVE_INFINITY
    for (let i = 0; i <= alpha; i++) {
      const lc = Math.log(binomialReal(alpha, i)) + i * Math.log(q) + (alpha - i) * Math.log(1 - q)
      la = logAdd(la, lc + (i * i - i) / (2 * sigma * sigma))
    }
    return la
  }
  let a0 = Number.NEGATIVE_INFINITY
  let a1 = Number.NEGATIVE_INFINITY
  const z0 = sigma * sigma * Math.log(1 / q - 1) + 0.5
  for (let i = 0; i < 100000; i++) {
    const coef = binomialReal(alpha, i)
    const lc = Math.log(Math.abs(coef))
    const j = alpha - i
    const t0 = lc + i * Math.log(q) + j * Math.log(1 - q)
    const t1 = lc + j * Math.log(q) + i * Math.log(1 - q)
    const e0 = Math.log(0.5) + logErfc((i - z0) / (Math.SQRT2 * sigma))
    const e1 = Math.log(0.5) + logErfc((z0 - j) / (Math.SQRT2 * sigma))
    const s0 = t0 + (i * i - i) / (2 * sigma * sigma) + e0
    const s1 = t1 + (j * j - j) / (2 * sigma * sigma) + e1
    if (coef > 0) {
      a0 = logAdd(a0, s0)
      a1 = logAdd(a1, s1)
    } else {
      a0 = logSub(a0, s0)
      a1 = logSub(a1, s1)
    }
    if (Math.max(s0, s1) < -30) break
  }
  return logAdd(a0, a1)
}

/** The default RDP orders (opacus's): 1.1 to 10.9 by 0.1, then 12 to 63. */
export const DEFAULT_RDP_ORDERS: readonly number[] = [
  ...Array.from({ length: 99 }, (_, x) => 1 + (x + 1) / 10),
  ...Array.from({ length: 52 }, (_, x) => 12 + x),
]

/**
 * The Renyi differential privacy of the subsampled Gaussian mechanism at each order (Mironov 2017; Mironov, Talwar
 * and Zhang 2019; opacus's compute_rdp): Poisson sampling with rate q, noise multiplier sigma, composed over steps.
 *
 * @param q - The sampling rate in [0, 1].
 * @param noiseMultiplier - sigma, the noise standard deviation over the clipping norm.
 * @param steps - The number of steps.
 * @param orders - The RDP orders (default {@link DEFAULT_RDP_ORDERS}).
 * @returns The RDP epsilon at each order.
 * @example
 * ```ts
 * import { subsampledGaussianRdp } from 'mdml-engine/learn'
 * // Without subsampling the Gaussian mechanism has RDP alpha / (2 sigma^2).
 * subsampledGaussianRdp(1, 2, 1, [2, 4]) // => [0.25, 0.5]
 * ```
 */
export function subsampledGaussianRdp(
  q: number,
  noiseMultiplier: number,
  steps: number,
  orders: readonly number[] = DEFAULT_RDP_ORDERS,
): number[] {
  if (!(q >= 0 && q <= 1)) throw new MdmlError('value', 'subsampledGaussianRdp: q must lie in [0, 1]')
  return orders.map((alpha) => {
    if (q === 0) return 0
    if (noiseMultiplier === 0) return Number.POSITIVE_INFINITY
    if (q === 1) return (steps * alpha) / (2 * noiseMultiplier * noiseMultiplier)
    if (!Number.isFinite(alpha)) return Number.POSITIVE_INFINITY
    return (steps * logA(q, noiseMultiplier, alpha)) / (alpha - 1)
  })
}

/**
 * Convert RDP to (epsilon, delta) at the best order (opacus's get_privacy_spent; Balle et al. 2020's improved
 * conversion): epsilon = rdp - (ln delta + ln alpha) / (alpha - 1) + ln((alpha - 1) / alpha).
 *
 * @param orders - The orders.
 * @param rdp - The RDP at each order.
 * @param delta - The target delta.
 * @returns epsilon and the best order.
 * @example
 * ```ts
 * import { rdpToDp } from 'mdml-engine/learn'
 * rdpToDp([2, 4], [0.25, 0.5], 1e-5).order // => 4
 * ```
 */
export function rdpToDp(
  orders: readonly number[],
  rdp: readonly number[],
  delta: number,
): { readonly epsilon: number; readonly order: number } {
  if (orders.length !== rdp.length) throw new MdmlError('shape', 'rdpToDp: one RDP value per order')
  probability(delta, 'delta', 'rdpToDp')
  let best = -1
  let bestEps = Number.POSITIVE_INFINITY
  orders.forEach((a, i) => {
    const eps = (rdp[i] ?? Number.NaN) - (Math.log(delta) + Math.log(a)) / (a - 1) + Math.log((a - 1) / a)
    if (!Number.isNaN(eps) && (best < 0 || eps < bestEps)) {
      best = i
      bestEps = eps
    }
  })
  return best < 0
    ? { epsilon: Number.POSITIVE_INFINITY, order: Number.NaN }
    : { epsilon: bestEps, order: orders[best] ?? Number.NaN }
}

/** One phase of DP-SGD for {@link rdpAccountant}: a sampling rate, a noise multiplier and a number of steps. */
export interface GaussianEvent {
  readonly sampleRate: number
  readonly noiseMultiplier: number
  readonly steps: number
}

/** The result of {@link rdpAccountant}. */
export interface RdpAccount {
  readonly epsilon: number
  readonly order: number
  /** The accumulated RDP at each order. */
  readonly rdp: readonly number[]
  readonly orders: readonly number[]
}

/**
 * An RDP accountant for DP-SGD (Abadi et al. 2016; opacus's RDPAccountant): add up the subsampled Gaussian RDP of each
 * training phase and convert to (epsilon, delta). A traced algorithm with one step per phase, recording the running
 * epsilon.
 *
 * @param events - The training phases.
 * @param options - delta, orders (default {@link DEFAULT_RDP_ORDERS}) and tracing.
 * @returns epsilon, the best order and the RDP curve.
 * @example
 * ```ts
 * import { rdpAccountant } from 'mdml-engine/learn'
 * const a = rdpAccountant([{ sampleRate: 0.01, noiseMultiplier: 1.1, steps: 1000 }], { delta: 1e-5 })
 * a.epsilon > 0 && a.epsilon < 5 // => true
 * ```
 */
export function rdpAccountant(
  events: readonly GaussianEvent[],
  options: { readonly delta: number; readonly orders?: readonly number[] } & Pick<
    FitOptions,
    'tracer' | 'policy' | 'parent'
  >,
): RdpAccount {
  const orders = options.orders ?? DEFAULT_RDP_ORDERS
  type State = { readonly i: number; readonly rdp: readonly number[] }
  const algorithm: Algorithm<State> = {
    name: 'rdp-accountant',
    config: { events: events.length, delta: options.delta },
    schema: channels({
      metrics: { epsilon: { description: 'epsilon spent so far at the target delta', direction: 'lower' } },
    }),
    init: () => ({ i: 0, rdp: orders.map(() => 0) }),
    step: (s) => {
      const e = events[s.i] as GaussianEvent
      const add = subsampledGaussianRdp(e.sampleRate, e.noiseMultiplier, e.steps, orders)
      return { i: s.i + 1, rdp: s.rdp.map((v, k) => v + (add[k] ?? 0)) }
    },
    done: (s) => (s.i >= events.length ? 'completed' : undefined),
    observe: (s) => ({ metrics: s.i === 0 ? {} : { epsilon: rdpToDp(orders, s.rdp, options.delta).epsilon } }),
  }
  const run = {
    ...(options.parent === undefined ? {} : { parent: options.parent }),
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.policy === undefined ? {} : { policy: options.policy }),
  }
  const { state } = runAlgorithm(algorithm, { ...run, maxSteps: events.length })
  const r = rdpToDp(orders, state.rdp, options.delta)
  return { epsilon: r.epsilon, order: r.order, rdp: state.rdp, orders }
}
