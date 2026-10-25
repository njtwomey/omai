/**
 * Sequential testing, where data are examined as they arrive: Wald's sequential probability ratio test (also as a
 * traced algorithm), Lan-DeMets alpha-spending boundaries for group-sequential trials, and always-valid inference for a
 * normal mean (mixture SPRT p-values and confidence sequences) that stays valid however often it is checked.
 */

import { MdmlError, type TestResult } from '../../base/index.ts'
import { brentq } from '../../numerics/roots/index.ts'
import { type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { normCdf, normPpf, normSf, sample, vec } from '../_util.ts'
import { testResult } from './result.ts'

/** The decision of a sequential test so far. */
export type SequentialDecision = 'reject H0' | 'accept H0' | 'continue'

/** Options of {@link sprt}. */
export interface SprtOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** The log-likelihood ratio log f1(x) / f0(x) of one observation. */
  readonly logLikelihoodRatio: (x: number) => number
  /** Type I error (default 0.05). */
  readonly alpha?: number
  /** Type II error (default 0.2). */
  readonly beta?: number
}

/** The state of {@link sprtAlgorithm}. */
export interface SprtState {
  readonly n: number
  readonly llr: number
  readonly path: readonly number[]
  readonly decision: SequentialDecision
}

/**
 * Wald's sequential probability ratio test as a traced algorithm: each step adds one observation's log-likelihood
 * ratio and stops when the sum leaves (log(beta / (1 - alpha)), log((1 - beta) / alpha)).
 *
 * @param observations - The data, in arrival order.
 * @param options - logLikelihoodRatio, alpha and beta.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { sprtAlgorithm, sprtNormal } from 'mdml-engine/stats/tests'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = sprtAlgorithm([2, 2, 2, 2], { logLikelihoodRatio: sprtNormal(0, 1, 1) })
 * runAlgorithm(alg, { maxSteps: 10 }).state.decision // => 'reject H0'
 * ```
 */
export function sprtAlgorithm(observations: TensorLike, options: SprtOptions): Algorithm<SprtState> {
  const x = sample('sprt', observations, 1)
  const alpha = options.alpha ?? 0.05
  const beta = options.beta ?? 0.2
  if (!(alpha > 0 && alpha < 1 && beta > 0 && beta < 1))
    throw new MdmlError('value', 'sprt: alpha and beta lie in (0, 1)')
  const upper = Math.log((1 - beta) / alpha)
  const lower = Math.log(beta / (1 - alpha))
  return {
    name: 'sprt',
    config: { alpha, beta, upper, lower },
    schema: channels({ metrics: { llr: { description: 'cumulative log-likelihood ratio' } } }),
    init: () => ({ n: 0, llr: 0, path: [], decision: 'continue' }),
    step: (s) => {
      const llr = s.llr + options.logLikelihoodRatio(x[s.n] ?? 0)
      const decision: SequentialDecision = llr >= upper ? 'reject H0' : llr <= lower ? 'accept H0' : 'continue'
      return { n: s.n + 1, llr, path: [...s.path, llr], decision }
    },
    done: (s) => (s.decision !== 'continue' ? 'converged' : s.n >= x.length ? 'completed' : undefined),
    observe: (s) => ({ metrics: { llr: s.llr } }),
  }
}

/** The result of {@link sprt}. */
export interface SprtResult extends TestResult {
  readonly decision: SequentialDecision
  /** Observations used before stopping. */
  readonly n: number
  /** The cumulative log-likelihood ratio after each observation. */
  readonly path: Tensor
  readonly boundaries: readonly [number, number]
}

/**
 * Wald's sequential probability ratio test: stop at the first observation whose cumulative log-likelihood ratio
 * crosses log((1 - beta) / alpha) (reject H0) or log(beta / (1 - alpha)) (accept H0). `statistic` is the final log
 * likelihood ratio; `pValue` is NaN (the SPRT decides rather than measures).
 *
 * @param observations - The data, in arrival order.
 * @param options - logLikelihoodRatio, alpha (default 0.05), beta (default 0.2), tracer.
 * @returns The decision, the number of observations used and the path.
 * @example
 * ```ts
 * import { sprt, sprtBernoulli } from 'mdml-engine/stats/tests'
 * sprt([1, 1, 1, 1, 1, 1, 1, 1], { logLikelihoodRatio: sprtBernoulli(0.3, 0.7) }).n // => 4
 * ```
 */
export function sprt(observations: TensorLike, options: SprtOptions): SprtResult {
  const alg = sprtAlgorithm(observations, options)
  const { logLikelihoodRatio: _l, alpha: _a, beta: _b, ...run } = options
  const r = runAlgorithm(alg, { ...run, maxSteps: Number.MAX_SAFE_INTEGER })
  const cfg = alg.config as { upper: number; lower: number }
  return {
    ...testResult({
      test: 'sequential probability ratio test',
      statistic: r.state.llr,
      pValue: Number.NaN,
      method: 'Wald',
    }),
    decision: r.state.decision,
    n: r.state.n,
    path: vec(r.state.path),
    boundaries: [cfg.lower, cfg.upper],
  }
}

/**
 * The log-likelihood ratio of one Bernoulli observation, p1 against p0.
 *
 * @param p0 - The null success probability.
 * @param p1 - The alternative.
 * @returns x => log f1(x) / f0(x).
 * @example
 * ```ts
 * import { sprtBernoulli } from 'mdml-engine/stats/tests'
 * sprtBernoulli(0.5, 0.5)(1) // => 0
 * ```
 */
export function sprtBernoulli(p0: number, p1: number): (x: number) => number {
  return (x) => (x === 1 ? Math.log(p1 / p0) : Math.log((1 - p1) / (1 - p0)))
}

/**
 * The log-likelihood ratio of one normal observation with known sigma, mean mu1 against mu0.
 *
 * @param mu0 - The null mean.
 * @param mu1 - The alternative mean.
 * @param sigma - The standard deviation.
 * @returns x => log f1(x) / f0(x).
 * @example
 * ```ts
 * import { sprtNormal } from 'mdml-engine/stats/tests'
 * sprtNormal(0, 1, 1)(0.5) // => 0
 * ```
 */
export function sprtNormal(mu0: number, mu1: number, sigma: number): (x: number) => number {
  return (x) => ((mu1 - mu0) * (x - (mu0 + mu1) / 2)) / (sigma * sigma)
}

/** A Lan-DeMets spending function family. */
export type SpendingKind = 'obrien-fleming' | 'pocock' | 'power' | 'hwang-shih-decani'

/**
 * A Lan-DeMets alpha-spending function: the cumulative type I error spent by information fraction t, approximating
 * O'Brien-Fleming (2 - 2 Phi(z_(1 - alpha/2) / sqrt t)) or Pocock (alpha log(1 + (e - 1) t)) boundaries, or the
 * Kim-DeMets power (alpha t^rho) and Hwang-Shih-DeCani (alpha (1 - e^(-gamma t)) / (1 - e^(-gamma))) families.
 *
 * @param kind - The family.
 * @param alpha - The total type I error.
 * @param options - rho (power, default 3) and gamma (Hwang-Shih-DeCani, default -4).
 * @returns t => alpha spent by t.
 * @example
 * ```ts
 * import { spendingFunction } from 'mdml-engine/stats/tests'
 * spendingFunction('pocock', 0.05)(1) // => 0.05
 * ```
 */
export function spendingFunction(
  kind: SpendingKind,
  alpha: number,
  options: { readonly rho?: number; readonly gamma?: number } = {},
): (t: number) => number {
  const z = normPpf(1 - alpha / 2)
  return (t) => {
    if (t <= 0) return 0
    const u = Math.min(1, t)
    switch (kind) {
      case 'obrien-fleming':
        return 2 * normSf(z / Math.sqrt(u))
      case 'pocock':
        return alpha * Math.log(1 + (Math.E - 1) * u)
      case 'power':
        return alpha * u ** (options.rho ?? 3)
      default: {
        const g = options.gamma ?? -4
        return g === 0 ? alpha * u : (alpha * (1 - Math.exp(-g * u))) / (1 - Math.exp(-g))
      }
    }
  }
}

/** A group-sequential design. */
export interface GroupSequentialDesign {
  readonly times: readonly number[]
  /** Cumulative alpha spent at each look. */
  readonly spent: readonly number[]
  /** Critical |z| (two-sided) or z (one-sided) at each look. */
  readonly boundaries: Tensor
  /** The nominal p-value threshold of each look's z test. */
  readonly nominalAlpha: Tensor
}

/**
 * Group-sequential efficacy boundaries from an alpha-spending function (Lan and DeMets 1983): at each look the
 * critical z whose crossing probability, given no earlier crossing, equals the alpha newly spent, computed by
 * recursive numerical integration of the score process (Armitage, McPherson and Rowe) on a Simpson grid. A two-sided
 * design is symmetric and spends alpha / 2 in each tail (so its O'Brien-Fleming boundaries for alpha 0.05 are those of
 * the one-sided alpha 0.025 design, as in Jennison and Turnbull's tables).
 *
 * @param times - Increasing information fractions in (0, 1].
 * @param options - alpha (default 0.05), kind (default `'obrien-fleming'`), sides (2, default, or 1), rho, gamma.
 * @returns The spent alpha, boundaries and nominal levels.
 * @example
 * ```ts
 * import { alphaSpending } from 'mdml-engine/stats/tests'
 * alphaSpending([1]).boundaries // => [1.959964]
 * ```
 */
export function alphaSpending(
  times: readonly number[],
  options: {
    readonly alpha?: number
    readonly kind?: SpendingKind
    readonly sides?: 1 | 2
    readonly rho?: number
    readonly gamma?: number
  } = {},
): GroupSequentialDesign {
  const alpha = options.alpha ?? 0.05
  const sides = options.sides ?? 2
  for (let i = 0; i < times.length; i++)
    if (!((times[i] ?? 0) > (i === 0 ? 0 : (times[i - 1] ?? 0)) && (times[i] ?? 0) <= 1))
      throw new MdmlError('value', 'alphaSpending: times must increase within (0, 1]')
  // A symmetric two-sided design spends alpha / 2 in each tail.
  const perSide = spendingFunction(options.kind ?? 'obrien-fleming', sides === 2 ? alpha / 2 : alpha, {
    ...(options.rho === undefined ? {} : { rho: options.rho }),
    ...(options.gamma === undefined ? {} : { gamma: options.gamma }),
  })
  const spent = times.map((t) => (sides === 2 ? 2 : 1) * perSide(t))
  const boundaries: number[] = []
  const G = 801
  // The sub-density of the score S on a grid over the continuation region of the previous look.
  let grid: Float64Array | undefined
  let dens: Float64Array | undefined
  let prevT = 0
  for (let k = 0; k < times.length; k++) {
    const t = times[k] ?? 1
    const inc = (spent[k] ?? 0) - (k === 0 ? 0 : (spent[k - 1] ?? 0))
    const dt = t - prevT
    const sd = Math.sqrt(dt)
    // Simpson weights of the previous grid.
    const crossing = (c: number): number => {
      const hiS = c * Math.sqrt(t)
      if (grid === undefined || dens === undefined) return sides === 2 ? 2 * normSf(hiS / sd) : normSf(hiS / sd)
      const h = ((grid[G - 1] ?? 0) - (grid[0] ?? 0)) / (G - 1)
      let s = 0
      for (let i = 0; i < G; i++) {
        const w = i === 0 || i === G - 1 ? 1 : i % 2 === 1 ? 4 : 2
        const u = grid[i] ?? 0
        const p = normSf((hiS - u) / sd) + (sides === 2 ? normCdf((-hiS - u) / sd) : 0)
        s += w * (dens[i] ?? 0) * p
      }
      return (s * h) / 3
    }
    let c: number
    if (!(inc > 0)) c = Number.POSITIVE_INFINITY
    else {
      const f = (z: number): number => crossing(z) - inc
      let hi = 10
      while (f(hi) > 0 && hi < 40) hi *= 1.5
      c = f(0) <= 0 ? 0 : brentq(f, 0, hi, { xtol: 1e-12 }).root
    }
    boundaries.push(c)
    // The continuation density at this look, on its own grid.
    const top = Number.isFinite(c) ? c * Math.sqrt(t) : 8 * Math.sqrt(t)
    const bottom = sides === 2 ? -top : -8 * Math.sqrt(t)
    const next = Float64Array.from({ length: G }, (_, i) => bottom + ((top - bottom) * i) / (G - 1))
    const nd = new Float64Array(G)
    for (let j = 0; j < G; j++) {
      const s = next[j] ?? 0
      if (grid === undefined || dens === undefined)
        nd[j] = Math.exp(-0.5 * (s / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))
      else {
        const h = ((grid[G - 1] ?? 0) - (grid[0] ?? 0)) / (G - 1)
        let acc = 0
        for (let i = 0; i < G; i++) {
          const w = i === 0 || i === G - 1 ? 1 : i % 2 === 1 ? 4 : 2
          const u = grid[i] ?? 0
          acc += w * (dens[i] ?? 0) * (Math.exp(-0.5 * ((s - u) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI)))
        }
        nd[j] = (acc * h) / 3
      }
    }
    grid = next
    dens = nd
    prevT = t
  }
  return {
    times: [...times],
    spent,
    boundaries: vec(boundaries),
    nominalAlpha: vec(boundaries.map((c) => (sides === 2 ? 2 : 1) * normSf(c))),
  }
}

/** Options of the normal-mean always-valid inference. */
export interface AlwaysValidOptions {
  /** The known standard deviation of an observation. */
  readonly sigma: number
  /** The null mean (default 0). */
  readonly mu0?: number
  /** The standard deviation of the normal mixing distribution over the effect (default sigma). */
  readonly tau?: number
}

/**
 * Always-valid p-values for a normal mean by the mixture SPRT (Johari, Pekelis and Walsh 2017): p_n = min(1, min_k<=n
 * 1 / Lambda_k) with the normal-mixture likelihood ratio Lambda_n = sqrt(sigma^2 / (sigma^2 + n tau^2)) exp(n^2 tau^2
 * (xbar - mu0)^2 / (2 sigma^2 (sigma^2 + n tau^2))); valid however the data are monitored and whenever one stops.
 *
 * @param x - The observations, in arrival order.
 * @param options - sigma, mu0 and tau.
 * @returns The always-valid p-value after each observation (non-increasing).
 * @example
 * ```ts
 * import { alwaysValidPValues } from 'mdml-engine/stats/tests'
 * alwaysValidPValues([0, 0, 0], { sigma: 1 }) // => [1, 1, 1]
 * ```
 */
export function alwaysValidPValues(x: TensorLike, options: AlwaysValidOptions): Tensor {
  const a = sample('alwaysValidPValues', x, 1)
  const s2 = options.sigma ** 2
  const tau2 = (options.tau ?? options.sigma) ** 2
  const mu0 = options.mu0 ?? 0
  let sum = 0
  let p = 1
  const out: number[] = []
  a.forEach((v, i) => {
    const n = i + 1
    sum += v
    const d = sum / n - mu0
    const logLambda = 0.5 * Math.log(s2 / (s2 + n * tau2)) + (n * n * tau2 * d * d) / (2 * s2 * (s2 + n * tau2))
    p = Math.min(p, Math.min(1, Math.exp(-logLambda)))
    out.push(p)
  })
  return vec(out)
}

/**
 * The mixture SPRT of a normal mean as a test: the always-valid p-value after the last observation.
 *
 * @param x - The observations.
 * @param options - sigma, mu0 and tau.
 * @returns The final mean as the statistic and the always-valid p-value.
 * @example
 * ```ts
 * import { msprt } from 'mdml-engine/stats/tests'
 * msprt([1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.7, 1.4], { sigma: 1 }).pValue < 0.05 // => true
 * ```
 */
export function msprt(x: TensorLike, options: AlwaysValidOptions): TestResult {
  const p = toFlat(alwaysValidPValues(x, options))
  const a = sample('msprt', x, 1)
  let m = 0
  for (const v of a) m += v / a.length
  return testResult({ test: 'mixture SPRT', statistic: m, pValue: p.at(-1) ?? 1, method: 'always valid' })
}

/**
 * A confidence sequence for a normal mean with known sigma (Robbins' normal mixture; Howard, Ramdas, McAuliffe and
 * Sekhon 2021): xbar_n -+ sqrt((n sigma^2 + rho) log((n sigma^2 + rho) / (rho alpha^2))) / n, covering the mean at
 * every n simultaneously with probability 1 - alpha. rho is tuned to be tightest near n = optimiseAt.
 *
 * @param x - The observations.
 * @param options - sigma, alpha (default 0.05) and optimiseAt (default 100).
 * @returns The lower and upper bounds after each observation.
 * @example
 * ```ts
 * import { confidenceSequence } from 'mdml-engine/stats/tests'
 * const cs = confidenceSequence([0.1, -0.2, 0.3], { sigma: 1 })
 * cs.lower.shape // => [3]
 * ```
 */
export function confidenceSequence(
  x: TensorLike,
  options: { readonly sigma: number; readonly alpha?: number; readonly optimiseAt?: number },
): { readonly lower: Tensor; readonly upper: Tensor } {
  const a = sample('confidenceSequence', x, 1)
  const alpha = options.alpha ?? 0.05
  const s2 = options.sigma ** 2
  const m = options.optimiseAt ?? 100
  const L = -2 * Math.log(alpha)
  const rho = (s2 * m) / (L + Math.log(L + 1))
  const lower: number[] = []
  const upper: number[] = []
  let sum = 0
  a.forEach((v, i) => {
    const n = i + 1
    sum += v
    const V = n * s2
    const r = Math.sqrt((V + rho) * Math.log((V + rho) / (rho * alpha * alpha))) / n
    lower.push(sum / n - r)
    upper.push(sum / n + r)
  })
  return { lower: vec(lower), upper: vec(upper) }
}
