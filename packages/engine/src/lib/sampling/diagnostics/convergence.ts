/**
 * Convergence diagnostics for MCMC chains, matching ArviZ: R-hat (the potential scale reduction factor, which compares
 * between- and within-chain variance to say whether chains agree), the effective sample size in its bulk, tail and
 * other variants, and the Monte Carlo standard error of posterior summaries.
 */

import { MdmlError, quantileOf } from '../../base/index.ts'
import { beta } from '../../dists/index.ts'
import type { TensorLike } from '../../tensor/index.ts'
import { toFlat } from '../../tensor/index.ts'
import {
  chainOf,
  type Draws,
  drawsOf,
  fold,
  invalid,
  medianOf,
  rowAutocovariance,
  splitHalves,
  zScale,
} from './_draws.ts'

function mean(x: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < x.length; i++) s += x[i] ?? 0
  return s / x.length
}

function variance(x: ArrayLike<number>, ddof: number): number {
  const m = mean(x)
  let s = 0
  for (let i = 0; i < x.length; i++) s += ((x[i] ?? 0) - m) ** 2
  return s / (x.length - ddof)
}

/** The potential scale reduction of `[chain, draw]` draws (Gelman-Rubin, ArviZ `_rhat`). */
function rhatCore(d: Draws): number {
  const n = d.draws
  const means = new Float64Array(d.chains)
  const vars = new Float64Array(d.chains)
  for (let c = 0; c < d.chains; c++) {
    const row = chainOf(d, c)
    means[c] = mean(row)
    vars[c] = variance(row, 1)
  }
  const between = n * variance(means, 1)
  const within = mean(vars)
  return Math.sqrt((between / within + n - 1) / n)
}

/** The ways to compute R-hat (ArviZ `rhat` methods). */
export type RhatMethod = 'rank' | 'split' | 'folded' | 'zScale' | 'identity'

/**
 * The potential scale reduction factor R-hat of `[chain, draw]` draws, ArviZ's `rhat`. The default `rank` method is
 * the rank-normalised split R-hat of Vehtari et al. (2021): chains are split in half, draws are replaced by normal
 * scores of their pooled ranks, and the larger of the R-hats of the scores and of the folded scores |x - median| is
 * reported, so it detects differences in location and in scale and works with heavy tails. `split` is the classic
 * split R-hat, `folded` the folded part alone, `zScale` the rank-normalised part alone and `identity` the original
 * Gelman-Rubin statistic without splitting. Values below 1.01 indicate convergence.
 *
 * @param x - Draws `[chain, draw]` (at least two chains and four draws).
 * @param options - The method (default `rank`).
 * @returns R-hat, NaN for too few chains or draws or for draws containing NaN.
 * @throws MdmlError with code `shape` for inputs that are not `[chain, draw]` or `[draw]`.
 * @concept r-hat
 * @reference A. Vehtari, A. Gelman, D. Simpson, B. Carpenter and P.-C. Burkner, Rank-normalization, folding, and
 *   localization: an improved R-hat for assessing convergence of MCMC (Bayesian Analysis 2021).
 * @example
 * ```ts
 * import { rhat } from 'mdml-engine/sampling'
 * // Two chains stuck in different places: R-hat far above 1.01 (the rank-normalised form is bounded).
 * const stuck = [[0, 0.1, -0.1, 0.05, 0, -0.05], [5, 5.1, 4.9, 5.05, 5, 4.95]]
 * rhat(stuck) > 1.5 // => true
 * rhat(stuck, { method: 'split' }) > 30 // => true
 * Number.isNaN(rhat([[1, 2, 3]])) // => true
 * ```
 */
export function rhat(x: TensorLike, options: { readonly method?: RhatMethod } = {}): number {
  const d = drawsOf(x, 'rhat')
  const method = options.method ?? 'rank'
  // As ArviZ: every method needs two chains before splitting.
  if (invalid(d, 2)) return Number.NaN
  switch (method) {
    case 'rank': {
      const s = splitHalves(d)
      return Math.max(rhatCore(zScale(s)), rhatCore(zScale(fold(s))))
    }
    case 'split':
      return rhatCore(splitHalves(d))
    case 'folded':
      return rhatCore(zScale(fold(splitHalves(d))))
    case 'zScale':
      return rhatCore(zScale(splitHalves(d)))
    case 'identity':
      return rhatCore(d)
    default:
      throw new MdmlError('value', `rhat: unknown method ${String(method)}`)
  }
}

/**
 * The effective sample size of `[chain, draw]` draws from the multi-chain autocorrelation (ArviZ `_ess`): Geyer's
 * initial positive and monotone sequences truncate the sum of autocorrelations, with the between-chain variance in
 * the variance estimate.
 */
function essCore(d: Draws, relative: boolean): number {
  const { chains, draws: n } = d
  let lo = Number.POSITIVE_INFINITY
  let hi = Number.NEGATIVE_INFINITY
  for (const v of d.data) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  if (hi - lo < 1e-15) return d.data.length
  const acov = rowAutocovariance(d)
  const lag = (t: number): number => {
    let s = 0
    for (let c = 0; c < chains; c++) s += acov[c * n + t] ?? 0
    return s / chains
  }
  const chainMeans = Float64Array.from({ length: chains }, (_, c) => mean(chainOf(d, c)))
  const meanVar = (lag(0) * n) / (n - 1)
  let varPlus = (meanVar * (n - 1)) / n
  if (chains > 1) varPlus += variance(chainMeans, 1)
  const rho = new Float64Array(n)
  let even = 1
  rho[0] = even
  let odd = 1 - (meanVar - lag(1)) / varPlus
  rho[1] = odd
  // Geyer's initial positive sequence.
  let t = 1
  while (t < n - 3 && even + odd > 0) {
    even = 1 - (meanVar - lag(t + 1)) / varPlus
    odd = 1 - (meanVar - lag(t + 2)) / varPlus
    if (even + odd >= 0) {
      rho[t + 1] = even
      rho[t + 2] = odd
    }
    t += 2
  }
  const maxT = t - 2
  if (even > 0) rho[maxT + 1] = even
  // Geyer's initial monotone sequence.
  t = 1
  while (t <= maxT - 2) {
    if ((rho[t + 1] ?? 0) + (rho[t + 2] ?? 0) > (rho[t - 1] ?? 0) + (rho[t] ?? 0)) {
      rho[t + 1] = ((rho[t - 1] ?? 0) + (rho[t] ?? 0)) / 2
      rho[t + 2] = rho[t + 1] ?? 0
    }
    t += 2
  }
  const total = chains * n
  let tau = -1
  for (let k = 0; k <= maxT; k++) tau += 2 * (rho[k] ?? 0)
  tau += rho[maxT + 1] ?? 0
  tau = Math.max(tau, 1 / Math.log10(total))
  for (const r of rho) if (Number.isNaN(r)) return Number.NaN
  return (relative ? 1 : total) / tau
}

/** The ways to compute an effective sample size (ArviZ `ess` methods). */
export type EssMethod =
  | 'bulk'
  | 'tail'
  | 'mean'
  | 'sd'
  | 'quantile'
  | 'median'
  | 'mad'
  | 'local'
  | 'folded'
  | 'zScale'
  | 'identity'

/** Options of {@link ess}. */
export interface EssOptions {
  /** The method (default `bulk`). */
  readonly method?: EssMethod
  /**
   * The probability of `quantile`, the tail probability of `tail` (both tails p and 1 - p; default 0.05) or the
   * interval of `local`.
   */
  readonly prob?: number | readonly [number, number]
  /** Divide by the number of draws (default false). */
  readonly relative?: boolean
}

function indicator(d: Draws, test: (v: number) => boolean): Draws {
  return { ...d, data: d.data.map((v) => (test(v) ? 1 : 0)) }
}

function essQuantile(d: Draws, prob: number, relative: boolean): number {
  const q = quantileOf(d.data, prob)
  return essCore(splitHalves(indicator(d, (v) => v <= q)), relative)
}

function scalarProb(prob: EssOptions['prob'], method: string): number {
  if (typeof prob !== 'number') throw new MdmlError('value', `ess: the ${method} method needs a probability`)
  return prob
}

/**
 * The effective sample size of `[chain, draw]` draws, ArviZ's `ess`. `bulk` (the default) is the split ESS of
 * rank-normalised draws, for the centre of the distribution; `tail` the smaller ESS of the indicators x <= q_p and
 * x <= q_{1-p} (p = 0.05 by default), for interval endpoints; `mean` and `sd` the split ESS of the draws and of the
 * squared deviations; `quantile` and `median` that of x <= q; `local` that of q_a <= x <= q_b; `mad` that of the
 * median absolute deviation indicator; `folded` that of the rank-normalised folded draws; `zScale` that of the
 * rank-normalised draws (bulk); `identity` the ESS without splitting. Constant draws give the number of draws.
 *
 * @param x - Draws `[chain, draw]` or `[draw]` (at least four draws).
 * @param options - The method, a probability for `tail`, `quantile` and `local`, and `relative`.
 * @returns The effective sample size (relative to the number of draws with `relative`), NaN for too few draws or
 *   draws containing NaN.
 * @throws MdmlError with code `value` when `quantile` or `local` lacks its probability.
 * @concept effective-sample-size
 * @reference A. Vehtari et al., Rank-normalization, folding, and localization (Bayesian Analysis 2021); C. J. Geyer,
 *   Practical Markov chain Monte Carlo (Statistical Science 1992).
 * @example
 * ```ts
 * import { ess } from 'mdml-engine/sampling'
 * // Alternating draws are anti-correlated: more effective draws (19.3) than draws (16).
 * ess([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], { method: 'identity' }) > 16 // => true
 * ess([3, 3, 3, 3, 3, 3]) // => 6
 * ```
 */
export function ess(x: TensorLike, options: EssOptions = {}): number {
  const d = drawsOf(x, 'ess')
  const method = options.method ?? 'bulk'
  const relative = options.relative ?? false
  if (invalid(d, 1)) return Number.NaN
  switch (method) {
    case 'bulk':
    case 'zScale':
      return essCore(zScale(splitHalves(d)), relative)
    case 'tail': {
      const p = options.prob ?? 0.05
      const [a, b] = typeof p === 'number' ? ([Math.min(p, 1 - p), Math.max(p, 1 - p)] as const) : p
      return Math.min(essQuantile(d, a, relative), essQuantile(d, b, relative))
    }
    case 'mean':
      return essCore(splitHalves(d), relative)
    case 'sd': {
      const m = mean(d.data)
      return essCore(splitHalves({ ...d, data: d.data.map((v) => (v - m) ** 2) }), relative)
    }
    case 'quantile':
      return essQuantile(d, scalarProb(options.prob, 'quantile'), relative)
    case 'median':
      return essQuantile(d, 0.5, relative)
    case 'local': {
      const p = options.prob
      if (typeof p === 'number' || p === undefined) {
        throw new MdmlError('value', 'ess: the local method needs an interval [low, high] of probabilities')
      }
      const lo = quantileOf(d.data, p[0])
      const hi = quantileOf(d.data, p[1])
      return essCore(splitHalves(indicator(d, (v) => lo <= v && v <= hi)), relative)
    }
    case 'mad': {
      const m = medianOf(d.data)
      const dev = d.data.map((v) => Math.abs(v - m))
      const md = medianOf(dev)
      return essCore(zScale(splitHalves({ ...d, data: dev.map((v) => (v <= md ? 1 : 0)) })), relative)
    }
    case 'folded':
      return essCore(zScale(fold(splitHalves(d))), relative)
    case 'identity':
      return essCore(d, relative)
    default:
      throw new MdmlError('value', `ess: unknown method ${String(method)}`)
  }
}

/** The quantities whose Monte Carlo standard error {@link mcse} estimates. */
export type McseMethod = 'mean' | 'sd' | 'median' | 'quantile'

/**
 * The Monte Carlo standard error of a posterior summary from `[chain, draw]` draws, ArviZ's `mcse`: for the `mean`,
 * sd / sqrt(ESS_mean); for the `sd`, the delta method on the variance with the ESS of the squared deviations; for a
 * `quantile` (and the `median`), half the width of the order statistics bracketing a central 68% interval of the
 * Beta(ESS p + 1, ESS (1 - p) + 1) distribution of the quantile's rank, with ESS that of the quantile indicator.
 *
 * @param x - Draws `[chain, draw]` or `[draw]` (at least four draws).
 * @param options - The quantity (default `mean`) and the probability of `quantile`.
 * @returns The standard error, NaN for too few draws or draws containing NaN.
 * @throws MdmlError with code `value` when `quantile` lacks its probability.
 * @concept monte-carlo-standard-error
 * @reference A. Vehtari et al., Rank-normalization, folding, and localization (Bayesian Analysis 2021), section 4.
 * @example
 * ```ts
 * import { ess, mcse } from 'mdml-engine/sampling'
 * const x = [0.3, -1.2, 0.8, 1.9, -0.4, 0.1, -0.9, 1.1]
 * const sd = Math.sqrt(x.reduce((s, v) => s + (v - 0.2125) ** 2, 0) / 7)
 * Math.abs(mcse(x) - sd / Math.sqrt(ess(x, { method: 'mean' }))) < 1e-12 // => true
 * ```
 */
export function mcse(x: TensorLike, options: { readonly method?: McseMethod; readonly prob?: number } = {}): number {
  const d = drawsOf(x, 'mcse')
  if (invalid(d, 1)) return Number.NaN
  const method = options.method ?? 'mean'
  switch (method) {
    case 'mean':
      return Math.sqrt(variance(d.data, 1)) / Math.sqrt(essCore(splitHalves(d), false))
    case 'sd': {
      const m = mean(d.data)
      const sq = d.data.map((v) => (v - m) ** 2)
      const essSq = essCore(splitHalves({ ...d, data: sq }), false)
      const evar = mean(sq)
      const varvar = (mean(sq.map((v) => v * v)) - evar * evar) / essSq
      return Math.sqrt(varvar / evar / 4)
    }
    case 'median':
      return mcseQuantile(d, 0.5)
    case 'quantile': {
      const p = options.prob
      if (p === undefined) throw new MdmlError('value', 'mcse: the quantile method needs a probability')
      return mcseQuantile(d, p)
    }
    default:
      throw new MdmlError('value', `mcse: unknown method ${String(method)}`)
  }
}

function mcseQuantile(d: Draws, p: number): number {
  const e = essQuantile(d, p, false)
  const sorted = Float64Array.from(d.data).sort()
  const size = sorted.length
  let lo = 0
  let hi = size - 1
  if (Number.isFinite(e)) {
    const ppf = toFlat(beta(e * p + 1, e * (1 - p) + 1).ppf([0.1586553, 0.8413447]))
    const a = (ppf[0] ?? Number.NaN) * size - 1
    const b = (ppf[1] ?? Number.NaN) * size - 1
    lo = Math.floor(Number.isNaN(a) ? 0 : Math.max(a, 0))
    hi = Math.ceil(Number.isNaN(b) ? size - 1 : Math.min(b, size - 1))
  }
  return ((sorted[hi] ?? Number.NaN) - (sorted[lo] ?? Number.NaN)) / 2
}
