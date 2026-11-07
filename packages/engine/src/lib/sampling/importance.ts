/**
 * Importance sampling, which estimates expectations under a target from weighted draws of a simpler proposal: weight
 * normalisation, Kish's effective sample size, self-normalised estimates and log-evidence with standard errors, the
 * full `importanceSampling` routine, and Pareto-smoothed importance sampling (PSIS, as ArviZ) with its k-hat diagnostic
 * of whether the weights can be trusted.
 */

import { type Estimate, logSumExpOf, MdmlError } from '../base/index.ts'
import type { Key } from '../random/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { ess as essOf } from './diagnostics/convergence.ts'
import type { ChainProposal, LogDensity } from './target.ts'

function values(x: TensorLike, name: string): Float64Array {
  const v = Float64Array.from(toFlat(asTensor(x) as Tensor))
  if (v.length === 0) throw new MdmlError('value', `${name}: no values`)
  return v.map((u) => (Number.isNaN(u) ? Number.NEGATIVE_INFINITY : u))
}

/**
 * Normalise log importance weights: w_i = exp(l_i) / sum_j exp(l_j), computed stably (a softmax). NaN reads as
 * -Infinity (zero weight); when every weight is zero the result is NaN.
 *
 * @param logWeights - Unnormalised log weights.
 * @returns The normalised weights.
 * @throws MdmlError with code `value` for an empty input.
 * @concept importance-sampling
 * @example
 * ```ts
 * import { normaliseLogWeights } from 'mdml-engine/sampling'
 * Array.from(normaliseLogWeights([0, Math.log(3)])) // => [0.25, 0.75]
 * ```
 */
export function normaliseLogWeights(logWeights: TensorLike): Float64Array {
  const l = values(logWeights, 'normaliseLogWeights')
  const z = logSumExpOf(l)
  return l.map((v) => Math.exp(v - z))
}

/**
 * Kish's effective sample size of importance weights, (sum w)^2 / sum w^2: n for equal weights, 1 when one weight
 * dominates.
 *
 * @param logWeights - Log weights (normalised or not).
 * @returns The effective sample size.
 * @concept importance-sampling
 * @example
 * ```ts
 * import { importanceEss } from 'mdml-engine/sampling'
 * importanceEss([0, 0, 0, 0]) // => 4
 * ```
 */
export function importanceEss(logWeights: TensorLike): number {
  const w = normaliseLogWeights(logWeights)
  let s = 0
  for (const v of w) s += v * v
  return 1 / s
}

/**
 * The self-normalised importance sampling estimate of E_pi[f] = sum_i w_i f(x_i) / sum_i w_i, with its delta-method
 * standard error sqrt(sum_i wbar_i^2 (f_i - mu)^2) (wbar normalised) and Kish's ESS. It needs pi only up to a
 * constant, and is consistent but biased at O(1/n).
 *
 * @param values - f(x_i) for each draw.
 * @param logWeights - log pi(x_i) - log q(x_i), unnormalised.
 * @returns The estimate (value, standard error, normal 95% interval) with the effective sample size.
 * @throws MdmlError with code `shape` when the lengths differ.
 * @concept importance-sampling
 * @reference A. B. Owen, Monte Carlo theory, methods and examples (2013), chapter 9.
 * @example
 * ```ts
 * import { selfNormalisedEstimate } from 'mdml-engine/sampling'
 * selfNormalisedEstimate([1, 2, 3], [0, 0, Math.log(2)]).value // => 2.25
 * ```
 */
export function selfNormalisedEstimate(
  values: TensorLike,
  logWeights: TensorLike,
): Estimate & { readonly ess: number } {
  const f = Float64Array.from(toFlat(asTensor(values) as Tensor))
  const w = normaliseLogWeights(logWeights)
  if (f.length !== w.length) {
    throw new MdmlError('shape', `selfNormalisedEstimate: ${f.length} values for ${w.length} weights`)
  }
  let mu = 0
  for (let i = 0; i < f.length; i++) mu += (w[i] ?? 0) * (f[i] ?? 0)
  let v = 0
  let s2 = 0
  for (let i = 0; i < f.length; i++) {
    v += (w[i] ?? 0) ** 2 * ((f[i] ?? 0) - mu) ** 2
    s2 += (w[i] ?? 0) ** 2
  }
  const se = Math.sqrt(v)
  return {
    value: mu,
    standardError: se,
    interval: [mu - 1.959964 * se, mu + 1.959964 * se],
    level: 0.95,
    method: 'self-normalised importance sampling',
    ess: 1 / s2,
  }
}

/**
 * The log of the normalising constant from importance weights, log Zhat = log (1/n) sum_i exp(l_i) (unbiased for Z on
 * the natural scale), with the delta-method standard error sd(w) / (sqrt(n) mean(w)) on the log scale.
 *
 * @param logWeights - log pi~(x_i) - log q(x_i) for an unnormalised pi~.
 * @returns The estimate of log Z.
 * @concept importance-sampling
 * @example
 * ```ts
 * import { logEvidence } from 'mdml-engine/sampling'
 * logEvidence([Math.log(2), Math.log(4)]).value === Math.log(3) // => true
 * ```
 */
export function logEvidence(logWeights: TensorLike): Estimate {
  const l = values(logWeights, 'logEvidence')
  const n = l.length
  const value = logSumExpOf(l) - Math.log(n)
  const w = l.map((v) => Math.exp(v - value))
  let s = 0
  for (const v of w) s += (v - 1) ** 2
  const se = n > 1 ? Math.sqrt(s / (n - 1)) / Math.sqrt(n) : Number.NaN
  return { value, standardError: se, method: 'importance sampling' }
}

/** The result of {@link importanceSampling}. */
export interface ImportanceResult {
  /** The draws from the proposal `[n, dim]`. */
  readonly draws: Tensor<'float64'>
  /** log pi(x_i) - log q(x_i). */
  readonly logWeights: Tensor<'float64'>
  /** The normalised weights. */
  readonly weights: Tensor<'float64'>
  readonly logEvidence: Estimate
  /** Kish's effective sample size. */
  readonly ess: number
}

/**
 * Importance sampling: draw x_i ~ q, weight by pi(x_i) / q(x_i), and return the weighted sample, the evidence
 * estimate and the effective sample size. Combine with {@link selfNormalisedEstimate} for expectations and
 * {@link paretoSmoothedWeights} to stabilise heavy-tailed weights.
 *
 * @param target - The target (unnormalised is fine).
 * @param q - The proposal (heavier tails than the target keep the weights' variance finite).
 * @param options - Key and number of draws.
 * @returns Draws, log weights, normalised weights, log evidence and ESS.
 * @throws MdmlError with code `shape` when the proposal's dimension differs from the target's, and `value` for a
 *   number of draws that is not a positive integer.
 * @concept importance-sampling
 * @example
 * ```ts
 * import { importanceSampling, numericTarget, proposalFromDistribution } from 'mdml-engine/sampling'
 * import { normal } from 'mdml-engine/dists'
 * import { key } from 'mdml-engine/random'
 * // An unnormalised N(0, 1): Z = sqrt(2 pi).
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1)
 * const r = importanceSampling(t, proposalFromDistribution(normal(0, 2)), { key: key(0), n: 4000 })
 * Math.abs(r.logEvidence.value - 0.5 * Math.log(2 * Math.PI)) < 0.05 // => true
 * ```
 */
export function importanceSampling(
  target: LogDensity,
  q: ChainProposal,
  options: { readonly key: Key; readonly n: number },
): ImportanceResult {
  if (q.dim !== target.dim) {
    throw new MdmlError('shape', `importanceSampling: the proposal has dimension ${q.dim}, the target ${target.dim}`)
  }
  const { n } = options
  if (!(Number.isInteger(n) && n >= 1)) throw new MdmlError('value', 'importanceSampling: n must be a positive integer')
  const dim = target.dim
  const xs = q.sampleMany(options.key, n)
  const lq = q.logProbMany(xs)
  const lw = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const v = target.logDensity(xs.subarray(i * dim, (i + 1) * dim)) - (lq[i] ?? 0)
    lw[i] = Number.isNaN(v) ? Number.NEGATIVE_INFINITY : v
  }
  return {
    draws: fromStorage(xs, [n, dim], 'float64'),
    logWeights: fromStorage(lw, [n], 'float64'),
    weights: fromStorage(normaliseLogWeights(lw), [n], 'float64'),
    logEvidence: logEvidence(lw),
    ess: importanceEss(lw),
  }
}

/**
 * The empirical-Bayes fit of a generalised Pareto distribution to sorted exceedances (Zhang and Stephens 2009, with
 * the weakly informative prior on k of Vehtari et al. 2024; ArviZ `_gpdfit`).
 */
function gpdFit(x: Float64Array): { readonly k: number; readonly sigma: number } {
  const n = x.length
  const priorBs = 3
  const priorK = 10
  const m = 30 + Math.floor(Math.sqrt(n))
  const xstar = x[Math.trunc(n / 4 + 0.5) - 1] ?? Number.NaN
  if (!(xstar > (x[0] ?? Number.NaN))) return { k: Number.NaN, sigma: Number.NaN }
  const last = x[n - 1] ?? Number.NaN
  const b = Float64Array.from(
    { length: m },
    (_, i) => (1 - Math.sqrt(m / (i + 1 - 0.5))) / (priorBs * xstar) + 1 / last,
  )
  const kArr = b.map((bi) => {
    let s = 0
    for (const v of x) s += Math.log1p(-bi * v)
    return s / n
  })
  const lenScale = b.map((bi, i) => n * (Math.log(-bi / (kArr[i] ?? 1)) - (kArr[i] ?? 0) - 1))
  const z = logSumExpOf(lenScale)
  const all = lenScale.map((v) => Math.exp(v - z))
  // Drop negligible weights, as ArviZ does before normalising.
  const keep = Array.from(all, (v) => v >= 10 * Number.EPSILON)
  const w = all.filter((_, i) => keep[i] === true)
  const bs = b.filter((_, i) => keep[i] === true)
  const total = w.reduce((a, c) => a + c, 0)
  let bPost = 0
  for (let i = 0; i < w.length; i++) bPost += (bs[i] ?? 0) * ((w[i] ?? 0) / total)
  let k = 0
  for (const v of x) k += Math.log1p(-bPost * v)
  k /= n
  const sigma = -k / bPost
  k = (n * k + priorK * 0.5) / (n + priorK)
  if (Number.isNaN(k)) return { k: Number.POSITIVE_INFINITY, sigma: Number.NaN }
  return { k, sigma }
}

/** The quantile function of a generalised Pareto distribution (ArviZ `_gpinv`). */
function gpdQuantile(p: number, k: number, sigma: number, mu: number): number {
  if (!(sigma > 0)) return Number.NaN
  return k === 0 ? mu - sigma * Math.log1p(-p) : mu + (sigma * Math.expm1(-k * Math.log1p(-p))) / k
}

/** Indices that sort `x` ascending. */
function order(x: Float64Array): number[] {
  return Array.from({ length: x.length }, (_, i) => i).sort((a, b) => (x[a] ?? 0) - (x[b] ?? 0))
}

/**
 * Fit the right tail of `x` and optionally replace it by the fitted quantiles (ArviZ `_ps_tail`); with `logScale`
 * the values are log weights, fitted on the exponential scale.
 */
function paretoTail(
  x: Float64Array,
  tailLength: number,
  smooth: boolean,
  logScale: boolean,
): { readonly values: Float64Array; readonly k: number } {
  const n = x.length
  let a = Float64Array.from(x)
  if (logScale) {
    let mx = Number.NEGATIVE_INFINITY
    for (const v of a) if (v > mx) mx = v
    a = a.map((v) => v - mx)
  }
  const ord = order(a)
  const start = n - tailLength
  let tail = Float64Array.from({ length: tailLength }, (_, i) => a[ord[start + i] ?? 0] ?? 0)
  let cutoff = a[ord[start - 1] ?? 0] ?? 0
  const maxTail = tail[tailLength - 1] ?? 0
  const minTail = tail[0] ?? 0
  if (!(Math.abs(maxTail - minTail) >= 2.2250738585072014e-308)) {
    throw new MdmlError('value', 'paretoSmoothedWeights: all tail values are the same')
  }
  if (logScale) {
    tail = tail.map(Math.exp)
    cutoff = Math.exp(cutoff)
  }
  const { k, sigma } = gpdFit(tail.map((v) => v - cutoff))
  if (Number.isFinite(k) && smooth) {
    for (let i = 0; i < tailLength; i++) {
      let s = gpdQuantile((i + 0.5) / tailLength, k, sigma, cutoff)
      if (logScale) s = Math.log(s)
      if (s > maxTail) s = maxTail
      a[ord[start + i] ?? 0] = s
    }
  }
  return { values: a, k }
}

function tailLengthFor(n: number, rEff: number): number {
  return n * rEff > 225 ? Math.floor(3 * Math.sqrt(n / rEff)) : Math.floor(n / 5)
}

/** The result of {@link paretoSmoothedWeights}. */
export interface ParetoSmoothedWeights {
  /** Normalised, smoothed log weights. */
  readonly logWeights: Tensor<'float64'>
  /** The estimated shape k of the weights' tail: below 0.7 (or 1 - 1/log10 n) the estimates are reliable. */
  readonly paretoK: number
  /** The number of largest weights that were smoothed. */
  readonly tailLength: number
}

/**
 * Pareto-smoothed importance sampling (Vehtari, Simpson, Gelman, Yao and Gabry 2024; ArviZ `psislw`): fit a
 * generalised Pareto distribution to the M = min(n / 5, 3 sqrt(n / r_eff)) largest weights, replace them by the
 * fitted quantiles (capped at the largest raw weight), and normalise. The fitted shape k diagnoses the estimate:
 * k < 0.7 is reliable, k > 1 means the importance-sampling variance is infinite.
 *
 * @param logRatios - Log importance ratios log pi(x_i) - log q(x_i) (at least 5 per tail).
 * @param options - The relative efficiency r_eff of the draws (default 1: independent draws).
 * @returns The smoothed, normalised log weights, k-hat and the tail length.
 * @throws MdmlError with code `value` for fewer than 25 ratios or a constant tail.
 * @concept pareto-smoothed-importance-sampling
 * @reference A. Vehtari, D. Simpson, A. Gelman, Y. Yao and J. Gabry, Pareto smoothed importance sampling (JMLR 2024).
 * @example
 * ```ts
 * import { paretoSmoothedWeights } from 'mdml-engine/sampling'
 * import { key, normal } from 'mdml-engine/random'
 * const r = paretoSmoothedWeights(normal(key(0), [400], { scale: 0.5 }).data)
 * r.paretoK < 0.7 // => true
 * r.tailLength // => 60
 * ```
 */
export function paretoSmoothedWeights(
  logRatios: TensorLike,
  options: { readonly rEff?: number } = {},
): ParetoSmoothedWeights {
  const x = values(logRatios, 'paretoSmoothedWeights')
  const n = x.length
  const rEff = options.rEff ?? 1
  const tailLength = tailLengthFor(n, rEff)
  if (tailLength < 5) throw new MdmlError('value', 'paretoSmoothedWeights: need at least 25 draws (5 in the tail)')
  const { values: smoothed, k } = paretoTail(x, tailLength, true, true)
  const z = logSumExpOf(smoothed)
  return {
    logWeights: fromStorage(
      smoothed.map((v) => v - z),
      [n],
      'float64',
    ),
    paretoK: k,
    tailLength,
  }
}

/** Options of {@link paretoKHat}. */
export interface ParetoKOptions {
  /** Which tail to fit: `both` (default, the larger k), `right` or `left`. */
  readonly tail?: 'both' | 'right' | 'left'
  /** The relative efficiency of the draws (default the relative tail ESS at 0.05, as ArviZ). */
  readonly rEff?: number
}

/**
 * The Pareto k-hat diagnostic of draws (ArviZ `pareto_khat`): the shape of a generalised Pareto fit to a tail of the
 * draws, which tells how many finite moments the distribution has (k < 1/2: finite variance, k < 1: finite mean) and
 * whether Monte Carlo estimates from these draws can be trusted. For importance weights use the k-hat of
 * {@link paretoSmoothedWeights}, which fits the weights themselves.
 *
 * @param draws - Draws `[chain, draw]` or `[draw]`.
 * @param options - Tail and relative efficiency.
 * @returns k-hat.
 * @throws MdmlError with code `value` for too few draws or a constant tail.
 * @concept pareto-smoothed-importance-sampling
 * @example
 * ```ts
 * import { paretoKHat } from 'mdml-engine/sampling'
 * import { key, normal, studentT } from 'mdml-engine/random'
 * paretoKHat(normal(key(0), [1000]).data) < paretoKHat(studentT(key(0), [1000], 1).data) // => true
 * ```
 */
export function paretoKHat(draws: TensorLike, options: ParetoKOptions = {}): number {
  const x = Float64Array.from(toFlat(asTensor(draws) as Tensor))
  const tail = options.tail ?? 'both'
  const rEff = options.rEff ?? essOf(draws, { method: 'tail', prob: 0.05, relative: true })
  const n = x.length
  let m = tailLengthFor(n, rEff)
  if (tail === 'both') {
    m = Math.min(m, Math.floor(n / 2))
    m = Math.max(m, 5)
  }
  if (m < 5 || m >= n) throw new MdmlError('value', 'paretoKHat: too few draws for a tail fit')
  if (tail === 'both') {
    return Math.max(
      paretoTail(
        x.map((v) => -v),
        m,
        false,
        false,
      ).k,
      paretoTail(x, m, false, false).k,
    )
  }
  return paretoTail(tail === 'left' ? x.map((v) => -v) : x, m, false, false).k
}

/** The verdict of {@link weightDiagnostics}. */
export interface WeightDiagnostics {
  readonly n: number
  /** Kish's effective sample size and its fraction of n. */
  readonly ess: number
  readonly essFraction: number
  /** The largest normalised weight. */
  readonly maxWeight: number
  /** The coefficient of variation of the weights. */
  readonly cv: number
  /** The Pareto k-hat of the weights (NaN for fewer than 25 weights). */
  readonly paretoK: number
  /** The sample size at which k-hat says the estimate becomes reliable, 10^(1 / (1 - max(0, k))) (Inf for k >= 1). */
  readonly minSampleSize: number
  /** Whether k-hat is below min(1 - 1 / log10 n, 0.7). */
  readonly reliable: boolean
}

/**
 * Diagnostics of importance weights: Kish's ESS, the largest weight, the coefficient of variation, and the Pareto
 * k-hat with the sample size it asks for and its reliability threshold min(1 - 1 / log10 n, 0.7) (Vehtari et al.
 * 2024).
 *
 * @param logWeights - Log importance weights.
 * @returns The diagnostics.
 * @concept pareto-smoothed-importance-sampling
 * @example
 * ```ts
 * import { weightDiagnostics } from 'mdml-engine/sampling'
 * weightDiagnostics([0, 0, 0, 0]).ess // => 4
 * ```
 */
export function weightDiagnostics(logWeights: TensorLike): WeightDiagnostics {
  const l = values(logWeights, 'weightDiagnostics')
  const n = l.length
  const w = normaliseLogWeights(l)
  let s2 = 0
  let maxWeight = 0
  for (const v of w) {
    s2 += v * v
    if (v > maxWeight) maxWeight = v
  }
  const ess = 1 / s2
  // The weights' coefficient of variation: sd(n w) with mean 1.
  let c = 0
  for (const v of w) c += (n * v - 1) ** 2
  const cv = n > 1 ? Math.sqrt(c / (n - 1)) : Number.NaN
  let k = Number.NaN
  if (n >= 25) {
    try {
      k = paretoSmoothedWeights(l).paretoK
    } catch {
      k = Number.NaN
    }
  }
  const minSampleSize = !(k < 1) ? Number.POSITIVE_INFINITY : 10 ** (1 / (1 - Math.max(0, k)))
  return {
    n,
    ess,
    essFraction: ess / n,
    maxWeight,
    cv,
    paretoK: k,
    minSampleSize,
    reliable: k < Math.min(1 - 1 / Math.log10(n), 0.7),
  }
}
