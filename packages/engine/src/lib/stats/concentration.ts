/**
 * Distribution-free concentration inequalities and simultaneous confidence bands: Markov's and Chebyshev's bounds from
 * moments alone, Chernoff's sub-Gaussian tail bound, Hoeffding's bounded-range inequality and sample-size inversion,
 * Bernstein's and Bennett's variance-sensitive bounds, the empirical Bernstein interval of Maurer and Pontil
 * (`empiricalBernsteinInterval`, the bound with the sample variance in place of the true one), Chernoff's KL bound for
 * sums of Bernoulli variables (`chernoffKlBound`), McDiarmid's bounded-differences inequality, and the
 * Dvoretzky-Kiefer-Wolfowitz (DKW) simultaneous ECDF confidence band with Massart's tight constant.
 */

import {
  type ConfidenceLevel,
  type Estimate,
  MdmlError,
  meanOf,
  resolveConfidenceLevel,
  varianceOf,
} from '../base/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { sample, sortedCopy, vec } from './_util.ts'

/**
 * Options for tail concentration bounds that may be evaluated one-sided ($\mathbb{P}(X - \mu \ge t)$) or two-sided
 * ($\mathbb{P}(|X - \mu| \ge t)$).
 */
export interface TailBoundOptions {
  /** When true (default), multiplies the one-sided tail bound by 2 and clamps to `[0, 1]`. */
  readonly twoSided?: boolean
}

function clampProb(p: number): number {
  if (p <= 0) return 0
  if (p >= 1) return 1
  return p
}

/**
 * Markov's inequality for a non-negative random variable $X \ge 0$:
 * $\mathbb{P}(X \ge a) \le \min(1, \mathbb{E}[X] / a)$.
 *
 * @param mean - Expected value $\mathbb{E}[X] \ge 0$.
 * @param threshold - Positive threshold $a > 0$.
 * @returns Upper bound on $\mathbb{P}(X \ge a)$ in $[0, 1]$.
 * @throws MdmlError with code `domain` if `mean < 0` or `threshold <= 0`.
 * @example
 * ```ts
 * import { markovBound } from 'mdml-engine/stats'
 * markovBound(2, 10) // => 0.2
 * markovBound(15, 10) // => 1
 * ```
 */
export function markovBound(mean: number, threshold: number): number {
  if (!(mean >= 0) || !Number.isFinite(mean)) {
    throw new MdmlError('domain', `markovBound: mean must be finite and >= 0, got ${mean}`)
  }
  if (!(threshold > 0) || !Number.isFinite(threshold)) {
    throw new MdmlError('domain', `markovBound: threshold must be finite and > 0, got ${threshold}`)
  }
  return clampProb(mean / threshold)
}

/**
 * Chebyshev's inequality from variance $\sigma^2$:
 * two-sided $\mathbb{P}(|X - \mu| \ge t) \le \min(1, \sigma^2 / t^2)$, or one-sided Cantelli bound
 * $\mathbb{P}(X - \mu \ge t) \le \sigma^2 / (\sigma^2 + t^2)$ when `twoSided: false`.
 *
 * @param variance - Variance $\sigma^2 \ge 0$ of the random variable (for a sample mean of $n$ i.i.d. draws, pass $\sigma^2 / n$).
 * @param t - Positive deviation threshold $t > 0$.
 * @param options - `twoSided` (default `true`; when `false`, returns Cantelli's one-sided bound $\sigma^2 / (\sigma^2 + t^2)$).
 * @returns Upper bound on the tail probability in $[0, 1]$.
 * @throws MdmlError with code `domain` if `variance < 0` or `t <= 0`.
 * @example
 * ```ts
 * import { chebyshevBound } from 'mdml-engine/stats'
 * chebyshevBound(1, 2) // => 0.25
 * chebyshevBound(1, 2, { twoSided: false }) // => 0.2
 * ```
 */
export function chebyshevBound(variance: number, t: number, options: TailBoundOptions = {}): number {
  if (!(variance >= 0) || !Number.isFinite(variance)) {
    throw new MdmlError('domain', `chebyshevBound: variance must be finite and >= 0, got ${variance}`)
  }
  if (!(t > 0) || !Number.isFinite(t)) {
    throw new MdmlError('domain', `chebyshevBound: deviation t must be finite and > 0, got ${t}`)
  }
  const twoSided = options.twoSided ?? true
  if (twoSided) {
    return clampProb(variance / (t * t))
  }
  return clampProb(variance / (variance + t * t))
}

/**
 * Options for the sub-Gaussian Chernoff tail bound.
 */
export interface ChernoffOptions extends TailBoundOptions {
  /** Sample size $n \ge 1$ when bounding the sample mean $\bar{X}_n - \mu$ of $n$ independent $\sigma$-sub-Gaussian variables (default 1). */
  readonly n?: number
}

/**
 * Chernoff bound for a $\sigma$-sub-Gaussian random variable (or the sample mean of $n$ independent $\sigma$-sub-Gaussian
 * variables): $\mathbb{P}(\bar{X}_n - \mu \ge t) \le \exp(-n t^2 / (2 \sigma^2))$, doubled when `twoSided` is true.
 *
 * @param sigma - Sub-Gaussian scale parameter $\sigma > 0$.
 * @param t - Positive deviation $t > 0$.
 * @param options - `n` (sample size, default `1`) and `twoSided` (default `true`).
 * @returns Upper bound on the tail probability in $[0, 1]$.
 * @throws MdmlError with code `domain` if `sigma <= 0`, `t <= 0`, or `n < 1`.
 * @example
 * ```ts
 * import { chernoffGaussianBound } from 'mdml-engine/stats'
 * Number(chernoffGaussianBound(1, 2, { twoSided: false }).toFixed(6)) // => 0.135335
 * ```
 */
export function chernoffGaussianBound(sigma: number, t: number, options: ChernoffOptions = {}): number {
  if (!(sigma > 0) || !Number.isFinite(sigma)) {
    throw new MdmlError('domain', `chernoffGaussianBound: sigma must be finite and > 0, got ${sigma}`)
  }
  if (!(t > 0) || !Number.isFinite(t)) {
    throw new MdmlError('domain', `chernoffGaussianBound: t must be finite and > 0, got ${t}`)
  }
  const n = options.n ?? 1
  if (!(n >= 1) || !Number.isFinite(n)) {
    throw new MdmlError('domain', `chernoffGaussianBound: n must be >= 1, got ${n}`)
  }
  const twoSided = options.twoSided ?? true
  const oneSided = Math.exp((-n * t * t) / (2 * sigma * sigma))
  return clampProb(twoSided ? 2 * oneSided : oneSided)
}

/**
 * Alias for {@link chernoffGaussianBound}.
 *
 * @param sigma - Sub-Gaussian scale parameter $\sigma > 0$.
 * @param t - Positive deviation $t > 0$.
 * @param options - `n` (sample size, default `1`) and `twoSided` (default `true`).
 * @returns Upper bound on the tail probability in $[0, 1]$.
 * @example
 * ```ts
 * import { chernoffBound } from 'mdml-engine/stats'
 * chernoffBound(1, 2, { twoSided: false }) < 0.14 // => true
 * ```
 */
export function chernoffBound(sigma: number, t: number, options: ChernoffOptions = {}): number {
  return chernoffGaussianBound(sigma, t, options)
}

/**
 * Options for Hoeffding's inequality on the sample mean of bounded independent random variables.
 */
export interface HoeffdingOptions extends TailBoundOptions {
  /** Common interval `[a, b]` or width `b - a` for all $n$ summands (default `[0, 1]`, width `1`). */
  readonly range?: readonly [number, number] | number
  /** Per-summand intervals `[a_i, b_i]` of length `n`, overriding `range` when summands have distinct ranges. */
  readonly ranges?: readonly (readonly [number, number])[]
}

function resolveRangeWidth(range: readonly [number, number] | number | undefined, name: string): number {
  if (range === undefined) return 1
  if (typeof range === 'number') {
    if (!(range > 0) || !Number.isFinite(range)) {
      throw new MdmlError('domain', `${name}: range width must be finite and > 0, got ${range}`)
    }
    return range
  }
  const [a, b] = range
  const width = b - a
  if (!(width > 0) || !Number.isFinite(width)) {
    throw new MdmlError('domain', `${name}: range [a, b] must have b > a, got [${a}, ${b}]`)
  }
  return width
}

/**
 * Hoeffding's inequality for the sample mean $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$ of $n$ independent bounded
 * random variables $X_i \in [a_i, b_i]$:
 * $\mathbb{P}(\bar{X}_n - \mu \ge t) \le \exp\!\left(-\frac{2 n^2 t^2}{\sum_{i=1}^n (b_i - a_i)^2}\right)$,
 * doubled when `twoSided` is true (default).
 *
 * @param n - Sample size $n \ge 1$.
 * @param t - Positive deviation $t > 0$ of the sample mean from its expectation.
 * @param options - Common `range` (`[a, b]` or width, default `[0, 1]`), per-variable `ranges`, and `twoSided` (default `true`).
 * @returns Upper bound on the tail probability in $[0, 1]$.
 * @throws MdmlError with code `domain` for invalid `n`, `t`, or interval widths.
 * @example
 * ```ts
 * import { hoeffdingBound } from 'mdml-engine/stats'
 * Number(hoeffdingBound(100, 0.1).toFixed(6)) // => 0.270671
 * ```
 */
export function hoeffdingBound(n: number, t: number, options: HoeffdingOptions = {}): number {
  if (!Number.isInteger(n) || n < 1) {
    throw new MdmlError('domain', `hoeffdingBound: n must be a positive integer, got ${n}`)
  }
  if (!(t > 0) || !Number.isFinite(t)) {
    throw new MdmlError('domain', `hoeffdingBound: t must be finite and > 0, got ${t}`)
  }
  let sumSqWidths = 0
  if (options.ranges !== undefined) {
    if (options.ranges.length !== n) {
      throw new MdmlError('shape', `hoeffdingBound: ranges length ${options.ranges.length} must equal n=${n}`)
    }
    for (let i = 0; i < n; i++) {
      const w = resolveRangeWidth(options.ranges[i], 'hoeffdingBound')
      sumSqWidths += w * w
    }
  } else {
    const w = resolveRangeWidth(options.range, 'hoeffdingBound')
    sumSqWidths = n * w * w
  }
  const oneSided = Math.exp((-2 * n * n * t * t) / sumSqWidths)
  const twoSided = options.twoSided ?? true
  return clampProb(twoSided ? 2 * oneSided : oneSided)
}

/**
 * Minimum sample size $n$ required by Hoeffding's inequality to guarantee
 * $\mathbb{P}(|\bar{X}_n - \mu| \ge t) \le \delta$ (or one-sided when `twoSided: false`) for variables bounded in a
 * range of width $R = b - a$:
 * $n = \left\lceil \frac{R^2 \ln(2 / \delta)}{2 t^2} \right\rceil$.
 *
 * @param t - Target accuracy margin $t > 0$.
 * @param delta - Allowed failure probability $\delta \in (0, 1)$.
 * @param options - `range` (`[a, b]` or width $R$, default `1`) and `twoSided` (default `true`).
 * @returns Integer sample size $n \ge 1$.
 * @throws MdmlError with code `domain` if `t <= 0` or `delta` is not in `(0, 1)`.
 * @example
 * ```ts
 * import { hoeffdingSampleSize } from 'mdml-engine/stats'
 * hoeffdingSampleSize(0.05, 0.05) // => 738
 * ```
 */
export function hoeffdingSampleSize(
  t: number,
  delta: number,
  options: { readonly range?: readonly [number, number] | number; readonly twoSided?: boolean } = {},
): number {
  if (!(t > 0) || !Number.isFinite(t)) {
    throw new MdmlError('domain', `hoeffdingSampleSize: t must be finite and > 0, got ${t}`)
  }
  if (!(delta > 0 && delta < 1)) {
    throw new MdmlError('domain', `hoeffdingSampleSize: delta must be in (0, 1), got ${delta}`)
  }
  const width = resolveRangeWidth(options.range, 'hoeffdingSampleSize')
  const twoSided = options.twoSided ?? true
  const numer = (twoSided ? Math.log(2 / delta) : Math.log(1 / delta)) * width * width
  return Math.max(1, Math.ceil(numer / (2 * t * t)))
}

/**
 * Bernstein's inequality for the sample mean $\bar{X}_n$ of $n$ independent zero-mean (or centered) random variables
 * with average variance $\sigma^2 = \frac{1}{n}\sum_{i=1}^n \operatorname{Var}(X_i)$ and almost-sure bound
 * $|X_i - \mathbb{E}[X_i]| \le M$:
 * $\mathbb{P}(\bar{X}_n - \mu \ge t) \le \exp\!\left(-\frac{n t^2 / 2}{\sigma^2 + M t / 3}\right)$,
 * doubled when `twoSided` is true (default).
 *
 * @param n - Sample size $n \ge 1$.
 * @param variance - Average per-variable variance $\sigma^2 \ge 0$.
 * @param maxAbsBound - Almost-sure centered bound $M > 0$ with $|X_i - \mu_i| \le M$.
 * @param t - Positive deviation $t > 0$ of the sample mean.
 * @param options - `twoSided` (default `true`).
 * @returns Upper bound on the tail probability in $[0, 1]$.
 * @throws MdmlError with code `domain` if `n < 1`, `variance < 0`, `maxAbsBound <= 0`, or `t <= 0`.
 * @example
 * ```ts
 * import { bernsteinBound } from 'mdml-engine/stats'
 * Number(bernsteinBound(100, 0.05, 1, 0.1).toFixed(6)) // => 0.004958
 * ```
 */
export function bernsteinBound(
  n: number,
  variance: number,
  maxAbsBound: number,
  t: number,
  options: TailBoundOptions = {},
): number {
  if (!Number.isInteger(n) || n < 1) {
    throw new MdmlError('domain', `bernsteinBound: n must be a positive integer, got ${n}`)
  }
  if (!(variance >= 0) || !Number.isFinite(variance)) {
    throw new MdmlError('domain', `bernsteinBound: variance must be finite and >= 0, got ${variance}`)
  }
  if (!(maxAbsBound > 0) || !Number.isFinite(maxAbsBound)) {
    throw new MdmlError('domain', `bernsteinBound: maxAbsBound must be finite and > 0, got ${maxAbsBound}`)
  }
  if (!(t > 0) || !Number.isFinite(t)) {
    throw new MdmlError('domain', `bernsteinBound: t must be finite and > 0, got ${t}`)
  }
  const denom = variance + (maxAbsBound * t) / 3
  const oneSided = denom === 0 ? 0 : Math.exp((-0.5 * n * t * t) / denom)
  const twoSided = options.twoSided ?? true
  return clampProb(twoSided ? 2 * oneSided : oneSided)
}

/**
 * Chernoff's relative-entropy bound for a sum of independent Bernoulli($p$) variables:
 * $\mathbb{P}(\bar{X}_n \ge q) \le \exp(-n\,\mathrm{KL}(q \,\|\, p))$ for $q \ge p$ (and the mirrored lower tail for
 * $q \le p$), with $\mathrm{KL}(q\|p) = q\ln(q/p) + (1-q)\ln((1-q)/(1-p))$. It is the exponent Hoeffding's bound weakens
 * to $2(q-p)^2$.
 *
 * @param n - Sample size $n \ge 1$.
 * @param p - Success probability in $(0, 1)$.
 * @param q - Threshold for the sample proportion in $[0, 1]$.
 * @returns Upper bound on $\mathbb{P}(\bar{X}_n \ge q)$ when $q \ge p$, or on $\mathbb{P}(\bar{X}_n \le q)$ when $q < p$.
 * @throws MdmlError with code `domain` for invalid arguments.
 * @example
 * ```ts
 * import { chernoffKlBound } from 'mdml-engine/stats'
 * Number(chernoffKlBound(100, 0.5, 0.6).toFixed(6)) // => 0.133514
 * ```
 */
export function chernoffKlBound(n: number, p: number, q: number): number {
  if (!Number.isInteger(n) || n < 1) {
    throw new MdmlError('domain', `chernoffKlBound: n must be a positive integer, got ${n}`)
  }
  if (!(p > 0 && p < 1)) throw new MdmlError('domain', `chernoffKlBound: p must be in (0, 1), got ${p}`)
  if (!(q >= 0 && q <= 1)) throw new MdmlError('domain', `chernoffKlBound: q must be in [0, 1], got ${q}`)
  const term = (a: number, b: number): number => (a === 0 ? 0 : a * Math.log(a / b))
  const kl = term(q, p) + term(1 - q, 1 - p)
  return clampProb(Math.exp(-n * kl))
}

/**
 * The empirical Bernstein confidence interval of Maurer and Pontil (2009, Theorem 4) for the mean of a bounded sample:
 * with probability at least $1 - \delta$,
 * $|\mu - \bar{X}_n| \le \sqrt{2 S_n^2 \ln(4/\delta) / n} + 7 R \ln(4/\delta) / (3(n - 1))$, where $S_n^2$ is the
 * sample variance and $R$ the range width; unlike Bernstein's inequality it needs no knowledge of the variance, and it
 * is tighter than Hoeffding's interval whenever $S_n^2 \ll R^2$.
 *
 * @param x - A sample of at least two values, each inside the range.
 * @param options - `range` (`[a, b]` or width $R$, default `[0, 1]`) and the failure probability `delta` (default `0.05`).
 * @returns An `Estimate`: the sample mean with the symmetric interval at level $1 - \delta$.
 * @throws MdmlError with code `domain` for invalid `range` or `delta`, `value` for fewer than two observations.
 * @example
 * ```ts
 * import { empiricalBernsteinInterval } from 'mdml-engine/stats'
 * const e = empiricalBernsteinInterval([0.5, 0.5, 0.5, 0.5, 0.5], { delta: 0.05 })
 * Number((e.interval?.[1] ?? 0).toFixed(4)) // => 3.0562
 * ```
 */
export function empiricalBernsteinInterval(
  x: TensorLike,
  options: { readonly range?: readonly [number, number] | number; readonly delta?: number } = {},
): Estimate {
  const xs = sample('empiricalBernsteinInterval', x, 2)
  const delta = options.delta ?? 0.05
  if (!(delta > 0 && delta < 1)) {
    throw new MdmlError('domain', `empiricalBernsteinInterval: delta must be in (0, 1), got ${delta}`)
  }
  const width = resolveRangeWidth(options.range, 'empiricalBernsteinInterval')
  const n = xs.length
  const mean = meanOf(xs)
  const logTerm = Math.log(4 / delta)
  const half = Math.sqrt((2 * varianceOf(xs, 1) * logTerm) / n) + (7 * width * logTerm) / (3 * (n - 1))
  return { value: mean, interval: [mean - half, mean + half], level: 1 - delta, method: 'empirical Bernstein' }
}

/**
 * Bennett's inequality for the sample mean $\bar{X}_n$ of $n$ independent variables with average variance $\sigma^2$
 * and centered bound $|X_i - \mu_i| \le M$:
 * $\mathbb{P}(\bar{X}_n - \mu \ge t) \le \exp\!\left(-\frac{n \sigma^2}{M^2}\, h\!\left(\frac{M t}{\sigma^2}\right)\right)$
 * where $h(u) = (1 + u)\ln(1 + u) - u$. Tighter than Bernstein's inequality for all $t > 0$.
 *
 * @param n - Sample size $n \ge 1$.
 * @param variance - Average per-variable variance $\sigma^2 > 0$.
 * @param maxAbsBound - Almost-sure centered bound $M > 0$ with $|X_i - \mu_i| \le M$.
 * @param t - Positive deviation $t > 0$ of the sample mean.
 * @param options - `twoSided` (default `true`).
 * @returns Upper bound on the tail probability in $[0, 1]$.
 * @throws MdmlError with code `domain` if `n < 1`, `variance <= 0`, `maxAbsBound <= 0`, or `t <= 0`.
 * @example
 * ```ts
 * import { bennettBound, bernsteinBound } from 'mdml-engine/stats'
 * bennettBound(100, 0.05, 1, 0.1) < bernsteinBound(100, 0.05, 1, 0.1) // => true
 * ```
 */
export function bennettBound(
  n: number,
  variance: number,
  maxAbsBound: number,
  t: number,
  options: TailBoundOptions = {},
): number {
  if (!Number.isInteger(n) || n < 1) {
    throw new MdmlError('domain', `bennettBound: n must be a positive integer, got ${n}`)
  }
  if (!(variance > 0) || !Number.isFinite(variance)) {
    throw new MdmlError('domain', `bennettBound: variance must be finite and > 0, got ${variance}`)
  }
  if (!(maxAbsBound > 0) || !Number.isFinite(maxAbsBound)) {
    throw new MdmlError('domain', `bennettBound: maxAbsBound must be finite and > 0, got ${maxAbsBound}`)
  }
  if (!(t > 0) || !Number.isFinite(t)) {
    throw new MdmlError('domain', `bennettBound: t must be finite and > 0, got ${t}`)
  }
  const u = (maxAbsBound * t) / variance
  const h = (1 + u) * Math.log1p(u) - u
  const oneSided = Math.exp(((-n * variance) / (maxAbsBound * maxAbsBound)) * h)
  const twoSided = options.twoSided ?? true
  return clampProb(twoSided ? 2 * oneSided : oneSided)
}

/**
 * McDiarmid's bounded-differences inequality for a function $f(X_1, \dots, X_n)$ of independent variables where
 * changing the $i$-th coordinate changes $f$ by at most $c_i > 0$:
 * $\mathbb{P}(f(X) - \mathbb{E}[f(X)] \ge t) \le \exp\!\left(-\frac{2 t^2}{\sum_{i=1}^n c_i^2}\right)$,
 * doubled when `twoSided` is true (default).
 *
 * @param boundedDifferences - Vector of coordinate sensitivities $c_1, \dots, c_n > 0$.
 * @param t - Positive deviation threshold $t > 0$.
 * @param options - `twoSided` (default `true`).
 * @returns Upper bound on the tail probability in $[0, 1]$.
 * @throws MdmlError with code `domain` if any $c_i \le 0$ or `t <= 0`.
 * @example
 * ```ts
 * import { mcDiarmidBound } from 'mdml-engine/stats'
 * Number(mcDiarmidBound([0.1, 0.1, 0.1, 0.1], 0.2).toFixed(6)) // => 0.270671
 * ```
 */
export function mcDiarmidBound(boundedDifferences: TensorLike, t: number, options: TailBoundOptions = {}): number {
  const c = sample('mcDiarmidBound', boundedDifferences, 1)
  if (!(t > 0) || !Number.isFinite(t)) {
    throw new MdmlError('domain', `mcDiarmidBound: t must be finite and > 0, got ${t}`)
  }
  let sumSq = 0
  for (let i = 0; i < c.length; i++) {
    const ci = c[i] ?? 0
    if (!(ci > 0) || !Number.isFinite(ci)) {
      throw new MdmlError('domain', `mcDiarmidBound: all bounded differences must be > 0, got ${ci} at index ${i}`)
    }
    sumSq += ci * ci
  }
  const oneSided = Math.exp((-2 * t * t) / sumSq)
  const twoSided = options.twoSided ?? true
  return clampProb(twoSided ? 2 * oneSided : oneSided)
}

/**
 * Simultaneous ECDF confidence band from the Dvoretzky-Kiefer-Wolfowitz (DKW) inequality with Massart's tight constant.
 */
export interface DkwBandResult {
  /** Sorted unique sample values at which the ECDF jumps. */
  readonly values: Tensor
  /** Empirical CDF $\hat{F}_n(x)$ at each entry of `values`. */
  readonly ecdf: Tensor
  /** Simultaneous lower confidence band $\max(0, \hat{F}_n(x) - \varepsilon_n)$ at each entry of `values`. */
  readonly lower: Tensor
  /** Simultaneous upper confidence band $\min(1, \hat{F}_n(x) + \varepsilon_n)$ at each entry of `values`. */
  readonly upper: Tensor
  /** Half-width $\varepsilon_n = \sqrt{\ln(2 / \alpha) / (2 n)}$ of the simultaneous band. */
  readonly epsilon: number
  /** Significance level $\alpha = 1 - \text{level}$. */
  readonly alpha: number
  /** Sample size $n$. */
  readonly n: number
  /** Evaluate `{ ecdf, lower, upper }` at any point $t \in \mathbb{R}$. */
  readonly evaluate: (t: number) => { readonly ecdf: number; readonly lower: number; readonly upper: number }
}

/**
 * Options for {@link dkwBand}.
 */
export interface DkwBandOptions {
  /** Significance level $\alpha \in (0, 1)$ (overrides `level` when supplied). */
  readonly alpha?: number
  /** Simultaneous confidence level $1 - \alpha \in (0, 1)$ (default `0.95`). */
  readonly level?: ConfidenceLevel
}

/**
 * Dvoretzky-Kiefer-Wolfowitz (DKW) simultaneous $(1 - \alpha)$ confidence band for the cumulative distribution function
 * $F(x)$ around the empirical CDF $\hat{F}_n(x)$, using Massart's tight constant:
 * $\mathbb{P}\!\left(\sup_{x \in \mathbb{R}} |\hat{F}_n(x) - F(x)| > \varepsilon_n\right) \le 2 e^{-2 n \varepsilon_n^2} = \alpha$,
 * so $\varepsilon_n = \sqrt{\ln(2 / \alpha) / (2 n)}$.
 *
 * @param x - 1-D sample of observations.
 * @param options - `alpha` (significance level in `(0, 1)`) or `level` (confidence level, default `0.95`).
 * @returns Simultaneous band with sorted unique `values`, `ecdf`, `lower`, `upper`, half-width `epsilon`, and `evaluate(t)`.
 * @throws MdmlError with code `domain` if `alpha` is outside `(0, 1)`.
 * @example
 * ```ts
 * import { dkwBand } from 'mdml-engine/stats'
 * const band = dkwBand([1, 2, 3, 4, 5], { level: 0.95 })
 * Number(band.epsilon.toFixed(4)) // => 0.6074
 * band.evaluate(3).ecdf // => 0.6
 * ```
 */
export function dkwBand(x: TensorLike, options: DkwBandOptions = {}): DkwBandResult {
  const s = sortedCopy(sample('dkwBand', x, 1))
  const n = s.length
  let alpha: number
  if (options.alpha !== undefined) {
    if (!(options.alpha > 0 && options.alpha < 1) || !Number.isFinite(options.alpha)) {
      throw new MdmlError('domain', `dkwBand: alpha must be in (0, 1), got ${options.alpha}`)
    }
    alpha = options.alpha
  } else {
    const level = resolveConfidenceLevel(options.level)
    alpha = 1 - level
  }
  const epsilon = Math.sqrt(Math.log(2 / alpha) / (2 * n))

  // Deduplicate sorted values and record the cumulative count after ties.
  const uniqVals: number[] = []
  const uniqEcdf: number[] = []
  let i = 0
  while (i < n) {
    const v = s[i] ?? 0
    let j = i + 1
    while (j < n && s[j] === v) j++
    uniqVals.push(v)
    uniqEcdf.push(j / n)
    i = j
  }

  const m = uniqVals.length
  const valuesArr = new Float64Array(m)
  const ecdfArr = new Float64Array(m)
  const lowerArr = new Float64Array(m)
  const upperArr = new Float64Array(m)
  for (let k = 0; k < m; k++) {
    const f = uniqEcdf[k] ?? 0
    valuesArr[k] = uniqVals[k] ?? 0
    ecdfArr[k] = f
    lowerArr[k] = Math.max(0, f - epsilon)
    upperArr[k] = Math.min(1, f + epsilon)
  }

  const evaluate = (t: number): { readonly ecdf: number; readonly lower: number; readonly upper: number } => {
    // Binary search for the number of sample elements <= t.
    let lo = 0
    let hi = n
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if ((s[mid] ?? 0) <= t) lo = mid + 1
      else hi = mid
    }
    const f = lo / n
    return {
      ecdf: f,
      lower: Math.max(0, f - epsilon),
      upper: Math.min(1, f + epsilon),
    }
  }

  return {
    values: vec(valuesArr),
    ecdf: vec(ecdfArr),
    lower: vec(lowerArr),
    upper: vec(upperArr),
    epsilon,
    alpha,
    n,
    evaluate,
  }
}
