/**
 * Effect sizes, which say how large a difference or association is rather than whether it is significant, each with a
 * confidence interval: standardised mean differences (Cohen's d, Hedges' g, Glass's delta), Cohen's h for proportions,
 * Cramer's V for contingency tables, eta and omega squared for one-way layouts, and Cliff's delta.
 */

import { type Estimate, MdmlError, meanOf, resolveConfidenceLevel, varianceOf } from '../base/index.ts'
import { noncentralChi2, noncentralF } from '../dists/index.ts'
import { item, type TensorLike } from '../tensor/index.ts'
import { chi2Cdf, fCdf, groupsOf, normPpf, rootAbove, rowsOf, sample, tPpf } from './_util.ts'

/** An effect size: an estimate with the name of the measure. */
export type EffectSize = Estimate & { readonly name: string }

/** Options of the standardised mean differences. */
export interface MeanDifferenceOptions {
  /** Treat x and y as paired measurements (default false). */
  readonly paired?: boolean
  /** The interval's confidence level (default 0.95). */
  readonly level?: number
}

function effect(
  name: string,
  value: number,
  interval: readonly [number, number],
  level: number,
  method: string,
): EffectSize {
  return { name, value, interval, level, method }
}

// pingouin's compute_esci for d and g: d -+ t_{dof} se with Hedges and Olkin's standard error.
function dInterval(d: number, nx: number, ny: number, paired: boolean, level: number): [number, number] {
  const one = ny === 1 || paired
  const se = one ? Math.sqrt(1 / nx + (d * d) / (2 * nx)) : Math.sqrt((nx + ny) / (nx * ny) + (d * d) / (2 * (nx + ny)))
  const crit = Math.abs(tPpf(one ? nx - 1 : nx + ny - 2, (1 - level) / 2))
  return [d - crit * se, d + crit * se]
}

function rawD(
  name: string,
  x: TensorLike,
  y: TensorLike | number,
  paired: boolean,
): { d: number; nx: number; ny: number } {
  const a = sample(name, x, 2)
  if (typeof y === 'number') return { d: (meanOf(a) - y) / Math.sqrt(varianceOf(a, 1)), nx: a.length, ny: 1 }
  const b = sample(name, y, 2)
  if (paired && a.length !== b.length) throw new MdmlError('shape', `${name}: paired samples differ in length`)
  const sd = paired
    ? Math.sqrt((varianceOf(a, 1) + varianceOf(b, 1)) / 2)
    : Math.sqrt(((a.length - 1) * varianceOf(a, 1) + (b.length - 1) * varianceOf(b, 1)) / (a.length + b.length - 2))
  return { d: (meanOf(a) - meanOf(b)) / sd, nx: a.length, ny: b.length }
}

/**
 * Cohen's d (pingouin.compute_effsize): the mean difference over the pooled standard deviation; for one sample against
 * a value, (mean - mu) / sd; for paired samples Cumming's d_av, which divides by the root mean of the two variances.
 * The interval is d -+ t sqrt((nx + ny) / (nx ny) + d^2 / (2 (nx + ny))) (Hedges and Olkin), as pingouin's
 * compute_esci.
 *
 * @param x - First sample.
 * @param y - Second sample, or a reference value for one sample.
 * @param options - paired and level.
 * @returns The effect size with its interval.
 * @example
 * ```ts
 * import { cohensD } from 'mdml-engine/stats'
 * cohensD([2, 4, 6], [1, 3, 5]).value // => 0.5
 * cohensD([2, 4, 6], 2).value // => 1
 * ```
 */
export function cohensD(x: TensorLike, y: TensorLike | number, options: MeanDifferenceOptions = {}): EffectSize {
  const level = resolveConfidenceLevel(options.level)
  const paired = options.paired ?? false
  const { d, nx, ny } = rawD('cohensD', x, y, paired)
  return effect("Cohen's d", d, dInterval(d, nx, ny, paired, level), level, 'Hedges-Olkin standard error')
}

/**
 * Hedges' g: Cohen's d times the small-sample correction 1 - 3 / (4 (nx + ny) - 9) (pingouin's convert_effsize), with
 * the same form of interval.
 *
 * @param x - First sample.
 * @param y - Second sample, or a reference value.
 * @param options - paired and level.
 * @returns The effect size with its interval.
 * @example
 * ```ts
 * import { hedgesG } from 'mdml-engine/stats'
 * hedgesG([2, 4, 6], [1, 3, 5]).value // => 0.4
 * ```
 */
export function hedgesG(x: TensorLike, y: TensorLike | number, options: MeanDifferenceOptions = {}): EffectSize {
  const level = resolveConfidenceLevel(options.level)
  const paired = options.paired ?? false
  const { d, nx, ny } = rawD('hedgesG', x, y, paired)
  const g = d * (1 - 3 / (4 * (nx + ny) - 9))
  return effect("Hedges' g", g, dInterval(g, nx, ny, paired, level), level, 'Hedges-Olkin standard error')
}

/**
 * Glass's delta: the mean difference over the control group's standard deviation (y is the control), with the normal
 * interval of Hedges and Olkin, se = sqrt((nx + ny) / (nx ny) + delta^2 / (2 (ny - 1))).
 *
 * @param x - Treatment sample.
 * @param y - Control sample.
 * @param options - level.
 * @returns The effect size with its interval.
 * @example
 * ```ts
 * import { glassDelta } from 'mdml-engine/stats'
 * glassDelta([3, 5, 7], [1, 2, 3]).value // => 3
 * ```
 */
export function glassDelta(x: TensorLike, y: TensorLike, options: { readonly level?: number } = {}): EffectSize {
  const level = resolveConfidenceLevel(options.level)
  const a = sample('glassDelta', x, 2)
  const b = sample('glassDelta', y, 2)
  const delta = (meanOf(a) - meanOf(b)) / Math.sqrt(varianceOf(b, 1))
  const nx = a.length
  const ny = b.length
  const se = Math.sqrt((nx + ny) / (nx * ny) + (delta * delta) / (2 * (ny - 1)))
  const z = normPpf(0.5 + level / 2)
  return effect("Glass's delta", delta, [delta - z * se, delta + z * se], level, 'Hedges-Olkin standard error')
}

/**
 * Cohen's h for two proportions, 2 asin(sqrt(p1)) - 2 asin(sqrt(p2)), with the normal interval h -+ z sqrt(1/n1 +
 * 1/n2) when the sample sizes are given.
 *
 * @param p1 - First proportion.
 * @param p2 - Second proportion.
 * @param options - Sample sizes n1, n2 (for the interval) and level.
 * @returns The effect size (with an interval when n1 and n2 are given).
 * @example
 * ```ts
 * import { cohensH } from 'mdml-engine/stats'
 * cohensH(0.5, 0.5).value // => 0
 * cohensH(1, 0).value // => 3.141592653589793
 * ```
 */
export function cohensH(
  p1: number,
  p2: number,
  options: { readonly n1?: number; readonly n2?: number; readonly level?: number } = {},
): EffectSize {
  for (const p of [p1, p2]) if (!(p >= 0 && p <= 1)) throw new MdmlError('domain', 'cohensH: proportions lie in [0, 1]')
  const h = 2 * Math.asin(Math.sqrt(p1)) - 2 * Math.asin(Math.sqrt(p2))
  if (options.n1 === undefined || options.n2 === undefined) return { name: "Cohen's h", value: h }
  const level = resolveConfidenceLevel(options.level)
  const se = Math.sqrt(1 / options.n1 + 1 / options.n2)
  const z = normPpf(0.5 + level / 2)
  return effect("Cohen's h", h, [h - z * se, h + z * se], level, 'normal approximation')
}

/**
 * The noncentrality bounds of an observed statistic by inverting its noncentral CDF in the noncentrality (Steiger
 * and Fouladi's method): lambda_L with CDF(obs; lambda_L) = (1 + level) / 2 and lambda_U with CDF = (1 - level) / 2,
 * clipped at zero.
 */
function noncentralityBounds(cdf: (lambda: number) => number, level: number, scale: number): [number, number] {
  const hi = (1 + level) / 2
  const lo = (1 - level) / 2
  const lower = cdf(0) <= hi ? 0 : rootAbove((l) => cdf(l) - hi, 0, Math.max(1, scale))
  const upper = cdf(0) <= lo ? 0 : rootAbove((l) => cdf(l) - lo, 0, Math.max(1, scale))
  return [lower, upper]
}

/**
 * Cramer's V for a contingency table, sqrt(chi^2 / (n (min(r, c) - 1))) (scipy.stats.contingency.association), or
 * Bergsma's bias-corrected version; the interval inverts the noncentral chi-square distribution of the statistic
 * (Smithson 2003): the population V = sqrt(lambda / (n (min(r, c) - 1))) at the noncentrality bounds.
 *
 * @param table - An r x c table of counts.
 * @param options - Bias correction (default false) and level.
 * @returns The effect size with its interval.
 * @example
 * ```ts
 * import { cramersV } from 'mdml-engine/stats'
 * cramersV([[10, 0], [0, 10]]).value // => 1
 * cramersV([[5, 5], [5, 5]]).value // => 0
 * ```
 */
export function cramersV(
  table: TensorLike,
  options: { readonly correction?: boolean; readonly level?: number } = {},
): EffectSize {
  const level = resolveConfidenceLevel(options.level)
  const t = rowsOf('cramersV', table)
  const { rows: r, cols: c, data } = t
  let n = 0
  const rowSums = new Float64Array(r)
  const colSums = new Float64Array(c)
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) {
      const v = data[i * c + j] ?? 0
      rowSums[i] = (rowSums[i] ?? 0) + v
      colSums[j] = (colSums[j] ?? 0) + v
      n += v
    }
  let chi2 = 0
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) {
      const e = ((rowSums[i] ?? 0) * (colSums[j] ?? 0)) / n
      chi2 += ((data[i * c + j] ?? 0) - e) ** 2 / e
    }
  const k = Math.min(r, c)
  const df = (r - 1) * (c - 1)
  let value: number
  if (options.correction === true) {
    const phi2 = Math.max(0, chi2 / n - df / (n - 1))
    const rr = r - (r - 1) ** 2 / (n - 1)
    const cc = c - (c - 1) ** 2 / (n - 1)
    value = Math.sqrt(phi2 / Math.min(rr - 1, cc - 1))
  } else value = Math.sqrt(chi2 / (n * (k - 1)))
  const cdf = (lambda: number): number =>
    lambda === 0 ? chi2Cdf(df, chi2) : item(noncentralChi2(df, lambda).cdf(chi2))
  const [lo, hi] = noncentralityBounds(cdf, level, chi2 + df)
  const toV = (lambda: number): number => Math.min(1, Math.sqrt(lambda / (n * (k - 1))))
  return effect("Cramer's V", value, [toV(lo), toV(hi)], level, 'noncentral chi-square inversion')
}

/** Sums of squares of a one-way layout. */
function oneWay(name: string, groups: readonly TensorLike[]): { ssb: number; ssw: number; k: number; n: number } {
  const gs = groupsOf(name, groups, 1)
  const all = gs.flatMap((g) => Array.from(g))
  const grand = meanOf(all)
  let ssb = 0
  let ssw = 0
  for (const g of gs) {
    const m = meanOf(g)
    ssb += g.length * (m - grand) ** 2
    for (const v of g) ssw += (v - m) ** 2
  }
  return { ssb, ssw, k: gs.length, n: all.length }
}

/** Population eta-squared bounds lambda / (lambda + N) from the noncentral F inversion of a one-way F statistic. */
function etaBounds(F: number, df1: number, df2: number, level: number): [number, number] {
  // No variance within the groups: eta squared is exactly 1 (or undefined when there is none between either).
  if (!Number.isFinite(F)) return Number.isNaN(F) ? [Number.NaN, Number.NaN] : [1, 1]
  const cdf = (lambda: number): number =>
    lambda === 0 ? fCdf(df1, df2, F) : item(noncentralF(df1, df2, lambda).cdf(F))
  const [lo, hi] = noncentralityBounds(cdf, level, F * df1 + 1)
  const N = df1 + df2 + 1
  return [lo / (lo + N), hi / (hi + N)]
}

/**
 * Eta squared of a one-way layout, SS_between / SS_total (the proportion of variance explained), with the interval
 * of Steiger (2004) from the noncentral F distribution: lambda / (lambda + N) at the noncentrality bounds.
 *
 * @param groups - Two or more samples.
 * @param options - level.
 * @returns The effect size with its interval.
 * @example
 * ```ts
 * import { etaSquared } from 'mdml-engine/stats'
 * etaSquared([[1, 2, 3], [4, 5, 6]]).value // => 0.7714285714285715
 * ```
 */
export function etaSquared(groups: readonly TensorLike[], options: { readonly level?: number } = {}): EffectSize {
  const level = resolveConfidenceLevel(options.level)
  const { ssb, ssw, k, n } = oneWay('etaSquared', groups)
  const F = ssb / (k - 1) / (ssw / (n - k))
  return effect('eta squared', ssb / (ssb + ssw), etaBounds(F, k - 1, n - k, level), level, 'noncentral F inversion')
}

/**
 * Omega squared of a one-way layout, (SS_between - (k - 1) MS_within) / (SS_total + MS_within), the less biased
 * estimate of the variance explained, with the noncentral-F interval of the population proportion it estimates.
 *
 * @param groups - Two or more samples.
 * @param options - level.
 * @returns The effect size with its interval.
 * @example
 * ```ts
 * import { omegaSquared } from 'mdml-engine/stats'
 * omegaSquared([[1, 2, 3], [4, 5, 6]]).value // => 0.6756756756756757
 * ```
 */
export function omegaSquared(groups: readonly TensorLike[], options: { readonly level?: number } = {}): EffectSize {
  const level = resolveConfidenceLevel(options.level)
  const { ssb, ssw, k, n } = oneWay('omegaSquared', groups)
  const msw = ssw / (n - k)
  const value = (ssb - (k - 1) * msw) / (ssb + ssw + msw)
  const F = ssb / (k - 1) / msw
  return effect('omega squared', value, etaBounds(F, k - 1, n - k, level), level, 'noncentral F inversion')
}

/**
 * Cliff's delta, P(X > Y) - P(X < Y) over all pairs, with Cliff's (1993) asymmetric interval from the consistent
 * variance estimate.
 *
 * @param x - First sample.
 * @param y - Second sample.
 * @param options - level.
 * @returns The effect size in [-1, 1] with its interval.
 * @reference N. Cliff, Dominance statistics: ordinal analyses to answer ordinal questions, Psychological Bulletin 114
 * (1993).
 * @example
 * ```ts
 * import { cliffsDelta } from 'mdml-engine/stats'
 * cliffsDelta([3, 4, 5], [1, 2, 3]).value // => 0.8888888888888888
 * ```
 */
export function cliffsDelta(x: TensorLike, y: TensorLike, options: { readonly level?: number } = {}): EffectSize {
  const level = resolveConfidenceLevel(options.level)
  const a = sample('cliffsDelta', x, 2)
  const b = sample('cliffsDelta', y, 2)
  const nx = a.length
  const ny = b.length
  const di = new Float64Array(nx)
  const dj = new Float64Array(ny)
  let total = 0
  for (let i = 0; i < nx; i++)
    for (let j = 0; j < ny; j++) {
      const s = Math.sign((a[i] ?? 0) - (b[j] ?? 0))
      di[i] = (di[i] ?? 0) + s
      dj[j] = (dj[j] ?? 0) + s
      total += s
    }
  const d = total / (nx * ny)
  let si = 0
  for (const v of di) si += (v / ny - d) ** 2
  si /= nx - 1
  let sj = 0
  for (const v of dj) sj += (v / nx - d) ** 2
  sj /= ny - 1
  let sij = 0
  for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) sij += (Math.sign((a[i] ?? 0) - (b[j] ?? 0)) - d) ** 2
  sij /= (nx - 1) * (ny - 1)
  const variance = Math.max(0, ((ny - 1) * si + (nx - 1) * sj + sij) / (nx * ny))
  const s = Math.sqrt(variance)
  const z = normPpf(0.5 + level / 2)
  const root = z * s * Math.sqrt((1 - d * d) ** 2 + z * z * variance)
  const den = 1 - d * d + z * z * variance
  // A degenerate sample (complete separation or no spread) has no variance to build an interval from.
  const lo = den > 0 && variance > 0 ? (d - d ** 3 - root) / den : d
  const hi = den > 0 && variance > 0 ? (d - d ** 3 + root) / den : d
  return {
    name: "Cliff's delta",
    value: d,
    standardError: s,
    interval: [Math.max(-1, lo), Math.min(1, hi)],
    level,
    method: "Cliff's consistent variance",
  }
}
