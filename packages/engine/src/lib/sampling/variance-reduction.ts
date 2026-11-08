/**
 * Variance reduction for Monte Carlo estimates of an expectation: control variates (one or several, with the optimal
 * regression coefficients), antithetic pairs, stratified sampling with proportional or Neyman allocation, conditioning
 * (Rao-Blackwellisation), and randomised quasi-Monte Carlo over scrambled Sobol or shifted Halton points. Each returns an
 * `Estimate` with a standard error that is valid for the estimator it describes, plus the plain Monte Carlo standard
 * error for the same sample or budget so the reduction is a number; `varianceReductionAlgorithm` runs four of them side
 * by side on the same integrand, one batch of draws per step. Importance sampling is in `importance.ts`.
 */

import { type Estimate, type Json, meanOf, OmaiError, varianceOf } from '../base/index.ts'
import { solve } from '../numerics/linalg/index.ts'
import { foldIn, halton, type Key, normal, scrambledSobol, uniform } from '../random/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'

const Z95 = 1.959964

function normalEstimate(value: number, standardError: number, method: string): Estimate {
  return {
    value,
    standardError,
    interval: [value - Z95 * standardError, value + Z95 * standardError],
    level: 0.95,
    method,
  }
}

function checkN(n: number, minimum: number, who: string): void {
  if (!(Number.isInteger(n) && n >= minimum))
    throw new OmaiError('value', `${who}: n must be an integer of at least ${minimum}, got ${n}`)
}

/** A draw of `n` independent samples from the distribution being integrated against. */
export type Sampler<T> = (key: Key, n: number) => readonly T[]

const uniformSampler: Sampler<number> = (key, n) => Array.from(uniform(key, [n]).data)

/** An estimate with the standard error that plain Monte Carlo would have had on the same evaluations. */
export interface ReducedEstimate extends Estimate {
  /** Function evaluations used. */
  readonly evaluations: number
  /** The plain Monte Carlo standard error for the same budget. */
  readonly plainStandardError: number
  /** `plainStandardError^2 / standardError^2`: how many times more plain samples the same accuracy would have cost. */
  readonly varianceReductionFactor: number
}

function reduced(value: number, se: number, plainSe: number, evaluations: number, method: string): ReducedEstimate {
  return {
    ...normalEstimate(value, se, method),
    evaluations,
    plainStandardError: plainSe,
    varianceReductionFactor: se > 0 ? (plainSe / se) ** 2 : Number.POSITIVE_INFINITY,
  }
}

/** Options for {@link monteCarloEstimate}. */
export interface MonteCarloEstimateOptions<T = number> {
  /** The integrand `f(x)`. */
  readonly fn: (x: T) => number
  /** Draws `n` samples from the integrating distribution (default `U[0, 1)`). */
  readonly sample?: Sampler<T>
  /** Random key. */
  readonly key: Key
  /** Number of samples (at least 2). */
  readonly n: number
}

/**
 * The plain Monte Carlo estimate `mean f(X_i)` of `E f(X)` with its standard error `s / sqrt(n)`, the baseline the other
 * estimators here are measured against.
 *
 * @param options - Integrand, sampler (default uniform on `[0, 1)`), key and `n`.
 * @returns The estimate with its 95% normal interval.
 * @throws OmaiError with code `value` for `n < 2`.
 * @concept monte-carlo
 * @example
 * ```ts
 * import { monteCarloEstimate } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * const e = monteCarloEstimate({ fn: (u) => u * u, key: key(0), n: 20000 })
 * Math.abs(e.value - 1 / 3) < 4 * (e.standardError ?? 0) // => true
 * ```
 */
export function monteCarloEstimate<T = number>(options: MonteCarloEstimateOptions<T>): Estimate {
  checkN(options.n, 2, 'monteCarloEstimate')
  const draw = (options.sample ?? (uniformSampler as unknown as Sampler<T>))(options.key, options.n)
  const f = draw.map((x) => options.fn(x))
  return normalEstimate(meanOf(f), Math.sqrt(varianceOf(f, 1) / options.n), 'Monte Carlo')
}

/** A control variate: a function of the draw with a known expectation. */
export interface ControlVariate<T = number> {
  /** The control `g(x)`. */
  readonly fn: (x: T) => number
  /** Its exact expectation `E g(X)`. */
  readonly mean: number
}

/** Options for {@link controlVariateEstimate}. */
export interface ControlVariateOptions<T = number> {
  /** The integrand `f(x)`. */
  readonly fn: (x: T) => number
  /** One or more controls with known means. */
  readonly controls: readonly ControlVariate<T>[]
  /** Draws `n` samples (default `U[0, 1)`). */
  readonly sample?: Sampler<T>
  /** Random key. */
  readonly key: Key
  /** Number of samples (more than the number of controls + 1). */
  readonly n: number
}

/** The result of {@link controlVariateEstimate}. */
export interface ControlVariateResult extends ReducedEstimate {
  /** The fitted coefficients `beta = C_gg^{-1} c_gf`: the estimate is `mean(f) - beta . (mean(g) - E g)`. */
  readonly beta: readonly number[]
  /** The multiple correlation `R` between `f` and the controls; the variance falls by the factor `1 - R^2`. */
  readonly correlation: number
}

/**
 * The control-variate estimate `mean(f) - beta . (mean(g) - E g)` with the regression coefficients `beta` that minimise
 * the variance. The estimator is unbiased up to `O(1/n)` (the coefficients are fitted from the same sample) and its
 * variance is the residual variance of `f` regressed on the controls, `(1 - R^2) Var f / n`.
 *
 * @param options - Integrand, controls with their means, sampler, key and `n`.
 * @returns The estimate with `beta`, `R`, and the plain standard error for comparison.
 * @throws OmaiError with code `value` for fewer than `controls + 2` samples or no controls.
 * @concept control-variates
 * @reference A. B. Owen, Monte Carlo theory, methods and examples (2013), chapter 8.
 * @example
 * ```ts
 * import { controlVariateEstimate } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * const e = controlVariateEstimate({
 *   fn: (u) => Math.exp(u),
 *   controls: [{ fn: (u) => u, mean: 0.5 }],
 *   key: key(0),
 *   n: 4000,
 * })
 * [Math.abs(e.value - (Math.E - 1)) < 4 * (e.standardError ?? 0), e.varianceReductionFactor > 20] // => [true, true]
 * ```
 */
export function controlVariateEstimate<T = number>(options: ControlVariateOptions<T>): ControlVariateResult {
  const { fn, controls, n } = options
  const p = controls.length
  if (p === 0) throw new OmaiError('value', 'controlVariateEstimate: at least one control is needed')
  checkN(n, p + 2, 'controlVariateEstimate')
  const draw = (options.sample ?? (uniformSampler as unknown as Sampler<T>))(options.key, n)
  const f = draw.map((x) => fn(x))
  const g = controls.map((c) => draw.map((x) => c.fn(x) - c.mean))
  const fBar = meanOf(f)
  const gBar = g.map((col) => meanOf(col))
  const cov = (a: number[], ma: number, b: number[], mb: number): number => {
    let s = 0
    for (let i = 0; i < n; i++) s += ((a[i] ?? 0) - ma) * ((b[i] ?? 0) - mb)
    return s / (n - 1)
  }
  const cgg = new Float64Array(p * p)
  const cgf = new Float64Array(p)
  for (let a = 0; a < p; a++) {
    cgf[a] = cov(g[a] as number[], gBar[a] as number, f, fBar)
    for (let b = 0; b < p; b++)
      cgg[a * p + b] = cov(g[a] as number[], gBar[a] as number, g[b] as number[], gBar[b] as number)
  }
  const beta = toFlat(
    solve(
      tensor(Array.from({ length: p }, (_, a) => Array.from(cgg.subarray(a * p, (a + 1) * p)))),
      tensor(Array.from(cgf)),
    ),
  )
  const adjusted = f.map((v, i) => v - beta.reduce((s, b, j) => s + b * (g[j]?.[i] ?? 0), 0))
  const value = meanOf(adjusted)
  const plainVar = varianceOf(f, 1)
  const resVar = varianceOf(adjusted, 1)
  return {
    ...reduced(value, Math.sqrt(resVar / n), Math.sqrt(plainVar / n), n, 'control variates'),
    beta,
    correlation: plainVar > 0 ? Math.sqrt(Math.max(0, 1 - resVar / plainVar)) : 0,
  }
}

/** Options for {@link antitheticEstimate}. */
export interface AntitheticOptions {
  /** The integrand of a point `x` of length `dim`. */
  readonly fn: (x: Float64Array) => number
  /** Dimension of the point. */
  readonly dim: number
  /** The base distribution and its reflection: `'uniform'` pairs `u` with `1 - u`, `'normal'` pairs `z` with `-z`. */
  readonly distribution?: 'uniform' | 'normal'
  /** Random key. */
  readonly key: Key
  /** Number of antithetic pairs (the evaluations are twice this; at least 2). */
  readonly pairs: number
}

/** The result of {@link antitheticEstimate}. */
export interface AntitheticResult extends ReducedEstimate {
  /** The correlation between `f(X)` and `f(X')` over the pairs: negative correlation is what helps. */
  readonly pairCorrelation: number
}

/**
 * The antithetic estimate (Hammersley and Morton 1956): the mean over pairs of `(f(X) + f(X')) / 2`, where `X'` is the
 * reflection of `X` (`1 - u` for uniforms, `-z` for standard normals). The pairs are independent, so the standard error
 * is `sd(pair means) / sqrt(pairs)`; it is smaller than plain Monte Carlo on the same `2 pairs` evaluations by the factor
 * `1 + corr(f(X), f(X'))` in variance, which is below 1 for monotone integrands.
 *
 * @param options - Integrand, dimension, distribution, key and number of pairs.
 * @returns The estimate with the pair correlation and the plain standard error for the same evaluations.
 * @throws OmaiError with code `value` for fewer than 2 pairs or a non-positive dimension.
 * @concept antithetic-variates
 * @example
 * ```ts
 * import { antitheticEstimate } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * const e = antitheticEstimate({ fn: (x) => Math.exp(x[0] ?? 0), dim: 1, key: key(0), pairs: 2000 })
 * [Math.abs(e.value - (Math.E - 1)) < 4 * (e.standardError ?? 0), e.pairCorrelation < -0.9] // => [true, true]
 * ```
 */
export function antitheticEstimate(options: AntitheticOptions): AntitheticResult {
  const { fn, dim, pairs } = options
  checkN(pairs, 2, 'antitheticEstimate')
  if (!(Number.isInteger(dim) && dim >= 1))
    throw new OmaiError('value', `antitheticEstimate: dim must be a positive integer, got ${dim}`)
  const kind = options.distribution ?? 'uniform'
  const base = kind === 'uniform' ? uniform(options.key, [pairs, dim]).data : normal(options.key, [pairs, dim]).data
  const first = new Float64Array(pairs)
  const second = new Float64Array(pairs)
  for (let i = 0; i < pairs; i++) {
    const x = Float64Array.from(base.subarray(i * dim, (i + 1) * dim))
    first[i] = fn(x)
    second[i] = fn(x.map((v) => (kind === 'uniform' ? 1 - v : -v)))
  }
  const means = Array.from(first, (v, i) => (v + (second[i] ?? 0)) / 2)
  const all = Array.from(first).concat(Array.from(second))
  const mf = meanOf(first)
  const ms = meanOf(second)
  let cross = 0
  for (let i = 0; i < pairs; i++) cross += ((first[i] ?? 0) - mf) * ((second[i] ?? 0) - ms)
  const denom = Math.sqrt(varianceOf(first, 1) * varianceOf(second, 1)) * (pairs - 1)
  return {
    ...reduced(
      meanOf(means),
      Math.sqrt(varianceOf(means, 1) / pairs),
      Math.sqrt(varianceOf(all, 1) / (2 * pairs)),
      2 * pairs,
      'antithetic variates',
    ),
    pairCorrelation: denom > 0 ? cross / denom : 0,
  }
}

/** One stratum of a partition of the sample space. */
export interface Stratum<T = number> {
  /** The probability of the stratum `w_s`; the weights sum to 1. */
  readonly weight: number
  /** Draws `n` samples from the distribution conditioned on the stratum. */
  readonly sample: Sampler<T>
  /** The within-stratum standard deviation when known (for Neyman allocation); estimated by a pilot otherwise. */
  readonly std?: number
}

/**
 * `count` equal strata of `U[0, 1)`: stratum `j` is `[j / count, (j + 1) / count)` with weight `1 / count`, sampled
 * uniformly within.
 *
 * @param count - Number of strata.
 * @returns The strata.
 * @throws OmaiError with code `value` for a non-positive count.
 * @concept stratified-sampling
 * @example
 * ```ts
 * import { uniformStrata } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * const s = uniformStrata(4)
 * [s.length, s[2]?.weight, (s[2]?.sample(key(0), 1)[0] ?? 0) >= 0.5] // => [4, 0.25, true]
 * ```
 */
export function uniformStrata(count: number): Stratum<number>[] {
  if (!(Number.isInteger(count) && count >= 1))
    throw new OmaiError('value', `uniformStrata: count must be a positive integer, got ${count}`)
  return Array.from({ length: count }, (_, j) => ({
    weight: 1 / count,
    sample: (key: Key, n: number) => Array.from(uniform(key, [n]).data, (u) => (j + u) / count),
  }))
}

/** Options for {@link stratifiedEstimate}. */
export interface StratifiedOptions<T = number> {
  /** The strata (a partition: weights positive and summing to 1). */
  readonly strata: readonly Stratum<T>[]
  /** The integrand. */
  readonly fn: (x: T) => number
  /** Total sample budget `N` (at least two per stratum). */
  readonly n: number
  /** `'proportional'` (`n_s` proportional to `w_s`, default) or `'neyman'` (`n_s` proportional to `w_s sigma_s`). */
  readonly allocation?: 'proportional' | 'neyman'
  /** Pilot draws per stratum to estimate `sigma_s` for Neyman allocation when `std` is not given (default 20). */
  readonly pilot?: number
  /** Random key. */
  readonly key: Key
}

/** The result of {@link stratifiedEstimate}. */
export interface StratifiedResult extends ReducedEstimate {
  /** Samples per stratum. */
  readonly counts: readonly number[]
  /** Sample mean within each stratum. */
  readonly means: readonly number[]
  /** Sample variance within each stratum. */
  readonly variances: readonly number[]
}

function allocate(weights: readonly number[], n: number): number[] {
  const total = weights.reduce((a, b) => a + b, 0)
  const ideal = weights.map((w) => (w / total) * n)
  const counts = ideal.map((v) => Math.max(2, Math.floor(v)))
  let left = n - counts.reduce((a, b) => a + b, 0)
  const order = ideal.map((v, i) => [v - Math.floor(v), i] as const).sort((a, b) => b[0] - a[0])
  for (let t = 0; left > 0; t = (t + 1) % order.length, left--)
    counts[order[t]?.[1] ?? 0] = (counts[order[t]?.[1] ?? 0] ?? 0) + 1
  // The floor of two per stratum can overshoot the budget; take the excess from the largest strata.
  while (left < 0) {
    let big = 0
    for (let i = 1; i < counts.length; i++) if ((counts[i] ?? 0) > (counts[big] ?? 0)) big = i
    counts[big] = (counts[big] ?? 0) - 1
    left += 1
  }
  return counts
}

/**
 * The stratified estimate `sum_s w_s mean_s` with variance `sum_s w_s^2 s_s^2 / n_s`. Proportional allocation never does
 * worse than plain Monte Carlo; Neyman allocation `n_s ~ w_s sigma_s` minimises the variance for the budget. The plain
 * standard error reported is for the same budget with the law of total variance,
 * `(sum_s w_s (s_s^2 + (m_s - m)^2)) / N`.
 *
 * @param options - Strata, integrand, budget `n`, allocation, pilot size and key.
 * @returns The estimate, the per-stratum counts, means and variances.
 * @throws OmaiError with code `value` for weights that are not a partition or too small a budget.
 * @concept stratified-sampling
 * @example
 * ```ts
 * import { stratifiedEstimate, uniformStrata } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * const e = stratifiedEstimate({ strata: uniformStrata(10), fn: (u) => u * u, n: 1000, key: key(0) })
 * [Math.abs(e.value - 1 / 3) < 0.01, e.varianceReductionFactor > 10] // => [true, true]
 * ```
 */
export function stratifiedEstimate<T = number>(options: StratifiedOptions<T>): StratifiedResult {
  const { strata, fn, n } = options
  const S = strata.length
  if (S === 0) throw new OmaiError('value', 'stratifiedEstimate: at least one stratum is needed')
  const wsum = strata.reduce((a, s) => a + s.weight, 0)
  if (!strata.every((s) => s.weight > 0) || Math.abs(wsum - 1) > 1e-9) {
    throw new OmaiError('value', `stratifiedEstimate: weights must be positive and sum to 1, got sum ${wsum}`)
  }
  checkN(n, 2 * S, 'stratifiedEstimate')
  let counts: number[]
  if ((options.allocation ?? 'proportional') === 'neyman') {
    const pilot = options.pilot ?? 20
    const sigma = strata.map((s, j) =>
      s.std !== undefined
        ? s.std
        : Math.sqrt(
            varianceOf(
              s.sample(foldIn(options.key, 1_000_003 + j), pilot).map((x) => fn(x)),
              1,
            ),
          ),
    )
    counts = allocate(
      strata.map((s, j) => s.weight * Math.max(sigma[j] ?? 0, 1e-12)),
      n,
    )
  } else {
    counts = allocate(
      strata.map((s) => s.weight),
      n,
    )
  }
  const means: number[] = []
  const variances: number[] = []
  strata.forEach((s, j) => {
    const f = s.sample(foldIn(options.key, j), counts[j] ?? 2).map((x) => fn(x))
    means.push(meanOf(f))
    variances.push(varianceOf(f, 1))
  })
  const value = strata.reduce((a, s, j) => a + s.weight * (means[j] ?? 0), 0)
  const variance = strata.reduce((a, s, j) => a + (s.weight ** 2 * (variances[j] ?? 0)) / (counts[j] ?? 1), 0)
  const total = strata.reduce((a, s, j) => a + s.weight * ((variances[j] ?? 0) + ((means[j] ?? 0) - value) ** 2), 0)
  return {
    ...reduced(
      value,
      Math.sqrt(variance),
      Math.sqrt(total / n),
      n,
      `stratified (${options.allocation ?? 'proportional'})`,
    ),
    counts,
    means,
    variances,
  }
}

/** Options for {@link raoBlackwellEstimate}. */
export interface RaoBlackwellOptions<T> {
  /** Draws `n` joint samples `s` (containing the conditioning statistic and the nuisance variables). */
  readonly sample: Sampler<T>
  /** The raw estimator `h(s)`, which uses the nuisance variables. */
  readonly raw: (s: T) => number
  /** Its conditional expectation `E[h | statistic](s)`, integrating the nuisance variables out analytically. */
  readonly conditional: (s: T) => number
  /** Random key. */
  readonly key: Key
  /** Number of joint samples (at least 2). */
  readonly n: number
}

/** The result of {@link raoBlackwellEstimate}. */
export interface RaoBlackwellResult extends ReducedEstimate {
  /** The raw estimator's mean over the same draws. */
  readonly rawValue: number
  /** `Var(E[h | S])`, the variance that remains. */
  readonly varianceOfConditionalMean: number
  /** `E[Var(h | S)]`, estimated by `mean((h - E[h | S])^2)`: the variance conditioning removes. */
  readonly expectedConditionalVariance: number
}

/**
 * The Rao-Blackwellised estimate: the mean of `E[h | S]` in place of the mean of `h`, with the law of total variance
 * `Var h = Var E[h | S] + E Var(h | S)` showing exactly what was removed. Never worse than the raw estimator.
 *
 * @param options - Sampler of joint draws, the raw and conditional estimators, key and `n`.
 * @returns The estimate with both parts of the variance decomposition.
 * @throws OmaiError with code `value` for `n < 2`.
 * @concept rao-blackwellisation
 * @example
 * ```ts
 * import { raoBlackwellEstimate } from 'omai-engine/sampling'
 * import { key, uniform } from 'omai-engine/random'
 * // h = X + Y with X, Y uniform: E[h | X] = X + 1/2.
 * const e = raoBlackwellEstimate<[number, number]>({
 *   sample: (k, n) => {
 *     const d = uniform(k, [n, 2]).data
 *     return Array.from({ length: n }, (_, i) => [d[2 * i] ?? 0, d[2 * i + 1] ?? 0] as [number, number])
 *   },
 *   raw: ([x, y]) => x + y,
 *   conditional: ([x]) => x + 0.5,
 *   key: key(0),
 *   n: 1000,
 * })
 * e.varianceReductionFactor > 1 // => true
 * ```
 */
export function raoBlackwellEstimate<T>(options: RaoBlackwellOptions<T>): RaoBlackwellResult {
  checkN(options.n, 2, 'raoBlackwellEstimate')
  const draw = options.sample(options.key, options.n)
  const raw = draw.map((s) => options.raw(s))
  const cond = draw.map((s) => options.conditional(s))
  const residual = raw.map((v, i) => (v - (cond[i] ?? 0)) ** 2)
  return {
    ...reduced(
      meanOf(cond),
      Math.sqrt(varianceOf(cond, 1) / options.n),
      Math.sqrt(varianceOf(raw, 1) / options.n),
      options.n,
      'Rao-Blackwell',
    ),
    rawValue: meanOf(raw),
    varianceOfConditionalMean: varianceOf(cond, 1),
    expectedConditionalVariance: meanOf(residual),
  }
}

/** Options for {@link quasiMonteCarloEstimate}. */
export interface QuasiMonteCarloOptions {
  /** The integrand of a point `u` in `[0, 1)^dim`. */
  readonly fn: (u: Float64Array) => number
  /** Dimension (1 to 1111 for Sobol). */
  readonly dim: number
  /** Points per replicate (a power of two suits Sobol). */
  readonly n: number
  /** Independent randomisations (default 16, at least 2). */
  readonly replicates?: number
  /** `'sobol'` with Owen-style digital scrambling (default) or `'halton'` with Cranley-Patterson random shifts. */
  readonly sequence?: 'sobol' | 'halton'
  /** Random key for the scrambles or shifts. */
  readonly key: Key
}

/**
 * A randomised quasi-Monte Carlo estimate: the mean of `fn` over a low-discrepancy point set, randomised `replicates`
 * times independently (a scrambled Sobol net, or a Halton sequence under a random shift mod 1). Each replicate is
 * unbiased and the replicates are independent, so the standard error is their spread; for smooth integrands the error
 * falls close to `1 / n` rather than `1 / sqrt(n)`. The plain standard error is for `n replicates` independent draws.
 *
 * @param options - Integrand, dimension, points, replicates, sequence and key.
 * @returns The estimate with the plain Monte Carlo standard error for the same number of evaluations.
 * @throws OmaiError with code `value` for fewer than two replicates.
 * @concept quasi-monte-carlo
 * @reference A. B. Owen, Randomized Halton sequences and Sobol nets, in Monte Carlo and Quasi-Monte Carlo Methods (2003).
 * @example
 * ```ts
 * import { quasiMonteCarloEstimate } from 'omai-engine/sampling'
 * import { key } from 'omai-engine/random'
 * const e = quasiMonteCarloEstimate({ fn: (u) => (u[0] ?? 0) * (u[1] ?? 0), dim: 2, n: 1024, key: key(0) })
 * Math.abs(e.value - 0.25) < 1e-3 // => true
 * ```
 */
export function quasiMonteCarloEstimate(options: QuasiMonteCarloOptions): ReducedEstimate {
  const { fn, dim, n } = options
  const replicates = options.replicates ?? 16
  checkN(n, 2, 'quasiMonteCarloEstimate')
  checkN(replicates, 2, 'quasiMonteCarloEstimate')
  const sequence = options.sequence ?? 'sobol'
  const base = sequence === 'halton' ? halton(n, dim).data : undefined
  const means: number[] = []
  const firstValues: number[] = []
  for (let r = 0; r < replicates; r++) {
    const k = foldIn(options.key, r)
    const pts =
      base === undefined
        ? scrambledSobol(k, n, dim).data
        : (() => {
            const shift = uniform(k, [dim]).data
            return base.map((v, i) => (v + (shift[i % dim] ?? 0)) % 1)
          })()
    let total = 0
    for (let i = 0; i < n; i++) {
      const v = fn(Float64Array.from(pts.subarray(i * dim, (i + 1) * dim)))
      total += v
      if (r === 0) firstValues.push(v)
    }
    means.push(total / n)
  }
  const plainSe = Math.sqrt(varianceOf(firstValues, 1) / (n * replicates))
  return reduced(
    meanOf(means),
    Math.sqrt(varianceOf(means, 1) / replicates),
    plainSe,
    n * replicates,
    `randomised quasi-Monte Carlo (${sequence})`,
  )
}

/** The four estimators {@link varianceReductionAlgorithm} runs side by side. */
export const VARIANCE_REDUCTION_TECHNIQUES = ['plain', 'antithetic', 'control', 'stratified'] as const

/** One of {@link VARIANCE_REDUCTION_TECHNIQUES}. */
export type VarianceReductionTechnique = (typeof VARIANCE_REDUCTION_TECHNIQUES)[number]

/** The state of {@link varianceReductionAlgorithm}: running sums of each technique's observations. */
export interface VarianceReductionState {
  /** Batches done. */
  readonly batches: number
  /** Plain: `[n, sum f, sum f^2]`. */
  readonly plain: readonly number[]
  /** Antithetic: `[pairs, sum a, sum a^2]` for the pair means `a`. */
  readonly antithetic: readonly number[]
  /** Control: `[n, sum f, sum f^2, sum g, sum g^2, sum fg]` with `g` centred at its mean. */
  readonly control: readonly number[]
  /** Stratified: per stratum `sum f` (first half) and `sum f^2` (second half). */
  readonly strata: readonly number[]
}

/** A running estimate and its standard error. */
export interface RunningEstimate {
  /** The estimate. */
  readonly value: number
  /** Its standard error (infinite until two observations are in). */
  readonly se: number
}

/** Input of {@link varianceReductionAlgorithm}. */
export interface VarianceReductionOptions {
  /** The integrand of `u ~ U[0, 1)` (default `exp`). */
  readonly fn?: (u: number) => number
  /** The control variate (default the identity). */
  readonly control?: ControlVariate<number>
  /** Evaluations per technique per step (even, at least 2; default 10). The stratified one uses this many strata. */
  readonly batch?: number
  /** The true integral, if known, logged as the error. */
  readonly truth?: number
  /** Steps to run (the algorithm is done after this many; default 1000). */
  readonly batches?: number
}

function sumsEstimate(s: readonly number[]): { value: number; se: number } {
  const n = s[0] ?? 0
  if (n < 2) return { value: n === 0 ? 0 : (s[1] ?? 0) / n, se: Number.POSITIVE_INFINITY }
  const mean = (s[1] ?? 0) / n
  const variance = Math.max(0, ((s[2] ?? 0) - n * mean * mean) / (n - 1))
  return { value: mean, se: Math.sqrt(variance / n) }
}

/**
 * The running estimate and standard error of each technique in a {@link VarianceReductionState}: the plain mean, the
 * mean of antithetic pair means, the control-variate regression estimate, and the stratified mean over equal strata.
 *
 * @param s - A state of {@link varianceReductionAlgorithm}.
 * @returns The four estimates.
 * @concept variance-reduction
 * @example
 * ```ts
 * import { varianceReductionAlgorithm, varianceReductionEstimates } from 'omai-engine/sampling'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const res = runAlgorithm(varianceReductionAlgorithm({ batches: 100 }), { seed: 3, maxSteps: 1000 })
 * Math.abs(varianceReductionEstimates(res.state).stratified.value - (Math.E - 1)) < 0.01 // => true
 * ```
 */
export function varianceReductionEstimates(
  s: VarianceReductionState,
): Record<VarianceReductionTechnique, RunningEstimate> {
  const m = s.strata.length / 2
  const t = s.batches
  const c = s.control
  const n = c[0] ?? 0
  let control: RunningEstimate = { value: n === 0 ? 0 : (c[1] ?? 0) / n, se: Number.POSITIVE_INFINITY }
  if (n >= 3) {
    const fBar = (c[1] ?? 0) / n
    const gBar = (c[3] ?? 0) / n
    const cff = ((c[2] ?? 0) - n * fBar * fBar) / (n - 1)
    const cgg = ((c[4] ?? 0) - n * gBar * gBar) / (n - 1)
    const cgf = ((c[5] ?? 0) - n * fBar * gBar) / (n - 1)
    const beta = cgg > 0 ? cgf / cgg : 0
    control = { value: fBar - beta * gBar, se: Math.sqrt(Math.max(0, cff - beta * cgf) / n) }
  }
  let stratumVar = 0
  let stratifiedValue = 0
  for (let j = 0; j < m; j++) {
    const sum = s.strata[j] ?? 0
    const sq = s.strata[m + j] ?? 0
    stratifiedValue += sum / Math.max(t, 1) / m
    if (t >= 2) stratumVar += Math.max(0, (sq - (sum * sum) / t) / (t - 1)) / t / (m * m)
  }
  return {
    plain: sumsEstimate(s.plain),
    antithetic: sumsEstimate(s.antithetic),
    control,
    stratified: { value: stratifiedValue, se: t >= 2 ? Math.sqrt(stratumVar) : Number.POSITIVE_INFINITY },
  }
}

/**
 * Plain, antithetic, control-variate and stratified Monte Carlo on the same integrand, each given the same number of
 * evaluations per step from independent draws, logging every technique's running estimate and standard error: the
 * standard errors fall as `1 / sqrt(t)` for all four, with the reduced ones below the plain one by their variance
 * factors. The integrand defaults to `exp(u)` on `U[0, 1)` with the control `u` (known mean 1/2), whose truth is `e - 1`.
 *
 * @param options - Integrand, control, batch size, truth and number of batches.
 * @returns The algorithm.
 * @throws OmaiError with code `value` for an odd or too small batch.
 * @concept variance-reduction
 * @example
 * ```ts
 * import { varianceReductionAlgorithm, varianceReductionEstimates } from 'omai-engine/sampling'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const res = runAlgorithm(varianceReductionAlgorithm({ batches: 200 }), { seed: 1, maxSteps: 1000 })
 * const { plain, control } = varianceReductionEstimates(res.state)
 * [res.state.batches, control.se < plain.se / 5] // => [200, true]
 * ```
 */
export function varianceReductionAlgorithm(options: VarianceReductionOptions = {}): Algorithm<VarianceReductionState> {
  const fn = options.fn ?? Math.exp
  const controlVariate = options.control ?? { fn: (u: number) => u, mean: 0.5 }
  const m = options.batch ?? 10
  const total = options.batches ?? 1000
  if (!(Number.isInteger(m) && m >= 2 && m % 2 === 0))
    throw new OmaiError('value', `varianceReductionAlgorithm: batch must be an even integer of at least 2, got ${m}`)
  const metrics = Object.fromEntries(
    VARIANCE_REDUCTION_TECHNIQUES.flatMap((name) => [
      [`${name}/value`, { description: `the ${name} estimate` }],
      [`${name}/se`, { description: `the ${name} standard error`, direction: 'lower' as const }],
    ]),
  )
  return {
    name: 'variance-reduction',
    config: { batch: m, batches: total },
    schema: channels({
      metrics:
        options.truth === undefined
          ? metrics
          : {
              ...metrics,
              'plain/error': { description: 'plain estimate minus truth' },
              'stratified/error': { description: 'stratified estimate minus truth' },
            },
    }),
    init: () => {
      return {
        batches: 0,
        plain: [0, 0, 0],
        antithetic: [0, 0, 0],
        control: [0, 0, 0, 0, 0, 0],
        strata: new Array<number>(2 * m).fill(0),
      }
    },
    step: (s, ctx) => {
      const u = uniform(ctx.key, [(7 * m) / 2]).data
      const plain = [...s.plain]
      const antithetic = [...s.antithetic]
      const control = [...s.control]
      const strata = [...s.strata]
      for (let i = 0; i < m; i++) {
        const f = fn(u[i] ?? 0)
        plain[0] = (plain[0] ?? 0) + 1
        plain[1] = (plain[1] ?? 0) + f
        plain[2] = (plain[2] ?? 0) + f * f
      }
      for (let i = 0; i < m / 2; i++) {
        const x = u[m + i] ?? 0
        const a = (fn(x) + fn(1 - x)) / 2
        antithetic[0] = (antithetic[0] ?? 0) + 1
        antithetic[1] = (antithetic[1] ?? 0) + a
        antithetic[2] = (antithetic[2] ?? 0) + a * a
      }
      for (let i = 0; i < m; i++) {
        const x = u[(3 * m) / 2 + i] ?? 0
        const f = fn(x)
        const g = controlVariate.fn(x) - controlVariate.mean
        control[0] = (control[0] ?? 0) + 1
        control[1] = (control[1] ?? 0) + f
        control[2] = (control[2] ?? 0) + f * f
        control[3] = (control[3] ?? 0) + g
        control[4] = (control[4] ?? 0) + g * g
        control[5] = (control[5] ?? 0) + f * g
      }
      for (let j = 0; j < m; j++) {
        const f = fn((j + (u[(5 * m) / 2 + j] ?? 0)) / m)
        strata[j] = (strata[j] ?? 0) + f
        strata[m + j] = (strata[m + j] ?? 0) + f * f
      }
      const batches = s.batches + 1
      return { batches, plain, antithetic, control, strata }
    },
    done: (s) => (s.batches >= total ? 'completed' : undefined),
    observe: (s) => {
      const e = varianceReductionEstimates(s)
      return {
        metrics: {
          ...Object.fromEntries(
            VARIANCE_REDUCTION_TECHNIQUES.flatMap((name) => [
              [`${name}/value`, e[name].value],
              [`${name}/se`, Number.isFinite(e[name].se) ? e[name].se : 0],
            ]),
          ),
          ...(options.truth === undefined
            ? {}
            : { 'plain/error': e.plain.value - options.truth, 'stratified/error': e.stratified.value - options.truth }),
        },
      }
    },
    serialise: (s) => stateToJson(s),
    deserialise: (j: Json) => stateFromJson(j) as VarianceReductionState,
  }
}
