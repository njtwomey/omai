/**
 * Variance-stabilising power transforms that make data closer to normal: Box-Cox for positive data and Yeo-Johnson for
 * data of any sign, with lambda fitted by maximum likelihood and its profile-likelihood interval, their log-likelihoods
 * and inverses (as scipy.stats), and standardisation to zero mean and unit variance.
 */

import { meanOf, varianceOf, WhiteboxError } from '../base/index.ts'
import { chi2 } from '../dists/index.ts'
import { brentq } from '../numerics/roots/index.ts'
import {
  abs,
  add,
  asTensor,
  div,
  expm1,
  greaterEqual,
  item,
  log,
  log1p,
  mean,
  mul,
  primalOf,
  square,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  type Value,
  where,
} from '../tensor/index.ts'
import { maximiseSmooth, slopeOf } from './_optimise.ts'
import { mat, rowsOf, sample, vec } from './_util.ts'

// expm1(lambda L) / lambda as a differentiable function of lambda, by its series L + lambda L^2 / 2 + lambda^2 L^3 / 6
// near lambda = 0 (where the ratio is L and its derivative L^2 / 2).
function expm1Ratio(lambda: Value, L: Value): Value {
  if (Math.abs(item(primalOf(lambda))) > 1e-6) return div(expm1(mul(lambda, L)), lambda)
  const L2 = mul(L, L)
  return add(add(L, mul(mul(lambda, L2), 0.5)), mul(mul(mul(lambda, lambda), mul(L2, L)), 1 / 6))
}

/**
 * The Box-Cox log-likelihood of lambda (scipy.stats.boxcox_llf): (lambda - 1) sum log x - n / 2 log var(y_lambda),
 * with the biased variance of the transformed data.
 *
 * @param lambda - The power.
 * @param x - Positive data.
 * @returns The profile log-likelihood.
 * @example
 * ```ts
 * import { boxcoxLlf } from 'whitebox-engine/stats'
 * boxcoxLlf(1, [1, 2, 3, 4]) // => -0.446287
 * ```
 */
export function boxcoxLlf(lambda: number, x: TensorLike): number {
  const a = positive('boxcoxLlf', x)
  return llfBoxcox(Float64Array.from(a, Math.log), lambda)
}

function positive(name: string, x: TensorLike): Float64Array {
  const a = sample(name, x, 2)
  for (const v of a) if (!(v > 0)) throw new WhiteboxError('domain', `${name}: data must be positive`)
  return a
}

function llfBoxcox(logx: Float64Array, lambda: number): number {
  const n = logx.length
  let sumLog = 0
  for (const v of logx) sumLog += v
  const y = lambda === 0 ? logx : Float64Array.from(logx, (v) => Math.expm1(lambda * v) / lambda)
  return (lambda - 1) * sumLog - (n / 2) * Math.log(varianceOf(y))
}

/** The maximiser of a smooth profile log-likelihood, bracketed by doubling [-2, 2] until its slope changes sign. */
function maximise(d: (lambda: number) => number, llf: (lambda: number) => number): number {
  let lo = -2
  let hi = 2
  for (let i = 0; i < 60 && Math.sign(d(lo)) === Math.sign(d(hi)); i++) {
    lo *= 2
    hi *= 2
  }
  if (Math.sign(d(lo)) === Math.sign(d(hi)))
    throw new WhiteboxError('convergence', 'normmax: the log-likelihood has no interior maximum')
  return maximiseSmooth(llf, d, lo, hi)
}

/**
 * The maximum-likelihood Box-Cox lambda (scipy.stats.boxcox_normmax with method 'mle'): the root of the derivative of
 * the profile log-likelihood (by reverse-mode differentiation), found by Brent's method.
 *
 * @param x - Positive data.
 * @returns The lambda maximising the log-likelihood.
 * @example
 * ```ts
 * import { boxcoxNormmax } from 'whitebox-engine/stats'
 * boxcoxNormmax([1, 2, 4, 8, 16]) // => 0
 * ```
 */
export function boxcoxNormmax(x: TensorLike): number {
  const a = positive('boxcoxNormmax', x)
  const logx = Float64Array.from(a, Math.log)
  const lt = vec(logx)
  const sumLog = logx.reduce((s, v) => s + v, 0)
  const f = (lambda: Value): Value => {
    const y = expm1Ratio(lambda, lt)
    const m = mean(y)
    const v = mean(square(sub(y, m)))
    return sub(mul(sub(lambda, 1), sumLog), mul(log(v), a.length / 2))
  }
  const lam = maximise(
    slopeOf((l) => sum(f(l))),
    (l) => llfBoxcox(logx, l),
  )
  return Math.abs(lam) < 1e-12 ? 0 : lam
}

/** A transformed sample with its lambda (and the profile-likelihood interval of lambda when requested). */
export interface PowerTransformResult {
  readonly transformed: Tensor
  readonly lambda: number
  readonly interval?: readonly [number, number]
}

/**
 * The Box-Cox power transform (scipy.stats.boxcox): (x^lambda - 1) / lambda, log x at lambda 0, with lambda given or
 * fitted by maximum likelihood; with `level`, the profile-likelihood interval {lambda : llf >= max - chi2_1(level) /
 * 2}.
 *
 * @param x - Positive data.
 * @param options - lambda (fitted when omitted) and level for the interval.
 * @returns The transformed data and lambda.
 * @example
 * ```ts
 * import { boxcox } from 'whitebox-engine/stats'
 * boxcox([1, 4, 9], { lambda: 0.5 }).transformed // => [0, 2, 4]
 * ```
 */
export function boxcox(
  x: TensorLike,
  options: { readonly lambda?: number; readonly level?: number } = {},
): PowerTransformResult {
  const a = positive('boxcox', x)
  const lambda = options.lambda ?? boxcoxNormmax(a)
  const logx = Float64Array.from(a, Math.log)
  const transformed = vec(lambda === 0 ? logx : Float64Array.from(logx, (v) => Math.expm1(lambda * v) / lambda))
  if (options.level === undefined) return { transformed, lambda }
  const interval = profileInterval((l) => llfBoxcox(logx, l), lambda, options.level)
  return { transformed, lambda, interval }
}

function profileInterval(llf: (l: number) => number, at: number, level: number): [number, number] {
  const q = item(chi2(1).ppf(level))
  const target = llf(at) - q / 2
  const f = (l: number): number => llf(l) - target
  let lo = at - 1
  for (let i = 0; i < 60 && f(lo) > 0; i++) lo = at - (at - lo) * 2
  let hi = at + 1
  for (let i = 0; i < 60 && f(hi) > 0; i++) hi = at + (hi - at) * 2
  return [brentq(f, lo, at).root, brentq(f, at, hi).root]
}

/**
 * The inverse Box-Cox transform (scipy.special.inv_boxcox): (lambda y + 1)^(1 / lambda), exp(y) at lambda 0.
 *
 * @param y - Transformed values.
 * @param lambda - The power.
 * @returns The original values.
 * @example
 * ```ts
 * import { invBoxcox } from 'whitebox-engine/stats'
 * invBoxcox([0, 2, 4], 0.5) // => [1, 4, 9]
 * ```
 */
export function invBoxcox(y: TensorLike, lambda: number): Tensor {
  const v = sample('invBoxcox', y, 0)
  return vec(Array.from(v, (t) => (lambda === 0 ? Math.exp(t) : Math.exp(Math.log1p(lambda * t) / lambda))))
}

function yeoValue(x: number, lambda: number): number {
  if (x >= 0) return Math.abs(lambda) < 1e-300 ? Math.log1p(x) : Math.expm1(lambda * Math.log1p(x)) / lambda
  return Math.abs(lambda - 2) < 1e-300 ? -Math.log1p(-x) : -Math.expm1((2 - lambda) * Math.log1p(-x)) / (2 - lambda)
}

function llfYeo(a: Float64Array, lambda: number): number {
  const n = a.length
  let s = 0
  for (const v of a) s += Math.sign(v) * Math.log1p(Math.abs(v))
  const y = Float64Array.from(a, (v) => yeoValue(v, lambda))
  return -(n / 2) * Math.log(varianceOf(y)) + (lambda - 1) * s
}

/**
 * The Yeo-Johnson log-likelihood of lambda (scipy.stats.yeojohnson_llf): -n / 2 log var(y_lambda) + (lambda - 1) sum
 * sign(x) log(1 + |x|).
 *
 * @param lambda - The power.
 * @param x - Data (any sign).
 * @returns The profile log-likelihood.
 * @example
 * ```ts
 * import { yeojohnsonLlf } from 'whitebox-engine/stats'
 * yeojohnsonLlf(1, [-1, 0, 1]) // => 0.608198
 * ```
 */
export function yeojohnsonLlf(lambda: number, x: TensorLike): number {
  return llfYeo(sample('yeojohnsonLlf', x, 2), lambda)
}

/**
 * The maximum-likelihood Yeo-Johnson lambda (scipy.stats.yeojohnson_normmax).
 *
 * @param x - Data (any sign).
 * @returns The lambda maximising the log-likelihood.
 * @example
 * ```ts
 * import { yeojohnsonNormmax } from 'whitebox-engine/stats'
 * Math.abs(yeojohnsonNormmax([-2, -1, 0, 1, 2]) - 1) < 1e-8 // => true
 * ```
 */
export function yeojohnsonNormmax(x: TensorLike): number {
  const a = sample('yeojohnsonNormmax', x, 2)
  const xt = vec(a)
  const nonneg = greaterEqual(xt, 0)
  const lp = log1p(abs(xt))
  const ln = log1p(abs(xt))
  let s = 0
  for (const v of a) s += Math.sign(v) * Math.log1p(Math.abs(v))
  const f = (lambda: Value): Value => {
    const pos = expm1Ratio(lambda, lp)
    const neg = mul(expm1Ratio(sub(2, lambda), ln), -1)
    const y = where(nonneg, pos, neg)
    const m = mean(y)
    const v = mean(square(sub(y, m)))
    return sum(sub(mul(sub(lambda, 1), s), mul(log(v), a.length / 2)))
  }
  return maximise(slopeOf(f), (l) => llfYeo(a, l))
}

/**
 * The Yeo-Johnson power transform (scipy.stats.yeojohnson), defined for data of any sign, with lambda given or fitted
 * by maximum likelihood (and its profile-likelihood interval with `level`).
 *
 * @param x - Data.
 * @param options - lambda and level.
 * @returns The transformed data and lambda.
 * @example
 * ```ts
 * import { yeojohnson } from 'whitebox-engine/stats'
 * yeojohnson([-1, 0, 3], { lambda: 1 }).transformed // => [-1, 0, 3]
 * ```
 */
export function yeojohnson(
  x: TensorLike,
  options: { readonly lambda?: number; readonly level?: number } = {},
): PowerTransformResult {
  const a = sample('yeojohnson', x, 2)
  const lambda = options.lambda ?? yeojohnsonNormmax(a)
  const transformed = vec(Array.from(a, (v) => yeoValue(v, lambda)))
  if (options.level === undefined) return { transformed, lambda }
  return { transformed, lambda, interval: profileInterval((l) => llfYeo(a, l), lambda, options.level) }
}

/**
 * The inverse Yeo-Johnson transform (sklearn's PowerTransformer inverse).
 *
 * @param y - Transformed values.
 * @param lambda - The power.
 * @returns The original values.
 * @example
 * ```ts
 * import { invYeojohnson, yeojohnson } from 'whitebox-engine/stats'
 * invYeojohnson(yeojohnson([-1.5, 0, 2], { lambda: 0.4 }).transformed, 0.4) // => [-1.5, 0, 2]
 * ```
 */
export function invYeojohnson(y: TensorLike, lambda: number): Tensor {
  const v = sample('invYeojohnson', y, 0)
  return vec(
    Array.from(v, (t) => {
      if (t >= 0) return Math.abs(lambda) < 1e-300 ? Math.expm1(t) : Math.expm1(Math.log1p(lambda * t) / lambda)
      return Math.abs(lambda - 2) < 1e-300 ? -Math.expm1(-t) : -Math.expm1(Math.log1p(-(2 - lambda) * t) / (2 - lambda))
    }),
  )
}

/**
 * Standardise to zero mean and unit variance (sklearn's StandardScaler, scipy.stats.zscore along columns): a sample,
 * or each column of a matrix; constant columns are only centred.
 *
 * @param x - A sample `[N]` or a matrix `[N, P]`.
 * @param options - ddof of the scale (default 0).
 * @returns The standardised data, with the means and scales used.
 * @example
 * ```ts
 * import { standardise } from 'whitebox-engine/stats'
 * standardise([[1, 10], [3, 30]]).transformed // => [[-1, -1], [1, 1]]
 * ```
 */
export function standardise(
  x: TensorLike,
  options: { readonly ddof?: number } = {},
): { readonly transformed: Tensor; readonly mean: Tensor; readonly scale: Tensor } {
  const t = asTensor(x)
  const m = rowsOf('standardise', x)
  const { rows: n, cols: p } = m
  const means = new Float64Array(p)
  const scales = new Float64Array(p)
  const out = new Float64Array(n * p)
  for (let j = 0; j < p; j++) {
    const col = Float64Array.from({ length: n }, (_, i) => m.data[i * p + j] ?? 0)
    const mu = meanOf(col)
    const sd = Math.sqrt(varianceOf(col, options.ddof ?? 0))
    means[j] = mu
    scales[j] = sd > 0 ? sd : 1
    for (let i = 0; i < n; i++) out[i * p + j] = ((col[i] ?? 0) - mu) / (scales[j] ?? 1)
  }
  return {
    transformed: t.shape.length === 1 ? vec(out) : mat(out, n, p),
    mean: vec(means),
    scale: vec(scales),
  }
}
