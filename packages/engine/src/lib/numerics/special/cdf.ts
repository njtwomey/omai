/**
 * Distribution functions in scipy.special's naming: the Student t CDF and its inverse, chi-square and F CDFs and
 * survival functions accurate in the tails, and the Kolmogorov and Smirnov distributions behind the Kolmogorov-Smirnov
 * test.
 */

import { jvp } from '../../autodiff/index.ts'
import {
  abs,
  add,
  div,
  exp,
  type FloatDType,
  full,
  greater,
  less,
  log,
  log1p,
  mul,
  neg,
  promoteFloat,
  sign,
  square,
  sub,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { special } from './_elementwise.ts'
import * as inc from './_incomplete.ts'
import * as misc from './_misc.ts'
import { betaincDa, betaincDb } from './_partials.ts'
import { lbeta, lgamma } from './gamma.ts'
import { betainc, gammainc, gammaincc } from './incomplete.ts'

const tDensity = (df: Value, t: Value): Value =>
  exp(
    sub(
      sub(sub(lgamma(mul(add(df, 1), 0.5)), lgamma(mul(df, 0.5))), mul(log(mul(df, Math.PI)), 0.5)),
      mul(mul(add(df, 1), 0.5), log1p(div(square(t), df))),
    ),
  )

/**
 * The Student t CDF with `df` degrees of freedom.
 *
 * @param df - Degrees of freedom (positive, need not be an integer).
 * @param t - Argument.
 * @returns P(T <= t), differentiable in both arguments.
 * @concept special-functions
 * @example
 * ```ts
 * import { stdtr } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(stdtr(3, 0)) // => 0.5
 * ```
 */
export function stdtr(df: TensorLike, t: TensorLike): Value {
  return stdtrP(df, t)
}

// d/ddf of the lower tail I_x(df/2, 1/2) / 2, x = df / (df + t^2): the shape term plus the density times dx/ddf, the
// latter written as x^{df/2} (1 - x)^{1/2} / (B df) so that it is finite (zero) at t = 0.
const tailDdf = (d: Value, t: Value): Value => {
  const t2 = square(t)
  const x = div(d, add(d, t2))
  const a = mul(d, 0.5)
  const shape = mul(betaincDa(a, 0.5, x), 0.5)
  const chain = div(exp(sub(add(mul(a, log(x)), mul(log1p(neg(x)), 0.5)), lbeta(a, 0.5))), d)
  return mul(add(shape, chain), 0.5)
}

// d/ddf of the centre form 1/2 + sign(t) I_z(1/2, df/2) / 2, z = t^2 / (df + t^2), used where it is the accurate one
// (|F - 1/2| < 1/4, which includes tiny t, where df / (df + t^2) rounds to 1 and the tail form loses everything).
const centreDdf = (d: Value, t: Value): Value => {
  const t2 = square(t)
  const z = div(t2, add(d, t2))
  const a = mul(d, 0.5)
  const shape = mul(betaincDb(0.5, a, z), 0.5)
  const chain = div(exp(sub(add(mul(log(z), 0.5), mul(a, log1p(neg(z)))), lbeta(0.5, a))), d)
  return mul(mul(sign(t), sub(shape, chain)), 0.5)
}

const stdtrP = special('stdtr', inc.stdtr, [
  ([d, t], out) => {
    const dd = d as Value
    const tt = t as Value
    const tail = where(greater(tt, 0), neg(tailDdf(dd, tt)), tailDdf(dd, tt))
    return where(less(abs(sub(out, 0.5)), 0.25), centreDdf(dd, tt), tail)
  },
  ([d, t]) => tDensity(d as Value, t as Value),
])

const stdtritP = special('stdtrit', inc.stdtrit, [
  // dt/ddf = -(dF/ddf) / f(t), with dF/ddf from forward-mode differentiation of stdtr.
  ([df], t) => {
    const d = df as Value
    const ones = full(d.shape, 1, { dtype: d.dtype as FloatDType })
    const dF = jvp((x: Value) => stdtr(x, t), d, ones).tangent
    return neg(div(dF, tDensity(d, t)))
  },
  ([df], t) => div(1, tDensity(df as Value, t)),
])

/**
 * The inverse of {@link stdtr}: the t with P(T <= t) = p.
 *
 * @param df - Degrees of freedom.
 * @param p - Probability.
 * @returns t, differentiable in both arguments (implicit function theorem).
 * @example
 * ```ts
 * import { stdtrit } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(stdtrit(1, 0.75)) - 1) < 1e-14 // => true
 * ```
 */
export function stdtrit(df: TensorLike, p: TensorLike): Value {
  return stdtritP(df, p)
}

/**
 * The chi-square CDF with `k` degrees of freedom.
 *
 * @param k - Degrees of freedom.
 * @param x - Argument.
 * @returns P(X <= x), differentiable.
 * @example
 * ```ts
 * import { chdtr } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(chdtr(2, 2)) - (1 - Math.exp(-1))) < 1e-15 // => true
 * ```
 */
export function chdtr(k: TensorLike, x: TensorLike): Value {
  const [kk, xx] = promoteFloat(k, x) as [Value, Value]
  const pos = greater(xx, 0)
  return where(pos, gammainc(mul(kk, 0.5), mul(where(pos, xx, 1), 0.5)), 0)
}

/**
 * The chi-square survival function, accurate where it is tiny.
 *
 * @param k - Degrees of freedom.
 * @param x - Argument.
 * @returns P(X > x), differentiable.
 * @example
 * ```ts
 * import { chdtrc } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(chdtrc(2, 2)) - Math.exp(-1)) < 1e-15 // => true
 * ```
 */
export function chdtrc(k: TensorLike, x: TensorLike): Value {
  const [kk, xx] = promoteFloat(k, x) as [Value, Value]
  const pos = greater(xx, 0)
  return where(pos, gammaincc(mul(kk, 0.5), mul(where(pos, xx, 1), 0.5)), 1)
}

/**
 * The F distribution CDF with `d1` and `d2` degrees of freedom.
 *
 * @param d1 - Numerator degrees of freedom.
 * @param d2 - Denominator degrees of freedom.
 * @param x - Argument.
 * @returns P(F <= x), differentiable.
 * @example
 * ```ts
 * import { fdtr } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(fdtr(4, 4, 1)) // => 0.5
 * ```
 */
export function fdtr(d1: TensorLike, d2: TensorLike, x: TensorLike): Value {
  const [a, b, xx] = promoteFloat(d1, d2, x) as [Value, Value, Value]
  const pos = greater(xx, 0)
  const s = where(pos, xx, 1)
  return where(pos, betainc(mul(a, 0.5), mul(b, 0.5), div(mul(a, s), add(mul(a, s), b))), 0)
}

/**
 * The F distribution survival function, accurate where it is tiny.
 *
 * @param d1 - Numerator degrees of freedom.
 * @param d2 - Denominator degrees of freedom.
 * @param x - Argument.
 * @returns P(F > x), differentiable.
 * @example
 * ```ts
 * import { fdtrc } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(fdtrc(4, 4, 1)) // => 0.5
 * ```
 */
export function fdtrc(d1: TensorLike, d2: TensorLike, x: TensorLike): Value {
  const [a, b, xx] = promoteFloat(d1, d2, x) as [Value, Value, Value]
  const pos = greater(xx, 0)
  const s = where(pos, xx, 1)
  return where(pos, betainc(mul(b, 0.5), mul(a, 0.5), div(b, add(b, mul(a, s)))), 1)
}

const kolmogorovPdfP = special('kolmogorovPdf', misc.kolmogorovPdf, [null])
const kolmogorovP = special('kolmogorov', misc.kolmogorov, [([x]) => neg(kolmogorovPdfP(x as Value))])
const smirnovPs = new Map<number, (d: TensorLike) => Value>()

/**
 * The survival function of the Kolmogorov distribution, the limiting distribution of sqrt(n) times the two-sided
 * Kolmogorov-Smirnov statistic (scipy `kolmogorov`).
 *
 * @param x - Argument.
 * @returns P(K > x), differentiable (the derivative is minus the Kolmogorov density).
 * @example
 * ```ts
 * import { kolmogorov } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(kolmogorov(0)) // => 1
 * ```
 */
export function kolmogorov(x: TensorLike): Value {
  return kolmogorovP(x)
}

/**
 * The density of the Kolmogorov distribution (scipy `kstwobign.pdf`).
 *
 * @param x - Argument.
 * @returns The density, differentiable (by finite differences).
 * @example
 * ```ts
 * import { kolmogorovPdf } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(kolmogorovPdf(0)) // => 0
 * ```
 */
export function kolmogorovPdf(x: TensorLike): Value {
  return kolmogorovPdfP(x)
}

/**
 * The exact survival function of the one-sided Kolmogorov-Smirnov statistic for sample size n, P(D_n^+ >= d)
 * (Birnbaum and Tingey, 1951).
 *
 * @param n - Sample size (a positive integer).
 * @param d - Statistic in [0, 1].
 * @returns The probability, differentiable in d (by finite differences).
 * @example
 * ```ts
 * import { smirnov } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(smirnov(1, 0.25)) // => 0.75
 * ```
 */
export function smirnov(n: number, d: TensorLike): Value {
  let f = smirnovPs.get(n)
  if (f === undefined) {
    f = special(`smirnov${n}`, (x) => misc.smirnov(n, x), [null])
    smirnovPs.set(n, f)
  }
  return f(d)
}
