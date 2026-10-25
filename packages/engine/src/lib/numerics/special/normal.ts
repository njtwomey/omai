/**
 * Standard normal special functions: the CDF `ndtr` and its logarithm (accurate deep in the tail), the quantile
 * `ndtri`, the density and log density, Owen's T function for bivariate normal and skew-normal probabilities, and the
 * log-probability of an interval, as truncated normals and ordinal regression need.
 */

import {
  add,
  div,
  exp,
  greater,
  mul,
  neg,
  promoteFloat,
  square,
  sub,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'
import { special } from './_elementwise.ts'
import * as k from './_kernels.ts'
import * as misc from './_misc.ts'
import { erf } from './erf.ts'
import { log1mexp } from './logistic.ts'

const phi = (x: Value): Value => exp(sub(mul(square(x), -0.5), k.LOG_SQRT_2PI))

const ndtrP = special('ndtr', k.ndtr, [([x]) => phi(x as Value)])
const logNdtrP = special('logNdtr', k.logNdtr, [
  ([x], out) => exp(sub(sub(mul(square(x as Value), -0.5), k.LOG_SQRT_2PI), out)),
])
const ndtriP = special('ndtri', k.ndtri, [(_, out) => exp(add(mul(square(out), 0.5), k.LOG_SQRT_2PI))])
const owensTP = special('owensT', misc.owensT, [
  // dT/dh = -phi(h) erf(a h / sqrt 2) / 2.
  ([h, a]) => mul(mul(phi(h as Value), erf(mul(mul(a as Value, h as Value), Math.SQRT1_2))), -0.5),
  // dT/da = exp(-h^2 (1 + a^2) / 2) / (2 pi (1 + a^2)).
  ([h, a]) => {
    const q = add(square(a as Value), 1)
    return div(exp(mul(mul(square(h as Value), q), -0.5)), mul(q, 2 * Math.PI))
  },
])

/**
 * The standard normal CDF Phi(x).
 *
 * @param x - Input.
 * @returns Phi(x), differentiable.
 * @concept special-functions
 * @example
 * ```ts
 * import { ndtr } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(ndtr(0)) // => 0.5
 * ```
 */
export function ndtr(x: TensorLike): Value {
  return ndtrP(x)
}

/**
 * log Phi(x), accurate far into the lower tail (where Phi underflows) and near 1.
 *
 * @param x - Input.
 * @returns log Phi(x), differentiable.
 * @example
 * ```ts
 * import { logNdtr } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(logNdtr(-40)) < -800 // => true
 * ```
 */
export function logNdtr(x: TensorLike): Value {
  return logNdtrP(x)
}

/**
 * The standard normal quantile, the inverse of {@link ndtr}.
 *
 * @param p - Probability.
 * @returns The x with Phi(x) = p, differentiable.
 * @example
 * ```ts
 * import { ndtri } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(ndtri(0.5)) // => 0
 * ```
 */
export function ndtri(p: TensorLike): Value {
  return ndtriP(p)
}

/**
 * The standard normal density.
 *
 * @param x - Input.
 * @returns phi(x), differentiable.
 * @example
 * ```ts
 * import { normalPdf } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(normalPdf(0)) // => 0.398942280401433
 * ```
 */
export function normalPdf(x: TensorLike): Value {
  return phi(promoteFloat(x)[0] as Value)
}

/**
 * The log of the standard normal density.
 *
 * @param x - Input.
 * @returns log phi(x), differentiable.
 * @example
 * ```ts
 * import { normalLogPdf } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * item(normalLogPdf(0)) // => -0.918938533204673
 * ```
 */
export function normalLogPdf(x: TensorLike): Value {
  return sub(mul(square(promoteFloat(x)[0] as Value), -0.5), k.LOG_SQRT_2PI)
}

/**
 * Owen's T function T(h, a) = (1 / 2 pi) int_0^a exp(-h^2 (1 + t^2) / 2) / (1 + t^2) dt, which gives bivariate normal
 * orthant probabilities and the skew-normal CDF.
 *
 * @param h - First argument.
 * @param a - Second argument.
 * @returns T(h, a), differentiable in both arguments.
 * @example
 * ```ts
 * import { owensT } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(owensT(0, 1)) - 0.125) < 1e-15 // => true
 * ```
 */
export function owensT(h: TensorLike, a: TensorLike): Value {
  return owensTP(h, a)
}

/**
 * log(Phi(b) - Phi(a)) for a <= b, accurate in either tail (the log-probability of a normal falling in an interval,
 * as truncated normals and ordinal regression need).
 *
 * @param a - Lower limit.
 * @param b - Upper limit.
 * @returns The log interval probability, differentiable.
 * @example
 * ```ts
 * import { logIntervalProbability } from 'whitebox-engine/numerics/special'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(logIntervalProbability(-Infinity, 0)) - Math.log(0.5)) < 1e-15 // => true
 * ```
 */
export function logIntervalProbability(a: TensorLike, b: TensorLike): Value {
  const [lo, hi] = promoteFloat(a, b) as [Value, Value]
  // In the upper tail use symmetry: Phi(b) - Phi(a) = Phi(-a) - Phi(-b).
  const upper = greater(lo, 0)
  const x = where(upper, neg(hi), lo)
  const y = where(upper, neg(lo), hi)
  const lx = logNdtr(x)
  const ly = logNdtr(y)
  return add(ly, log1mexp(sub(lx, ly)))
}
