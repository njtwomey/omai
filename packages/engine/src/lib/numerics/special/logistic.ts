/**
 * Logistic and log-domain helpers, scipy.special style, computed without overflow: the sigmoid `expit` and `logit`,
 * softplus and log-sigmoid, `log1mexp`, `logaddexp`, and the entropy terms `xlogy`, `xlog1py`, `entr`, `relEntr` and
 * `klDiv` that are defined as 0 where x is 0.
 */

import {
  abs,
  add,
  div,
  equal,
  exp,
  expm1,
  greater,
  greaterEqual,
  log,
  log1p,
  logicalAnd,
  maximum,
  mul,
  neg,
  promoteFloat,
  sigmoid,
  softplus as softplusOp,
  sub,
  type TensorLike,
  type Value,
  where,
} from '../../tensor/index.ts'

/**
 * The logistic sigmoid 1 / (1 + exp(-x)) (scipy `expit`).
 *
 * @param x - Input.
 * @returns expit(x), differentiable.
 * @example
 * ```ts
 * import { expit } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(expit(0)) // => 0.5
 * ```
 */
export function expit(x: TensorLike): Value {
  return sigmoid(x)
}

/**
 * The log-odds log(p / (1 - p)), the inverse of {@link expit}.
 *
 * @param p - Probability.
 * @returns logit(p), differentiable.
 * @example
 * ```ts
 * import { logit } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(logit(0.5)) // => 0
 * ```
 */
export function logit(p: TensorLike): Value {
  const [q] = promoteFloat(p) as [Value]
  return sub(log(q), log1p(neg(q)))
}

/**
 * softplus(x) = log(1 + exp(x)), without overflow.
 *
 * @param x - Input.
 * @returns softplus(x), differentiable.
 * @example
 * ```ts
 * import { softplus } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(softplus(0)) === Math.LN2 // => true
 * ```
 */
export function softplus(x: TensorLike): Value {
  return softplusOp(x)
}

/**
 * log(1 + exp(x)), the same function as {@link softplus} under its scipy-style name.
 *
 * @param x - Input.
 * @returns log1pexp(x), differentiable.
 * @example
 * ```ts
 * import { log1pexp } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(log1pexp(-1000)) // => 0
 * ```
 */
export function log1pexp(x: TensorLike): Value {
  return softplusOp(x)
}

/**
 * The log of the sigmoid, -softplus(-x), accurate for very negative x.
 *
 * @param x - Input.
 * @returns log(expit(x)), differentiable.
 * @example
 * ```ts
 * import { logSigmoid } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(logSigmoid(-1000)) // => -1000
 * ```
 */
export function logSigmoid(x: TensorLike): Value {
  return neg(softplusOp(neg(promoteFloat(x)[0] as Value)))
}

/**
 * log(1 - exp(x)) for x <= 0, choosing between `log(-expm1(x))` and `log1p(-exp(x))` for accuracy (Machler 2012).
 *
 * @param x - Input (non-positive).
 * @returns log1mexp(x), differentiable.
 * @example
 * ```ts
 * import { log1mexp } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(log1mexp(-Math.LN2)) === Math.log(0.5) // => true
 * ```
 */
export function log1mexp(x: TensorLike): Value {
  const [v] = promoteFloat(x) as [Value]
  const near = greater(v, -Math.LN2)
  // Each branch gets an argument where it is finite, so the unselected branch never contributes NaN gradients.
  const a = where(near, v, -1)
  const b = where(near, -1, v)
  return where(near, log(neg(expm1(a))), log1p(neg(exp(b))))
}

/**
 * log(exp(a) + exp(b)), without overflow.
 *
 * @param a - First input.
 * @param b - Second input.
 * @returns logaddexp(a, b), differentiable.
 * @example
 * ```ts
 * import { logaddexp } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(logaddexp(0, 0)) === Math.LN2 // => true
 * ```
 */
export function logaddexp(a: TensorLike, b: TensorLike): Value {
  const [x, y] = promoteFloat(a, b) as [Value, Value]
  // x - y is NaN for two equal infinities, whose log-sum is the infinity itself: the gap is 0 whenever x equals y.
  return add(maximum(x, y), log1p(exp(neg(abs(where(equal(x, y), 0, sub(x, y)))))))
}

/**
 * x log(y), defined as 0 when x = 0 (whatever y is).
 *
 * @param x - Multiplier.
 * @param y - Argument of the logarithm.
 * @returns xlogy(x, y), differentiable.
 * @example
 * ```ts
 * import { xlogy } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(xlogy(0, 0)) // => 0
 * ```
 */
export function xlogy(x: TensorLike, y: TensorLike): Value {
  const [a, b] = promoteFloat(x, y) as [Value, Value]
  const zero = equal(a, 0)
  return where(zero, 0, mul(a, log(where(zero, 1, b))))
}

/**
 * x log(1 + y), defined as 0 when x = 0.
 *
 * @param x - Multiplier.
 * @param y - Argument.
 * @returns xlog1py(x, y), differentiable.
 * @example
 * ```ts
 * import { xlog1py } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(xlog1py(0, -1)) // => 0
 * ```
 */
export function xlog1py(x: TensorLike, y: TensorLike): Value {
  const [a, b] = promoteFloat(x, y) as [Value, Value]
  const zero = equal(a, 0)
  return where(zero, 0, mul(a, log1p(where(zero, 0, b))))
}

/**
 * Elementwise entropy -x log(x) (0 at 0, -inf for negative x).
 *
 * @param x - Input.
 * @returns entr(x), differentiable for x > 0.
 * @example
 * ```ts
 * import { entr } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(entr(1)) // => 0
 * ```
 */
export function entr(x: TensorLike): Value {
  const [v] = promoteFloat(x) as [Value]
  const pos = greater(v, 0)
  const safe = where(pos, v, 1)
  return where(pos, neg(mul(safe, log(safe))), where(equal(v, 0), 0, Number.NEGATIVE_INFINITY))
}

/**
 * Elementwise relative entropy x log(x / y) (0 when x = 0 and y >= 0, inf otherwise outside the domain).
 *
 * @param x - First input.
 * @param y - Second input.
 * @returns relEntr(x, y), differentiable on x, y > 0.
 * @example
 * ```ts
 * import { relEntr } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(relEntr(1, 1)) // => 0
 * ```
 */
export function relEntr(x: TensorLike, y: TensorLike): Value {
  const [a, b] = promoteFloat(x, y) as [Value, Value]
  const inside = logicalAnd(greater(a, 0), greater(b, 0))
  const sa = where(inside, a, 1)
  const sb = where(inside, b, 1)
  const edge = where(logicalAnd(equal(a, 0), greaterEqual(b, 0)), 0, Number.POSITIVE_INFINITY)
  return where(inside, mul(sa, log(div(sa, sb))), edge)
}

/**
 * The elementwise Kullback-Leibler term x log(x / y) - x + y (scipy `kl_div`), non-negative everywhere.
 *
 * @param x - First input.
 * @param y - Second input.
 * @returns klDiv(x, y), differentiable on x, y > 0.
 * @example
 * ```ts
 * import { klDiv } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(klDiv(2, 2)) // => 0
 * ```
 */
export function klDiv(x: TensorLike, y: TensorLike): Value {
  const [a, b] = promoteFloat(x, y) as [Value, Value]
  return add(sub(relEntr(a, b), a), b)
}
