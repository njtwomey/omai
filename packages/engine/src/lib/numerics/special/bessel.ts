/**
 * Bessel functions, scipy.special style: the modified Bessel functions I and K of real order with exponentially scaled
 * forms and an overflow-free log I (the normaliser of von Mises distributions), the ordinary Bessel functions J and Y,
 * and the common integer-order shortcuts `i0`, `i1`, `i0e` and `i1e`.
 */

import { abs, add, div, log, maximum, mul, sign, sub, type TensorLike, type Value } from '../../tensor/index.ts'
import * as b from './_bessel.ts'
import { special } from './_elementwise.ts'

// Derivatives in x by the recurrences (two neighbouring orders, which works at x = 0 and for every real order);
// derivatives in the order have no convenient closed form and use finite differences.
const ivP = special('iv', b.iv, [
  null,
  ([v, x]) => mul(add(iv(sub(v as Value, 1), x as Value), iv(add(v as Value, 1), x as Value)), 0.5),
])
const iveP = special('ive', b.ive, [
  null,
  ([v, x], out) =>
    sub(
      mul(add(ive(sub(v as Value, 1), x as Value), ive(add(v as Value, 1), x as Value)), 0.5),
      mul(sign(x as Value), out),
    ),
])
// The order derivative of log I by finite differences of log(e^{-x} I), which is O(log x) rather than O(x), so the
// difference keeps its digits for large x.
const logIveOrder = ([v, x]: readonly Value[]): Value => {
  const h = mul(maximum(abs(v as Value), 1), 1e-6)
  const up = log(ive(add(v as Value, h), x as Value))
  const down = log(ive(sub(v as Value, h), x as Value))
  return div(sub(up, down), mul(h, 2))
}
const logIvP = special('logIv', b.logIv, [
  logIveOrder,
  ([v, x]) =>
    div(
      add(ive(sub(v as Value, 1), x as Value), ive(add(v as Value, 1), x as Value)),
      mul(ive(v as Value, x as Value), 2),
    ),
])
const kvP = special('kv', b.kv, [
  null,
  ([v, x]) => mul(add(kv(sub(v as Value, 1), x as Value), kv(add(v as Value, 1), x as Value)), -0.5),
])
const kveP = special('kve', b.kve, [
  null,
  ([v, x], out) => sub(out, mul(add(kve(sub(v as Value, 1), x as Value), kve(add(v as Value, 1), x as Value)), 0.5)),
])
const jvP = special('jv', b.jv, [
  null,
  ([v, x]) => mul(sub(jv(sub(v as Value, 1), x as Value), jv(add(v as Value, 1), x as Value)), 0.5),
])
const yvP = special('yv', b.yv, [
  null,
  ([v, x]) => mul(sub(yv(sub(v as Value, 1), x as Value), yv(add(v as Value, 1), x as Value)), 0.5),
])

/**
 * The modified Bessel function of the first kind I_v(x), for real order v (negative x only for integer orders).
 *
 * @param v - Order.
 * @param x - Argument.
 * @returns I_v(x), differentiable (exactly in x, by finite differences in v).
 * @concept special-functions
 * @example
 * ```ts
 * import { iv } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(iv(0, 0)) // => 1
 * ```
 */
export function iv(v: TensorLike, x: TensorLike): Value {
  return ivP(v, x)
}

/**
 * The exponentially scaled modified Bessel function exp(-|x|) I_v(x), finite for large x.
 *
 * @param v - Order.
 * @param x - Argument.
 * @returns exp(-|x|) I_v(x), differentiable.
 * @example
 * ```ts
 * import { ive } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * // Large x: exp(-x) I_0(x) ~ (1 + 1 / (8x)) / sqrt(2 pi x).
 * Math.abs(item(ive(0, 1e6)) * Math.sqrt(2 * Math.PI * 1e6) - (1 + 1 / 8e6)) < 1e-12 // => true
 * ```
 */
export function ive(v: TensorLike, x: TensorLike): Value {
  return iveP(v, x)
}

/**
 * log I_v(x) for x > 0, without overflow (the normaliser of the von Mises and von Mises-Fisher distributions).
 *
 * @param v - Order.
 * @param x - Argument (positive).
 * @returns log I_v(x), differentiable.
 * @example
 * ```ts
 * import { logIv } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(logIv(0, 1e4)) > 9990 // => true
 * ```
 */
export function logIv(v: TensorLike, x: TensorLike): Value {
  return logIvP(v, x)
}

/**
 * The modified Bessel function of the second kind K_v(x), for x > 0.
 *
 * @param v - Order.
 * @param x - Argument.
 * @returns K_v(x), differentiable.
 * @example
 * ```ts
 * import { kv } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(kv(0.5, 1)) - Math.sqrt(Math.PI / 2) * Math.exp(-1)) < 1e-15 // => true
 * ```
 */
export function kv(v: TensorLike, x: TensorLike): Value {
  return kvP(v, x)
}

/**
 * The exponentially scaled modified Bessel function exp(x) K_v(x).
 *
 * @param v - Order.
 * @param x - Argument.
 * @returns exp(x) K_v(x), differentiable.
 * @example
 * ```ts
 * import { kve } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(kve(0.5, 2)) - Math.sqrt(Math.PI / 4)) < 1e-15 // => true
 * ```
 */
export function kve(v: TensorLike, x: TensorLike): Value {
  return kveP(v, x)
}

/**
 * The Bessel function of the first kind J_v(x), for real order v (negative x only for integer orders).
 *
 * @param v - Order.
 * @param x - Argument.
 * @returns J_v(x), differentiable.
 * @example
 * ```ts
 * import { jv } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(jv(0, 0)) // => 1
 * ```
 */
export function jv(v: TensorLike, x: TensorLike): Value {
  return jvP(v, x)
}

/**
 * The Bessel function of the second kind Y_v(x), for x > 0.
 *
 * @param v - Order.
 * @param x - Argument.
 * @returns Y_v(x), differentiable.
 * @example
 * ```ts
 * import { yv } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(yv(0, 0)) // => -Infinity
 * ```
 */
export function yv(v: TensorLike, x: TensorLike): Value {
  return yvP(v, x)
}

/**
 * I_0(x).
 *
 * @param x - Argument.
 * @returns I_0(x), differentiable.
 * @example
 * ```ts
 * import { i0 } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(i0(0)) // => 1
 * ```
 */
export function i0(x: TensorLike): Value {
  return iv(0, x)
}

/**
 * I_1(x).
 *
 * @param x - Argument.
 * @returns I_1(x), differentiable.
 * @example
 * ```ts
 * import { i1 } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(i1(0)) // => 0
 * ```
 */
export function i1(x: TensorLike): Value {
  return iv(1, x)
}

/**
 * exp(-|x|) I_0(x).
 *
 * @param x - Argument.
 * @returns The scaled value, differentiable.
 * @example
 * ```ts
 * import { i0e } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(i0e(0)) // => 1
 * ```
 */
export function i0e(x: TensorLike): Value {
  return ive(0, x)
}

/**
 * exp(-|x|) I_1(x).
 *
 * @param x - Argument.
 * @returns The scaled value, differentiable.
 * @example
 * ```ts
 * import { i1e } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(i1e(0)) // => 0
 * ```
 */
export function i1e(x: TensorLike): Value {
  return ive(1, x)
}
