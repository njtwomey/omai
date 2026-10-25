/**
 * The error function family as differentiable elementwise operations: `erf`, `erfc` (accurate in the far tail), the
 * scaled `erfcx`, the inverses `erfinv` and `erfcinv`, and Dawson's integral.
 */

import { add, exp, mul, neg, square, type TensorLike, type Value } from '../../tensor/index.ts'
import { special } from './_elementwise.ts'
import * as k from './_kernels.ts'
import * as misc from './_misc.ts'

const TWO_OVER_SQRT_PI = 2 / k.SQRT_PI

const erfP = special('erf', k.erf, [([x]) => mul(exp(neg(square(x as Value))), TWO_OVER_SQRT_PI)])
const erfcP = special('erfc', k.erfc, [([x]) => mul(exp(neg(square(x as Value))), -TWO_OVER_SQRT_PI)])
// erfcx'' = 2 erfcx + 2 x erfcx', written with operations on the accurate erfcx' kernel.
const erfcxPrimeP = special('erfcxPrime', k.erfcxPrime, [
  ([x], out) => add(mul(erfcxP(x as Value), 2), mul(mul(x as Value, out), 2)),
])
const erfcxP = special('erfcx', k.erfcx, [([x]) => erfcxPrimeP(x as Value)])
const erfinvP = special('erfinv', k.erfinv, [(_, out) => mul(exp(square(out)), k.SQRT_PI / 2)])
const erfcinvP = special('erfcinv', k.erfcinv, [(_, out) => mul(exp(square(out)), -k.SQRT_PI / 2)])
const dawsnP = special('dawsn', misc.dawsn, [([x], out) => add(1, mul(mul(x as Value, out), -2))])

/**
 * The error function erf(x) = (2 / sqrt(pi)) int_0^x exp(-t^2) dt.
 *
 * @param x - Input.
 * @returns erf(x), differentiable.
 * @concept special-functions
 * @example
 * ```ts
 * import { erf } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(erf(0)) // => 0
 * ```
 */
export function erf(x: TensorLike): Value {
  return erfP(x)
}

/**
 * The complementary error function 1 - erf(x), accurate where it is tiny (large x).
 *
 * @param x - Input.
 * @returns erfc(x), differentiable.
 * @example
 * ```ts
 * import { erfc } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(erfc(0)) // => 1
 * ```
 */
export function erfc(x: TensorLike): Value {
  return erfcP(x)
}

/**
 * The scaled complementary error function exp(x^2) erfc(x), finite for large positive x.
 *
 * @param x - Input.
 * @returns erfcx(x), differentiable.
 * @example
 * ```ts
 * import { erfcx } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(erfcx(0)) // => 1
 * ```
 */
export function erfcx(x: TensorLike): Value {
  return erfcxP(x)
}

/**
 * The inverse error function, on (-1, 1).
 *
 * @param y - Input.
 * @returns The x with erf(x) = y, differentiable.
 * @example
 * ```ts
 * import { erfinv } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(erfinv(0)) // => 0
 * ```
 */
export function erfinv(y: TensorLike): Value {
  return erfinvP(y)
}

/**
 * The inverse complementary error function, on (0, 2).
 *
 * @param y - Input.
 * @returns The x with erfc(x) = y, differentiable.
 * @example
 * ```ts
 * import { erfcinv } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(erfcinv(1)) // => 0
 * ```
 */
export function erfcinv(y: TensorLike): Value {
  return erfcinvP(y)
}

/**
 * Dawson's integral D(x) = exp(-x^2) int_0^x exp(t^2) dt.
 *
 * @param x - Input.
 * @returns D(x), differentiable (D' = 1 - 2 x D).
 * @example
 * ```ts
 * import { dawsn } from 'mdml-engine/numerics/special'
 * import { item } from 'mdml-engine/tensor'
 * item(dawsn(0)) // => 0
 * ```
 */
export function dawsn(x: TensorLike): Value {
  return dawsnP(x)
}
