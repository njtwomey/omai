import { bind1, type Primitive } from '../primitive.ts'
import type { TensorLike } from '../tensor.ts'
import type { Value } from '../traced.ts'
import { coerce2, floatify, val } from './_coerce.ts'
import { binaryPrimitive, unaryPrimitive } from './_factories.ts'
import { add, cast, div, mul, neg, square, sub } from './arithmetic.ts'
import { greater, where } from './compare.ts'

const prims = {
  exp: unaryPrimitive('exp', (t, _x, out) => mul(t, out)),
  expm1: unaryPrimitive('expm1', (t, _x, out) => mul(t, add(out, 1))),
  log: unaryPrimitive('log', (t, x) => div(t, x)),
  log1p: unaryPrimitive('log1p', (t, x) => div(t, add(x, 1))),
  sqrt: unaryPrimitive('sqrt', (t, _x, out) => div(t, mul(out, 2))),
  sin: unaryPrimitive('sin', (t, x) => mul(t, cos(x))),
  cos: unaryPrimitive('cos', (t, x) => neg(mul(t, sin(x)))),
  tan: unaryPrimitive('tan', (t, _x, out) => mul(t, add(square(out), 1))),
  asin: unaryPrimitive('asin', (t, x) => div(t, sqrt(sub(1, square(x))))),
  acos: unaryPrimitive('acos', (t, x) => neg(div(t, sqrt(sub(1, square(x)))))),
  atan: unaryPrimitive('atan', (t, x) => div(t, add(square(x), 1))),
  sinh: unaryPrimitive('sinh', (t, x) => mul(t, cosh(x))),
  cosh: unaryPrimitive('cosh', (t, x) => mul(t, sinh(x))),
  tanh: unaryPrimitive('tanh', (t, _x, out) => mul(t, sub(1, square(out)))),
  asinh: unaryPrimitive('asinh', (t, x) => div(t, sqrt(add(square(x), 1)))),
  acosh: unaryPrimitive('acosh', (t, x) => div(t, sqrt(sub(square(x), 1)))),
  atanh: unaryPrimitive('atanh', (t, x) => div(t, sub(1, square(x)))),
  // sigmoid' = sigmoid(x) sigmoid(-x): avoids forming 1 - sigmoid(x), which cancels for large x.
  sigmoid: unaryPrimitive('sigmoid', (t, x, out) => mul(t, mul(out, sigmoid(neg(x))))),
  floor: unaryPrimitive('floor'),
  ceil: unaryPrimitive('ceil'),
  round: unaryPrimitive('round'),
  trunc: unaryPrimitive('trunc'),
} satisfies Record<string, Primitive<null>>

// d atan2(y, x) = (x dy - y dx) / (x^2 + y^2)
const atan2Primitive: Primitive<null> = binaryPrimitive('atan2', {
  jvp: (y, x, ty, tx) => {
    const r2 = add(square(x), square(y))
    const a = ty === null ? null : mul(ty, x)
    const b = tx === null ? null : mul(tx, y)
    return div(a === null ? neg(b as Value) : b === null ? a : sub(a, b), r2)
  },
  vjp: (g, y, x) => {
    const r2 = add(square(x), square(y))
    return [div(mul(g, x), r2), neg(div(mul(g, y), r2))]
  },
})

function apply(p: Primitive<null>, x: TensorLike): Value {
  return bind1(p, [floatify(x)], null)
}

/**
 * Elementwise exponential.
 *
 * @param x - Input.
 * @returns `e^x`.
 * @example
 * ```ts
 * import { exp, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(exp(tensor([0, 1]))) // => [1, 2.718282]
 * ```
 */
export function exp(x: TensorLike): Value {
  return apply(prims.exp, x)
}

/**
 * Elementwise `e^x - 1`, accurate for small `x`.
 *
 * @param x - Input.
 * @returns `expm1(x)`.
 * @example
 * ```ts
 * import { expm1, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(expm1(tensor([1e-10]))) // => [1e-10]
 * ```
 */
export function expm1(x: TensorLike): Value {
  return apply(prims.expm1, x)
}

/**
 * Elementwise natural logarithm (NaN for negative input, -Infinity at 0).
 *
 * @param x - Input.
 * @returns `ln x`.
 * @example
 * ```ts
 * import { log, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(log(tensor([1, Math.E]))) // => [0, 1]
 * ```
 */
export function log(x: TensorLike): Value {
  return apply(prims.log, x)
}

/**
 * Elementwise `ln(1 + x)`, accurate for small `x`.
 *
 * @param x - Input.
 * @returns `log1p(x)`.
 * @example
 * ```ts
 * import { log1p, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(log1p(tensor([0]))) // => [0]
 * ```
 */
export function log1p(x: TensorLike): Value {
  return apply(prims.log1p, x)
}

/**
 * Elementwise base-2 logarithm.
 *
 * @param x - Input.
 * @returns `log2 x`.
 * @example
 * ```ts
 * import { log2, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(log2(tensor([8]))) // => [3]
 * ```
 */
export function log2(x: TensorLike): Value {
  return div(log(x), Math.LN2)
}

/**
 * Elementwise base-10 logarithm.
 *
 * @param x - Input.
 * @returns `log10 x`.
 * @example
 * ```ts
 * import { log10, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(log10(tensor([1000]))) // => [3]
 * ```
 */
export function log10(x: TensorLike): Value {
  return div(log(x), Math.LN10)
}

/**
 * Elementwise square root.
 *
 * @param x - Input.
 * @returns `sqrt(x)`.
 * @example
 * ```ts
 * import { sqrt, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(sqrt(tensor([4, 9]))) // => [2, 3]
 * ```
 */
export function sqrt(x: TensorLike): Value {
  return apply(prims.sqrt, x)
}

/**
 * Elementwise sine.
 *
 * @param x - Input in radians.
 * @returns `sin x`.
 * @example
 * ```ts
 * import { sin, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(sin(tensor([0, Math.PI / 2]))) // => [0, 1]
 * ```
 */
export function sin(x: TensorLike): Value {
  return apply(prims.sin, x)
}

/**
 * Elementwise cosine.
 *
 * @param x - Input in radians.
 * @returns `cos x`.
 * @example
 * ```ts
 * import { cos, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(cos(tensor([0]))) // => [1]
 * ```
 */
export function cos(x: TensorLike): Value {
  return apply(prims.cos, x)
}

/**
 * Elementwise tangent.
 *
 * @param x - Input in radians.
 * @returns `tan x`.
 * @example
 * ```ts
 * import { tan, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(tan(tensor([Math.PI / 4]))) // => [1]
 * ```
 */
export function tan(x: TensorLike): Value {
  return apply(prims.tan, x)
}

/**
 * Elementwise inverse sine.
 *
 * @param x - Input in [-1, 1].
 * @returns `arcsin x`.
 * @example
 * ```ts
 * import { arcsin, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(arcsin(tensor([1]))) // => [1.570796]
 * ```
 */
export function arcsin(x: TensorLike): Value {
  return apply(prims.asin, x)
}

/**
 * Elementwise inverse cosine.
 *
 * @param x - Input in [-1, 1].
 * @returns `arccos x`.
 * @example
 * ```ts
 * import { arccos, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(arccos(tensor([1]))) // => [0]
 * ```
 */
export function arccos(x: TensorLike): Value {
  return apply(prims.acos, x)
}

/**
 * Elementwise inverse tangent.
 *
 * @param x - Input.
 * @returns `arctan x`.
 * @example
 * ```ts
 * import { arctan, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(arctan(tensor([1]))) // => [0.785398]
 * ```
 */
export function arctan(x: TensorLike): Value {
  return apply(prims.atan, x)
}

/**
 * Elementwise hyperbolic sine.
 *
 * @param x - Input.
 * @returns `sinh x`.
 * @example
 * ```ts
 * import { sinh, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(sinh(tensor([0]))) // => [0]
 * ```
 */
export function sinh(x: TensorLike): Value {
  return apply(prims.sinh, x)
}

/**
 * Elementwise hyperbolic cosine.
 *
 * @param x - Input.
 * @returns `cosh x`.
 * @example
 * ```ts
 * import { cosh, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(cosh(tensor([0]))) // => [1]
 * ```
 */
export function cosh(x: TensorLike): Value {
  return apply(prims.cosh, x)
}

/**
 * Elementwise hyperbolic tangent.
 *
 * @param x - Input.
 * @returns `tanh x`.
 * @example
 * ```ts
 * import { tanh, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(tanh(tensor([0]))) // => [0]
 * ```
 */
export function tanh(x: TensorLike): Value {
  return apply(prims.tanh, x)
}

/**
 * Elementwise inverse hyperbolic sine.
 *
 * @param x - Input.
 * @returns `arcsinh x`.
 * @example
 * ```ts
 * import { arcsinh, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(arcsinh(tensor([0]))) // => [0]
 * ```
 */
export function arcsinh(x: TensorLike): Value {
  return apply(prims.asinh, x)
}

/**
 * Elementwise inverse hyperbolic cosine.
 *
 * @param x - Input at least 1.
 * @returns `arccosh x`.
 * @example
 * ```ts
 * import { arccosh, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(arccosh(tensor([1]))) // => [0]
 * ```
 */
export function arccosh(x: TensorLike): Value {
  return apply(prims.acosh, x)
}

/**
 * Elementwise inverse hyperbolic tangent.
 *
 * @param x - Input in (-1, 1).
 * @returns `arctanh x`.
 * @example
 * ```ts
 * import { arctanh, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(arctanh(tensor([0]))) // => [0]
 * ```
 */
export function arctanh(x: TensorLike): Value {
  return apply(prims.atanh, x)
}

/**
 * Elementwise logistic sigmoid, computed without overflow for large `|x|`.
 *
 * @param x - Input.
 * @returns `1 / (1 + e^-x)`.
 * @example
 * ```ts
 * import { sigmoid, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(sigmoid(tensor([0, 1000, -1000]))) // => [0.5, 1, 0]
 * ```
 */
export function sigmoid(x: TensorLike): Value {
  return apply(prims.sigmoid, x)
}

/**
 * Elementwise softplus `log(1 + e^x)`, computed stably as `x + log1p(e^-x)` for x > 0 and `log1p(e^x)` otherwise (each
 * branch on clamped inputs, so neither overflows and the derivative is sigmoid(x) everywhere, 1/2 at 0).
 *
 * @param x - Input.
 * @returns `softplus(x)`.
 * @example
 * ```ts
 * import { softplus, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(softplus(tensor([0, 1000]))) // => [0.693147, 1000]
 * ```
 */
export function softplus(x: TensorLike): Value {
  const v = floatify(x)
  const positive = greater(v, 0)
  const pos = where(positive, v, 0)
  const negative = where(positive, 0, v)
  return where(positive, add(pos, log1p(exp(neg(pos)))), log1p(exp(negative)))
}

/**
 * Elementwise rectified linear unit (subgradient 0 at 0).
 *
 * @param x - Input.
 * @returns `max(x, 0)`.
 * @example
 * ```ts
 * import { relu, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(relu(tensor([-1, 2]))) // => [0, 2]
 * ```
 */
export function relu(x: TensorLike): Value {
  const v = val(x)
  return where(greater(v, 0), v, 0)
}

/**
 * Elementwise Heaviside step: 1 where `x > 0`, else 0 (the derivative of {@link relu}).
 *
 * @param x - Input.
 * @returns The step function of `x` in the input's float dtype.
 * @example
 * ```ts
 * import { step, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(step(tensor([-1, 0, 2]))) // => [0, 0, 1]
 * ```
 */
export function step(x: TensorLike): Value {
  const v = floatify(x)
  return cast(where(greater(v, 0), 1, 0), v.dtype)
}

/**
 * Elementwise floor. Its derivative is zero.
 *
 * @param x - Input.
 * @returns The largest integer not above `x`.
 * @example
 * ```ts
 * import { floor, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(floor(tensor([1.5, -1.5]))) // => [1, -2]
 * ```
 */
export function floor(x: TensorLike): Value {
  return apply(prims.floor, x)
}

/**
 * Elementwise ceiling. Its derivative is zero.
 *
 * @param x - Input.
 * @returns The smallest integer not below `x`.
 * @example
 * ```ts
 * import { ceil, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(ceil(tensor([1.5, -1.5]))) // => [2, -1]
 * ```
 */
export function ceil(x: TensorLike): Value {
  return apply(prims.ceil, x)
}

/**
 * Elementwise rounding to the nearest integer, halves to even (numpy semantics). Its derivative is zero.
 *
 * @param x - Input.
 * @returns The rounded values.
 * @example
 * ```ts
 * import { round, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(round(tensor([0.5, 1.5, 2.5, -0.5]))) // => [0, 2, 2, -0]
 * ```
 */
export function round(x: TensorLike): Value {
  return apply(prims.round, x)
}

/**
 * Elementwise truncation towards zero. Its derivative is zero.
 *
 * @param x - Input.
 * @returns The integer part of `x`.
 * @example
 * ```ts
 * import { trunc, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(trunc(tensor([1.7, -1.7]))) // => [1, -1]
 * ```
 */
export function trunc(x: TensorLike): Value {
  return apply(prims.trunc, x)
}

/**
 * Elementwise two-argument arctangent (the angle of the point `(x, y)`).
 *
 * @param y - Ordinate.
 * @param x - Abscissa.
 * @returns `atan2(y, x)` in `(-pi, pi]`.
 * @example
 * ```ts
 * import { arctan2, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(arctan2(tensor([1]), tensor([-1]))) // => [2.356194]
 * ```
 */
export function arctan2(y: TensorLike, x: TensorLike): Value {
  return bind1(atan2Primitive, coerce2(y, x, 'float'), null)
}
