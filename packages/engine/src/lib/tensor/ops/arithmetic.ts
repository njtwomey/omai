import { type DType, isFloat } from '../dtype.ts'
import { arg, bind1, definePrimitive, type Primitive } from '../primitive.ts'
import type { TensorLike } from '../tensor.ts'
import type { Value } from '../traced.ts'
import { coerce2, val } from './_coerce.ts'
import { binaryPrimitive, toShape, unaryPrimitive } from './_factories.ts'
import { equal, greater, less, where } from './compare.ts'
import { floor, log } from './math.ts'

/** Elementwise sum. @internal */
export const addPrimitive: Primitive<null> = binaryPrimitive('add', {
  jvp: (_a, _b, ta, tb, out) =>
    ta === null ? toShape(tb as Value, out.shape) : tb === null ? toShape(ta, out.shape) : add(ta, tb),
  vjp: (g) => [g, g],
})

/** Elementwise difference. @internal */
export const subPrimitive: Primitive<null> = binaryPrimitive('sub', {
  jvp: (_a, _b, ta, tb, out) =>
    ta === null ? toShape(neg(tb as Value), out.shape) : tb === null ? toShape(ta, out.shape) : sub(ta, tb),
  vjp: (g) => [g, neg(g)],
})

/** Elementwise product. @internal */
export const mulPrimitive: Primitive<null> = binaryPrimitive('mul', {
  jvp: (a, b, ta, tb) => (ta === null ? mul(a, tb as Value) : tb === null ? mul(ta, b) : add(mul(ta, b), mul(a, tb))),
  vjp: (g, a, b) => [mul(g, b), mul(g, a)],
})

/** Elementwise quotient. @internal */
export const divPrimitive: Primitive<null> = binaryPrimitive('div', {
  // d(a / b) = (da - db * out) / b
  jvp: (_a, b, ta, tb, out) =>
    div(ta === null ? neg(mul(tb as Value, out)) : tb === null ? ta : sub(ta, mul(tb, out)), b),
  vjp: (g, _a, b, out) => [div(g, b), neg(div(mul(g, out), b))],
})

// d/db a^b = a^b log a, taken as 0 where a = 0 (the limit for b > 0).
function powLogTerm(a: Value, out: Value): Value {
  return where(equal(a, 0), 0, mul(out, log(a)))
}

/** Elementwise power. @internal */
export const powPrimitive: Primitive<null> = binaryPrimitive('pow', {
  jvp: (a, b, ta, tb, out) => {
    const da = ta === null ? null : mul(ta, mul(b, pow(a, sub(b, 1))))
    const db = tb === null ? null : mul(tb, powLogTerm(a, out))
    return da === null ? db : db === null ? da : add(da, db)
  },
  vjp: (g, a, b, out) => [mul(g, mul(b, pow(a, sub(b, 1)))), mul(g, powLogTerm(a, out))],
})

// Ties share the derivative equally, as in JAX.
function maxShare(a: Value, b: Value, pick: (x: Value, y: Value) => Value): Value {
  return cast(add(where(pick(a, b), 1, 0), where(equal(a, b), 0.5, 0)), a.dtype)
}

/** Elementwise maximum. @internal */
export const maxPrimitive: Primitive<null> = binaryPrimitive('max', {
  jvp: (a, b, ta, tb) => {
    const s = maxShare(a, b, greater)
    const da = ta === null ? null : mul(ta, s)
    const db = tb === null ? null : mul(tb, sub(1, s))
    return da === null ? db : db === null ? da : add(da, db)
  },
  vjp: (g, a, b) => {
    const s = maxShare(a, b, greater)
    return [mul(g, s), mul(g, sub(1, s))]
  },
})

/** Elementwise minimum. @internal */
export const minPrimitive: Primitive<null> = binaryPrimitive('min', {
  jvp: (a, b, ta, tb) => {
    const s = maxShare(a, b, less)
    const da = ta === null ? null : mul(ta, s)
    const db = tb === null ? null : mul(tb, sub(1, s))
    return da === null ? db : db === null ? da : add(da, db)
  },
  vjp: (g, a, b) => {
    const s = maxShare(a, b, less)
    return [mul(g, s), mul(g, sub(1, s))]
  },
})

/** Python-style modulo (sign of the divisor). @internal */
export const modPrimitive: Primitive<null> = binaryPrimitive('mod', {
  jvp: (a, b, ta, tb, out) => {
    const q = floor(div(a, b))
    return ta === null ? neg(mul(tb as Value, q)) : tb === null ? toShape(ta, out.shape) : sub(ta, mul(tb, q))
  },
  vjp: (g, a, b) => [g, neg(mul(g, floor(div(a, b))))],
})

const floorDivPrimitive = binaryPrimitive('floorDiv')
const negPrimitive = unaryPrimitive('neg', (t) => neg(t))
const absPrimitive = unaryPrimitive('abs', (t, x) => mul(t, sign(x)))
const signPrimitive = unaryPrimitive('sign')

/** Convert to another dtype. @internal */
export const castPrimitive: Primitive<{ readonly dtype: DType }> = definePrimitive<{ readonly dtype: DType }>({
  name: 'cast',
  abstractEval: (xs, p) => [{ shape: arg(xs, 0).shape, dtype: p.dtype }],
  impl: (xs, p) => [arg(xs, 0).backend.cast(arg(xs, 0), p.dtype)],
  jvp: (xs, ts, p) => {
    const t = arg(ts, 0)
    return [t === null || !isFloat(p.dtype) || !isFloat(arg(xs, 0).dtype) ? null : cast(t, p.dtype)]
  },
  vjp: (gs, xs, p) => {
    const g = arg(gs, 0)
    const from = arg(xs, 0).dtype
    return [g === null || !isFloat(from) || !isFloat(p.dtype) ? null : cast(g, from)]
  },
  describe: (p) => p.dtype,
})

/** Identity whose derivative is zero. @internal */
export const stopGradientPrimitive: Primitive<null> = definePrimitive<null>({
  name: 'stopGradient',
  abstractEval: (xs) => [arg(xs, 0)],
  impl: (xs) => [arg(xs, 0)],
  zeroDerivative: true,
})

/**
 * Elementwise sum with broadcasting. Plain numbers are weak: they adopt the other operand's dtype.
 *
 * @param a - Left operand.
 * @param b - Right operand.
 * @returns `a + b`.
 * @concept broadcasting
 * @example
 * ```ts
 * import { add, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(add(tensor([[1], [2]]), tensor([10, 20]))) // => [[11, 21], [12, 22]]
 * ```
 */
export function add(a: TensorLike, b: TensorLike): Value {
  return bind1(addPrimitive, coerce2(a, b), null)
}

/**
 * Elementwise difference with broadcasting.
 *
 * @param a - Left operand.
 * @param b - Right operand.
 * @returns `a - b`.
 * @example
 * ```ts
 * import { sub, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(sub(tensor([5, 7]), 2)) // => [3, 5]
 * ```
 */
export function sub(a: TensorLike, b: TensorLike): Value {
  return bind1(subPrimitive, coerce2(a, b), null)
}

/**
 * Elementwise product with broadcasting.
 *
 * @param a - Left operand.
 * @param b - Right operand.
 * @returns `a * b`.
 * @example
 * ```ts
 * import { mul, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(mul(tensor([1, 2]), tensor([3, 4]))) // => [3, 8]
 * ```
 */
export function mul(a: TensorLike, b: TensorLike): Value {
  return bind1(mulPrimitive, coerce2(a, b), null)
}

/**
 * Elementwise true division with broadcasting (integers give float64; division by zero follows IEEE rules).
 *
 * @param a - Numerator.
 * @param b - Denominator.
 * @returns `a / b`.
 * @example
 * ```ts
 * import { div, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(div(tensor([1, 3]), 2)) // => [0.5, 1.5]
 * ```
 */
export function div(a: TensorLike, b: TensorLike): Value {
  return bind1(divPrimitive, coerce2(a, b, 'float'), null)
}

/**
 * Elementwise power with broadcasting (computed in floating point).
 *
 * @param a - Base.
 * @param b - Exponent.
 * @returns `a ** b`.
 * @example
 * ```ts
 * import { pow, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(pow(tensor([2, 3]), 2)) // => [4, 9]
 * ```
 */
export function pow(a: TensorLike, b: TensorLike): Value {
  return bind1(powPrimitive, coerce2(a, b, 'float'), null)
}

/**
 * Elementwise maximum with broadcasting (NaN propagates; ties split the gradient equally).
 *
 * @param a - Left operand.
 * @param b - Right operand.
 * @returns `max(a, b)`.
 * @example
 * ```ts
 * import { maximum, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(maximum(tensor([-1, 2]), 0)) // => [0, 2]
 * ```
 */
export function maximum(a: TensorLike, b: TensorLike): Value {
  return bind1(maxPrimitive, coerce2(a, b), null)
}

/**
 * Elementwise minimum with broadcasting.
 *
 * @param a - Left operand.
 * @param b - Right operand.
 * @returns `min(a, b)`.
 * @example
 * ```ts
 * import { minimum, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(minimum(tensor([-1, 2]), 0)) // => [-1, 0]
 * ```
 */
export function minimum(a: TensorLike, b: TensorLike): Value {
  return bind1(minPrimitive, coerce2(a, b), null)
}

/**
 * Elementwise floor division (rounds towards negative infinity).
 *
 * @param a - Dividend.
 * @param b - Divisor.
 * @returns `floor(a / b)`.
 * @example
 * ```ts
 * import { floorDiv, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(floorDiv(tensor([7, -7]), 2)) // => [3, -4]
 * ```
 */
export function floorDiv(a: TensorLike, b: TensorLike): Value {
  return bind1(floorDivPrimitive, coerce2(a, b), null)
}

/**
 * Elementwise modulo with the sign of the divisor (Python and numpy semantics).
 *
 * @param a - Dividend.
 * @param b - Divisor.
 * @returns `a - b * floor(a / b)`.
 * @example
 * ```ts
 * import { mod, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(mod(tensor([7, -7]), 3)) // => [1, 2]
 * ```
 */
export function mod(a: TensorLike, b: TensorLike): Value {
  return bind1(modPrimitive, coerce2(a, b), null)
}

/**
 * Elementwise negation.
 *
 * @param x - Input.
 * @returns `-x`.
 * @example
 * ```ts
 * import { neg, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(neg(tensor([1, -2]))) // => [-1, 2]
 * ```
 */
export function neg(x: TensorLike): Value {
  const v = val(x)
  return bind1(negPrimitive, [v.dtype === 'bool' ? cast(v, 'int32') : v], null)
}

/**
 * Elementwise absolute value (subgradient 0 at 0).
 *
 * @param x - Input.
 * @returns `|x|`.
 * @example
 * ```ts
 * import { abs, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(abs(tensor([-3, 2]))) // => [3, 2]
 * ```
 */
export function abs(x: TensorLike): Value {
  return bind1(absPrimitive, [val(x)], null)
}

/**
 * Elementwise sign (-1, 0 or 1; NaN stays NaN). Its derivative is zero.
 *
 * @param x - Input.
 * @returns `sign(x)`.
 * @example
 * ```ts
 * import { sign, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(sign(tensor([-3, 0, 2]))) // => [-1, 0, 1]
 * ```
 */
export function sign(x: TensorLike): Value {
  return bind1(signPrimitive, [val(x)], null)
}

/**
 * Elementwise square.
 *
 * @param x - Input.
 * @returns `x * x`.
 * @example
 * ```ts
 * import { square, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(square(tensor([-3, 2]))) // => [9, 4]
 * ```
 */
export function square(x: TensorLike): Value {
  const v = val(x)
  return mul(v, v)
}

/**
 * Elementwise reciprocal.
 *
 * @param x - Input.
 * @returns `1 / x`.
 * @example
 * ```ts
 * import { reciprocal, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(reciprocal(tensor([2, 4]))) // => [0.5, 0.25]
 * ```
 */
export function reciprocal(x: TensorLike): Value {
  return div(1, x)
}

/**
 * Limit values to an interval (gradient 1 inside, 0 outside).
 *
 * @param x - Input.
 * @param lo - Lower bound.
 * @param hi - Upper bound.
 * @returns `min(max(x, lo), hi)`.
 * @example
 * ```ts
 * import { clip, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(clip(tensor([-2, 0.5, 3]), 0, 1)) // => [0, 0.5, 1]
 * ```
 */
export function clip(x: TensorLike, lo: TensorLike, hi: TensorLike): Value {
  return minimum(maximum(x, lo), hi)
}

/**
 * Convert to another dtype (floats truncate towards zero when cast to int32; bool is `x != 0`).
 *
 * @param x - Input.
 * @param dtype - Target dtype.
 * @returns The converted value (the input itself when the dtype already matches).
 * @example
 * ```ts
 * import { cast, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(cast(tensor([1.7, -1.7]), 'int32')) // => [1, -1]
 * ```
 */
export function cast(x: TensorLike, dtype: DType): Value {
  const v = val(x)
  return v.dtype === dtype ? v : bind1(castPrimitive, [v], { dtype })
}

/**
 * The identity function with a zero derivative: values pass through, gradients stop.
 *
 * @param x - Input.
 * @returns `x`, treated as a constant by differentiation.
 * @concept stop-gradient
 * @example
 * ```ts
 * import { stopGradient, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(stopGradient(tensor([1, 2]))) // => [1, 2]
 * ```
 */
export function stopGradient(x: TensorLike): Value {
  return bind1(stopGradientPrimitive, [val(x)], null)
}
