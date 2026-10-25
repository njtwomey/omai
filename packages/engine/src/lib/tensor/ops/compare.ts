/**
 * Comparisons and logic on tensors: elementwise comparisons with broadcasting, NaN and infinity tests, logical
 * connectives, `where` for choosing elements by a condition (differentiable in the chosen branches), `any` and `all`
 * over axes, and numpy's `allclose`.
 */

import type { CompareKernel, PredicateKernel } from '../backend.ts'
import { batchElementwise } from '../batching.ts'
import { arg, bind1, definePrimitive, type Primitive } from '../primitive.ts'
import { broadcastShapes } from '../shape.ts'
import { isTensor, type TensorLike, toFlat } from '../tensor.ts'
import type { Value } from '../traced.ts'
import { asDtype, coerce2, val } from './_coerce.ts'
import { toShape } from './_factories.ts'
import { abs, add, cast, mul, sub } from './arithmetic.ts'
import { max, min } from './reduce.ts'
import { sumToShape } from './shape.ts'

function comparePrimitive(op: CompareKernel): Primitive<null> {
  const primitive: Primitive<null> = definePrimitive<null>({
    name: op,
    abstractEval: (xs) => [{ shape: broadcastShapes(arg(xs, 0).shape, arg(xs, 1).shape), dtype: 'bool' }],
    impl: (xs) => [arg(xs, 0).backend.compare(op, arg(xs, 0), arg(xs, 1))],
    batch: (xs, ds) => batchElementwise(primitive, xs, ds, null),
    zeroDerivative: true,
  })
  return primitive
}

function predicatePrimitive(op: PredicateKernel): Primitive<null> {
  const primitive: Primitive<null> = definePrimitive<null>({
    name: op,
    abstractEval: (xs) => [{ shape: arg(xs, 0).shape, dtype: 'bool' }],
    impl: (xs) => [arg(xs, 0).backend.predicate(op, arg(xs, 0))],
    batch: (xs, ds) => batchElementwise(primitive, xs, ds, null),
    zeroDerivative: true,
  })
  return primitive
}

const prims = {
  eq: comparePrimitive('eq'),
  ne: comparePrimitive('ne'),
  lt: comparePrimitive('lt'),
  le: comparePrimitive('le'),
  gt: comparePrimitive('gt'),
  ge: comparePrimitive('ge'),
  isnan: predicatePrimitive('isnan'),
  isfinite: predicatePrimitive('isfinite'),
  isinf: predicatePrimitive('isinf'),
}

/** Select between two values by a bool condition, with broadcasting. @internal */
export const wherePrimitive: Primitive<null> = definePrimitive<null>({
  name: 'where',
  abstractEval: (xs) => [
    {
      shape: broadcastShapes(arg(xs, 0).shape, broadcastShapes(arg(xs, 1).shape, arg(xs, 2).shape)),
      dtype: arg(xs, 1).dtype,
    },
  ],
  impl: (xs) => [arg(xs, 1).backend.where(arg(xs, 0) as never, arg(xs, 1), arg(xs, 2))],
  jvp: (xs, ts, _p, outs) => {
    const ta = arg(ts, 1)
    const tb = arg(ts, 2)
    if (ta === null && tb === null) return [null]
    const shape = arg(outs, 0).shape
    const c = arg(xs, 0)
    return [where(c, ta === null ? 0 : toShape(ta, shape), tb === null ? 0 : toShape(tb, shape))]
  },
  vjp: (gs, xs) => {
    const g = arg(gs, 0)
    if (g === null) return [null, null, null]
    const c = arg(xs, 0)
    return [null, sumToShape(where(c, g, 0), arg(xs, 1).shape), sumToShape(where(c, 0, g), arg(xs, 2).shape)]
  },
  batch: (xs, ds) => batchElementwise(wherePrimitive, xs, ds, null),
})

function cmp(p: Primitive<null>, a: TensorLike, b: TensorLike): Value {
  return bind1(p, coerce2(a, b, 'any'), null)
}

/**
 * Elementwise `a == b` with broadcasting.
 *
 * @param a - Left operand.
 * @param b - Right operand.
 * @returns A bool value.
 * @example
 * ```ts
 * import { equal, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(equal(tensor([1, 2]), 2)) // => [0, 1]
 * ```
 */
export function equal(a: TensorLike, b: TensorLike): Value {
  return cmp(prims.eq, a, b)
}

/**
 * Elementwise `a != b` with broadcasting.
 *
 * @param a - Left operand.
 * @param b - Right operand.
 * @returns A bool value.
 * @example
 * ```ts
 * import { notEqual, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(notEqual(tensor([1, 2]), 2)) // => [1, 0]
 * ```
 */
export function notEqual(a: TensorLike, b: TensorLike): Value {
  return cmp(prims.ne, a, b)
}

/**
 * Elementwise `a < b` with broadcasting.
 *
 * @param a - Left operand.
 * @param b - Right operand.
 * @returns A bool value.
 * @example
 * ```ts
 * import { less, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(less(tensor([1, 3]), 2)) // => [1, 0]
 * ```
 */
export function less(a: TensorLike, b: TensorLike): Value {
  return cmp(prims.lt, a, b)
}

/**
 * Elementwise `a <= b` with broadcasting.
 *
 * @param a - Left operand.
 * @param b - Right operand.
 * @returns A bool value.
 * @example
 * ```ts
 * import { lessEqual, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(lessEqual(tensor([1, 2, 3]), 2)) // => [1, 1, 0]
 * ```
 */
export function lessEqual(a: TensorLike, b: TensorLike): Value {
  return cmp(prims.le, a, b)
}

/**
 * Elementwise `a > b` with broadcasting.
 *
 * @param a - Left operand.
 * @param b - Right operand.
 * @returns A bool value.
 * @example
 * ```ts
 * import { greater, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(greater(tensor([1, 3]), 2)) // => [0, 1]
 * ```
 */
export function greater(a: TensorLike, b: TensorLike): Value {
  return cmp(prims.gt, a, b)
}

/**
 * Elementwise `a >= b` with broadcasting.
 *
 * @param a - Left operand.
 * @param b - Right operand.
 * @returns A bool value.
 * @example
 * ```ts
 * import { greaterEqual, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(greaterEqual(tensor([1, 2, 3]), 2)) // => [0, 1, 1]
 * ```
 */
export function greaterEqual(a: TensorLike, b: TensorLike): Value {
  return cmp(prims.ge, a, b)
}

/**
 * Elementwise NaN test.
 *
 * @param x - Input.
 * @returns A bool value.
 * @example
 * ```ts
 * import { isnan, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(isnan(tensor([1, Number.NaN]))) // => [0, 1]
 * ```
 */
export function isnan(x: TensorLike): Value {
  return bind1(prims.isnan, [val(x)], null)
}

/**
 * Elementwise finiteness test.
 *
 * @param x - Input.
 * @returns A bool value.
 * @example
 * ```ts
 * import { isfinite, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(isfinite(tensor([1, Infinity, Number.NaN]))) // => [1, 0, 0]
 * ```
 */
export function isfinite(x: TensorLike): Value {
  return bind1(prims.isfinite, [val(x)], null)
}

/**
 * Elementwise infinity test.
 *
 * @param x - Input.
 * @returns A bool value.
 * @example
 * ```ts
 * import { isinf, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(isinf(tensor([1, -Infinity]))) // => [0, 1]
 * ```
 */
export function isinf(x: TensorLike): Value {
  return bind1(prims.isinf, [val(x)], null)
}

/**
 * Elementwise logical and (inputs are tested for non-zero).
 *
 * @param a - Left operand.
 * @param b - Right operand.
 * @returns A bool value.
 * @example
 * ```ts
 * import { logicalAnd, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(logicalAnd(tensor([true, true]), tensor([true, false]))) // => [1, 0]
 * ```
 */
export function logicalAnd(a: TensorLike, b: TensorLike): Value {
  return equal(count(a, b), 2)
}

/**
 * Elementwise logical or.
 *
 * @param a - Left operand.
 * @param b - Right operand.
 * @returns A bool value.
 * @example
 * ```ts
 * import { logicalOr, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(logicalOr(tensor([true, false]), tensor([false, false]))) // => [1, 0]
 * ```
 */
export function logicalOr(a: TensorLike, b: TensorLike): Value {
  return greater(count(a, b), 0)
}

/**
 * Elementwise logical exclusive or.
 *
 * @param a - Left operand.
 * @param b - Right operand.
 * @returns A bool value.
 * @example
 * ```ts
 * import { logicalXor, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(logicalXor(tensor([true, true]), tensor([true, false]))) // => [0, 1]
 * ```
 */
export function logicalXor(a: TensorLike, b: TensorLike): Value {
  return equal(count(a, b), 1)
}

/**
 * Elementwise logical not.
 *
 * @param x - Input (tested for non-zero).
 * @returns A bool value.
 * @example
 * ```ts
 * import { logicalNot, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(logicalNot(tensor([true, false]))) // => [0, 1]
 * ```
 */
export function logicalNot(x: TensorLike): Value {
  return equal(x, 0)
}

// Number of true operands per element, as int32.
function count(a: TensorLike, b: TensorLike): Value {
  return add(cast(notEqual(a, 0), 'int32'), cast(notEqual(b, 0), 'int32'))
}

/**
 * Choose elements from `a` where `condition` holds, otherwise from `b` (like `numpy.where`), with broadcasting.
 * The gradient flows to the chosen branch only.
 *
 * @param condition - A condition (tested for non-zero).
 * @param a - Values where true.
 * @param b - Values where false.
 * @returns The selection.
 * @example
 * ```ts
 * import { where, greater, tensor, toArray } from 'whitebox-engine/tensor'
 * const x = tensor([-1, 2])
 * toArray(where(greater(x, 0), x, 0)) // => [0, 2]
 * ```
 */
export function where(condition: TensorLike, a: TensorLike, b: TensorLike): Value {
  const c = asDtype(val(condition), 'bool')
  const [x, y] = coerce2(a, b, 'any')
  return bind1(wherePrimitive, [c, x, y], null)
}

/**
 * Whether any element is non-zero, over axes (all by default).
 *
 * @param x - Input.
 * @param axis - An axis or axes.
 * @returns A bool value.
 * @example
 * ```ts
 * import { any, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(any(tensor([0, 0, 1]))) // => 1
 * ```
 */
export function any(x: TensorLike, axis?: number | readonly number[]): Value {
  return greater(max(cast(notEqual(x, 0), 'int32'), axis), 0)
}

/**
 * Whether every element is non-zero, over axes (all by default).
 *
 * @param x - Input.
 * @param axis - An axis or axes.
 * @returns A bool value.
 * @example
 * ```ts
 * import { all, tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(all(tensor([1, 1, 0]))) // => 0
 * ```
 */
export function all(x: TensorLike, axis?: number | readonly number[]): Value {
  return greater(min(cast(notEqual(x, 0), 'int32'), axis), 0)
}

/**
 * Whether two concrete arrays are elementwise close (numpy `allclose`: `|a - b| <= atol + rtol |b|`, shapes broadcast).
 *
 * @param a - First array.
 * @param b - Second array (the reference).
 * @param rtol - Relative tolerance (default 1e-5).
 * @param atol - Absolute tolerance (default 1e-8).
 * @returns `true` when every element is close.
 * @example
 * ```ts
 * import { allclose, tensor } from 'whitebox-engine/tensor'
 * allclose(tensor([1, 2]), tensor([1, 2.000001])) // => true
 * ```
 */
export function allclose(a: TensorLike, b: TensorLike, rtol = 1e-5, atol = 1e-8): boolean {
  const [x, y] = coerce2(a, b, 'float')
  const close = lessEqual(abs(sub(x, y)), add(mul(abs(y), rtol), atol))
  // Equal infinities are close even though their difference is NaN.
  const ok = logicalOr(close, equal(x, y))
  return isTensor(ok) && toFlat(ok).every((v) => v === 1)
}
