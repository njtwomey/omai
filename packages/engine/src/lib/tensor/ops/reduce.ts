/**
 * Reductions over axes: sums, products, extrema (ties share the gradient), means, two-pass variances and standard
 * deviations, argmax and argmin, cumulative sums and products, the stable `logsumexp`, `softmax` and `logSoftmax`, and
 * vector norms of any order.
 *
 * @index function Sums, means, extrema, argmax, cumulative ops, softmax and logsumexp over axes.
 */

import { MdmlError } from '../../base/index.ts'
import type { ReduceKernel } from '../backend.ts'
import { batchedAxis } from '../batching.ts'
import { arg, type BatchDim, type BatchResult, bind1, definePrimitive, type Primitive } from '../primitive.ts'
import { normaliseAxes, normaliseAxis, sizeOf } from '../shape.ts'
import type { TensorLike } from '../tensor.ts'
import type { Value } from '../traced.ts'
import { floatify, val } from './_coerce.ts'
import { abs, add, cast, div, mul, pow, square, stopGradient, sub } from './arithmetic.ts'
import { equal, isfinite, where } from './compare.ts'
import { exp, log, sqrt } from './math.ts'
import { broadcastTo, reshape } from './shape.ts'

interface ReduceParams {
  readonly axes: readonly number[]
  readonly keepDims: boolean
}

function keptShape(shape: readonly number[], axes: readonly number[]): number[] {
  return shape.map((n, a) => (axes.includes(a) ? 1 : n))
}

function reducedShape(shape: readonly number[], p: ReduceParams): number[] {
  return p.keepDims ? keptShape(shape, p.axes) : shape.filter((_, a) => !p.axes.includes(a))
}

/** Bring a reduced value back to the input's shape (keeping reduced axes as length 1, then broadcasting). */
function expandBack(g: Value, shape: readonly number[], axes: readonly number[]): Value {
  return broadcastTo(reshape(g, keptShape(shape, axes)), shape)
}

/**
 * The product of all the other elements of each reduced group, at every position: `out / x` where the group has no
 * zero, the product of the rest where it has exactly one (at the zero itself), and 0 everywhere else. Built from
 * safe divisions, so there is no 0/0 and the rule stays differentiable.
 */
function prodOfOthers(x: Value, axes: readonly number[]): Value {
  const isZero = equal(x, 0)
  const safe = where(isZero, 1, x)
  const nonZeroProduct = prod(safe, axes, true)
  const zeros = sum(isZero, axes, true)
  return where(isZero, where(equal(zeros, 1), nonZeroProduct, 0), where(equal(zeros, 0), div(nonZeroProduct, safe), 0))
}

/** `sqrt` with a zero gradient at zero (the subgradient of a norm) instead of 0 * Infinity; NaN still propagates. */
function sqrtZeroGradient(s: Value): Value {
  const isZero = equal(s, 0)
  return where(isZero, 0, sqrt(where(isZero, 1, s)))
}

function reducePrimitive(op: ReduceKernel): Primitive<ReduceParams> {
  const primitive: Primitive<ReduceParams> = definePrimitive<ReduceParams>({
    name: `reduce_${op}`,
    abstractEval: (xs, p) => [{ shape: reducedShape(arg(xs, 0).shape, p), dtype: arg(xs, 0).dtype }],
    impl: (xs, p) => [arg(xs, 0).backend.reduce(op, arg(xs, 0), p.axes, p.keepDims)],
    jvp: (xs, ts, p, outs) => {
      const t = arg(ts, 0)
      if (t === null) return [null]
      const x = arg(xs, 0)
      if (op === 'sum') return [sum(t, p.axes, p.keepDims)]
      const out = expandBack(arg(outs, 0), x.shape, p.axes)
      if (op === 'prod') return [sum(mul(t, prodOfOthers(x, p.axes)), p.axes, p.keepDims)]
      const mask = cast(equal(x, out), x.dtype)
      const share = div(mask, expandBack(sum(mask, p.axes, true), x.shape, p.axes))
      return [sum(mul(t, share), p.axes, p.keepDims)]
    },
    vjp: (gs, xs, p, outs) => {
      const g = arg(gs, 0)
      if (g === null) return [null]
      const x = arg(xs, 0)
      const gx = expandBack(g, x.shape, p.axes)
      if (op === 'sum') return [gx]
      const out = expandBack(arg(outs, 0), x.shape, p.axes)
      if (op === 'prod') return [mul(gx, prodOfOthers(x, p.axes))]
      const mask = cast(equal(x, out), x.dtype)
      return [mul(gx, div(mask, expandBack(sum(mask, p.axes, true), x.shape, p.axes)))]
    },
    // Reduce the same axes of every example in place; the batch axis shifts left past dropped axes.
    batch: (xs, ds, p) => {
      const d = arg(ds, 0) ?? 0
      const axes = p.axes.map((a) => batchedAxis(a, d))
      const dim = p.keepDims ? d : d - p.axes.filter((a) => a < d).length
      return { outputs: [bind1(primitive, [arg(xs, 0)], { axes, keepDims: p.keepDims })], dims: [dim] }
    },
    describe: (p) => `axes=[${p.axes.join(',')}]`,
  })
  return primitive
}

const sumPrimitive = reducePrimitive('sum')
const prodPrimitive = reducePrimitive('prod')
const maxPrimitive = reducePrimitive('max')
const minPrimitive = reducePrimitive('min')

function argPrimitive(op: 'max' | 'min'): Primitive<{ readonly axis: number; readonly keepDims: boolean }> {
  const primitive: Primitive<{ readonly axis: number; readonly keepDims: boolean }> = definePrimitive<{
    readonly axis: number
    readonly keepDims: boolean
  }>({
    name: `arg${op}`,
    abstractEval: (xs, p) => {
      const shape = arg(xs, 0).shape
      return [
        {
          shape: p.keepDims ? shape.map((n, a) => (a === p.axis ? 1 : n)) : shape.filter((_, a) => a !== p.axis),
          dtype: 'int32',
        },
      ]
    },
    impl: (xs, p) => [arg(xs, 0).backend.argReduce(op, arg(xs, 0), p.axis, p.keepDims)],
    batch: (xs, ds, p) => {
      const d = arg(ds, 0) ?? 0
      const axis = batchedAxis(p.axis, d)
      const dim = p.keepDims || p.axis >= d ? d : d - 1
      return { outputs: [bind1(primitive, [arg(xs, 0)], { axis, keepDims: p.keepDims })], dims: [dim] }
    },
    zeroDerivative: true,
  })
  return primitive
}

const argmaxPrimitive = argPrimitive('max')
const argminPrimitive = argPrimitive('min')

interface ScanParams {
  readonly axis: number
  readonly reverse: boolean
}

const cumsumPrimitive: Primitive<ScanParams> = definePrimitive<ScanParams>({
  name: 'cumsum',
  abstractEval: (xs) => [arg(xs, 0)],
  impl: (xs, p) => [arg(xs, 0).backend.scan('sum', arg(xs, 0), p.axis, p.reverse)],
  jvp: (_xs, ts, p) => {
    const t = arg(ts, 0)
    return [t === null ? null : bind1(cumsumPrimitive, [t], p)]
  },
  vjp: (gs, _xs, p) => {
    const g = arg(gs, 0)
    return [g === null ? null : bind1(cumsumPrimitive, [g], { axis: p.axis, reverse: !p.reverse })]
  },
  batch: (xs, ds, p) => scanBatch(cumsumPrimitive, xs, ds, p),
})

const cumprodPrimitive: Primitive<ScanParams> = definePrimitive<ScanParams>({
  name: 'cumprod',
  abstractEval: (xs) => [arg(xs, 0)],
  impl: (xs, p) => [arg(xs, 0).backend.scan('prod', arg(xs, 0), p.axis, p.reverse)],
  jvp: (xs, ts, p, outs) => {
    const t = arg(ts, 0)
    return [t === null ? null : mul(arg(outs, 0), bind1(cumsumPrimitive, [div(t, arg(xs, 0))], p))]
  },
  vjp: (gs, xs, p, outs) => {
    const g = arg(gs, 0)
    if (g === null) return [null]
    const back = bind1(cumsumPrimitive, [mul(g, arg(outs, 0))], { axis: p.axis, reverse: !p.reverse })
    return [div(back, arg(xs, 0))]
  },
  batch: (xs, ds, p) => scanBatch(cumprodPrimitive, xs, ds, p),
})

function scanBatch(
  primitive: Primitive<ScanParams>,
  xs: readonly Value[],
  ds: readonly BatchDim[],
  p: ScanParams,
): BatchResult {
  const d = arg(ds, 0) ?? 0
  return { outputs: [bind1(primitive, [arg(xs, 0)], { axis: batchedAxis(p.axis, d), reverse: p.reverse })], dims: [d] }
}

function reduceWith(
  p: Primitive<ReduceParams>,
  x: Value,
  axis: number | readonly number[] | undefined,
  keepDims: boolean,
): Value {
  const axes = normaliseAxes(axis, x.shape.length)
  if (axes.length === 0) return x
  return bind1(p, [x], { axes, keepDims })
}

/**
 * Sum over axes (all axes by default). Bool sums count as int32.
 *
 * @param x - Input.
 * @param axis - An axis or list of axes; negative values count from the end.
 * @param keepDims - Keep reduced axes with length 1.
 * @returns The sum.
 * @example
 * ```ts
 * import { sum, tensor, toArray } from 'mdml-engine/tensor'
 * const x = tensor([[1, 2], [3, 4]])
 * toArray(sum(x)) // => 10
 * toArray(sum(x, 0)) // => [4, 6]
 * ```
 */
export function sum(x: TensorLike, axis?: number | readonly number[], keepDims = false): Value {
  return reduceWith(sumPrimitive, intForBool(val(x)), axis, keepDims)
}

/**
 * Product over axes (all axes by default).
 *
 * @param x - Input.
 * @param axis - An axis or list of axes.
 * @param keepDims - Keep reduced axes with length 1.
 * @returns The product.
 * @example
 * ```ts
 * import { prod, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(prod(tensor([2, 3, 4]))) // => 24
 * ```
 */
export function prod(x: TensorLike, axis?: number | readonly number[], keepDims = false): Value {
  return reduceWith(prodPrimitive, intForBool(val(x)), axis, keepDims)
}

/**
 * Maximum over axes (all axes by default); NaN propagates, ties share the gradient.
 *
 * @param x - Input.
 * @param axis - An axis or list of axes.
 * @param keepDims - Keep reduced axes with length 1.
 * @returns The maximum.
 * @example
 * ```ts
 * import { max, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(max(tensor([[1, 5], [3, 2]]), 1)) // => [5, 3]
 * ```
 */
export function max(x: TensorLike, axis?: number | readonly number[], keepDims = false): Value {
  return reduceWith(maxPrimitive, val(x), axis, keepDims)
}

/**
 * Minimum over axes (all axes by default); NaN propagates, ties share the gradient.
 *
 * @param x - Input.
 * @param axis - An axis or list of axes.
 * @param keepDims - Keep reduced axes with length 1.
 * @returns The minimum.
 * @example
 * ```ts
 * import { min, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(min(tensor([[1, 5], [3, 2]]), 1)) // => [1, 2]
 * ```
 */
export function min(x: TensorLike, axis?: number | readonly number[], keepDims = false): Value {
  return reduceWith(minPrimitive, val(x), axis, keepDims)
}

/**
 * Arithmetic mean over axes (all axes by default); integers give float64.
 *
 * @param x - Input.
 * @param axis - An axis or list of axes.
 * @param keepDims - Keep reduced axes with length 1.
 * @returns The mean.
 * @example
 * ```ts
 * import { mean, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(mean(tensor([[1, 2], [3, 4]]), 0)) // => [2, 3]
 * ```
 */
export function mean(x: TensorLike, axis?: number | readonly number[], keepDims = false): Value {
  const v = floatify(x)
  const axes = normaliseAxes(axis, v.shape.length)
  const count = sizeOf(axes.map((a) => v.shape[a] ?? 1))
  return div(sum(v, axes, keepDims), count)
}

/** Options for {@link variance} and {@link std}. */
export interface VarianceOptions {
  readonly ddof?: number
  readonly keepDims?: boolean
}

/**
 * Variance over axes, computed in two passes (mean first) for accuracy (numpy `var`).
 *
 * @param x - Input.
 * @param axis - An axis or list of axes (all by default).
 * @param options - `ddof` (delta degrees of freedom, default 0) and `keepDims`.
 * @returns The variance.
 * @example
 * ```ts
 * import { variance, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(variance(tensor([1, 2, 3, 4]))) // => 1.25
 * toArray(variance(tensor([1, 2, 3, 4]), undefined, { ddof: 1 })) // => 1.666667
 * ```
 */
export function variance(x: TensorLike, axis?: number | readonly number[], options: VarianceOptions = {}): Value {
  const v = floatify(x)
  const axes = normaliseAxes(axis, v.shape.length)
  const n = sizeOf(axes.map((a) => v.shape[a] ?? 1))
  const ddof = options.ddof ?? 0
  if (n - ddof <= 0) throw new MdmlError('value', `variance: ${n} values leave no degrees of freedom with ddof ${ddof}`)
  const centred = sub(v, mean(v, axes, true))
  return div(sum(square(centred), axes, options.keepDims ?? false), n - ddof)
}

/**
 * Standard deviation over axes (the square root of {@link variance}; its gradient is 0, not NaN, where the variance
 * is 0).
 *
 * @param x - Input.
 * @param axis - An axis or list of axes.
 * @param options - `ddof` and `keepDims`.
 * @returns The standard deviation.
 * @example
 * ```ts
 * import { std, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(std(tensor([2, 4, 4, 4, 5, 5, 7, 9]))) // => 2
 * ```
 */
export function std(x: TensorLike, axis?: number | readonly number[], options: VarianceOptions = {}): Value {
  return sqrtZeroGradient(variance(x, axis, options))
}

/**
 * Index of the maximum along an axis (the first one on ties; NaN wins). Flattens when no axis is given.
 *
 * @param x - Input.
 * @param axis - The axis (default: over the flattened input).
 * @param keepDims - Keep the reduced axis with length 1.
 * @returns int32 indices.
 * @example
 * ```ts
 * import { argmax, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(argmax(tensor([[1, 5], [3, 2]]), 1)) // => [1, 0]
 * ```
 */
export function argmax(x: TensorLike, axis?: number, keepDims = false): Value {
  const v = val(x)
  if (axis === undefined) return bind1(argmaxPrimitive, [reshape(v, [-1])], { axis: 0, keepDims: false })
  return bind1(argmaxPrimitive, [v], { axis: normaliseAxis(axis, v.shape.length), keepDims })
}

/**
 * Index of the minimum along an axis (the first one on ties). Flattens when no axis is given.
 *
 * @param x - Input.
 * @param axis - The axis (default: over the flattened input).
 * @param keepDims - Keep the reduced axis with length 1.
 * @returns int32 indices.
 * @example
 * ```ts
 * import { argmin, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(argmin(tensor([3, 1, 2]))) // => 1
 * ```
 */
export function argmin(x: TensorLike, axis?: number, keepDims = false): Value {
  const v = val(x)
  if (axis === undefined) return bind1(argminPrimitive, [reshape(v, [-1])], { axis: 0, keepDims: false })
  return bind1(argminPrimitive, [v], { axis: normaliseAxis(axis, v.shape.length), keepDims })
}

/** Bool input counts as int32 in sums and products (a bool running total would wrap at 256). */
function intForBool(v: Value): Value {
  return v.dtype === 'bool' ? cast(v, 'int32') : v
}

/**
 * Cumulative sum along an axis (flattened when no axis is given). Bool input counts as int32.
 *
 * @param x - Input.
 * @param axis - The axis.
 * @param reverse - Accumulate from the end.
 * @returns The running sums.
 * @example
 * ```ts
 * import { cumsum, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(cumsum(tensor([1, 2, 3]))) // => [1, 3, 6]
 * ```
 */
export function cumsum(x: TensorLike, axis?: number, reverse = false): Value {
  const v = intForBool(val(x))
  const input = axis === undefined ? reshape(v, [-1]) : v
  return bind1(cumsumPrimitive, [input], {
    axis: axis === undefined ? 0 : normaliseAxis(axis, v.shape.length),
    reverse,
  })
}

/**
 * Cumulative product along an axis (flattened when no axis is given). Bool input counts as int32. Its derivative
 * divides by the input, so it is exact away from zeros.
 *
 * @param x - Input.
 * @param axis - The axis.
 * @param reverse - Accumulate from the end.
 * @returns The running products.
 * @example
 * ```ts
 * import { cumprod, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(cumprod(tensor([1, 2, 3]))) // => [1, 2, 6]
 * ```
 */
export function cumprod(x: TensorLike, axis?: number, reverse = false): Value {
  const v = intForBool(val(x))
  const input = axis === undefined ? reshape(v, [-1]) : v
  return bind1(cumprodPrimitive, [input], {
    axis: axis === undefined ? 0 : normaliseAxis(axis, v.shape.length),
    reverse,
  })
}

/**
 * `log(sum(exp(x)))` over axes, computed stably by shifting by the maximum (the shift carries no gradient, so the
 * gradient is the softmax).
 *
 * @param x - Input.
 * @param axis - An axis or axes (default: all).
 * @param keepDims - Keep reduced axes with length 1.
 * @returns The log-sum-exp.
 * @concept log-sum-exp
 * @example
 * ```ts
 * import { logsumexp, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(logsumexp(tensor([1000, 1000]))) // => 1000.693147
 * ```
 */
export function logsumexp(x: TensorLike, axis?: number | readonly number[], keepDims = false): Value {
  const v = floatify(x)
  const axes = normaliseAxes(axis, v.shape.length)
  // Rows that are entirely -Infinity (or contain +Infinity) are shifted by 0 so the result stays exact, and so are
  // empty reductions (no maximum exists; the result is log(0) = -Infinity, as in scipy).
  const empty = axes.some((a) => v.shape[a] === 0)
  const m = empty ? undefined : stopGradient(max(v, axes, true))
  const shift = m === undefined ? 0 : where(isfinite(m), m, 0)
  const out = add(log(sum(exp(sub(v, shift)), axes, true)), shift)
  return keepDims
    ? out
    : reshape(
        out,
        v.shape.filter((_, a) => !axes.includes(a)),
      )
}

/**
 * Softmax along an axis, computed stably through {@link logsumexp}.
 *
 * @param x - Logits.
 * @param axis - The axis (default -1).
 * @returns Probabilities that sum to one along `axis`.
 * @concept softmax
 * @example
 * ```ts
 * import { softmax, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(softmax(tensor([0, Math.log(3)]))) // => [0.25, 0.75]
 * ```
 */
export function softmax(x: TensorLike, axis = -1): Value {
  const v = floatify(x)
  return exp(sub(v, logsumexp(v, axis, true)))
}

/**
 * Log-softmax along an axis (`x - logsumexp(x)`), the stable input to cross-entropy.
 *
 * @param x - Logits.
 * @param axis - The axis (default -1).
 * @returns Log-probabilities.
 * @example
 * ```ts
 * import { logSoftmax, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(logSoftmax(tensor([0, 0]))) // => [-0.693147, -0.693147]
 * ```
 */
export function logSoftmax(x: TensorLike, axis = -1): Value {
  const v = floatify(x)
  return sub(v, logsumexp(v, axis, true))
}

/**
 * Vector norm over axes (all by default): `ord` 2 (Euclidean, the default), 1, `Infinity`, `-Infinity` or any p > 0.
 * The gradient of the 2-norm at the zero vector is 0. Matrix norms other than Frobenius are in `numerics/linalg`.
 *
 * @param x - Input.
 * @param ord - The order (default 2).
 * @param axis - An axis or axes.
 * @param keepDims - Keep reduced axes with length 1.
 * @returns The norm.
 * @example
 * ```ts
 * import { norm, tensor, toArray } from 'mdml-engine/tensor'
 * toArray(norm(tensor([3, 4]))) // => 5
 * toArray(norm(tensor([3, -4]), 1)) // => 7
 * ```
 */
export function norm(x: TensorLike, ord = 2, axis?: number | readonly number[], keepDims = false): Value {
  const v = floatify(x)
  if (ord === 2) return sqrtZeroGradient(sum(square(v), axis, keepDims))
  if (ord === 1) return sum(abs(v), axis, keepDims)
  if (ord === Infinity) return max(abs(v), axis, keepDims)
  if (ord === -Infinity) return min(abs(v), axis, keepDims)
  if (!(ord > 0)) throw new MdmlError('value', `norm: order must be positive, got ${ord}`)
  return pow(sum(pow(abs(v), ord), axis, keepDims), 1 / ord)
}
