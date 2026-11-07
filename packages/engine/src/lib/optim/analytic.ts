/**
 * Objectives with hand-written derivatives, as differentiable functions. A model whose likelihood is coded over plain
 * numbers (a softmax over classes, ordinal thresholds, a profiled mixed-model deviance) with its analytic gradient, and
 * a Hessian when it has one, becomes an `Objective` the solvers of `smoothSolvers` accept: the gradient is the
 * function's reverse rule and the Hessian its forward rule, so Newton's method gets the exact curvature and the
 * quasi-Newton, conjugate-gradient and stochastic solvers get the gradient, all on the same code. Without a Hessian
 * the Newton mode differences the gradient.
 */

import {
  arg,
  bind1,
  definePrimitive,
  fromStorage,
  matmul,
  mul,
  primalOf,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  toVector,
  type Value,
} from '../tensor/index.ts'

/** A model's objective over plain numbers, with its derivatives. */
export interface AnalyticObjective {
  /** The objective at a parameter vector. */
  value(x: Float64Array): number
  /** Its gradient. */
  gradient(x: Float64Array): ArrayLike<number>
  /** Its Hessian, row-major `[m, m]` (default: central differences of the gradient). */
  hessian?(x: Float64Array): ArrayLike<number>
}

/**
 * The Hessian by central differences of an analytic gradient, symmetrised. Takes the point as a tensor-like and returns
 * the `[m, m]` matrix as a tensor (migration, D-009: `x` was a `Float64Array`, the result a row-major `Float64Array`);
 * the gradient stays a function over plain numbers, like the rest of this module's callbacks.
 *
 * @param gradient - The gradient.
 * @param point - The point, a vector.
 * @returns The Hessian, an `[m, m]` tensor.
 * @throws MdmlError with code `shape` unless `point` has rank 1, and `not-differentiable` for a traced one.
 * @example
 * ```ts
 * import { finiteDifferenceHessian } from 'mdml-engine/optim'
 * import { toFlat } from 'mdml-engine/tensor'
 * const h = finiteDifferenceHessian((x) => Float64Array.from([2 * x[0], 6 * x[1]]), [1, 1])
 * toFlat(h).map((v) => Math.round(v)) // => [2, 0, 0, 6]
 * ```
 */
export function finiteDifferenceHessian(
  gradient: (x: Float64Array) => ArrayLike<number>,
  point: TensorLike,
): Tensor<'float64', [number, number]> {
  const x = Float64Array.from(toVector(point, 'finiteDifferenceHessian'))
  return fromStorage(hessianByDifferences(gradient, x), [x.length, x.length], 'float64')
}

/** The kernel of {@link finiteDifferenceHessian} on plain numbers, row-major. */
function hessianByDifferences(gradient: (x: Float64Array) => ArrayLike<number>, x: Float64Array): Float64Array {
  const m = x.length
  const H = new Float64Array(m * m)
  for (let j = 0; j < m; j++) {
    const h = 1e-5 * Math.max(1, Math.abs(x[j] ?? 0))
    const up = Float64Array.from(x)
    const dn = Float64Array.from(x)
    up[j] = (up[j] ?? 0) + h
    dn[j] = (dn[j] ?? 0) - h
    const gu = gradient(up)
    const gd = gradient(dn)
    for (let i = 0; i < m; i++) H[i * m + j] = ((gu[i] ?? 0) - (gd[i] ?? 0)) / (2 * h)
  }
  for (let i = 0; i < m; i++)
    for (let j = 0; j < i; j++) {
      const v = 0.5 * ((H[i * m + j] ?? 0) + (H[j * m + i] ?? 0))
      H[i * m + j] = v
      H[j * m + i] = v
    }
  return H
}

const flatOf = (x: Value): Float64Array => Float64Array.from(toFlat(primalOf(x)))

/**
 * An objective from its value, gradient and (optionally) Hessian over plain numbers, as a function of a parameter
 * vector the minimisers differentiate: reverse mode returns the gradient, forward mode the Hessian-vector product.
 *
 * @param spec - The value, the gradient and the Hessian.
 * @returns The objective, a scalar function of a vector.
 * @example
 * ```ts
 * import { analyticObjective, flatObjective } from 'mdml-engine/optim'
 * const f = analyticObjective({
 *   value: (x) => x[0] ** 2 + 3 * x[1] ** 2,
 *   gradient: (x) => [2 * x[0], 6 * x[1]],
 *   hessian: () => [2, 0, 0, 6],
 * })
 * const flat = flatObjective(f, [1, 1])
 * [flat.value([1, 1]), Array.from(flat.valueAndGrad([1, 1]).grad), Array.from(flat.hessian([1, 1]))] // => [4, [2, 6], [2, 0, 0, 6]]
 * ```
 */
export function analyticObjective(spec: AnalyticObjective): (x: Value) => Value {
  const hessianAt = (x: Float64Array): ArrayLike<number> => spec.hessian?.(x) ?? hessianByDifferences(spec.gradient, x)
  const gradientPrimitive = definePrimitive<null>({
    name: 'analytic-gradient',
    abstractEval: ([x]) => [{ shape: (x as { shape: readonly number[] }).shape, dtype: 'float64' }],
    impl: ([x]) => {
      const point = flatOf(x as Value)
      return [fromStorage(Float64Array.from(spec.gradient(point)), [point.length], 'float64')]
    },
    jvp: (primals, tangents) => {
      const t = arg(tangents, 0)
      if (t === null) return [null]
      const x = flatOf(arg(primals, 0))
      return [matmul(fromStorage(Float64Array.from(hessianAt(x)), [x.length, x.length], 'float64'), t)]
    },
    vjp: (cotangents, inputs) => {
      const g = arg(cotangents, 0)
      if (g === null) return [null]
      const x = flatOf(arg(inputs, 0))
      return [matmul(fromStorage(Float64Array.from(hessianAt(x)), [x.length, x.length], 'float64'), g)]
    },
  })
  const valuePrimitive = definePrimitive<null>({
    name: 'analytic-value',
    abstractEval: () => [{ shape: [], dtype: 'float64' }],
    impl: ([x]) => [fromStorage(Float64Array.of(spec.value(flatOf(x as Value))), [], 'float64')],
    jvp: (primals, tangents) => {
      const t = arg(tangents, 0)
      if (t === null) return [null]
      return [sum(mul(bind1(gradientPrimitive, [arg(primals, 0)], null), t))]
    },
    vjp: (cotangents, inputs) => {
      const g = arg(cotangents, 0)
      if (g === null) return [null]
      return [mul(g, bind1(gradientPrimitive, [arg(inputs, 0)], null))]
    },
  })
  return (x) => bind1(valuePrimitive, [x], null)
}
