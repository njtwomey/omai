/**
 * Losses for physics-informed neural networks (PINNs), which fit a network to a differential equation by penalising its
 * residual: field derivatives at collocation points by automatic differentiation, the PDE residual, boundary-condition
 * and initial-condition terms, and two schemes for balancing the terms (gradient-norm learning-rate annealing and
 * neural-tangent-kernel weights).
 */

import { grad, vjp } from '../autodiff/index.ts'
import { OmaiError } from '../base/index.ts'
import {
  abs,
  asTensor,
  fromStorage,
  mul,
  promoteFloat,
  reshape,
  sliceAxis,
  square,
  stack,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  tensor,
  toFlat,
  toVector,
  treeLeaves,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { type LossOptions, reduce } from './_reduce.ts'

/** A scalar field evaluated pointwise: points `[N, d]` to values `[N]` (a network, or a closed form). */
export type Field = (points: Value) => Value

/** A field and its derivatives at a batch of points, by automatic differentiation. */
export interface FieldDerivatives {
  /** u at each point `[N]`. */
  readonly value: Value
  /** The gradient du/dx `[N, d]`. */
  readonly gradient: Value
  /** The diagonal second derivatives d2u/dx_k^2 `[N, d]` (present when `order` is 2). */
  readonly secondDerivatives?: Value
  /** The Laplacian sum_k d2u/dx_k^2 `[N]` (present when `order` is 2). */
  readonly laplacian?: Value
}

/**
 * The value, gradient and (with order 2) diagonal second derivatives and Laplacian of a field at collocation points,
 * by reverse-mode differentiation through the field. Points are independent, so one backward pass per derivative
 * gives every point's derivative. The results stay differentiable in whatever the field closes over (network
 * weights), as physics-informed training needs.
 *
 * @param u - The field.
 * @param points - Collocation points `[N, d]`.
 * @param options - order (1 or 2, default 2).
 * @returns The derivatives.
 * @concept physics-informed-networks
 * @example
 * ```ts
 * import { fieldDerivatives } from 'omai-engine/losses'
 * import { mul, sum, toArray, type Value } from 'omai-engine/tensor'
 * const u = (x: Value) => sum(mul(x, x), 1)
 * toArray(fieldDerivatives(u, [[1, 2]]).laplacian as Value) // => [4]
 * ```
 */
export function fieldDerivatives(
  u: Field,
  points: TensorLike,
  options: { readonly order?: 1 | 2 } = {},
): FieldDerivatives {
  const [x] = promoteFloat(points) as [Value]
  if (x.shape.length !== 2) throw new OmaiError('shape', 'fieldDerivatives: points must be [N, d]')
  const d = x.shape[1] ?? 1
  const value = u(x)
  const gradientOf = (p: Value): Value => grad((v: Value) => sum(u(v)))(p)
  const gradient = gradientOf(x)
  if ((options.order ?? 2) === 1) return { value, gradient }
  const columns: Value[] = []
  for (let k = 0; k < d; k++) {
    const dk = grad((v: Value) => sum(sliceAxis(gradientOf(v), 1, k, k + 1)))(x)
    columns.push(reshape(sliceAxis(dk, 1, k, k + 1), [x.shape[0] ?? 0]))
  }
  const second = stack(columns, 1)
  return { value, gradient, secondDerivatives: second, laplacian: sum(second, 1) }
}

/**
 * The mean squared residual of a differential equation at collocation points: the physics term of a PINN loss
 * (Raissi, Perdikaris and Karniadakis, 2019). The residual function receives the field's derivatives and the points,
 * for example `(d) => sub(d.laplacian, f)` for a Poisson equation.
 *
 * @param u - The field.
 * @param points - Collocation points `[N, d]`.
 * @param residual - The equation's residual from the derivatives (and points) `[N]`.
 * @param options - order of derivatives needed (default 2), reduction and weights.
 * @returns The loss, differentiable in the field's parameters.
 * @example
 * ```ts
 * import { pdeResidualLoss } from 'omai-engine/losses'
 * import { item, mul, sub, sum, type Value } from 'omai-engine/tensor'
 * const u = (x: Value) => sum(mul(x, x), 1)
 * item(pdeResidualLoss(u, [[0.1, 0.2], [0.3, 0.4]], (d) => sub(d.laplacian as Value, 4))) // => 0
 * ```
 */
export function pdeResidualLoss(
  u: Field,
  points: TensorLike,
  residual: (derivatives: FieldDerivatives, points: Value) => Value,
  options: LossOptions & { readonly order?: 1 | 2 } = {},
): Value {
  const x = promoteFloat(points)[0] as Value
  return reduce(square(residual(fieldDerivatives(u, x, { order: options.order ?? 2 }), x)), options)
}

/** Options of {@link boundaryLoss}. */
export interface BoundaryOptions extends LossOptions {
  /** 'dirichlet' matches u (default); 'neumann' matches the normal derivative du/dn = grad u . n. */
  readonly kind?: 'dirichlet' | 'neumann'
  /** Outward unit normals `[N, d]` for Neumann conditions. */
  readonly normals?: TensorLike
}

/**
 * The boundary-condition term of a PINN loss: the mean squared mismatch between the field (Dirichlet) or its normal
 * derivative (Neumann) and prescribed values at boundary points.
 *
 * @param u - The field.
 * @param points - Boundary points `[N, d]`.
 * @param values - Prescribed values `[N]`.
 * @param options - kind, normals, reduction and weights.
 * @returns The loss, differentiable in the field's parameters.
 * @throws OmaiError with code `value` for a Neumann condition without normals.
 * @example
 * ```ts
 * import { boundaryLoss } from 'omai-engine/losses'
 * import { item, mul, sum, type Value } from 'omai-engine/tensor'
 * const u = (x: Value) => sum(mul(x, x), 1)
 * item(boundaryLoss(u, [[1, 0]], [2], { kind: 'neumann', normals: [[1, 0]] })) // => 0
 * ```
 */
export function boundaryLoss(u: Field, points: TensorLike, values: TensorLike, options: BoundaryOptions = {}): Value {
  const [x, g] = promoteFloat(points, values) as [Value, Value]
  if ((options.kind ?? 'dirichlet') === 'dirichlet') return reduce(square(sub(u(x), g)), options)
  if (options.normals === undefined) throw new OmaiError('value', 'boundaryLoss: Neumann conditions need normals')
  const n = promoteFloat(options.normals)[0] as Value
  const normal = sum(mul(fieldDerivatives(u, x, { order: 1 }).gradient, n), 1)
  return reduce(square(sub(normal, g)), options)
}

/**
 * The initial-condition term of a PINN loss for space-time fields u(x, t): the mean squared mismatch of u (and, with
 * `derivative: 1`, of du/dt, for second-order-in-time equations such as the wave equation) at t = t0 points.
 *
 * @param u - The field over points `[N, d]` whose `timeAxis` column is time.
 * @param points - Initial points `[N, d]`.
 * @param values - Prescribed values `[N]`.
 * @param options - derivative (0, default, or 1), timeAxis (default the last column), reduction and weights.
 * @returns The loss, differentiable in the field's parameters.
 * @example
 * ```ts
 * import { initialConditionLoss } from 'omai-engine/losses'
 * import { item, mul, sum, type Value } from 'omai-engine/tensor'
 * const u = (p: Value) => sum(mul(p, p), 1)
 * item(initialConditionLoss(u, [[0.5, 0]], [0.25])) // => 0
 * ```
 */
export function initialConditionLoss(
  u: Field,
  points: TensorLike,
  values: TensorLike,
  options: LossOptions & { readonly derivative?: 0 | 1; readonly timeAxis?: number } = {},
): Value {
  const [x, g] = promoteFloat(points, values) as [Value, Value]
  if ((options.derivative ?? 0) === 0) return reduce(square(sub(u(x), g)), options)
  const d = x.shape[1] ?? 1
  const axis = options.timeAxis ?? d - 1
  const dt = reshape(sliceAxis(fieldDerivatives(u, x, { order: 1 }).gradient, 1, axis, axis + 1), [x.shape[0] ?? 0])
  return reduce(square(sub(dt, g)), options)
}

/** A loss term as a function of the model parameters (any pytree). */
export type LossTerm<P> = (params: P) => Value

const vectorOf = (a: ArrayLike<number>): Tensor<'float64', [number]> =>
  fromStorage(Float64Array.from(a), [a.length], 'float64')

const flatAbs = (tree: unknown): number[] => treeLeaves(tree).flatMap((leaf) => toFlat(abs(asTensor(leaf))))

/**
 * Learning-rate annealing weights for multi-term PINN losses (Wang, Teng and Perdikaris, 2021): with the first term the
 * PDE residual, lambda_i = max |grad L_0| / mean |grad L_i| for every other term, blended with the previous weights as
 * lambda <- (1 - alpha) lambda_prev + alpha lambda_hat. The weights are numbers (no gradient), recomputed every few
 * steps. Takes the previous weights as a tensor-like and returns a tensor (migration, D-009: `previous` was
 * `readonly number[]`, the result a `number[]`).
 *
 * @param terms - Loss terms as functions of the parameters, the residual first.
 * @param params - The current parameters.
 * @param options - previous weights (a vector) and alpha (default 0.9).
 * @returns One weight per term (the first is 1), a `[terms]` tensor.
 * @throws OmaiError with code `shape` unless `previous` has rank 1, and `not-differentiable` for a traced one.
 * @example
 * ```ts
 * import { gradientNormWeights } from 'omai-engine/losses'
 * import { mul, sum, tensor, toFlat, type Value } from 'omai-engine/tensor'
 * const terms = [(p: Value) => sum(mul(p, 4)), (p: Value) => sum(mul(p, 2))]
 * toFlat(gradientNormWeights(terms, tensor([1, 1]), { alpha: 1 })) // => [1, 2]
 * ```
 */
export function gradientNormWeights<P>(
  terms: readonly LossTerm<P>[],
  params: P,
  options: { readonly previous?: TensorLike; readonly alpha?: number } = {},
): Tensor<'float64', [number]> {
  const previous = options.previous === undefined ? undefined : toVector(options.previous, 'gradientNormWeights')
  if (terms.length === 0) return vectorOf([])
  const alpha = options.alpha ?? 0.9
  const grads = terms.map((t) => flatAbs(grad(t)(params)))
  let top = 0
  for (const g of grads[0] ?? []) top = Math.max(top, g)
  const weights = grads.map((g, i) => {
    if (i === 0) return 1
    const meanAbs = g.reduce((s, v) => s + v, 0) / Math.max(g.length, 1)
    const hat = meanAbs > 0 ? top / meanAbs : 1
    const prev = previous?.[i]
    return prev === undefined ? hat : (1 - alpha) * prev + alpha * hat
  })
  return vectorOf(weights)
}

/** A pointwise model output (a residual or a prediction at points) as a function of the parameters `[N_i]`. */
export type PointwiseTerm<P> = (params: P) => Value

/**
 * Neural-tangent-kernel weights for multi-term PINN losses (Wang, Yu and Perdikaris, 2022): lambda_i = Tr(K) / Tr(K_i)
 * where Tr(K_i) = sum over the term's points of ||d f_i(x) / d theta||^2 is the trace of its NTK block and Tr(K) is the
 * sum over terms. Terms whose outputs move slowly under gradient descent get larger weights. Each term returns its
 * pointwise outputs (residuals at collocation points, predictions at boundary points).
 *
 * @param terms - Pointwise terms as functions of the parameters.
 * @param params - The current parameters.
 * @returns One weight per term, a `[terms]` tensor (migration, D-009: was a `number[]`).
 * @example
 * ```ts
 * import { ntkWeights } from 'omai-engine/losses'
 * import { mul, tensor, toFlat, type Value } from 'omai-engine/tensor'
 * toFlat(ntkWeights([(p: Value) => mul(p, 1), (p: Value) => mul(p, 3)], tensor([1, 2]))) // => [10, 1.1111111111111112]
 * ```
 */
export function ntkWeights<P>(terms: readonly PointwiseTerm<P>[], params: P): Tensor<'float64', [number]> {
  const traces = terms.map((t) => {
    const { out, pullback } = vjp(t, params)
    const n = toFlat(asTensor(out)).length
    let trace = 0
    for (let i = 0; i < n; i++) {
      const basis = toFlat(zeros([n])).map((_, j) => (j === i ? 1 : 0))
      const g = pullback(reshape(tensor(basis), out.shape) as Value)
      for (const leaf of treeLeaves(g)) for (const v of toFlat(asTensor(leaf))) trace += v * v
    }
    return trace
  })
  const total = traces.reduce((s, v) => s + v, 0)
  return vectorOf(traces.map((t) => (t > 0 ? total / t : 1)))
}
