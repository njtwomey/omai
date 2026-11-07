/**
 * Implicit differentiation through solutions: derivatives of a root, a fixed point or a minimiser with respect to its
 * parameters from the implicit function theorem, never by unrolling the solver. The basis of optimisation layers, deep
 * equilibrium models, bilevel problems and hyperparameter gradients.
 */

import { grad, jacobian, jvp, vjp } from '../autodiff/index.ts'
import { MdmlError } from '../base/index.ts'
import { solve as denseSolve } from '../numerics/linalg/index.ts'
import { fixedPoint } from '../numerics/roots/index.ts'
import { cg, gmres, type LinearOperator } from '../numerics/sparse/index.ts'
import {
  arg,
  asTensor,
  bind,
  definePrimitive,
  neg,
  type Primitive,
  primalOf,
  promoteFloat,
  reshape,
  sizeOf,
  sub,
  swapaxes,
  type Tensor,
  type TensorLike,
  type Value,
  zerosLike,
} from '../tensor/index.ts'
import { minimize } from './minimize/minimizer.ts'

/** Options of the implicit-differentiation wrappers. */
export interface ImplicitOptions {
  /**
   * How to solve the linear systems with dF/dx: `dense` (the Jacobian by autodiff and an LU solve, itself
   * differentiable, so higher derivatives work; the default), `gmres` (matrix-free, products by vjp and jvp) or `cg`
   * (matrix-free, for symmetric positive definite dF/dx such as the Hessian of an argmin). The Krylov solves run on
   * primal values: first derivatives only.
   */
  readonly linearSolver?: 'dense' | 'gmres' | 'cg'
  /** Relative tolerance of the Krylov solves (default 1e-12). */
  readonly tol?: number
}

/** The residual F(x, theta_1, ..., theta_k) whose root x*(theta) is differentiated. */
export type ImplicitResidual = (x: Value, ...thetas: Value[]) => Value

/** A solver: x*(theta) from concrete parameters. */
export type ImplicitSolver = (...thetas: Tensor[]) => TensorLike

interface ImplicitParams {
  readonly F: ImplicitResidual
  readonly options: ImplicitOptions
}

const flatOf = (v: Value): Value => reshape(v, [-1])

/** Solve (dF/dx)^T lambda = g or (dF/dx) y = r at (x, thetas). */
function linearSolve(
  params: ImplicitParams,
  x: Value,
  thetas: readonly Value[],
  rhs: Value,
  transpose: boolean,
): Value {
  const fx = (v: Value): Value => flatOf(params.F(reshape(v, x.shape), ...thetas))
  const n = sizeOf(x.shape)
  const method = params.options.linearSolver ?? 'dense'
  const b = flatOf(rhs)
  if (method === 'dense') {
    const j = reshape(jacobian(fx)(flatOf(x)), [n, n])
    return reshape(denseSolve(transpose ? swapaxes(j, 0, 1) : j, b), x.shape)
  }
  const point = flatOf(primalOf(x))
  const op: LinearOperator = {
    shape: [n, n],
    matvec: transpose ? (v) => vjp(fx, point).pullback(v) : (v) => jvp(fx, point, v).tangent,
  }
  const solver = method === 'cg' ? cg : gmres
  const r = solver(op, primalOf(b), { rtol: params.options.tol ?? 1e-12, maxIterations: 10 * n + 10 })
  if (!r.converged) throw new MdmlError('convergence', `implicit differentiation: the ${method} solve did not converge`)
  return reshape(r.x, x.shape)
}

const implicitPrimitive: Primitive<ImplicitParams> = definePrimitive<ImplicitParams>({
  name: 'implicitSolution',
  abstractEval: ([x]) => [{ shape: (x as { shape: readonly number[] }).shape, dtype: 'float64' }],
  impl: ([x]) => [x as Tensor],
  vjp: (gs, inputs, params, outs) => {
    const g = arg(gs, 0)
    // The output, not the constant solution input: at higher orders it carries x*'s own dependence on theta.
    const x = arg(outs, 0)
    const thetas = inputs.slice(1)
    if (g === null) return inputs.map(() => null)
    // lambda solves (dF/dx)^T lambda = g; then theta_bar = -(dF/dtheta)^T lambda.
    const lambda = linearSolve(params, x, thetas, g, true)
    const { pullback } = vjp((ts: Value[]) => params.F(x, ...ts), thetas)
    const bars = pullback(lambda).map((t) => neg(t))
    return [null, ...bars]
  },
  jvp: (inputs, tangents, params, outs) => {
    const x = arg(outs, 0)
    const thetas = inputs.slice(1)
    const ts = tangents.slice(1)
    if (ts.every((t) => t === null)) return [null]
    // dx = -(dF/dx)^{-1} sum_i (dF/dtheta_i) dtheta_i.
    const filled = thetas.map((th, i) => ts[i] ?? zerosLike(th))
    const { tangent } = jvp((args: Value[]) => params.F(x, ...args), thetas as Value[], filled as Value[])
    return [neg(linearSolve(params, x, thetas, tangent, false))]
  },
})

/**
 * Differentiate through the root of an equation: x*(theta) solves F(x, theta) = 0, and its derivatives come from the
 * implicit function theorem, dx/dtheta = -(dF/dx)^{-1} dF/dtheta, never from the solver's iterations (any solver, even
 * a black box, works). Reverse mode solves (dF/dx)^T lambda = g; forward mode solves (dF/dx) dx = -dF/dtheta dtheta.
 *
 * @param F - The residual F(x, ...thetas), with an output the size of x.
 * @param solver - Computes the root from concrete parameters.
 * @param options - The linear solver for the adjoint and tangent systems.
 * @returns x*(...thetas), usable inside and outside transformations.
 * @concept implicit-differentiation
 * @example
 * ```ts
 * import { implicitRoot } from 'mdml-engine/optim'
 * import { grad } from 'mdml-engine/autodiff'
 * import { mul, sqrt, sub, sum, tensor, toArray, type Value } from 'mdml-engine/tensor'
 * // x^2 = theta: x* = sqrt(theta), dx/dtheta = 1 / (2 sqrt(theta)).
 * const root = implicitRoot((x, t) => sub(mul(x, x), t as Value), (t) => sqrt(t))
 * toArray(grad((t: Value) => sum(root(t)))(tensor([4]))) // => [0.25]
 * ```
 */
export function implicitRoot(
  F: ImplicitResidual,
  solver: ImplicitSolver,
  options: ImplicitOptions = {},
): (...thetas: Value[]) => Value {
  const params: ImplicitParams = { F, options }
  return (...thetas) => {
    const concrete = thetas.map((t) => primalOf(promoteFloat(t)[0] as Value) as Tensor)
    const x = primalOf(promoteFloat(solver(...concrete))[0] as Value) as Tensor
    const out = primalOf(F(x, ...concrete))
    if (sizeOf(out.shape) !== sizeOf(x.shape))
      throw new MdmlError('shape', 'implicitRoot: F(x, theta) must have as many entries as x')
    return arg(bind(implicitPrimitive, [x, ...thetas.map((t) => promoteFloat(t)[0] as Value)], params), 0)
  }
}

/** Options of {@link implicitFixedPoint}. */
export interface ImplicitFixedPointOptions extends ImplicitOptions {
  /** The starting point of the default solver (scipy's fixed_point by Steffensen's method). */
  readonly x0?: TensorLike
  /** The default solver's tolerance and iteration cap. */
  readonly xtol?: number
  readonly maxIterations?: number
}

/**
 * Differentiate through a fixed point x* = T(x*, theta) (deep equilibrium models, value iteration, mean-field
 * updates): the residual x - T(x, theta) with {@link implicitRoot}. The default solver is the numerics fixed-point
 * iteration from `x0`.
 *
 * @param T - The map T(x, ...thetas).
 * @param solver - A solver from concrete parameters (default: fixed-point iteration from `options.x0`).
 * @param options - The linear solver, and the default solver's start, tolerance and iteration cap.
 * @returns x*(...thetas).
 * @throws MdmlError with code `convergence` when the default solver does not converge (the result must be a solution).
 * @concept implicit-differentiation
 * @example
 * ```ts
 * import { implicitFixedPoint } from 'mdml-engine/optim'
 * import { grad } from 'mdml-engine/autodiff'
 * import { add, mul, sum, tensor, toArray, type Value } from 'mdml-engine/tensor'
 * // x = x / 2 + theta: x* = 2 theta, dx/dtheta = 2.
 * const fp = implicitFixedPoint((x, t) => add(mul(x, 0.5), t as Value), undefined, { x0: [0] })
 * toArray(grad((t: Value) => sum(fp(t)))(tensor([1]))).map((v) => Math.round(v * 1e8) / 1e8) // => [2]
 * ```
 */
export function implicitFixedPoint(
  T: ImplicitResidual,
  solver?: ImplicitSolver,
  options: ImplicitFixedPointOptions = {},
): (...thetas: Value[]) => Value {
  const solve: ImplicitSolver =
    solver ??
    ((...thetas) => {
      if (options.x0 === undefined) throw new MdmlError('value', 'implicitFixedPoint: give a solver or x0')
      const start = primalOf(asTensor(options.x0))
      const r = fixedPoint((v) => flatOf(T(reshape(v, start.shape), ...thetas)), flatOf(start), {
        xtol: options.xtol ?? 1e-14,
        maxIterations: options.maxIterations ?? 10000,
      })
      if (!r.converged)
        throw new MdmlError('convergence', `implicitFixedPoint: no convergence after ${r.iterations} iterations`)
      return reshape(r.x, start.shape)
    })
  return implicitRoot((x, ...thetas) => sub(x, T(x, ...thetas)), solve, options)
}

/** Options of {@link implicitArgmin}. */
export interface ImplicitArgminOptions extends ImplicitOptions {
  /** The starting point of the default solver ({@link minimize} with its default method). */
  readonly x0?: TensorLike
}

/**
 * Differentiate through the minimiser of an objective (optimisation layers, bilevel problems, hyperparameter
 * gradients): x*(theta) = argmin_x f(x, theta) is the root of the stationarity condition grad_x f = 0, so
 * dx/dtheta = -H^{-1} d^2 f / dx dtheta, with H the Hessian (`cg` suits it: it is symmetric positive definite at a
 * strict minimum).
 *
 * @param f - The objective f(x, ...thetas), a scalar.
 * @param solver - A solver from concrete parameters (default: {@link minimize} from `options.x0`).
 * @param options - The linear solver and the default solver's start.
 * @returns x*(...thetas).
 * @concept implicit-differentiation
 * @example
 * ```ts
 * import { implicitArgmin } from 'mdml-engine/optim'
 * import { grad } from 'mdml-engine/autodiff'
 * import { add, div, mul, square, sub, sum, tensor, toArray, type Value } from 'mdml-engine/tensor'
 * // argmin (x - 3)^2 + theta x^2 = 3 / (1 + theta); at theta = 1 the derivative is -3 / 4.
 * const xs = implicitArgmin((x, t) => sum(add(square(sub(x, 3)), mul(t as Value, square(x)))), (t) => div(3, add(t, 1)))
 * toArray(grad((t: Value) => sum(xs(t)))(tensor([1]))) // => [-0.75]
 * ```
 */
export function implicitArgmin(
  f: ImplicitResidual,
  solver?: ImplicitSolver,
  options: ImplicitArgminOptions = {},
): (...thetas: Value[]) => Value {
  const solve: ImplicitSolver =
    solver ??
    ((...thetas) => {
      if (options.x0 === undefined) throw new MdmlError('value', 'implicitArgmin: give a solver or x0')
      return minimize((x) => f(x as Value, ...thetas), primalOf(asTensor(options.x0))).x as Value
    })
  const stationarity: ImplicitResidual = (x, ...thetas) => grad((v: Value) => f(v, ...thetas))(x)
  return implicitRoot(stationarity, solve, options)
}
