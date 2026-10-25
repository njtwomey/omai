/**
 * First-order methods for nonsmooth and structured convex problems: the projected subgradient method with the classic
 * step-size rules, mirror descent with entropic and Euclidean mirror maps (exponentiated gradient on the simplex), and
 * dual ascent, the method of multipliers and dual decomposition for equality-constrained and separable problems, each a
 * traced algorithm.
 */

import { MdmlError } from '../../base/index.ts'
import { daxpy, ddot, dgemv, dnrm2 } from '../../numerics/linalg/index.ts'
import {
  add,
  asTensor,
  fromStorage,
  matmul,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { configOf, observeVector, runVector, type VectorState, vectorDone, vectorSchema } from './_shared.ts'
import { minimize } from './minimizer.ts'
import {
  flatObjective,
  infNorm,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'

type Vec = Float64Array

const flat = (v: Value): Vec => Float64Array.from(toFlat(primalOf(v)))

/**
 * A step-size rule for the subgradient and mirror descent methods (Boyd's notes): `constant` a,
 * `constantLength` a / |g|, `squareSummable` a / (k + 1), `diminishing` a / sqrt(k + 1), `polyak`
 * (f(x) - fStar) / |g|^2, or a function of the iteration, the objective and the subgradient norm.
 */
export type StepRule =
  | 'constant'
  | 'constantLength'
  | 'squareSummable'
  | 'diminishing'
  | 'polyak'
  | ((k: number, value: number, subgradientNorm: number) => number)

function stepOf(rule: StepRule, a: number, k: number, value: number, gnorm: number, fStar: number | undefined): number {
  if (typeof rule === 'function') return rule(k, value, gnorm)
  switch (rule) {
    case 'constant':
      return a
    case 'constantLength':
      return a / Math.max(gnorm, 1e-300)
    case 'squareSummable':
      return a / (k + 1)
    case 'diminishing':
      return a / Math.sqrt(k + 1)
    case 'polyak':
      if (fStar === undefined) throw new MdmlError('value', "the 'polyak' step rule needs fStar")
      return (value - fStar) / Math.max(gnorm * gnorm, 1e-300)
  }
}

/** Options of {@link subgradientMethod}. */
export interface SubgradientOptions extends VectorMethodOptions {
  /** The step rule (default `diminishing`). */
  readonly stepRule?: StepRule
  /** The rule's constant a (default 1). */
  readonly stepSize?: number
  /** The optimal value, for Polyak's step (and to stop once f - fStar <= tol). */
  readonly fStar?: number
  readonly tol?: number
  /** A projection onto the feasible set (projected subgradient method). */
  readonly project?: (x: Value) => Value
  /** A subgradient oracle (default: the autodiff derivative, a valid subgradient for abs, max, relu and norms). */
  readonly subgradient?: (x: Value) => Value
}

/** State of {@link subgradientAlgorithm}: the iterate and the best point so far. */
export interface SubgradientState extends VectorState {
  readonly best: Vec
  readonly bestValue: number
}

/**
 * The (projected) subgradient method for nonsmooth convex objectives, as a traced algorithm: x+ = P(x - a_k g) with g a
 * subgradient and a_k from a step rule. Not a descent method, so it tracks the best point.
 *
 * @param f - The objective.
 * @param x0 - The starting point.
 * @param options - Step rule and constant, fStar, projection and subgradient oracle.
 * @returns The algorithm.
 * @concept subgradient-method
 * @example
 * ```ts
 * import { subgradientAlgorithm } from 'mdml-engine/optim'
 * import { abs, sum, type Value } from 'mdml-engine/tensor'
 * subgradientAlgorithm((x) => sum(abs(x as Value)), [1, -2]).name // => 'subgradient'
 * ```
 */
export function subgradientAlgorithm(
  f: Objective,
  x0: Parameters,
  options: SubgradientOptions = {},
): Algorithm<SubgradientState> {
  const obj = flatObjective(f, x0)
  const rule = options.stepRule ?? 'diminishing'
  const a = options.stepSize ?? 1
  const tol = options.tol ?? 0
  const project = options.project
  const sub = options.subgradient
  const evaluate = (x: Vec): { value: number; grad: Vec } => {
    if (sub === undefined) return obj.valueAndGrad(x)
    return { value: obj.value(x), grad: flat(sub(obj.unravel(x) as Value)) }
  }
  const projected = (x: Vec): Vec => (project === undefined ? x : flat(project(obj.unravel(x) as Value)))
  return {
    name: 'subgradient',
    config: configOf({ ...options, stepRule: typeof rule === 'string' ? rule : 'custom' }),
    schema: vectorSchema({ metrics: { best: { description: 'best objective so far', direction: 'lower' } } }),
    init: () => {
      const x = projected(Float64Array.from(obj.x0))
      const { value, grad } = evaluate(x)
      return { x, value, grad, nfev: 1, ngev: 1, nhev: 0, step: 0, best: x, bestValue: value }
    },
    step: (s, ctx) => {
      const gnorm = dnrm2(s.grad)
      if (gnorm === 0) return { ...s, stop: { status: 'converged', message: 'zero subgradient: x is optimal' } }
      const alpha = stepOf(rule, a, ctx.t, s.value, gnorm, options.fStar)
      const x = projected(daxpy(-alpha, s.grad, s.x))
      const { value, grad } = evaluate(x)
      const better = value < s.bestValue
      const bestValue = better ? value : s.bestValue
      const stop =
        options.fStar !== undefined && bestValue - options.fStar <= tol
          ? ({ status: 'converged', message: 'objective within tol of fStar' } as const)
          : undefined
      return {
        x,
        value,
        grad,
        nfev: s.nfev + 1,
        ngev: s.ngev + 1,
        nhev: 0,
        step: alpha,
        best: better ? x : s.best,
        bestValue,
        ...(stop === undefined ? {} : { stop }),
      }
    },
    done: vectorDone,
    observe: (s) => observeVector(s, { metrics: { best: s.bestValue } }),
  }
}

/** The best point of a subgradient-type run, reported as the result. */
function bestResult(r: MinimizeResult, best: Vec, bestValue: number, unravel: (x: Vec) => unknown): MinimizeResult {
  return { ...r, x: unravel(best), fun: bestValue }
}

/**
 * Minimise a nonsmooth convex function by the (projected) subgradient method.
 *
 * @param f - The objective.
 * @param x0 - The starting point.
 * @param options - Step rule, constant, fStar and tol, projection, subgradient oracle, iteration cap (default 1000).
 * @returns The best point found.
 * @concept subgradient-method
 * @example
 * ```ts
 * import { subgradientMethod } from 'mdml-engine/optim'
 * import { abs, sub, sum, toArray, type Value } from 'mdml-engine/tensor'
 * const r = subgradientMethod((x) => sum(abs(sub(x as Value, [1, -2]))), [0, 0], { stepRule: 'polyak', fStar: 0, tol: 1e-12 })
 * toArray(r.x as Value).map((v) => Math.round(v * 1e6) / 1e6) // => [1, -2]
 * ```
 */
export function subgradientMethod(f: Objective, x0: Parameters, options: SubgradientOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  let last: SubgradientState | undefined
  const alg = subgradientAlgorithm(f, x0, options)
  const tracked: Algorithm<SubgradientState> = {
    ...alg,
    init: (k) => {
      last = alg.init(k)
      return last
    },
    step: (s, ctx) => {
      last = alg.step(s, ctx)
      return last
    },
  }
  const r = runVector(tracked, obj, options, 1000)
  return last === undefined ? r : bestResult(r, last.best, last.bestValue, obj.unravel)
}

/**
 * A mirror map for mirror descent: the gradient of the distance-generating function psi (primal to dual) and of its
 * conjugate followed by the Bregman projection onto the feasible set (dual to primal).
 */
export interface Mirror {
  readonly name: string
  readonly toDual: (x: Vec) => Vec
  readonly toPrimal: (theta: Vec) => Vec
}

/**
 * The entropic mirror map on the simplex {x >= 0, sum x = radius}: psi(x) = sum x log x, so mirror descent is the
 * exponentiated gradient x+ proportional to x exp(-a g).
 *
 * @param radius - The simplex's total (default 1).
 * @returns The mirror map.
 * @example
 * ```ts
 * import { entropicMirror } from 'mdml-engine/optim'
 * Array.from(entropicMirror().toPrimal(new Float64Array([0, 0]))) // => [0.5, 0.5]
 * ```
 */
export function entropicMirror(radius = 1): Mirror {
  return {
    name: 'entropic',
    toDual: (x) => Float64Array.from(x, (v) => Math.log(Math.max(v, 1e-300))),
    toPrimal: (theta) => {
      let m = -Infinity
      for (const v of theta) m = Math.max(m, v)
      const e = Float64Array.from(theta, (v) => Math.exp(v - m))
      let total = 0
      for (const v of e) total += v
      return Float64Array.from(e, (v) => (radius * v) / total)
    },
  }
}

/**
 * The Euclidean mirror map psi = |x|^2 / 2 with a Euclidean projection: mirror descent becomes projected
 * (sub)gradient descent.
 *
 * @param project - The projection onto the feasible set (default none).
 * @returns The mirror map.
 * @example
 * ```ts
 * import { euclideanMirror } from 'mdml-engine/optim'
 * Array.from(euclideanMirror().toPrimal(new Float64Array([1, 2]))) // => [1, 2]
 * ```
 */
export function euclideanMirror(project?: (x: Vec) => ArrayLike<number>): Mirror {
  return {
    name: 'euclidean',
    toDual: (x) => Float64Array.from(x),
    toPrimal: (theta) => (project === undefined ? Float64Array.from(theta) : Float64Array.from(project(theta))),
  }
}

/** Options of {@link mirrorDescent}. */
export interface MirrorDescentOptions extends VectorMethodOptions {
  /** The mirror map (default entropic on the unit simplex). */
  readonly mirror?: Mirror
  /** The step rule (default `diminishing`) and its constant (default 1). */
  readonly stepRule?: StepRule
  readonly stepSize?: number
  readonly fStar?: number
  /** Stop once the relative change of x falls to this (default 0: run to the iteration cap). */
  readonly tol?: number
  /** Report the last, the best or the step-weighted average iterate (default best). */
  readonly output?: 'last' | 'best' | 'average'
}

/** State of {@link mirrorDescentAlgorithm}. */
export interface MirrorDescentState extends SubgradientState {
  readonly average: Vec
  readonly weight: number
}

/**
 * Mirror descent (Nemirovski and Yudin; Beck and Teboulle 2003), as a traced algorithm: theta = grad psi(x) - a g,
 * x+ = grad psi*(theta) with the Bregman projection. With the entropic map on the simplex it is the exponentiated
 * gradient method, whose rate depends on log n instead of n.
 *
 * @param f - The objective.
 * @param x0 - The starting point (feasible; in the simplex's interior for the entropic map).
 * @param options - Mirror map, step rule and constant, tolerance and output.
 * @returns The algorithm.
 * @concept mirror-descent
 * @example
 * ```ts
 * import { mirrorDescentAlgorithm } from 'mdml-engine/optim'
 * import { sum, square, type Value } from 'mdml-engine/tensor'
 * mirrorDescentAlgorithm((x) => sum(square(x as Value)), [0.5, 0.5]).name // => 'mirrorDescent'
 * ```
 */
export function mirrorDescentAlgorithm(
  f: Objective,
  x0: Parameters,
  options: MirrorDescentOptions = {},
): Algorithm<MirrorDescentState> {
  const obj = flatObjective(f, x0)
  const mirror = options.mirror ?? entropicMirror()
  const rule = options.stepRule ?? 'diminishing'
  const a = options.stepSize ?? 1
  const tol = options.tol ?? 0
  return {
    name: 'mirrorDescent',
    config: configOf({ ...options, mirror: mirror.name, stepRule: typeof rule === 'string' ? rule : 'custom' }),
    schema: vectorSchema({ metrics: { best: { description: 'best objective so far', direction: 'lower' } } }),
    init: () => {
      const x = Float64Array.from(obj.x0)
      const { value, grad } = obj.valueAndGrad(x)
      return {
        x,
        value,
        grad,
        nfev: 1,
        ngev: 1,
        nhev: 0,
        step: 0,
        best: x,
        bestValue: value,
        average: x,
        weight: 0,
      }
    },
    step: (s, ctx) => {
      const alpha = stepOf(rule, a, ctx.t, s.value, infNorm(s.grad), options.fStar)
      const x = mirror.toPrimal(daxpy(-alpha, s.grad, mirror.toDual(s.x)))
      const { value, grad } = obj.valueAndGrad(x)
      const better = value < s.bestValue
      const weight = s.weight + alpha
      const average = Float64Array.from(s.average, (v, i) => v + (alpha / weight) * ((x[i] ?? 0) - v))
      const change = dnrm2(daxpy(-1, s.x, x)) / Math.max(1, dnrm2(s.x))
      const stop =
        change <= tol ? ({ status: 'converged', message: 'relative change below tolerance' } as const) : undefined
      return {
        x,
        value,
        grad,
        nfev: s.nfev + 1,
        ngev: s.ngev + 1,
        nhev: 0,
        step: alpha,
        best: better ? x : s.best,
        bestValue: better ? value : s.bestValue,
        average,
        weight,
        ...(stop === undefined ? {} : { stop }),
      }
    },
    done: vectorDone,
    observe: (s) => observeVector(s, { metrics: { best: s.bestValue } }),
  }
}

/**
 * Minimise a convex function over a set with a mirror map (default: the simplex with the entropic map).
 *
 * @param f - The objective.
 * @param x0 - A feasible starting point.
 * @param options - Mirror map, step rule, tolerance, output, iteration cap (default 1000) and run options.
 * @returns The result (the best iterate by default).
 * @concept mirror-descent
 * @example
 * ```ts
 * import { mirrorDescent } from 'mdml-engine/optim'
 * import { mul, square, sub, sum, toArray, type Value } from 'mdml-engine/tensor'
 * // The closest point of the simplex to (0.5, 0.1, 0.4) is itself.
 * const r = mirrorDescent((x) => mul(sum(square(sub(x as Value, [0.5, 0.1, 0.4]))), 0.5), [1 / 3, 1 / 3, 1 / 3], { stepRule: 'constant', stepSize: 1, tol: 1e-14, maxIterations: 5000 })
 * toArray(r.x as Value).map((v) => Math.round(v * 1e6) / 1e6) // => [0.5, 0.1, 0.4]
 * ```
 */
export function mirrorDescent(f: Objective, x0: Parameters, options: MirrorDescentOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  let last: MirrorDescentState | undefined
  const alg = mirrorDescentAlgorithm(f, x0, options)
  const r = runVector(
    {
      ...alg,
      init: (k) => {
        last = alg.init(k)
        return last
      },
      step: (s, ctx) => {
        last = alg.step(s, ctx)
        return last
      },
    },
    obj,
    options,
    1000,
  )
  if (last === undefined) return r
  const output = options.output ?? 'best'
  if (output === 'last') return r
  if (output === 'best') return bestResult(r, last.best, last.bestValue, obj.unravel)
  return { ...r, x: obj.unravel(last.average), fun: obj.value(last.average) }
}

/** Options of {@link dualAscent}. */
export interface DualAscentOptions extends VectorMethodOptions {
  /** The dual step (default 1; with `augmented` the natural choice is rho). */
  readonly stepSize?: number
  /** The method of multipliers: add (rho / 2) |A x - b|^2 to the Lagrangian. */
  readonly augmented?: number
  /** The starting multipliers (default 0). */
  readonly y0?: ArrayLike<number>
  /** Stop once |A x - b| falls to this (default 1e-9). */
  readonly tol?: number
  /** argmin_x of a Lagrangian from a warm start (default: {@link minimize} with its default method). */
  readonly solve?: (lagrangian: Objective, start: Vec) => ArrayLike<number>
  /** The inner minimiser in closed form, given the multipliers (takes precedence over `solve`). */
  readonly argmin?: (y: Vec, start: Vec) => ArrayLike<number>
}

/** State of {@link dualAscentAlgorithm}. */
export interface DualAscentState extends VectorState {
  readonly y: Vec
  readonly violation: number
  readonly dualValue: number
}

/** Result of {@link dualAscent} and {@link dualDecomposition}. */
export interface DualResult extends MinimizeResult {
  /** The Lagrange multipliers of A x = b. */
  readonly multipliers: Tensor
  /** |A x - b| at the solution. */
  readonly violation: number
}

const defaultSolve = (lagrangian: Objective, start: Vec): Vec =>
  flat(minimize(lagrangian, fromStorage(Float64Array.from(start), [start.length], 'float64')).x as Value)

function matrixOf(a: TensorLike): { data: Vec; rows: number; cols: number } {
  const t = primalOf(asTensor(a)) as Tensor
  if (t.shape.length !== 2) throw new MdmlError('shape', 'dual ascent: A must be a matrix')
  return { data: Float64Array.from(t.data), rows: t.shape[0] ?? 0, cols: t.shape[1] ?? 0 }
}

/**
 * Dual ascent (and with `augmented`, the method of multipliers) for min f(x) subject to A x = b, as a traced
 * algorithm: x+ = argmin_x f(x) + y . (A x - b) [+ (rho / 2) |A x - b|^2], then y+ = y + a (A x+ - b). Records the
 * dual function value and the constraint violation.
 *
 * @param f - The objective of a vector x.
 * @param a - The constraint matrix `[m, n]`.
 * @param b - The right-hand side `[m]`.
 * @param x0 - The starting point `[n]`.
 * @param options - Dual step, augmentation, starting multipliers, tolerance and the inner solver.
 * @returns The algorithm.
 * @concept duality
 * @example
 * ```ts
 * import { dualAscentAlgorithm } from 'mdml-engine/optim'
 * import { square, sum, type Value } from 'mdml-engine/tensor'
 * dualAscentAlgorithm((x) => sum(square(x as Value)), [[1, 1]], [1], [0, 0]).name // => 'dualAscent'
 * ```
 */
export function dualAscentAlgorithm(
  f: Objective,
  a: TensorLike,
  b: TensorLike,
  x0: TensorLike,
  options: DualAscentOptions = {},
): Algorithm<DualAscentState> {
  const A = matrixOf(a)
  const bv = flat(asTensor(b))
  const obj = flatObjective(f, x0)
  if (A.cols !== obj.n || A.rows !== bv.length) throw new MdmlError('shape', 'dual ascent: A, b and x0 disagree')
  const rho = options.augmented ?? 0
  const alpha = options.stepSize ?? (rho > 0 ? rho : 1)
  const tol = options.tol ?? 1e-9
  const solve = options.solve ?? defaultSolve
  const Am = fromStorage(A.data, [A.rows, A.cols], 'float64')
  const lagrangian =
    (y: Vec): Objective =>
    (x) => {
      const xv = reshape(x as Value, [-1])
      const r = add(
        reshape(matmul(Am, reshape(xv, [-1, 1])), [-1]),
        fromStorage(
          Float64Array.from(bv, (v) => -v),
          [bv.length],
          'float64',
        ),
      )
      const linear = sum(mul(r, fromStorage(y, [y.length], 'float64')))
      const base = add(obj.ofValue(xv), linear)
      return rho > 0 ? add(base, mul(sum(mul(r, r)), rho / 2)) : base
    }
  const residual = (x: Vec): Vec => daxpy(-1, bv, dgemv(A.data, A.rows, A.cols, x))
  return {
    name: rho > 0 ? 'methodOfMultipliers' : 'dualAscent',
    config: configOf({ stepSize: alpha, augmented: rho, tol }),
    schema: vectorSchema({
      metrics: {
        violation: { description: '|A x - b|' },
        dual: { description: 'Lagrangian at the inner minimiser (the dual function when not augmented)' },
      },
      tensors: { y: { role: 'state', description: 'Lagrange multipliers' } },
    }),
    init: () => {
      const x = Float64Array.from(obj.x0)
      const y = options.y0 === undefined ? new Float64Array(A.rows) : Float64Array.from(options.y0)
      const r = residual(x)
      return {
        x,
        value: obj.value(x),
        grad: new Float64Array(0),
        nfev: 1,
        ngev: 0,
        nhev: 0,
        step: alpha,
        y,
        violation: dnrm2(r),
        dualValue: Number.NaN,
      }
    },
    step: (s) => {
      const x = Float64Array.from(options.argmin === undefined ? solve(lagrangian(s.y), s.x) : options.argmin(s.y, s.x))
      const r = residual(x)
      const value = obj.value(x)
      const dualValue = value + ddot(s.y, r) + (rho / 2) * ddot(r, r)
      const y = daxpy(alpha, r, s.y)
      const violation = dnrm2(r)
      const stop =
        violation <= tol
          ? ({ status: 'converged', message: 'constraint violation below tolerance' } as const)
          : undefined
      return {
        x,
        value,
        grad: new Float64Array(0),
        nfev: s.nfev + 1,
        ngev: 0,
        nhev: 0,
        step: alpha,
        y,
        violation,
        dualValue,
        ...(stop === undefined ? {} : { stop }),
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { violation: s.violation, dual: s.dualValue },
        tensors: { y: { role: 'state', value: s.y } },
      }),
  }
}

/**
 * Solve min f(x) subject to A x = b by dual ascent or the method of multipliers.
 *
 * @param f - The objective of a vector x.
 * @param a - The constraint matrix.
 * @param b - The right-hand side.
 * @param x0 - The starting point.
 * @param options - Dual step, augmentation, multipliers, tolerance, inner solver, iteration cap (default 500).
 * @returns The solution with its multipliers and constraint violation.
 * @concept duality
 * @example
 * ```ts
 * import { dualAscent } from 'mdml-engine/optim'
 * import { mul, square, sum, toArray, type Value } from 'mdml-engine/tensor'
 * // min |x|^2 / 2 subject to x1 + x2 = 1, with the closed-form inner minimiser x = -A^T y.
 * const r = dualAscent((x) => mul(sum(square(x as Value)), 0.5), [[1, 1]], [1], [0, 0], {
 *   stepSize: 0.5,
 *   argmin: (y) => [-(y[0] ?? 0), -(y[0] ?? 0)],
 * })
 * toArray(r.x as Value) // => [0.5, 0.5]
 * toArray(r.multipliers) // => [-0.5]
 * ```
 */
export function dualAscent(
  f: Objective,
  a: TensorLike,
  b: TensorLike,
  x0: TensorLike,
  options: DualAscentOptions = {},
): DualResult {
  const obj = flatObjective(f, x0)
  let last: DualAscentState | undefined
  const alg = dualAscentAlgorithm(f, a, b, x0, options)
  const r = runVector(
    {
      ...alg,
      init: (k) => {
        last = alg.init(k)
        return last
      },
      step: (s, ctx) => {
        last = alg.step(s, ctx)
        return last
      },
    },
    obj,
    options,
    500,
  )
  const y = last?.y ?? new Float64Array(0)
  return {
    ...r,
    multipliers: fromStorage(Float64Array.from(y), [y.length], 'float64'),
    violation: last?.violation ?? Infinity,
  }
}

/** One block of a separable problem for {@link dualDecomposition}: f_i(x_i) and its columns A_i of the coupling. */
export interface DecompositionBlock {
  readonly f: Objective
  readonly a: TensorLike
  readonly x0: TensorLike
  /** argmin_x f_i(x) + c . x in closed form (c = A_i^T y); otherwise the block is minimised numerically. */
  readonly argmin?: (c: Vec, start: Vec) => ArrayLike<number>
}

/**
 * Dual decomposition for min sum_i f_i(x_i) subject to sum_i A_i x_i = b: the Lagrangian separates, so each block is
 * minimised independently (a distributed step), and the multipliers follow the coupling residual. Runs as dual ascent
 * on the stacked variable with a block-separable inner solve.
 *
 * @param blocks - The blocks.
 * @param b - The coupling right-hand side.
 * @param options - Dual step, augmentation is not allowed (it couples the blocks), tolerance, the per-block solver and
 *   the iteration cap.
 * @returns The stacked solution with multipliers.
 * @concept duality
 * @example
 * ```ts
 * import { dualDecomposition } from 'mdml-engine/optim'
 * import { mul, square, sum, toArray, type Value } from 'mdml-engine/tensor'
 * // min (x1^2 + x2^2) / 2 subject to x1 + x2 = 2; each block's minimiser of x^2 / 2 + c x is -c.
 * const half = (x: unknown) => mul(sum(square(x as Value)), 0.5)
 * const block = { f: half, a: [[1]], x0: [0], argmin: (c: Float64Array) => [-(c[0] ?? 0)] }
 * const r = dualDecomposition([block, block], [2], { stepSize: 0.5 })
 * toArray(r.x as Value) // => [1, 1]
 * ```
 */
export function dualDecomposition(
  blocks: readonly DecompositionBlock[],
  b: TensorLike,
  options: Omit<DualAscentOptions, 'augmented'> = {},
): DualResult {
  if (blocks.length === 0) throw new MdmlError('value', 'dualDecomposition: give at least one block')
  const mats = blocks.map((bl) => matrixOf(bl.a))
  const starts = blocks.map((bl) => flat(asTensor(bl.x0)))
  const sizes = starts.map((s) => s.length)
  const offsets = sizes.map((_, i) => sizes.slice(0, i).reduce((p, q) => p + q, 0))
  const n = sizes.reduce((p, q) => p + q, 0)
  const m = mats[0]?.rows ?? 0
  const A = new Float64Array(m * n)
  for (const [k, mat] of mats.entries()) {
    if (mat.rows !== m || mat.cols !== sizes[k]) throw new MdmlError('shape', 'dualDecomposition: block shapes')
    for (let i = 0; i < m; i++)
      for (let j = 0; j < mat.cols; j++) A[i * n + (offsets[k] ?? 0) + j] = mat.data[i * mat.cols + j] ?? 0
  }
  const objs = blocks.map((bl) => flatObjective(bl.f, bl.x0))
  const total: Objective = (x) => {
    const v = reshape(x as Value, [-1])
    const parts = objs.map((o, k) => o.ofValue(sliceAxis(v, 0, offsets[k] ?? 0, (offsets[k] ?? 0) + (sizes[k] ?? 0))))
    return parts.reduce((p, q) => add(p, q))
  }
  const blockSolve = options.solve ?? defaultSolve
  const x0 = new Float64Array(n)
  for (const [k, s] of starts.entries()) x0.set(s, offsets[k] ?? 0)
  return dualAscent(total, fromStorage(A, [m, n], 'float64'), b, x0, {
    ...options,
    argmin: (y, start) => {
      const out = new Float64Array(n)
      for (const [k, o] of objs.entries()) {
        const mat = mats[k] as { data: Vec; rows: number; cols: number }
        const c = dgemv(mat.data, mat.rows, mat.cols, y, true)
        const piece = start.slice(offsets[k] ?? 0, (offsets[k] ?? 0) + (sizes[k] ?? 0))
        const closed = blocks[k]?.argmin
        const cv = fromStorage(c, [c.length], 'float64')
        const local: Objective = (x) =>
          add(o.ofValue(reshape(x as Value, [-1])), sum(mul(reshape(x as Value, [-1]), cv)))
        out.set(Array.from(closed === undefined ? blockSolve(local, piece) : closed(c, piece)), offsets[k] ?? 0)
      }
      return out
    },
  })
}
