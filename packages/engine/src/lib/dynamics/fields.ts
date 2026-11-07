/**
 * Vector-field analysis for phase portraits: Jacobians, divergence and curl by automatic differentiation; equilibria
 * found by Newton's method from many seeds and classified from their eigenvalues (saddle, node, spiral, centre);
 * nullclines by marching squares; streamlines and differentiable flow maps; and the stable and unstable manifolds of
 * saddles.
 *
 * @index function Vector-field analysis: equilibria, Jacobians, divergence for phase portraits.
 */

import { jacfwd } from '../autodiff/index.ts'
import { MdmlError } from '../base/index.ts'
import { contourLines } from '../numerics/geometry/index.ts'
import { eig } from '../numerics/linalg/index.ts'
import { fsolve } from '../numerics/roots/index.ts'
import { type Key, key as keyOf, uniform } from '../random/index.ts'
import { primalOf, reshape, sliceAxis, stack, type Tensor, type TensorLike, type Value } from '../tensor/index.ts'
import { floatState, fromNumbers, numbersOf, rms } from './ode/_vec.ts'
import type { OdeVectorField } from './ode/problem.ts'
import { type OdeMethodName, solveIvp } from './ode/solve-ivp.ts'

/**
 * The Jacobian df/dy of a vector field at a point (forward-mode autodiff, `[n, n]` over the flattened state).
 *
 * @param f - The field.
 * @param y - The point.
 * @param t - The time (default 0).
 * @returns The Jacobian.
 * @concept jacobian
 * @example
 * ```ts
 * import { fieldJacobian } from 'mdml-engine/dynamics'
 * import { mul, toArray } from 'mdml-engine/tensor'
 * toArray(fieldJacobian((_t, y) => mul(y, y), [3])) // => [[6]]
 * ```
 */
export function fieldJacobian(f: OdeVectorField, y: TensorLike, t = 0): Tensor {
  const v = floatState(y)
  const n = numbersOf(v).length
  const g = (x: Value): Value => reshape(f(t, reshape(x, v.shape)), [n])
  return primalOf(reshape(jacfwd(g)(reshape(v, [n])), [n, n]))
}

/**
 * The divergence of a vector field, the trace of its Jacobian (the local rate of phase-space volume change).
 *
 * @param f - The field.
 * @param y - The point.
 * @param t - The time (default 0).
 * @returns div f at y.
 * @example
 * ```ts
 * import { fieldDivergence } from 'mdml-engine/dynamics'
 * import { mul } from 'mdml-engine/tensor'
 * fieldDivergence((_t, y) => mul(y, -2), [1, 1, 1]) // => -6
 * ```
 */
export function fieldDivergence(f: OdeVectorField, y: TensorLike, t = 0): number {
  const J = fieldJacobian(f, y, t)
  const n = J.shape[0] ?? 0
  let s = 0
  for (let i = 0; i < n; i++) s += J.data[i * n + i] ?? 0
  return s
}

/**
 * The curl of a 2-D field (the scalar df2/dx - df1/dy) or a 3-D field (a vector).
 *
 * @param f - The field.
 * @param y - The point.
 * @param t - The time (default 0).
 * @returns The curl.
 * @throws MdmlError with code `shape` for other dimensions.
 * @example
 * ```ts
 * import { fieldCurl } from 'mdml-engine/dynamics'
 * import { stack, sliceAxis, neg, sum } from 'mdml-engine/tensor'
 * fieldCurl((_t, y) => stack([neg(sum(sliceAxis(y, 0, 1, 2))), sum(sliceAxis(y, 0, 0, 1))]), [0.3, 0.2]) // => [2]
 * ```
 */
export function fieldCurl(f: OdeVectorField, y: TensorLike, t = 0): number[] {
  const J = fieldJacobian(f, y, t)
  const n = J.shape[0] ?? 0
  const j = (r: number, c: number): number => J.data[r * n + c] ?? 0
  if (n === 2) return [j(1, 0) - j(0, 1) + 0]
  if (n === 3) return [j(2, 1) - j(1, 2) + 0, j(0, 2) - j(2, 0) + 0, j(1, 0) - j(0, 1) + 0]
  throw new MdmlError('shape', 'fieldCurl: the field must be 2-D or 3-D')
}

/** The kind of an equilibrium from its linearisation. */
export type FixedPointType =
  | 'stable-node'
  | 'unstable-node'
  | 'stable-spiral'
  | 'unstable-spiral'
  | 'saddle'
  | 'center'
  | 'degenerate'

/** A classified equilibrium. */
export interface FixedPointReport {
  readonly point: Tensor
  readonly type: FixedPointType
  /** Asymptotically stable (every eigenvalue has negative real part). */
  readonly stable: boolean
  readonly eigenvalues: { readonly re: readonly number[]; readonly im: readonly number[] }
  /** The residual |f| at the point. */
  readonly residual: number
}

/**
 * Classify an equilibrium from the eigenvalues of its Jacobian: saddle (real parts of both signs), node or spiral
 * (real parts of one sign, complex eigenvalues making a spiral), centre (purely imaginary), degenerate (a zero
 * eigenvalue). In 2-D this is the trace-determinant classification.
 *
 * @param jacobian - The Jacobian `[n, n]`.
 * @param tol - The tolerance for treating a real part as zero (default 1e-9).
 * @returns The type, stability and eigenvalues.
 * @concept fixed-point-classification
 * @example
 * ```ts
 * import { classifyFixedPoint } from 'mdml-engine/dynamics'
 * [classifyFixedPoint([[0, 1], [-1, 0]]).type, classifyFixedPoint([[1, 0], [0, -1]]).type] // => ['center', 'saddle']
 * ```
 */
export function classifyFixedPoint(jacobian: TensorLike, tol = 1e-9): Omit<FixedPointReport, 'point' | 'residual'> {
  const { values } = eig(jacobian)
  const re = Array.from(values.re.data)
  const im = Array.from(values.im.data)
  const scale = Math.max(
    1,
    re.reduce((m, v) => Math.max(m, Math.abs(v)), 0),
  )
  const zero = (v: number): boolean => Math.abs(v) <= tol * scale
  const complex = im.some((v) => !zero(v))
  const pos = re.some((v) => !zero(v) && v > 0)
  const neg = re.some((v) => !zero(v) && v < 0)
  const anyZeroReal = re.some((v) => zero(v))
  let type: FixedPointType
  if (pos && neg) type = 'saddle'
  else if (anyZeroReal) {
    const allImag = re.every((v, i) => zero(v) && !zero(im[i] ?? 0))
    type = allImag ? 'center' : 'degenerate'
  } else if (neg) type = complex ? 'stable-spiral' : 'stable-node'
  else type = complex ? 'unstable-spiral' : 'unstable-node'
  return { type, stable: !pos && !anyZeroReal && neg, eigenvalues: { re, im } }
}

/** Options of {@link fixedPoints}. */
export interface FixedPointOptions {
  /** Starting points for Newton's method. */
  readonly seeds?: readonly (readonly number[])[]
  /** Or random seeds in a box: one [lo, hi] per coordinate, with a count and a key. */
  readonly box?: readonly (readonly [number, number])[]
  readonly count?: number
  readonly key?: Key
  /** Residual tolerance (default 1e-10). */
  readonly tol?: number
  /** Two points closer than this are the same (default 1e-6). */
  readonly distinct?: number
  readonly t?: number
}

/**
 * Find and classify the equilibria f(y) = 0 of a vector field by Newton's method from many seeds (given, or uniform
 * random in a box from a key), keeping the distinct converged points.
 *
 * @param f - The field.
 * @param options - Seeds or a box with a count and key, tolerances and time.
 * @returns The distinct equilibria with their classification.
 * @throws MdmlError with code `value` when neither seeds nor a box is given.
 * @concept fixed-point-classification
 * @example
 * ```ts
 * import { fixedPoints } from 'mdml-engine/dynamics'
 * import { mul, stack, sub, sum, sliceAxis } from 'mdml-engine/tensor'
 * // x' = x - x^3, y' = -y: equilibria at x = -1, 0, 1.
 * const f = (_t, y) => { const x = sum(sliceAxis(y, 0, 0, 1)); return stack([sub(x, mul(mul(x, x), x)), mul(sum(sliceAxis(y, 0, 1, 2)), -1)]) }
 * fixedPoints(f, { seeds: [[-2, 1], [0.1, 0], [3, -1]] }).map((p) => p.type) // => ['stable-node', 'saddle', 'stable-node']
 * ```
 */
export function fixedPoints(f: OdeVectorField, options: FixedPointOptions): FixedPointReport[] {
  const t = options.t ?? 0
  let seeds: number[][]
  if (options.seeds !== undefined) seeds = options.seeds.map((s) => [...s])
  else if (options.box !== undefined) {
    const box = options.box
    const count = options.count ?? 20
    const u = uniform(options.key ?? keyOf(0), [count, box.length])
    seeds = Array.from({ length: count }, (_, i) =>
      box.map(([lo, hi], j) => lo + (hi - lo) * (u.data[i * box.length + j] ?? 0)),
    )
  } else throw new MdmlError('value', 'fixedPoints: give seeds or a box')
  const tol = options.tol ?? 1e-10
  const distinct = options.distinct ?? 1e-6
  const found: FixedPointReport[] = []
  for (const s of seeds) {
    const r = fsolve((v) => f(t, v), s, { ftol: tol, maxIterations: 100 })
    const x = Array.from(r.x.data)
    const res = rms(numbersOf(f(t, fromNumbers(x, [x.length]))))
    if (!(res <= Math.max(tol, 1e-8)) || !x.every(Number.isFinite)) continue
    if (
      found.some(
        (p) => rms(Float64Array.from(p.point.data, (v, i) => v - (x[i] ?? 0))) * Math.sqrt(x.length) < distinct,
      )
    )
      continue
    const point = fromNumbers(x, [x.length])
    found.push({ point, ...classifyFixedPoint(fieldJacobian(f, point, t)), residual: res })
  }
  return found
}

/**
 * The nullclines of a 2-D field: the curves f1 = 0 and f2 = 0 traced by marching squares on a grid.
 *
 * @param f - The field.
 * @param xRange - [x0, x1].
 * @param yRange - [y0, y1].
 * @param n - Grid points per side (default 101).
 * @returns Polylines of (x, y) points for each component.
 * @concept nullcline
 * @example
 * ```ts
 * import { nullclines } from 'mdml-engine/dynamics'
 * import { stack, sub, sum, sliceAxis } from 'mdml-engine/tensor'
 * // x' = y - x, y' = -y: the x-nullcline is the diagonal.
 * const lines = nullclines((_t, v) => stack([sub(sum(sliceAxis(v, 0, 1, 2)), sum(sliceAxis(v, 0, 0, 1))), sum(sliceAxis(v, 0, 1, 2))]), [-1, 1], [-1, 1], 11)
 * lines[0].length > 0 // => true
 * ```
 */
export function nullclines(
  f: OdeVectorField,
  xRange: readonly [number, number],
  yRange: readonly [number, number],
  n = 101,
): [number, number][][][] {
  const xs = Array.from({ length: n }, (_, i) => xRange[0] + ((xRange[1] - xRange[0]) * i) / (n - 1))
  const ys = Array.from({ length: n }, (_, i) => yRange[0] + ((yRange[1] - yRange[0]) * i) / (n - 1))
  const pts = new Float64Array(n * n * 2)
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      pts[2 * (r * n + c)] = xs[c] ?? 0
      pts[2 * (r * n + c) + 1] = ys[r] ?? 0
    }
  }
  // Evaluate point by point (fields are written for one state).
  const values = [new Float64Array(n * n), new Float64Array(n * n)]
  for (let k = 0; k < n * n; k++) {
    const v = numbersOf(f(0, fromNumbers([pts[2 * k] ?? 0, pts[2 * k + 1] ?? 0], [2])))
    ;(values[0] as Float64Array)[k] = v[0] ?? 0
    ;(values[1] as Float64Array)[k] = v[1] ?? 0
  }
  const toXY = (p: [number, number]): [number, number] => {
    const [r, c] = p
    return [xRange[0] + ((xRange[1] - xRange[0]) * c) / (n - 1), yRange[0] + ((yRange[1] - yRange[0]) * r) / (n - 1)]
  }
  return values.map((v) => contourLines(fromNumbers(v, [n, n]), 0).map((line) => line.map(toXY)))
}

/** Options of {@link streamline}. */
export interface StreamlineOptions {
  /** How far to follow the flow (default 10). */
  readonly tEnd?: number
  /** Also follow it backwards (default false). */
  readonly both?: boolean
  readonly method?: OdeMethodName
  readonly rtol?: number
  readonly atol?: number
  /** Points along each direction (default 100). */
  readonly points?: number
}

/**
 * A streamline (trajectory) of a vector field through a point, sampled at equal times, optionally in both directions.
 *
 * @param f - The field.
 * @param y0 - The starting point.
 * @param options - Duration, direction, method, tolerances and sampling.
 * @returns The points `[m, n]` in time order and their times.
 * @concept phase-portrait
 * @example
 * ```ts
 * import { streamline } from 'mdml-engine/dynamics'
 * import { neg } from 'mdml-engine/tensor'
 * streamline((_t, y) => neg(y), [1, 1], { tEnd: 1, points: 5 }).points.shape // => [5, 2]
 * ```
 */
export function streamline(
  f: OdeVectorField,
  y0: TensorLike,
  options: StreamlineOptions = {},
): { readonly t: readonly number[]; readonly points: Value } {
  const tEnd = options.tEnd ?? 10
  const m = options.points ?? 100
  const run = (end: number): { t: number[]; ys: Value[] } => {
    const times = Array.from({ length: m }, (_, i) => (end * i) / (m - 1))
    const sol = solveIvp(f, [0, end], y0, {
      method: options.method ?? 'RK45',
      rtol: options.rtol ?? 1e-6,
      atol: options.atol ?? 1e-9,
      tEval: times,
    })
    return { t: [...sol.t], ys: [...sol.ys] }
  }
  const fwd = run(tEnd)
  if (options.both !== true) return { t: fwd.t, points: stack(fwd.ys) }
  const back = run(-tEnd)
  return { t: [...back.t.slice(1).reverse(), ...fwd.t], points: stack([...back.ys.slice(1).reverse(), ...fwd.ys]) }
}

/**
 * The flow map phi_t: the states reached at time t from many initial points (differentiable in them through the
 * solver, so its Jacobian is the derivative of the flow).
 *
 * @param f - The field.
 * @param points - Initial points `[m, n]`.
 * @param t - The time.
 * @param options - Method and tolerances.
 * @returns phi_t of each point `[m, n]`.
 * @concept flow-map
 * @example
 * ```ts
 * import { flowMap } from 'mdml-engine/dynamics'
 * import { neg, toArray } from 'mdml-engine/tensor'
 * toArray(flowMap((_t, y) => neg(y), [[1], [2]], Math.log(2), { rtol: 1e-12, atol: 1e-14 })).map((r) => Math.round(r[0] * 1e9) / 1e9) // => [0.5, 1]
 * ```
 */
export function flowMap(
  f: OdeVectorField,
  points: TensorLike,
  t: number,
  options: { readonly method?: OdeMethodName; readonly rtol?: number; readonly atol?: number } = {},
): Value {
  const p = floatState(points)
  const m = p.shape[0] ?? 0
  const rows = Array.from({ length: m }, (_, i) => {
    const row = reshape(sliceRow(p, i), p.shape.slice(1))
    const sol = solveIvp(f, [0, t], row, {
      method: options.method ?? 'DOP853',
      rtol: options.rtol ?? 1e-9,
      atol: options.atol ?? 1e-12,
      tEval: [t],
    })
    return sol.ys[sol.ys.length - 1] as Value
  })
  return stack(rows)
}

function sliceRow(p: Value, i: number): Value {
  return sliceAxis(p, 0, i, i + 1)
}

/** One branch of a one-dimensional invariant manifold. */
export interface ManifoldBranch {
  readonly side: 1 | -1
  readonly points: Value
}

/**
 * The one-dimensional stable or unstable manifold of a saddle: start a small distance along the eigenvector of the
 * negative (stable) or positive (unstable) eigenvalue on both sides and integrate backwards (stable) or forwards
 * (unstable) in time.
 *
 * @param f - The field.
 * @param saddle - The equilibrium.
 * @param kind - `stable` or `unstable`.
 * @param options - Offset along the eigenvector (default 1e-6), duration (default 10) and sampling.
 * @returns The two branches.
 * @throws MdmlError with code `value` when the point has no real eigenvalue of the needed sign.
 * @concept invariant-manifold
 * @example
 * ```ts
 * import { invariantManifold } from 'mdml-engine/dynamics'
 * import { stack, sliceAxis, sum, neg } from 'mdml-engine/tensor'
 * // x' = x, y' = -y: the unstable manifold is the x-axis.
 * const f = (_t, y) => stack([sum(sliceAxis(y, 0, 0, 1)), neg(sum(sliceAxis(y, 0, 1, 2)))])
 * invariantManifold(f, [0, 0], 'unstable', { tEnd: 5 }).length // => 2
 * ```
 */
export function invariantManifold(
  f: OdeVectorField,
  saddle: TensorLike,
  kind: 'stable' | 'unstable',
  options: { readonly offset?: number; readonly tEnd?: number; readonly points?: number } = {},
): ManifoldBranch[] {
  const x = floatState(saddle)
  const J = fieldJacobian(f, x)
  const { values, vectors } = eig(J)
  const n = J.shape[0] ?? 0
  const re = Array.from(values.re.data)
  const im = Array.from(values.im.data)
  let pick = -1
  for (let i = 0; i < n; i++) {
    const real = Math.abs(im[i] ?? 0) < 1e-12
    if (real && (kind === 'stable' ? (re[i] ?? 0) < 0 : (re[i] ?? 0) > 0)) {
      if (pick < 0 || Math.abs(re[i] ?? 0) > Math.abs(re[pick] ?? 0)) pick = i
    }
  }
  if (pick < 0) throw new MdmlError('value', `invariantManifold: no real ${kind} eigenvalue at this point`)
  const v = Array.from({ length: n }, (_, r) => vectors.re.data[r * n + pick] ?? 0)
  const eps = options.offset ?? 1e-6
  const tEnd = (kind === 'stable' ? -1 : 1) * (options.tEnd ?? 10)
  const xs = numbersOf(x)
  return ([1, -1] as const).map((side) => {
    const start = fromNumbers(
      Float64Array.from(xs, (val, i) => val + side * eps * (v[i] ?? 0)),
      x.shape,
    )
    const { points } = streamline(f, start, { tEnd, points: options.points ?? 100 })
    return { side, points }
  })
}
