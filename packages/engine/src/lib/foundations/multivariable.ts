/**
 * Multivariable and vector calculus: gradients, directional derivatives, Jacobians and Hessians of fields by automatic
 * differentiation, linear and quadratic approximations and tangent planes, critical-point classification, Lagrange
 * multipliers, divergence and curl, line, flux and surface integrals, and numerical checks of Green's, Stokes' and the
 * divergence theorems.
 */

import { jvp } from '../autodiff/index.ts'
import { WhiteboxError } from '../base/index.ts'
import { eigvalsh, lstsq } from '../numerics/linalg/index.ts'
import { cubature, productGaussRule, quad } from '../numerics/quadrature/index.ts'
import { fsolve } from '../numerics/roots/index.ts'
import {
  abs,
  add,
  concat,
  item,
  matmul,
  mul,
  onesLike,
  promoteFloat,
  sliceAxis,
  sqrt,
  squeeze,
  stack,
  sub,
  sum,
  type TensorLike,
  toArray,
  toFlat,
  transpose,
  type Value,
  zerosLike,
} from '../tensor/index.ts'
import type { ParametricCurve } from './curves.ts'
import type { ParametricSurface } from './surfaces.ts'
import { crossProduct } from './vectors.ts'

/** A scalar field: points `[n, d]` to values `[n]`, written with tensor operations (vectorised over the points). */
export type ScalarField = (x: Value) => Value

/** A vector field: points `[n, d]` to vectors `[n, m]` (usually m = d), vectorised over the points. */
export type VectorField = (x: Value) => Value

function points(x: TensorLike): Value {
  const [v] = promoteFloat(x) as [Value]
  return v.shape.length === 1 ? stack([v]) : v
}

/** The unit direction e_j at every point. */
function basis(x: Value, j: number): Value {
  const d = x.shape[x.shape.length - 1] ?? 0
  const one = squeeze(onesLike(sliceAxis(x, -1, 0, 1)), -1)
  const zero = zerosLike(one)
  return stack(
    Array.from({ length: d }, (_, k) => (k === j ? one : zero)),
    -1,
  )
}

const dims = (x: Value): number => x.shape[x.shape.length - 1] ?? 0

/** The derivative of a field along e_j at every point (forward mode). */
const along = (f: (x: Value) => Value, x: Value, j: number): Value => jvp(f, x, basis(x, j)).tangent

/**
 * The gradient of a scalar field at points, `[df/dx_1, ..., df/dx_d]`, by forward-mode differentiation (one pass per
 * coordinate). It points uphill, with length the steepest rate of increase. Differentiable.
 *
 * @param f - The field.
 * @param x - Points `[n, d]` (or one point `[d]`).
 * @returns Gradients `[n, d]`.
 * @concept gradient
 * @example
 * ```ts
 * import { gradientAt } from 'whitebox-engine/foundations'
 * import { mul, sum, toArray, type Value } from 'whitebox-engine/tensor'
 * toArray(gradientAt((p: Value) => sum(mul(p, p), -1), [1, 2])) // => [[2, 4]]
 * ```
 */
export function gradientAt(f: ScalarField, x: TensorLike): Value {
  const p = points(x)
  return stack(
    Array.from({ length: dims(p) }, (_, j) => along(f, p, j)),
    -1,
  )
}

/**
 * The directional derivative `grad f . u / |u|` of a scalar field: its rate of change per unit distance along u.
 *
 * @param f - The field.
 * @param x - Points `[n, d]`.
 * @param direction - A direction `[d]` (normalised).
 * @returns Rates `[n]`.
 * @concept directional-derivative
 * @example
 * ```ts
 * import { directionalDerivative } from 'whitebox-engine/foundations'
 * import { mul, sum, toArray, type Value } from 'whitebox-engine/tensor'
 * toArray(directionalDerivative((p: Value) => sum(mul(p, p), -1), [1, 2], [3, 4])) // => [4.4]
 * ```
 */
export function directionalDerivative(f: ScalarField, x: TensorLike, direction: TensorLike): Value {
  const p = points(x)
  const [u] = promoteFloat(direction) as [Value]
  const unit = mul(u, 1 / Math.sqrt(toFlat(u).reduce((s, v) => s + v * v, 0)))
  return jvp(f, p, add(mul(p, 0), unit)).tangent
}

/**
 * The Jacobian matrices of a vector field at points: `J[n, i, j] = dF_i / dx_j`, by forward mode.
 *
 * @param F - The field.
 * @param x - Points `[n, d]`.
 * @returns Jacobians `[n, m, d]`.
 * @concept jacobian
 * @example
 * ```ts
 * import { jacobianAt } from 'whitebox-engine/foundations'
 * import { mul, toArray, type Value } from 'whitebox-engine/tensor'
 * toArray(jacobianAt((p: Value) => mul(p, p), [1, 3])) // => [[[2, 0], [0, 6]]]
 * ```
 */
export function jacobianAt(F: VectorField, x: TensorLike): Value {
  const p = points(x)
  return stack(
    Array.from({ length: dims(p) }, (_, j) => along(F, p, j)),
    -1,
  )
}

/**
 * The Hessian matrices of a scalar field at points: `H[n, i, j] = d^2 f / dx_i dx_j`, by nested forward mode.
 *
 * @param f - The field.
 * @param x - Points `[n, d]`.
 * @returns Hessians `[n, d, d]`.
 * @concept hessian
 * @example
 * ```ts
 * import { hessianAt } from 'whitebox-engine/foundations'
 * import { mul, prod, toArray, type Value } from 'whitebox-engine/tensor'
 * toArray(hessianAt((p: Value) => prod(p, -1), [2, 3])) // => [[[0, 1], [1, 0]]]
 * ```
 */
export function hessianAt(f: ScalarField, x: TensorLike): Value {
  const p = points(x)
  return jacobianAt((q) => gradientAt(f, q), p)
}

/** The local models of a scalar field at a point. */
export interface QuadraticModel {
  readonly point: number[]
  readonly value: number
  readonly gradient: number[]
  readonly hessian: number[][]
  /** The tangent (linear) approximation `f(a) + grad . (x - a)` at points `[n, d]`. */
  linear(x: TensorLike): Value
  /** The quadratic approximation `f(a) + grad . h + h^T H h / 2` at points `[n, d]`. */
  quadratic(x: TensorLike): Value
}

/**
 * The linear and quadratic approximations of a scalar field at a point (the multivariable Taylor polynomials of degree
 * one and two): the tangent plane and the osculating paraboloid of the graph.
 *
 * @param f - The field.
 * @param a - The point `[d]`.
 * @returns Value, gradient, Hessian and the two models.
 * @concept quadratic-approximation
 * @example
 * ```ts
 * import { quadraticApproximation } from 'whitebox-engine/foundations'
 * import { exp, sum, toArray, type Value } from 'whitebox-engine/tensor'
 * const q = quadraticApproximation((p: Value) => exp(sum(p, -1)), [0, 0])
 * q.gradient // => [1, 1]
 * toArray(q.quadratic([[0.1, 0.1]])) // => [1.22]
 * ```
 */
export function quadraticApproximation(f: ScalarField, a: readonly number[]): QuadraticModel {
  const p = points([...a])
  const value = item(f(p))
  const gradient = toFlat(gradientAt(f, p))
  const hessian = (toArray(hessianAt(f, p)) as number[][][])[0] ?? []
  const shift = (x: TensorLike): Value => sub(points(x), [...a])
  const lin = (h: Value): Value => add(value, sum(mul(h, gradient), -1))
  return {
    point: [...a],
    value,
    gradient,
    hessian,
    linear: (x) => lin(shift(x)),
    quadratic: (x) => {
      const h = shift(x)
      return add(lin(h), mul(sum(mul(h, matmul(h, hessian)), -1), 0.5))
    },
  }
}

/**
 * The tangent plane to the graph `z = f(x, y)` at a point: through `(a, b, f(a, b))` with normal `(f_x, f_y, -1)`.
 *
 * @param f - A scalar field of two variables.
 * @param a - The point `[2]`.
 * @returns The point on the graph, the (upward unit) normal and the plane's equation `z = z0 + f_x (x - a) + f_y (y - b)`.
 * @concept tangent-plane
 * @example
 * ```ts
 * import { graphTangentPlane } from 'whitebox-engine/foundations'
 * import { mul, sum, type Value } from 'whitebox-engine/tensor'
 * graphTangentPlane((p: Value) => sum(mul(p, p), -1), [1, 1]).slopes // => [2, 2]
 * ```
 */
export function graphTangentPlane(
  f: ScalarField,
  a: readonly [number, number],
): {
  readonly point: [number, number, number]
  readonly normal: [number, number, number]
  readonly slopes: [number, number]
} {
  const q = quadraticApproximation(f, a)
  const [fx = 0, fy = 0] = q.gradient
  const len = Math.hypot(fx, fy, 1)
  return { point: [a[0], a[1], q.value], normal: [-fx / len, -fy / len, 1 / len], slopes: [fx, fy] }
}

/** The kinds of critical point of a function of several variables. */
export type MultivariateCriticalKind = 'minimum' | 'maximum' | 'saddle' | 'degenerate'

/**
 * Classify a critical point by the eigenvalues of the Hessian: all positive a minimum, all negative a maximum, mixed
 * signs a saddle, any zero degenerate (the test is silent).
 *
 * @param f - The field.
 * @param a - A critical point `[d]`.
 * @param tol - Eigenvalues below this in magnitude count as zero (default 1e-10).
 * @returns The kind and the eigenvalues.
 * @concept second-derivative-test
 * @example
 * ```ts
 * import { classifyCriticalPoint } from 'whitebox-engine/foundations'
 * import { mul, sum, type Value } from 'whitebox-engine/tensor'
 * const saddle = (p: Value) => sum(mul(mul(p, p), [1, -1]), -1)
 * classifyCriticalPoint(saddle, [0, 0]).kind // => 'saddle'
 * ```
 */
export function classifyCriticalPoint(
  f: ScalarField,
  a: readonly number[],
  tol = 1e-10,
): { readonly kind: MultivariateCriticalKind; readonly eigenvalues: number[] } {
  const h = quadraticApproximation(f, a).hessian
  const eigenvalues = toFlat(eigvalsh(h))
  const kind: MultivariateCriticalKind = eigenvalues.some((e) => Math.abs(e) <= tol)
    ? 'degenerate'
    : eigenvalues.every((e) => e > 0)
      ? 'minimum'
      : eigenvalues.every((e) => e < 0)
        ? 'maximum'
        : 'saddle'
  return { kind, eigenvalues }
}

/** Result of {@link lagrangeMultipliers}. */
export interface LagrangeSolution {
  readonly x: number[]
  readonly multipliers: number[]
  readonly value: number
  readonly converged: boolean
  /** The residual norm of the Lagrange system after each Newton iteration. */
  readonly history: readonly number[]
}

/**
 * Solve a small equality-constrained problem `extremise f(x) subject to g_i(x) = 0` by the method of Lagrange
 * multipliers: Newton's method (Powell's hybrid, as scipy's fsolve) on the system `grad f = sum_i lambda_i grad g_i`,
 * `g(x) = 0`, from a starting point. Which stationary point it finds depends on the start.
 *
 * @param f - The objective, vectorised over points `[n, d]`.
 * @param constraints - The constraint functions, each vectorised the same way.
 * @param x0 - Starting point `[d]`.
 * @param lambda0 - Starting multipliers (default: the least-squares fit of grad f by the grad g_i at x0).
 * @returns The point, multipliers, objective value and convergence.
 * @concept lagrange-multipliers
 * @example
 * ```ts
 * import { lagrangeMultipliers } from 'whitebox-engine/foundations'
 * import { mul, sub, sum, type Value } from 'whitebox-engine/tensor'
 * // Maximise x + y on the unit circle.
 * const r = lagrangeMultipliers((p: Value) => sum(p, -1), [(p: Value) => sub(sum(mul(p, p), -1), 1)], [1, 0.5])
 * r.x.map((v) => Math.round(v * 1e10) / 1e10) // => [0.7071067812, 0.7071067812]
 * ```
 */
export function lagrangeMultipliers(
  f: ScalarField,
  constraints: readonly ScalarField[],
  x0: readonly number[],
  lambda0?: readonly number[],
): LagrangeSolution {
  const d = x0.length
  const k = constraints.length
  if (k === 0) throw new WhiteboxError('value', 'lagrangeMultipliers: give at least one constraint')
  const system = (z: Value): Value => {
    const x = stack([sliceAxis(z, 0, 0, d)])
    const lam = sliceAxis(z, 0, d, d + k)
    let station = squeeze(gradientAt(f, x), 0)
    for (const [i, g] of constraints.entries()) {
      station = sub(station, mul(squeeze(gradientAt(g, x), 0), sliceAxis(lam, 0, i, i + 1)))
    }
    return concat([station, ...constraints.map((g) => g(x))], 0)
  }
  // Without given multipliers, start from the least-squares fit of grad f by the constraint gradients at x0.
  const at0 = points([...x0])
  const fitted =
    lambda0 ??
    toFlat(
      lstsq(transpose(stack(constraints.map((g) => squeeze(gradientAt(g, at0), 0)))), squeeze(gradientAt(f, at0), 0)),
    )
  const start = [...x0, ...fitted]
  const r = fsolve(system, start, { xtol: 1e-13 })
  const z = toFlat(r.x)
  const x = z.slice(0, d)
  return { x, multipliers: z.slice(d), value: item(f(points(x))), converged: r.converged, history: r.history }
}

/**
 * The divergence `sum_i dF_i / dx_i` of a vector field at points: the net outflow per unit volume.
 *
 * @param F - A field `[n, d] -> [n, d]`.
 * @param x - Points `[n, d]`.
 * @returns Divergences `[n]`.
 * @concept divergence
 * @example
 * ```ts
 * import { divergence } from 'whitebox-engine/foundations'
 * import { toArray, type Value } from 'whitebox-engine/tensor'
 * toArray(divergence((p: Value) => p, [[1, 2, 3]])) // => [3]
 * ```
 */
export function divergence(F: VectorField, x: TensorLike): Value {
  const p = points(x)
  let total: Value | null = null
  for (let j = 0; j < dims(p); j++) {
    const dj = squeeze(sliceAxis(along(F, p, j), -1, j, j + 1), -1)
    total = total === null ? dj : add(total, dj)
  }
  if (total === null) throw new WhiteboxError('shape', 'divergence: points have no coordinates')
  return total
}

/**
 * The curl of a vector field at points: `curl F = (dF_z/dy - dF_y/dz, dF_x/dz - dF_z/dx, dF_y/dx - dF_x/dy)` in 3-D,
 * and the scalar `dF_y/dx - dF_x/dy` in 2-D: the circulation per unit area.
 *
 * @param F - A field in 2-D or 3-D.
 * @param x - Points `[n, 2]` or `[n, 3]`.
 * @returns Curls `[n, 3]`, or `[n]` in 2-D.
 * @throws WhiteboxError with code `shape` in other dimensions.
 * @concept curl
 * @example
 * ```ts
 * import { curl } from 'whitebox-engine/foundations'
 * import { mul, sliceAxis, squeeze, stack, toArray, type Value } from 'whitebox-engine/tensor'
 * const rotation = (p: Value) => stack([mul(squeeze(sliceAxis(p, -1, 1, 2), -1), -1), squeeze(sliceAxis(p, -1, 0, 1), -1)], -1)
 * toArray(curl(rotation, [[0.3, 0.4]])) // => [2]
 * ```
 */
export function curl(F: VectorField, x: TensorLike): Value {
  const p = points(x)
  const d = dims(p)
  const partial = (i: number, j: number): Value => squeeze(sliceAxis(along(F, p, j), -1, i, i + 1), -1)
  if (d === 2) return sub(partial(1, 0), partial(0, 1))
  if (d !== 3) throw new WhiteboxError('shape', 'curl: fields in 2-D or 3-D')
  return stack(
    [sub(partial(2, 1), partial(1, 2)), sub(partial(0, 2), partial(2, 0)), sub(partial(1, 0), partial(0, 1))],
    -1,
  )
}

const tangentOf = (r: ParametricCurve, t: Value): Value => jvp(r, t, onesLike(t)).tangent

/**
 * The line integral of a scalar field along a curve with respect to arc length, `int f(r(t)) |r'(t)| dt` (the mass of
 * a wire of density f), by adaptive quadrature.
 *
 * @param f - The scalar field.
 * @param r - The curve, parameters `[n]` to points `[n, d]`.
 * @param a - Start parameter.
 * @param b - End parameter.
 * @returns The integral.
 * @concept line-integrals
 * @example
 * ```ts
 * import { scalarLineIntegral } from 'whitebox-engine/foundations'
 * import { cos, onesLike, sin, sliceAxis, squeeze, stack, type Value } from 'whitebox-engine/tensor'
 * const circle = (t: Value) => stack([cos(t), sin(t)], -1)
 * scalarLineIntegral((p: Value) => squeeze(onesLike(sliceAxis(p, -1, 0, 1)), -1), circle, 0, Math.PI) // => 3.141592653589793
 * ```
 */
export function scalarLineIntegral(f: ScalarField, r: ParametricCurve, a: number, b: number): number {
  return item(
    quad(
      (t) => {
        const v = tangentOf(r, t)
        return mul(f(r(t)), sqrt(sum(mul(v, v), -1)))
      },
      a,
      b,
      { epsabs: 1e-12, epsrel: 1e-12 },
    ).value,
  )
}

/**
 * The line integral of a vector field along a curve, `int F(r(t)) . r'(t) dt`: the work done by a force, or the
 * circulation round a closed curve. It changes sign with the orientation and, for a gradient field, depends only on
 * the ends.
 *
 * @param F - The vector field.
 * @param r - The curve.
 * @param a - Start parameter.
 * @param b - End parameter.
 * @returns The integral.
 * @concept line-integrals
 * @example
 * ```ts
 * import { vectorLineIntegral } from 'whitebox-engine/foundations'
 * import { stack, type Value } from 'whitebox-engine/tensor'
 * // F = (x, y) along the segment from (0, 0) to (1, 1): the potential (x^2 + y^2) / 2 rises by 1.
 * vectorLineIntegral((p: Value) => p, (t: Value) => stack([t, t], -1), 0, 1) // => 1
 * ```
 */
export function vectorLineIntegral(F: VectorField, r: ParametricCurve, a: number, b: number): number {
  return item(quad((t) => sum(mul(F(r(t)), tangentOf(r, t)), -1), a, b, { epsabs: 1e-12, epsrel: 1e-12 }).value)
}

/**
 * The outward flux of a plane vector field across a closed curve traversed counter-clockwise,
 * `int F . n ds = int (F_x y' - F_y x') dt`.
 *
 * @param F - A plane field.
 * @param r - A closed plane curve, counter-clockwise.
 * @param a - Start parameter.
 * @param b - End parameter.
 * @returns The flux.
 * @concept flux
 * @example
 * ```ts
 * import { planeFlux } from 'whitebox-engine/foundations'
 * import { cos, sin, stack, type Value } from 'whitebox-engine/tensor'
 * planeFlux((p: Value) => p, (t: Value) => stack([cos(t), sin(t)], -1), 0, 2 * Math.PI) / Math.PI // => 2
 * ```
 */
export function planeFlux(F: VectorField, r: ParametricCurve, a: number, b: number): number {
  return item(
    quad(
      (t) => {
        const v = tangentOf(r, t)
        // The outward normal times speed is (y', -x') for a counter-clockwise curve.
        const n = stack([squeeze(sliceAxis(v, -1, 1, 2), -1), mul(squeeze(sliceAxis(v, -1, 0, 1), -1), -1)], -1)
        return sum(mul(F(r(t)), n), -1)
      },
      a,
      b,
      { epsabs: 1e-12, epsrel: 1e-12 },
    ).value,
  )
}

/** A parameter rectangle `[[u0, u1], [v0, v1]]`. */
export type ParameterBox = readonly [readonly [number, number], readonly [number, number]]

function surfacePartials(s: ParametricSurface, uv: Value): [Value, Value] {
  return [along(s, uv, 0), along(s, uv, 1)]
}

/**
 * The surface integral of a scalar field, `int int f(r(u, v)) |r_u x r_v| du dv`, by a product Gauss-Legendre rule.
 *
 * @param f - The scalar field on 3-D points.
 * @param s - The parametric surface.
 * @param box - The parameter rectangle.
 * @param n - Gauss points per axis (default 24).
 * @returns The integral.
 * @concept surface-integrals
 * @example
 * ```ts
 * import { scalarSurfaceIntegral, sphere } from 'whitebox-engine/foundations'
 * import { onesLike, sliceAxis, squeeze, type Value } from 'whitebox-engine/tensor'
 * const one = (p: Value) => squeeze(onesLike(sliceAxis(p, -1, 0, 1)), -1)
 * Math.round(scalarSurfaceIntegral(one, sphere(1), [[0, Math.PI], [0, 2 * Math.PI]]) / Math.PI * 1e12) / 1e12 // => 4
 * ```
 */
export function scalarSurfaceIntegral(f: ScalarField, s: ParametricSurface, box: ParameterBox, n = 24): number {
  const rule = productGaussRule(box, n)
  return item(
    cubature((uv) => {
      const [ru, rv] = surfacePartials(s, uv)
      const c = crossProduct(ru, rv)
      return mul(f(s(uv)), sqrt(sum(mul(c, c), -1)))
    }, rule),
  )
}

/**
 * The flux of a vector field through a parametric surface, `int int F(r) . (r_u x r_v) du dv` (oriented by
 * `r_u x r_v`), by a product Gauss-Legendre rule.
 *
 * @param F - The vector field on 3-D points.
 * @param s - The surface.
 * @param box - The parameter rectangle.
 * @param n - Gauss points per axis (default 24).
 * @returns The flux.
 * @concept flux
 * @example
 * ```ts
 * import { surfaceFlux, sphere } from 'whitebox-engine/foundations'
 * import { type Value } from 'whitebox-engine/tensor'
 * // F = x through the unit sphere (outward): 3 times the volume.
 * Math.round(surfaceFlux((p: Value) => p, sphere(1), [[0, Math.PI], [0, 2 * Math.PI]]) / Math.PI * 1e12) / 1e12 // => 4
 * ```
 */
export function surfaceFlux(F: VectorField, s: ParametricSurface, box: ParameterBox, n = 24): number {
  const rule = productGaussRule(box, n)
  return item(
    cubature((uv) => {
      const [ru, rv] = surfacePartials(s, uv)
      return sum(mul(F(s(uv)), crossProduct(ru, rv)), -1)
    }, rule),
  )
}

/** Both sides of an integral theorem, computed independently. */
export interface TheoremCheck {
  /** The boundary side (circulation or flux round the boundary). */
  readonly boundary: number
  /** The interior side (the integral of the curl or divergence). */
  readonly interior: number
  readonly difference: number
}

/**
 * Check Green's theorem numerically: the circulation `oint F . dr` round the boundary of a plane region equals
 * `int int (dQ/dx - dP/dy) dA`. The region is the image of a parameter rectangle under a map (such as polar
 * coordinates), and its boundary a closed counter-clockwise curve.
 *
 * @param F - A plane field `(P, Q)`.
 * @param boundary - The boundary curve and its parameter range.
 * @param region - The region as a map from a parameter rectangle to the plane.
 * @param n - Gauss points per axis for the area integral (default 24).
 * @returns Both sides and their difference.
 * @concept greens-theorem
 * @example
 * ```ts
 * import { greensTheorem, polarToCartesian } from 'whitebox-engine/foundations'
 * import { cos, mul, sin, sliceAxis, squeeze, stack, type Value } from 'whitebox-engine/tensor'
 * const F = (p: Value) => stack([mul(squeeze(sliceAxis(p, -1, 1, 2), -1), -1), squeeze(sliceAxis(p, -1, 0, 1), -1)], -1)
 * const disc = { map: polarToCartesian, box: [[0, 1], [0, 2 * Math.PI]] as const }
 * greensTheorem(F, { curve: (t: Value) => stack([cos(t), sin(t)], -1), range: [0, 2 * Math.PI] }, disc).difference < 1e-12 // => true
 * ```
 */
export function greensTheorem(
  F: VectorField,
  boundary: { readonly curve: ParametricCurve; readonly range: readonly [number, number] },
  region: { readonly map: (uv: Value) => Value; readonly box: ParameterBox },
  n = 24,
): TheoremCheck {
  const circulation = vectorLineIntegral(F, boundary.curve, boundary.range[0], boundary.range[1])
  const rule = productGaussRule(region.box, n)
  const interior = item(
    cubature((uv) => {
      const xy = region.map(uv)
      const ju = along(region.map, uv, 0)
      const jv = along(region.map, uv, 1)
      const area = abs(crossProduct(ju, jv))
      return mul(curl(F, xy), area)
    }, rule),
  )
  return { boundary: circulation, interior, difference: Math.abs(circulation - interior) }
}

/**
 * Check Stokes' theorem numerically: the circulation of F round the boundary curve equals the flux of `curl F` through
 * the surface, `oint F . dr = int int (curl F) . dS` (the boundary oriented by the right-hand rule with `r_u x r_v`).
 *
 * @param F - A 3-D field.
 * @param boundary - The boundary curve and its parameter range.
 * @param surface - The surface and its parameter rectangle.
 * @param n - Gauss points per axis (default 24).
 * @returns Both sides and their difference.
 * @concept stokes-theorem
 * @example
 * ```ts
 * import { sphere, stokesTheorem } from 'whitebox-engine/foundations'
 * import { cos, mul, sin, sliceAxis, squeeze, stack, zerosLike, type Value } from 'whitebox-engine/tensor'
 * const c = (p: Value, i: number) => squeeze(sliceAxis(p, -1, i, i + 1), -1)
 * const F = (p: Value) => stack([mul(c(p, 1), -1), c(p, 0), zerosLike(c(p, 2))], -1)
 * const rim = { curve: (t: Value) => stack([cos(t), sin(t), zerosLike(t)], -1), range: [0, 2 * Math.PI] as const }
 * stokesTheorem(F, rim, { surface: sphere(1), box: [[0, Math.PI / 2], [0, 2 * Math.PI]] }).difference < 1e-10 // => true
 * ```
 */
export function stokesTheorem(
  F: VectorField,
  boundary: { readonly curve: ParametricCurve; readonly range: readonly [number, number] },
  surface: { readonly surface: ParametricSurface; readonly box: ParameterBox },
  n = 24,
): TheoremCheck {
  const circulation = vectorLineIntegral(F, boundary.curve, boundary.range[0], boundary.range[1])
  // The upper hemisphere's r_u x r_v points outward (up), so the rim runs counter-clockwise seen from above.
  const flux = surfaceFlux((p) => curl(F, p), surface.surface, surface.box, n)
  return { boundary: circulation, interior: flux, difference: Math.abs(circulation - flux) }
}

/**
 * Check the divergence theorem numerically: the outward flux through a closed surface equals the integral of the
 * divergence over the solid it bounds. The solid is the image of a parameter box in three dimensions under a map (such
 * as spherical coordinates).
 *
 * @param F - A 3-D field.
 * @param surface - The closed surface (oriented outward) and its parameter rectangle.
 * @param solid - The solid as a map from a 3-D parameter box.
 * @param n - Gauss points per axis (default 16).
 * @returns Both sides and their difference.
 * @concept divergence-theorem
 * @example
 * ```ts
 * import { divergenceTheorem, sphere, sphericalToCartesian } from 'whitebox-engine/foundations'
 * import { mul, type Value } from 'whitebox-engine/tensor'
 * const ball = { map: sphericalToCartesian, box: [[0, 1], [0, Math.PI], [0, 2 * Math.PI]] as const }
 * divergenceTheorem((p: Value) => mul(p, p), { surface: sphere(1), box: [[0, Math.PI], [0, 2 * Math.PI]] }, ball).difference < 1e-10 // => true
 * ```
 */
export function divergenceTheorem(
  F: VectorField,
  surface: { readonly surface: ParametricSurface; readonly box: ParameterBox },
  solid: {
    readonly map: (uvw: Value) => Value
    readonly box: readonly [readonly [number, number], readonly [number, number], readonly [number, number]]
  },
  n = 16,
): TheoremCheck {
  const flux = surfaceFlux(F, surface.surface, surface.box, n)
  const rule = productGaussRule(solid.box, n)
  const interior = item(
    cubature((uvw) => {
      const xyz = solid.map(uvw)
      const [a, b, c] = [0, 1, 2].map((j) => along(solid.map, uvw, j)) as [Value, Value, Value]
      const volume = abs(sum(mul(a, crossProduct(b, c)), -1))
      return mul(divergence(F, xyz), volume)
    }, rule),
  )
  return { boundary: flux, interior, difference: Math.abs(flux - interior) }
}
