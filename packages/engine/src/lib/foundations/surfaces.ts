import { jvp } from '../autodiff/index.ts'
import { WhiteboxError } from '../base/index.ts'
import { cubature, productGaussRule } from '../numerics/quadrature/index.ts'
import { type Face3D, face3d, type Paint, type Style, type Vec3 } from '../scene/index.ts'
import {
  add,
  cos,
  div,
  item,
  linspace,
  mul,
  onesLike,
  promoteFloat,
  sin,
  sliceAxis,
  sqrt,
  squeeze,
  stack,
  sub,
  sum,
  type TensorLike,
  toArray,
  toFlat,
  type Value,
  zerosLike,
} from '../tensor/index.ts'
import { crossProduct, type Plane } from './vectors.ts'

/**
 * A parametric surface: a function from parameters `(u, v)` `[n, 2]` to points `[n, 3]`, written with tensor
 * operations so that it can be differentiated (each point may depend only on its own parameters).
 */
export type ParametricSurface = (uv: Value) => Value

const keep = (x: Value): Value => stack([x], -1)
const dotLast = (a: Value, b: Value): Value => sum(mul(a, b), -1)

function params(uv: TensorLike): Value {
  const [v] = promoteFloat(uv) as [Value]
  if (v.shape[v.shape.length - 1] !== 2) throw new WhiteboxError('shape', 'surface parameters must be [n, 2]')
  return v.shape.length === 1 ? stack([v]) : v
}

/** The direction (1, 0) or (0, 1) in parameter space, for every point. */
function direction(uv: Value, axis: 0 | 1): Value {
  const one = onesLike(sliceAxis(uv, -1, 0, 1))
  const zero = zerosLike(one)
  return axis === 0
    ? stack([squeeze(one, -1), squeeze(zero, -1)], -1)
    : stack([squeeze(zero, -1), squeeze(one, -1)], -1)
}

const partial =
  (r: ParametricSurface, axis: 0 | 1): ParametricSurface =>
  (uv) =>
    jvp(r, uv, direction(uv, axis)).tangent

/** The point and partial derivatives of a surface at parameters. */
export interface SurfaceDerivatives {
  readonly r: Value
  readonly ru: Value
  readonly rv: Value
  readonly ruu: Value
  readonly ruv: Value
  readonly rvv: Value
}

/**
 * The first and second partial derivatives of a parametric surface, by nested forward-mode differentiation.
 *
 * @param r - The surface.
 * @param uv - Parameters `[n, 2]` (or one pair).
 * @returns `r`, `r_u`, `r_v`, `r_uu`, `r_uv`, `r_vv`, each `[n, 3]`.
 * @concept parametric-surfaces
 * @example
 * ```ts
 * import { surfaceDerivatives } from 'whitebox-engine/foundations'
 * import { mul, sliceAxis, squeeze, stack, toArray, type Value } from 'whitebox-engine/tensor'
 * const saddle = (p: Value) => {
 *   const [u, v] = [0, 1].map((i) => squeeze(sliceAxis(p, -1, i, i + 1), -1))
 *   return stack([u, v, mul(u, v)], -1)
 * }
 * toArray(surfaceDerivatives(saddle, [1, 2]).ruv) // => [[0, 0, 1]]
 * ```
 */
export function surfaceDerivatives(r: ParametricSurface, uv: TensorLike): SurfaceDerivatives {
  const p = params(uv)
  const ru = partial(r, 0)
  const rv = partial(r, 1)
  return { r: r(p), ru: ru(p), rv: rv(p), ruu: partial(ru, 0)(p), ruv: partial(ru, 1)(p), rvv: partial(rv, 1)(p) }
}

/**
 * The unit normal `r_u x r_v / |r_u x r_v|` of a parametric surface.
 *
 * @param r - The surface.
 * @param uv - Parameters `[n, 2]`.
 * @returns Unit normals `[n, 3]`.
 * @concept surface-normal
 * @example
 * ```ts
 * import { graphSurface, surfaceNormal } from 'whitebox-engine/foundations'
 * import { mul, sum, toArray, type Value } from 'whitebox-engine/tensor'
 * toArray(surfaceNormal(graphSurface((p: Value) => sum(mul(p, p), -1)), [0, 0])) // => [[0, 0, 1]]
 * ```
 */
export function surfaceNormal(r: ParametricSurface, uv: TensorLike): Value {
  const p = params(uv)
  const n = crossProduct(partial(r, 0)(p), partial(r, 1)(p))
  return div(n, keep(sqrt(dotLast(n, n))))
}

/**
 * The tangent plane of a surface at one parameter pair: through the point, with the unit normal.
 *
 * @param r - The surface.
 * @param uv - One parameter pair `[2]`.
 * @returns The plane.
 * @concept tangent-plane
 * @example
 * ```ts
 * import { graphSurface, tangentPlane } from 'whitebox-engine/foundations'
 * import { mul, sum, type Value } from 'whitebox-engine/tensor'
 * tangentPlane(graphSurface((p: Value) => sum(mul(p, p), -1)), [1, 0]).point // => [1, 0, 1]
 * ```
 */
export function tangentPlane(r: ParametricSurface, uv: readonly [number, number]): Plane {
  const p = params([...uv])
  return { point: toFlat(r(p)), normal: toFlat(surfaceNormal(r, p)) }
}

/** The coefficients of a fundamental form at each parameter pair. */
export interface FundamentalForm {
  /** E or L: the uu coefficient. */
  readonly uu: Value
  /** F or M: the uv coefficient. */
  readonly uv: Value
  /** G or N: the vv coefficient. */
  readonly vv: Value
}

/**
 * The first fundamental form (the metric) `E = r_u . r_u`, `F = r_u . r_v`, `G = r_v . r_v`: lengths on the surface
 * are `ds^2 = E du^2 + 2F du dv + G dv^2`.
 *
 * @param r - The surface.
 * @param uv - Parameters `[n, 2]`.
 * @returns `{ uu: E, uv: F, vv: G }`, each `[n]`.
 * @concept fundamental-forms
 * @example
 * ```ts
 * import { firstFundamentalForm, sphere } from 'whitebox-engine/foundations'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(firstFundamentalForm(sphere(2), [Math.PI / 2, 0]).vv) // => [4]
 * ```
 */
export function firstFundamentalForm(r: ParametricSurface, uv: TensorLike): FundamentalForm {
  const p = params(uv)
  const ru = partial(r, 0)(p)
  const rv = partial(r, 1)(p)
  return { uu: dotLast(ru, ru), uv: dotLast(ru, rv), vv: dotLast(rv, rv) }
}

/**
 * The second fundamental form `L = r_uu . n`, `M = r_uv . n`, `N = r_vv . n` with n the unit normal: how the surface
 * bends away from its tangent plane.
 *
 * @param r - The surface.
 * @param uv - Parameters `[n, 2]`.
 * @returns `{ uu: L, uv: M, vv: N }`, each `[n]`.
 * @concept fundamental-forms
 * @example
 * ```ts
 * import { secondFundamentalForm, sphere } from 'whitebox-engine/foundations'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(secondFundamentalForm(sphere(2), [Math.PI / 2, 0]).uu) // => [-2]
 * ```
 */
export function secondFundamentalForm(r: ParametricSurface, uv: TensorLike): FundamentalForm {
  const d = surfaceDerivatives(r, uv)
  const n0 = crossProduct(d.ru, d.rv)
  const n = div(n0, keep(sqrt(dotLast(n0, n0))))
  return { uu: dotLast(d.ruu, n), uv: dotLast(d.ruv, n), vv: dotLast(d.rvv, n) }
}

/**
 * The Gaussian curvature `K = (LN - M^2) / (EG - F^2)`: the product of the principal curvatures, intrinsic (Gauss's
 * Theorema Egregium).
 *
 * @param r - The surface.
 * @param uv - Parameters `[n, 2]`.
 * @returns `K` `[n]`.
 * @concept gaussian-curvature
 * @example
 * ```ts
 * import { gaussianCurvature, sphere } from 'whitebox-engine/foundations'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(gaussianCurvature(sphere(2), [1, 0.5])) // => [0.25]
 * ```
 */
export function gaussianCurvature(r: ParametricSurface, uv: TensorLike): Value {
  const I = firstFundamentalForm(r, uv)
  const II = secondFundamentalForm(r, uv)
  return div(sub(mul(II.uu, II.vv), mul(II.uv, II.uv)), sub(mul(I.uu, I.vv), mul(I.uv, I.uv)))
}

/**
 * The mean curvature `H = (EN - 2FM + GL) / (2 (EG - F^2))`: the average of the principal curvatures (its sign follows
 * the orientation of the normal; zero for minimal surfaces).
 *
 * @param r - The surface.
 * @param uv - Parameters `[n, 2]`.
 * @returns `H` `[n]`.
 * @concept mean-curvature
 * @example
 * ```ts
 * import { meanCurvature, sphere } from 'whitebox-engine/foundations'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(meanCurvature(sphere(2), [1, 0.5])) // => [-0.5]
 * ```
 */
export function meanCurvature(r: ParametricSurface, uv: TensorLike): Value {
  const I = firstFundamentalForm(r, uv)
  const II = secondFundamentalForm(r, uv)
  const num = add(sub(mul(I.uu, II.vv), mul(2, mul(I.uv, II.uv))), mul(I.vv, II.uu))
  return div(num, mul(2, sub(mul(I.uu, I.vv), mul(I.uv, I.uv))))
}

/** Principal curvatures and directions at one parameter pair. */
export interface PrincipalCurvatures {
  /** k1 >= k2. */
  readonly curvatures: [number, number]
  /** The principal directions in parameter space (du, dv), normalised. */
  readonly parameterDirections: [[number, number], [number, number]]
  /** The principal directions as unit tangent vectors in space. */
  readonly directions: [number[], number[]]
}

/**
 * The principal curvatures and directions at a point: the eigenvalues and eigenvectors of the shape operator
 * `S = I^-1 II` (the extremes of the normal curvature over tangent directions).
 *
 * @param r - The surface.
 * @param uv - One parameter pair `[2]`.
 * @returns The curvatures (descending) and directions.
 * @concept principal-curvatures
 * @example
 * ```ts
 * import { principalCurvatures, torus } from 'whitebox-engine/foundations'
 * principalCurvatures(torus(2, 1), [0, 0]).curvatures.map((k) => Math.round(k * 1e12) / 1e12) // => [1, 0.333333333333]
 * ```
 */
export function principalCurvatures(r: ParametricSurface, uv: readonly [number, number]): PrincipalCurvatures {
  const p = params([...uv])
  const I = firstFundamentalForm(r, p)
  const II = secondFundamentalForm(r, p)
  const [E, Fv, G] = [item(I.uu), item(I.uv), item(I.vv)]
  const [L, M, N] = [item(II.uu), item(II.uv), item(II.vv)]
  const det = E * G - Fv * Fv
  // S = I^-1 II = [[a, b], [c, d]].
  const a = (G * L - Fv * M) / det
  const b = (G * M - Fv * N) / det
  const c = (E * M - Fv * L) / det
  const d = (E * N - Fv * M) / det
  const mid = (a + d) / 2
  const disc = Math.sqrt(Math.max(0, ((a - d) / 2) ** 2 + b * c))
  const k1 = mid + disc
  const k2 = mid - disc
  // (S - k I) v = 0: either row gives an eigenvector; take the better conditioned one (an umbilic has every direction).
  const eigen = (k: number, fallback: [number, number]): [number, number] => {
    const v1: [number, number] = [b, k - a]
    const v2: [number, number] = [k - d, c]
    const v = Math.hypot(...v1) >= Math.hypot(...v2) ? v1 : v2
    const len = Math.hypot(v[0], v[1])
    return len < 1e-14 ? fallback : [v[0] / len, v[1] / len]
  }
  const e1 = eigen(k1, [1, 0])
  const e2 = eigen(k2, [0, 1])
  const ru = toFlat(partial(r, 0)(p))
  const rv = toFlat(partial(r, 1)(p))
  const lift = (e: [number, number]): number[] => {
    const w = ru.map((x, i) => e[0] * x + e[1] * (rv[i] ?? 0))
    const len = Math.hypot(...w)
    return w.map((x) => x / len)
  }
  return { curvatures: [k1, k2], parameterDirections: [e1, e2], directions: [lift(e1), lift(e2)] }
}

/**
 * The surface `z = f(x, y)` (a graph) as a parametric surface with `(u, v) = (x, y)`.
 *
 * @param f - Heights, vectorised over points `[n, 2] -> [n]`.
 * @returns The surface.
 * @concept parametric-surfaces
 * @example
 * ```ts
 * import { graphSurface } from 'whitebox-engine/foundations'
 * import { mul, sum, tensor, toArray, type Value } from 'whitebox-engine/tensor'
 * toArray(graphSurface((p: Value) => sum(mul(p, p), -1))(tensor([[1, 2]]))) // => [[1, 2, 5]]
 * ```
 */
export function graphSurface(f: (xy: Value) => Value): ParametricSurface {
  return (uv) => stack([squeeze(sliceAxis(uv, -1, 0, 1), -1), squeeze(sliceAxis(uv, -1, 1, 2), -1), f(uv)], -1)
}

/**
 * The surface of revolution of a profile curve `(rho(t), z(t))` about the z-axis: `(u, v) = (t, angle)` maps to
 * `(rho cos v, rho sin v, z)`.
 *
 * @param profile - The profile, parameters `[n]` to `[n, 2]` as (distance from the axis, height).
 * @returns The surface.
 * @concept surfaces-of-revolution
 * @example
 * ```ts
 * import { surfaceOfRevolution } from 'whitebox-engine/foundations'
 * import { onesLike, stack, tensor, toArray, type Value } from 'whitebox-engine/tensor'
 * const cylinder = surfaceOfRevolution((t: Value) => stack([onesLike(t), t], -1))
 * toArray(cylinder(tensor([[2, 0]]))) // => [[1, 0, 2]]
 * ```
 */
export function surfaceOfRevolution(profile: (t: Value) => Value): ParametricSurface {
  return (uv) => {
    const t = squeeze(sliceAxis(uv, -1, 0, 1), -1)
    const angle = squeeze(sliceAxis(uv, -1, 1, 2), -1)
    const pr = profile(t)
    const rho = squeeze(sliceAxis(pr, -1, 0, 1), -1)
    return stack([mul(rho, cos(angle)), mul(rho, sin(angle)), squeeze(sliceAxis(pr, -1, 1, 2), -1)], -1)
  }
}

const uvParts = (uv: Value): [Value, Value] => [
  squeeze(sliceAxis(uv, -1, 0, 1), -1),
  squeeze(sliceAxis(uv, -1, 1, 2), -1),
]

/**
 * The sphere of radius R with `(u, v) = (polar angle theta from the z-axis, azimuth phi)`.
 *
 * @param radius - R (default 1).
 * @returns The surface.
 * @concept parametric-surfaces
 * @example
 * ```ts
 * import { sphere } from 'whitebox-engine/foundations'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(sphere(2)(tensor([[0, 0]]))) // => [[0, 0, 2]]
 * ```
 */
export function sphere(radius = 1): ParametricSurface {
  return (uv) => {
    const [t, f] = uvParts(uv)
    const s = mul(sin(t), radius)
    return stack([mul(s, cos(f)), mul(s, sin(f)), mul(cos(t), radius)], -1)
  }
}

/**
 * The torus with tube radius `a` around a circle of radius `R` in the xy-plane: `(u, v)` = (angle round the tube,
 * angle round the axis), `((R + a cos u) cos v, (R + a cos u) sin v, a sin u)`.
 *
 * @param major - R, the distance from the centre to the tube's centre.
 * @param minor - a, the tube radius.
 * @returns The surface.
 * @concept parametric-surfaces
 * @example
 * ```ts
 * import { torus } from 'whitebox-engine/foundations'
 * import { tensor, toArray } from 'whitebox-engine/tensor'
 * toArray(torus(2, 1)(tensor([[0, 0]]))) // => [[3, 0, 0]]
 * ```
 */
export function torus(major: number, minor: number): ParametricSurface {
  return (uv) => {
    const [u, v] = uvParts(uv)
    const rho = add(mul(cos(u), minor), major)
    return stack([mul(rho, cos(v)), mul(rho, sin(v)), mul(sin(u), minor)], -1)
  }
}

/**
 * The area `int int sqrt(EG - F^2) du dv` of a parametric surface over a parameter rectangle, by a product
 * Gauss-Legendre rule.
 *
 * @param r - The surface.
 * @param uRange - `[u0, u1]`.
 * @param vRange - `[v0, v1]`.
 * @param n - Gauss points per axis (default 24).
 * @returns The area.
 * @concept surface-area
 * @example
 * ```ts
 * import { sphere, surfaceArea } from 'whitebox-engine/foundations'
 * surfaceArea(sphere(1), [0, Math.PI], [0, 2 * Math.PI]) / Math.PI // => 4.000000000000001
 * ```
 */
export function surfaceArea(
  r: ParametricSurface,
  uRange: readonly [number, number],
  vRange: readonly [number, number],
  n = 24,
): number {
  const rule = productGaussRule([uRange, vRange], n)
  return item(
    cubature((pts) => {
      const I = firstFundamentalForm(r, pts)
      return sqrt(sub(mul(I.uu, I.vv), mul(I.uv, I.uv)))
    }, rule),
  )
}

/**
 * The Christoffel symbols of the second kind at one parameter pair: `Gamma^k_ij = (g^-1)^{kl} (r_ij . r_l)`, with
 * indices 0 = u and 1 = v. They are the connection that the geodesic equations use.
 *
 * @param r - The surface.
 * @param uv - One parameter pair `[2]`.
 * @returns `gamma[k][i][j]`.
 * @concept christoffel-symbols
 * @example
 * ```ts
 * import { christoffelSymbols, sphere } from 'whitebox-engine/foundations'
 * christoffelSymbols(sphere(1), [Math.PI / 4, 0])[0][1][1] // => -0.5000000000000001
 * ```
 */
export function christoffelSymbols(r: ParametricSurface, uv: readonly [number, number]): number[][][] {
  const d = surfaceDerivatives(r, [...uv])
  const ru = toFlat(d.ru)
  const rv = toFlat(d.rv)
  const second = [
    [toFlat(d.ruu), toFlat(d.ruv)],
    [toFlat(d.ruv), toFlat(d.rvv)],
  ]
  const dot = (a: readonly number[], b: readonly number[]): number => a.reduce((s, x, i) => s + x * (b[i] ?? 0), 0)
  const E = dot(ru, ru)
  const Fv = dot(ru, rv)
  const G = dot(rv, rv)
  const det = E * G - Fv * Fv
  const inv = [
    [G / det, -Fv / det],
    [-Fv / det, E / det],
  ]
  const tangents = [ru, rv]
  return [0, 1].map((k) =>
    [0, 1].map((i) =>
      [0, 1].map((j) =>
        [0, 1].reduce((s, l) => s + (inv[k]?.[l] ?? 0) * dot(second[i]?.[j] ?? [], tangents[l] ?? []), 0),
      ),
    ),
  )
}

/** A numerically integrated geodesic. */
export interface Geodesic {
  /** Parameter-space positions (u, v) at each step. */
  readonly parameters: [number, number][]
  /** The points on the surface. */
  readonly points: number[][]
}

/**
 * A geodesic from a point in a direction (both in parameter space), by integrating the geodesic equations
 * `u_k'' + Gamma^k_ij u_i' u_j' = 0` with the classical fourth-order Runge-Kutta method (the initial velocity is scaled
 * to unit speed on the surface, so the parameter is arc length).
 *
 * @param r - The surface.
 * @param start - Starting parameters `(u, v)`.
 * @param velocity - Initial direction `(du, dv)`.
 * @param options - Length to travel (default 1) and number of steps (default 100).
 * @returns The path in parameter space and on the surface.
 * @concept geodesics
 * @example
 * ```ts
 * import { geodesic, sphere } from 'whitebox-engine/foundations'
 * const g = geodesic(sphere(1), [Math.PI / 2, 0], [0, 1], { length: Math.PI / 2, steps: 50 })
 * g.points[50].map((x) => Math.round(x * 1e8) / 1e8) // => [0, 1, 0]
 * ```
 */
export function geodesic(
  r: ParametricSurface,
  start: readonly [number, number],
  velocity: readonly [number, number],
  options: { readonly length?: number; readonly steps?: number } = {},
): Geodesic {
  const steps = options.steps ?? 100
  const h = (options.length ?? 1) / steps
  const I = firstFundamentalForm(r, [...start])
  const speed = Math.sqrt(
    item(I.uu) * velocity[0] ** 2 + 2 * item(I.uv) * velocity[0] * velocity[1] + item(I.vv) * velocity[1] ** 2,
  )
  if (!(speed > 0)) throw new WhiteboxError('value', 'geodesic: the initial velocity is zero')
  // State (u, v, u', v').
  const rhs = (s: readonly number[]): number[] => {
    const g = christoffelSymbols(r, [s[0] ?? 0, s[1] ?? 0])
    const du = [s[2] ?? 0, s[3] ?? 0]
    const acc = [0, 1].map(
      (k) =>
        -[0, 1].reduce(
          (t, i) => t + [0, 1].reduce((q, j) => q + (g[k]?.[i]?.[j] ?? 0) * (du[i] ?? 0) * (du[j] ?? 0), 0),
          0,
        ),
    )
    return [du[0] ?? 0, du[1] ?? 0, acc[0] ?? 0, acc[1] ?? 0]
  }
  let state = [start[0], start[1], velocity[0] / speed, velocity[1] / speed]
  const parameters: [number, number][] = [[start[0], start[1]]]
  const axpy = (a: readonly number[], k: number, b: readonly number[]): number[] => a.map((x, i) => x + k * (b[i] ?? 0))
  for (let i = 0; i < steps; i++) {
    const k1 = rhs(state)
    const k2 = rhs(axpy(state, h / 2, k1))
    const k3 = rhs(axpy(state, h / 2, k2))
    const k4 = rhs(axpy(state, h, k3))
    state = state.map((x, j) => x + (h / 6) * ((k1[j] ?? 0) + 2 * (k2[j] ?? 0) + 2 * (k3[j] ?? 0) + (k4[j] ?? 0)))
    parameters.push([state[0] ?? 0, state[1] ?? 0])
  }
  const points = (toArray(r(promoteFloat(parameters)[0] as Value)) as number[][]).map((p) =>
    p.map((x) => (Object.is(x, -0) ? 0 : x)),
  )
  return { parameters, points }
}

/**
 * The faces of a parametric surface over a parameter grid, as 3-D scene faces for drawing (one quadrilateral per
 * cell), optionally coloured by a per-cell value such as the Gaussian curvature.
 *
 * @param r - The surface.
 * @param uRange - `[u0, u1]`.
 * @param vRange - `[v0, v1]`.
 * @param n - Cells per axis, `[nu, nv]` (default 24 each).
 * @param style - Face style; `fill` may be a function of the cell's centre parameters.
 * @returns `nu * nv` faces.
 * @concept parametric-surfaces
 * @example
 * ```ts
 * import { sphere, surfaceFaces } from 'whitebox-engine/foundations'
 * surfaceFaces(sphere(1), [0, Math.PI], [0, 2 * Math.PI], [4, 8]).length // => 32
 * ```
 */
export function surfaceFaces(
  r: ParametricSurface,
  uRange: readonly [number, number],
  vRange: readonly [number, number],
  n: readonly [number, number] = [24, 24],
  style: Omit<Style, 'fill'> & { readonly fill?: Paint | ((u: number, v: number) => Paint) } = {},
): Face3D[] {
  const [nu, nv] = n
  const us = toFlat(linspace(uRange[0], uRange[1], nu + 1))
  const vs = toFlat(linspace(vRange[0], vRange[1], nv + 1))
  const grid = us.flatMap((u) => vs.map((v) => [u, v]))
  const pts = toArray(r(promoteFloat(grid)[0] as Value)) as number[][]
  const at = (i: number, j: number): Vec3 => {
    const p = pts[i * (nv + 1) + j] ?? [0, 0, 0]
    return [p[0] ?? 0, p[1] ?? 0, p[2] ?? 0]
  }
  const { fill = 'tab0', ...rest } = style
  const faces: Face3D[] = []
  for (let i = 0; i < nu; i++) {
    for (let j = 0; j < nv; j++) {
      const paint =
        typeof fill === 'function'
          ? fill(((us[i] ?? 0) + (us[i + 1] ?? 0)) / 2, ((vs[j] ?? 0) + (vs[j + 1] ?? 0)) / 2)
          : fill
      faces.push(face3d([at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)], { ...rest, fill: paint }))
    }
  }
  return faces
}
