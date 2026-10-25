/**
 * Curves in the plane and in space: parametric curves with derivatives by forward-mode autodiff, sampling, speed and
 * arc length, curvature, torsion and the Frenet-Serret frame, osculating circles and evolutes; implicit curves traced
 * by marching squares with their tangent and curvature; conic sections classified from their equations; and Bezier
 * curves through de Casteljau's algorithm, with splitting, derivatives and degree elevation.
 */

import { grad, hessian, jvp } from '../autodiff/index.ts'
import { MdmlError, maxOf } from '../base/index.ts'
import { contourLines } from '../numerics/geometry/index.ts'
import { eigh } from '../numerics/linalg/index.ts'
import { quad } from '../numerics/quadrature/index.ts'
import {
  add,
  div,
  item,
  linspace,
  meshgrid,
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
  type Value,
} from '../tensor/index.ts'
import { crossProduct } from './vectors.ts'

/**
 * A parametric curve: a function from parameters `t` `[n]` to points `[n, d]`, written with tensor operations so that
 * it can be differentiated (each point may depend only on its own parameter).
 */
export type ParametricCurve = (t: Value) => Value

const keep = (x: Value): Value => stack([x], -1)
const dotLast = (a: Value, b: Value): Value => sum(mul(a, b), -1)
const norm = (a: Value): Value => sqrt(dotLast(a, a))

function params(t: TensorLike): Value {
  const [v] = promoteFloat(t) as [Value]
  return v.shape.length === 0 ? stack([v]) : v
}

/** The derivative of a curve with respect to its parameter, as another curve (forward mode). */
function derivativeCurve(r: ParametricCurve): ParametricCurve {
  return (t) => jvp(r, t, onesLike(t)).tangent
}

/**
 * The derivatives of a parametric curve at parameters: `[r, r', r'', ...]` up to `order`, by nested forward-mode
 * differentiation.
 *
 * @param r - The curve.
 * @param t - Parameters `[n]` (or a scalar).
 * @param order - The highest derivative (default 2).
 * @returns `order + 1` values `[n, d]`.
 * @concept parametric-curves
 * @example
 * ```ts
 * import { curveDerivatives } from 'mdml-engine/foundations'
 * import { cos, sin, stack, toArray, type Value } from 'mdml-engine/tensor'
 * const circle = (t: Value) => stack([cos(t), sin(t)], -1)
 * toArray(curveDerivatives(circle, 0, 2)[2]) // => [[-1, 0]]
 * ```
 */
export function curveDerivatives(r: ParametricCurve, t: TensorLike, order = 2): Value[] {
  const tv = params(t)
  const out: Value[] = []
  let f = r
  for (let k = 0; k <= order; k++) {
    out.push(f(tv))
    f = derivativeCurve(f)
  }
  return out
}

/** Points of a curve sampled at evenly spaced parameters, for drawing. */
export interface CurveSamples {
  readonly t: number[]
  readonly points: number[][]
}

/**
 * Sample a parametric curve at `n` evenly spaced parameters in `[a, b]`.
 *
 * @param r - The curve.
 * @param a - First parameter.
 * @param b - Last parameter.
 * @param n - Number of samples (default 100).
 * @returns The parameters and points.
 * @concept parametric-curves
 * @example
 * ```ts
 * import { sampleCurve } from 'mdml-engine/foundations'
 * import { mul, stack, type Value } from 'mdml-engine/tensor'
 * sampleCurve((t: Value) => stack([t, mul(t, t)], -1), 0, 2, 3).points // => [[0, 0], [1, 1], [2, 4]]
 * ```
 */
export function sampleCurve(r: ParametricCurve, a: number, b: number, n = 100): CurveSamples {
  const t = linspace(a, b, n)
  return { t: toFlat(t), points: toArray(r(t)) as number[][] }
}

/**
 * The speed `|r'(t)|` of a parametric curve.
 *
 * @param r - The curve.
 * @param t - Parameters `[n]`.
 * @returns Speeds `[n]`.
 * @concept arc-length
 * @example
 * ```ts
 * import { curveSpeed } from 'mdml-engine/foundations'
 * import { cos, mul, sin, stack, toArray, type Value } from 'mdml-engine/tensor'
 * toArray(curveSpeed((t: Value) => stack([mul(cos(t), 3), mul(sin(t), 3)], -1), [0, 1])) // => [3, 3]
 * ```
 */
export function curveSpeed(r: ParametricCurve, t: TensorLike): Value {
  return norm(derivativeCurve(r)(params(t)))
}

/**
 * The arc length `int_a^b |r'(t)| dt` of a parametric curve, by adaptive Gauss-Kronrod quadrature of the speed.
 *
 * @param r - The curve.
 * @param a - Start parameter.
 * @param b - End parameter.
 * @param options - Quadrature tolerances (default 1.49e-8 absolute and relative, as scipy).
 * @returns The length.
 * @concept arc-length
 * @example
 * ```ts
 * import { arcLength } from 'mdml-engine/foundations'
 * import { cos, sin, stack, type Value } from 'mdml-engine/tensor'
 * arcLength((t: Value) => stack([cos(t), sin(t)], -1), 0, Math.PI) // => 3.141592653589793
 * ```
 */
export function arcLength(
  r: ParametricCurve,
  a: number,
  b: number,
  options: { readonly epsabs?: number; readonly epsrel?: number } = {},
): number {
  return item(quad((t) => curveSpeed(r, t), a, b, options).value)
}

/**
 * The cumulative arc length `s(t_i) = int_{t_0}^{t_i} |r'|` at increasing parameters, segment by segment (adaptive
 * quadrature on each): the table behind arc-length reparameterisation.
 *
 * @param r - The curve.
 * @param t - Increasing parameters `[n]`.
 * @returns Arc lengths, starting at 0.
 * @concept arc-length
 * @example
 * ```ts
 * import { cumulativeArcLength } from 'mdml-engine/foundations'
 * import { mul, stack, type Value } from 'mdml-engine/tensor'
 * cumulativeArcLength((t: Value) => stack([mul(t, 3), mul(t, 4)], -1), [0, 1, 2]) // => [0, 5, 10]
 * ```
 */
export function cumulativeArcLength(r: ParametricCurve, t: readonly number[]): number[] {
  const out = [0]
  for (let i = 1; i < t.length; i++) out.push((out[i - 1] ?? 0) + arcLength(r, t[i - 1] ?? 0, t[i] ?? 0))
  return out
}

/**
 * The curvature of a parametric curve: for plane curves the signed curvature `(x'y'' - y'x'') / |r'|^3` (positive when
 * turning left), in any other dimension `sqrt(|r'|^2 |r''|^2 - (r' . r'')^2) / |r'|^3` (which is `|r' x r''| / |r'|^3`
 * in 3-D).
 *
 * @param r - The curve.
 * @param t - Parameters `[n]`.
 * @returns Curvatures `[n]`.
 * @concept curvature
 * @example
 * ```ts
 * import { curvature } from 'mdml-engine/foundations'
 * import { cos, mul, sin, stack, toArray, type Value } from 'mdml-engine/tensor'
 * toArray(curvature((t: Value) => stack([mul(cos(t), 2), mul(sin(t), 2)], -1), [0.3])) // => [0.5]
 * ```
 */
export function curvature(r: ParametricCurve, t: TensorLike): Value {
  const [, d1, d2] = curveDerivatives(r, t, 2) as [Value, Value, Value]
  const speed = norm(d1)
  const cube = mul(speed, mul(speed, speed))
  if (d1.shape[d1.shape.length - 1] === 2) return div(crossProduct(d1, d2), cube)
  const a = dotLast(d1, d1)
  const b = dotLast(d2, d2)
  const c = dotLast(d1, d2)
  return div(sqrt(sub(mul(a, b), mul(c, c))), cube)
}

/**
 * The torsion of a space curve, `(r' x r'') . r''' / |r' x r''|^2`: how fast it twists out of its osculating plane.
 *
 * @param r - A curve in 3-D.
 * @param t - Parameters `[n]`.
 * @returns Torsions `[n]` (NaN where the curvature vanishes).
 * @concept torsion
 * @example
 * ```ts
 * import { torsion } from 'mdml-engine/foundations'
 * import { cos, sin, stack, toArray, type Value } from 'mdml-engine/tensor'
 * toArray(torsion((t: Value) => stack([cos(t), sin(t), t], -1), [0])) // => [0.5]
 * ```
 */
export function torsion(r: ParametricCurve, t: TensorLike): Value {
  const [, d1, d2, d3] = curveDerivatives(r, t, 3) as [Value, Value, Value, Value]
  const b = crossProduct(d1, d2)
  return div(dotLast(b, d3), dotLast(b, b))
}

/** The Frenet-Serret frame of a curve at parameters, with curvature and torsion. */
export interface FrenetFrame {
  readonly point: number[][]
  readonly tangent: number[][]
  readonly normal: number[][]
  /** Only for space curves. */
  readonly binormal: number[][] | null
  readonly curvature: number[]
  /** Only for space curves. */
  readonly torsion: number[] | null
}

/**
 * The Frenet-Serret frame: unit tangent `T = r' / |r'|`, principal normal `N` (towards the centre of curvature) and,
 * in 3-D, binormal `B = T x N`, with the curvature and torsion that drive `T' = k N`, `N' = -k T + tau B`, `B' = -tau N`
 * (per unit arc length). For plane curves N is T turned left and the curvature is signed.
 *
 * @param r - A curve in 2-D or 3-D.
 * @param t - Parameters `[n]`.
 * @returns The frames.
 * @throws MdmlError with code `shape` for other dimensions.
 * @concept frenet-serret
 * @example
 * ```ts
 * import { frenetFrame } from 'mdml-engine/foundations'
 * import { cos, sin, stack, type Value } from 'mdml-engine/tensor'
 * frenetFrame((t: Value) => stack([cos(t), sin(t)], -1), [0]).normal // => [[-1, 0]]
 * ```
 */
export function frenetFrame(r: ParametricCurve, t: TensorLike): FrenetFrame {
  const [p, d1, d2] = curveDerivatives(r, t, 2) as [Value, Value, Value]
  const dim = d1.shape[d1.shape.length - 1]
  const tangent = div(d1, keep(norm(d1)))
  const clean = (v: Value): number[][] =>
    (toArray(v) as number[][]).map((row) => row.map((x) => (Object.is(x, -0) ? 0 : x)))
  if (dim === 2) {
    const tx = squeeze(sliceAxis(tangent, -1, 0, 1), -1)
    const ty = squeeze(sliceAxis(tangent, -1, 1, 2), -1)
    const normal = stack([mul(ty, -1), tx], -1)
    return {
      point: clean(p),
      tangent: clean(tangent),
      normal: clean(normal),
      binormal: null,
      curvature: toFlat(curvature(r, t)),
      torsion: null,
    }
  }
  if (dim !== 3) throw new MdmlError('shape', 'frenetFrame: curves in 2-D or 3-D')
  const b = crossProduct(d1, d2)
  const binormal = div(b, keep(norm(b)))
  const normal = crossProduct(binormal, tangent)
  return {
    point: clean(p),
    tangent: clean(tangent),
    normal: clean(normal),
    binormal: clean(binormal),
    curvature: toFlat(curvature(r, t)),
    torsion: toFlat(torsion(r, t)),
  }
}

/**
 * The osculating circle at each parameter: the circle of best contact, with radius `1 / k` and centre `r + N / k`.
 * In 3-D it lies in the osculating plane spanned by T and N.
 *
 * @param r - A curve in 2-D or 3-D.
 * @param t - Parameters `[n]`.
 * @returns Centres and radii (infinite radius where the curve is straight).
 * @concept osculating-circle
 * @example
 * ```ts
 * import { osculatingCircle } from 'mdml-engine/foundations'
 * import { mul, stack, type Value } from 'mdml-engine/tensor'
 * osculatingCircle((t: Value) => stack([t, mul(t, t)], -1), [0]) // => { centres: [[0, 0.5]], radii: [0.5] }
 * ```
 */
export function osculatingCircle(
  r: ParametricCurve,
  t: TensorLike,
): { readonly centres: number[][]; readonly radii: number[] } {
  const f = frenetFrame(r, t)
  const radii = f.curvature.map((k) => 1 / Math.abs(k))
  const centres = f.point.map((p, i) => {
    const k = f.curvature[i] ?? 0
    const n = f.normal[i] ?? []
    return p.map((x, j) => x + (n[j] ?? 0) / k)
  })
  return { centres, radii }
}

/**
 * The evolute of a plane curve: the locus of its centres of curvature, `r + N / k`.
 *
 * @param r - A plane curve.
 * @param a - First parameter.
 * @param b - Last parameter.
 * @param n - Number of samples (default 100).
 * @returns Points of the evolute `[n][2]`.
 * @concept evolute
 * @example
 * ```ts
 * import { evolute } from 'mdml-engine/foundations'
 * import { cos, mul, sin, stack, type Value } from 'mdml-engine/tensor'
 * evolute((t: Value) => stack([mul(cos(t), 2), sin(t)], -1), 0, 0, 1) // => [[1.5, 0]]
 * ```
 */
export function evolute(r: ParametricCurve, a: number, b: number, n = 100): number[][] {
  return osculatingCircle(r, linspace(a, b, n)).centres
}

/** An implicit plane curve `F(x, y) = 0`, with `F` vectorised over points `[n, 2] -> [n]`. */
export type ImplicitCurve = (points: Value) => Value

/**
 * Trace an implicit curve `F(x, y) = 0` over a box by marching squares on a grid of F values (as matplotlib's
 * `contour`), returning polylines in (x, y).
 *
 * @param f - The implicit function, vectorised over points `[n, 2]`.
 * @param box - `[[xmin, xmax], [ymin, ymax]]`.
 * @param n - Grid points per axis (default 101).
 * @returns Polylines (closed ones repeat their first point).
 * @concept implicit-curves
 * @example
 * ```ts
 * import { traceImplicitCurve } from 'mdml-engine/foundations'
 * import { mul, sub, sum, type Value } from 'mdml-engine/tensor'
 * const circle = (p: Value) => sub(sum(mul(p, p), -1), 1)
 * traceImplicitCurve(circle, [[-2, 2], [-2, 2]], 41).length // => 1
 * ```
 */
export function traceImplicitCurve(
  f: ImplicitCurve,
  box: readonly [readonly [number, number], readonly [number, number]],
  n = 101,
): [number, number][][] {
  const xs = linspace(box[0][0], box[0][1], n)
  const ys = linspace(box[1][0], box[1][1], n)
  const [gx, gy] = meshgrid([xs, ys]) as [Value, Value]
  const z = f(stack([gx, gy], -1))
  return contourLines(z, 0, { x: toFlat(xs), y: toFlat(ys) })
}

/**
 * The tangent, normal and curvature of an implicit curve `F = 0` at a point on it, from the gradient and Hessian
 * (automatic differentiation): `k = (F_y^2 F_xx - 2 F_x F_y F_xy + F_x^2 F_yy) / |grad F|^3`, the normal along
 * `grad F`.
 *
 * @param f - The implicit function, vectorised over points `[n, 2] -> [n]`.
 * @param point - A point `[2]` on (or near) the curve.
 * @returns The unit tangent, unit normal and signed curvature.
 * @concept implicit-curves
 * @example
 * ```ts
 * import { implicitCurveGeometry } from 'mdml-engine/foundations'
 * import { mul, sub, sum, type Value } from 'mdml-engine/tensor'
 * implicitCurveGeometry((p: Value) => sub(sum(mul(p, p), -1), 4), [2, 0]).curvature // => 0.5
 * ```
 */
export function implicitCurveGeometry(
  f: ImplicitCurve,
  point: readonly [number, number],
): { readonly tangent: [number, number]; readonly normal: [number, number]; readonly curvature: number } {
  const scalar = (p: Value): Value => sum(f(stack([p])))
  const g = toFlat(grad(scalar)(promoteFloat([...point])[0] as Value) as Value)
  const h = toArray(hessian(scalar)([...point])) as number[][]
  const [fx = 0, fy = 0] = g
  const fxx = h[0]?.[0] ?? 0
  const fxy = h[0]?.[1] ?? 0
  const fyy = h[1]?.[1] ?? 0
  const len = Math.hypot(fx, fy)
  if (len === 0) throw new MdmlError('value', 'implicitCurveGeometry: the gradient vanishes (a singular point)')
  const k = (fy * fy * fxx - 2 * fx * fy * fxy + fx * fx * fyy) / len ** 3
  return { tangent: [-fy / len, fx / len], normal: [fx / len, fy / len], curvature: k }
}

/** The kinds of conic section, including the degenerate ones. */
export type ConicType =
  | 'circle'
  | 'ellipse'
  | 'parabola'
  | 'hyperbola'
  | 'imaginary ellipse'
  | 'point'
  | 'intersecting lines'
  | 'parallel lines'
  | 'coincident lines'
  | 'empty'

/** A classified conic `A x^2 + B xy + C y^2 + D x + E y + F = 0`. */
export interface Conic {
  readonly type: ConicType
  readonly coefficients: readonly [number, number, number, number, number, number]
  /** The centre (null for parabolas and degenerate conics without one). */
  readonly centre: [number, number] | null
  /** The vertex of a parabola, or the two vertices on the major or transverse axis. */
  readonly vertices: [number, number][]
  /** Semi-axes a >= b of an ellipse, or (transverse, conjugate) of a hyperbola; the focal parameter p of a parabola. */
  readonly semiAxes: readonly number[]
  /** The angle of the major (transverse, or symmetry) axis with the x-axis, in `(-pi/2, pi/2]`. */
  readonly axisAngle: number
  readonly foci: [number, number][]
  readonly eccentricity: number
  /** Directrices as `[a, b, c]` for the lines `a x + b y = c` (none for circles and degenerate conics). */
  readonly directrices: [number, number, number][]
}

const angleOf = (v: readonly number[]): number => {
  let t = Math.atan2(v[1] ?? 0, v[0] ?? 0)
  if (t <= -Math.PI / 2) t += Math.PI
  if (t > Math.PI / 2) t -= Math.PI
  return t
}

/**
 * Classify a conic section from its equation `A x^2 + B xy + C y^2 + D x + E y + F = 0` and find its geometry: the type
 * (from the discriminant `B^2 - 4AC` and the determinant of the 3x3 matrix), centre, axes and their angle, vertices,
 * foci, eccentricity and directrices. The quadratic part is diagonalised by a rotation (its eigenvectors), and the
 * linear part removed by completing the square.
 *
 * @param coefficients - `[A, B, C, D, E, F]`.
 * @param tol - Relative tolerance for zero discriminants and determinants (default 1e-12).
 * @returns The conic.
 * @throws MdmlError with code `value` when A, B and C are all zero (a line, not a conic).
 * @concept conic-sections
 * @example
 * ```ts
 * import { classifyConic } from 'mdml-engine/foundations'
 * const c = classifyConic([1 / 25, 0, 1 / 9, 0, 0, -1])
 * c.type // => 'ellipse'
 * c.foci // => [[-4, 0], [4, 0]]
 * c.eccentricity // => 0.8
 * classifyConic([0, 0, 1, -4, 0, 0]).foci // => [[1, 0]]
 * ```
 */
export function classifyConic(
  coefficients: readonly [number, number, number, number, number, number],
  tol = 1e-12,
): Conic {
  const [A, B, C, D, E, F] = coefficients
  const scale = maxOf(coefficients.map(Math.abs))
  if (Math.max(Math.abs(A), Math.abs(B), Math.abs(C)) <= tol * scale) {
    throw new MdmlError('value', 'classifyConic: A, B and C are all zero')
  }
  const delta = A * C - (B * B) / 4
  const big =
    A * (C * F - (E * E) / 4) -
    (B / 2) * ((B / 2) * F - (E / 2) * (D / 2)) +
    (D / 2) * ((B / 2) * (E / 2) - C * (D / 2))
  const q = Math.max(Math.abs(A), Math.abs(B), Math.abs(C))
  const deltaZero = Math.abs(delta) <= tol * q * q
  const bigZero = Math.abs(big) <= tol * q * scale * scale
  // Rotate to the principal axes of the quadratic part (eigenvalues ascending).
  const { eigenvalues, eigenvectors } = eigh([
    [A, B / 2],
    [B / 2, C],
  ])
  const [l1 = 0, l2 = 0] = toFlat(eigenvalues)
  const vecs = toArray(eigenvectors) as number[][]
  const e1: [number, number] = [vecs[0]?.[0] ?? 0, vecs[1]?.[0] ?? 0]
  const e2: [number, number] = [vecs[0]?.[1] ?? 0, vecs[1]?.[1] ?? 0]
  const base = { coefficients: [A, B, C, D, E, F] as const }
  const at = (c: readonly [number, number], dir: readonly [number, number], s: number): [number, number] => [
    c[0] + s * dir[0],
    c[1] + s * dir[1],
  ]
  const directrix = (dir: readonly [number, number], point: readonly [number, number]): [number, number, number] => [
    dir[0],
    dir[1],
    dir[0] * point[0] + dir[1] * point[1],
  ]
  if (!deltaZero) {
    const x0 = (B * E - 2 * C * D) / (4 * delta)
    const y0 = (B * D - 2 * A * E) / (4 * delta)
    const centre: [number, number] = [x0, y0]
    const f0 = F + (D * x0 + E * y0) / 2
    const empty = { vertices: [], semiAxes: [], foci: [], directrices: [], eccentricity: Number.NaN }
    if (delta > 0) {
      // Same-sign eigenvalues: an ellipse if f0 has the opposite sign, a point if zero, nothing otherwise.
      if (bigZero) return { ...base, ...empty, type: 'point', centre, axisAngle: 0, vertices: [centre] }
      if (f0 * l1 > 0) return { ...base, ...empty, type: 'imaginary ellipse', centre, axisAngle: angleOf(e1) }
      // The smaller |eigenvalue| belongs to the major axis.
      const [lMajor, lMinor, dir] = Math.abs(l1) <= Math.abs(l2) ? [l1, l2, e1] : [l2, l1, e2]
      const a = Math.sqrt(-f0 / lMajor)
      const b = Math.sqrt(-f0 / lMinor)
      const circle = Math.abs(a - b) <= 1e-12 * a
      const c = circle ? 0 : Math.sqrt(a * a - b * b)
      const e = circle ? 0 : c / a
      return {
        ...base,
        type: circle ? 'circle' : 'ellipse',
        centre,
        vertices: [at(centre, dir, -a), at(centre, dir, a)],
        semiAxes: [a, b],
        axisAngle: circle ? 0 : angleOf(dir),
        foci: circle ? [centre] : [at(centre, dir, -c), at(centre, dir, c)],
        eccentricity: e,
        directrices: circle ? [] : [directrix(dir, at(centre, dir, -a / e)), directrix(dir, at(centre, dir, a / e))],
      }
    }
    if (bigZero) return { ...base, ...empty, type: 'intersecting lines', centre, axisAngle: 0 }
    // Hyperbola: the transverse axis is the eigenvector whose eigenvalue has the sign opposite to f0.
    const [lt, lc, dir] = -f0 / l1 > 0 ? [l1, l2, e1] : [l2, l1, e2]
    const a = Math.sqrt(-f0 / lt)
    const b = Math.sqrt(f0 / lc)
    const c = Math.hypot(a, b)
    const e = c / a
    return {
      ...base,
      type: 'hyperbola',
      centre,
      vertices: [at(centre, dir, -a), at(centre, dir, a)],
      semiAxes: [a, b],
      axisAngle: angleOf(dir),
      foci: [at(centre, dir, -c), at(centre, dir, c)],
      eccentricity: e,
      directrices: [directrix(dir, at(centre, dir, -a / e)), directrix(dir, at(centre, dir, a / e))],
    }
  }
  // Parabola or parallel lines: one eigenvalue is zero; u runs along its eigenvector (the axis), v across.
  const [lam, axis, across] = Math.abs(l1) > Math.abs(l2) ? [l1, e2, e1] : [l2, e1, e2]
  const dU = D * axis[0] + E * axis[1]
  const dV = D * across[0] + E * across[1]
  const v0 = -dV / (2 * lam)
  if (Math.abs(dU) <= tol * scale) {
    // lam (v - v0)^2 = rest: two parallel lines, one double line, or nothing.
    const rest = (dV * dV) / (4 * lam) - F
    const type: ConicType =
      Math.abs(rest) <= tol * scale ? 'coincident lines' : rest / lam > 0 ? 'parallel lines' : 'empty'
    return {
      ...base,
      type,
      centre: null,
      vertices: [],
      semiAxes: [],
      axisAngle: angleOf(axis),
      foci: [],
      eccentricity: Number.NaN,
      directrices: [],
    }
  }
  const u0 = ((dV * dV) / (4 * lam) - F) / dU
  // (v - v0)^2 = 4 p (u - u0)
  const p = -dU / (4 * lam)
  const toXY = (u: number, v: number): [number, number] => [u * axis[0] + v * across[0], u * axis[1] + v * across[1]]
  const vertex = toXY(u0, v0)
  const focus = toXY(u0 + p, v0)
  const onDirectrix = toXY(u0 - p, v0)
  // Orient the axis from the vertex towards the focus.
  const sign = p >= 0 ? 1 : -1
  const dir: [number, number] = [sign * axis[0], sign * axis[1]]
  return {
    ...base,
    type: 'parabola',
    centre: null,
    vertices: [vertex],
    semiAxes: [Math.abs(p)],
    axisAngle: angleOf(dir),
    foci: [focus],
    eccentricity: 1,
    directrices: [directrix(dir, onDirectrix)],
  }
}

/**
 * Points along a classified conic for drawing: the whole ellipse or circle, both branches of a hyperbola, or a
 * parabola around its vertex.
 *
 * @param conic - A conic from {@link classifyConic}.
 * @param n - Points per branch (default 100).
 * @param extent - How far the open conics run: the hyperbolic parameter range, or the distance along a parabola's
 *   axis (default 2).
 * @returns Branches of points (empty for degenerate conics).
 * @concept conic-sections
 * @example
 * ```ts
 * import { classifyConic, conicPoints } from 'mdml-engine/foundations'
 * conicPoints(classifyConic([1, 0, -1, 0, 0, -1]), 5).length // => 2
 * ```
 */
export function conicPoints(conic: Conic, n = 100, extent = 2): [number, number][][] {
  const c = Math.cos(conic.axisAngle)
  const s = Math.sin(conic.axisAngle)
  const place = (origin: readonly [number, number], u: number, v: number): [number, number] => [
    origin[0] + u * c - v * s,
    origin[1] + u * s + v * c,
  ]
  const ts = (lo: number, hi: number): number[] => toFlat(linspace(lo, hi, n))
  const [a = 0, b = 0] = conic.semiAxes
  if ((conic.type === 'ellipse' || conic.type === 'circle') && conic.centre !== null) {
    const centre = conic.centre
    return [ts(0, 2 * Math.PI).map((t) => place(centre, a * Math.cos(t), b * Math.sin(t)))]
  }
  if (conic.type === 'hyperbola' && conic.centre !== null) {
    const centre = conic.centre
    return [1, -1].map((side) =>
      ts(-extent, extent).map((t) => place(centre, side * a * Math.cosh(t), b * Math.sinh(t))),
    )
  }
  if (conic.type === 'parabola') {
    const vertex = conic.vertices[0] ?? [0, 0]
    const p = a
    // u = v^2 / (4p) along the axis.
    const vMax = Math.sqrt(4 * p * extent)
    return [ts(-vMax, vMax).map((v) => place(vertex, (v * v) / (4 * p), v))]
  }
  return []
}

/**
 * The Bernstein basis polynomial `b_{i,n}(t) = C(n, i) t^i (1 - t)^(n - i)`. Differentiable in t.
 *
 * @param n - The degree.
 * @param i - The index, `0 <= i <= n`.
 * @param t - Parameters.
 * @returns The values.
 * @concept bezier-curves
 * @example
 * ```ts
 * import { bernstein } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(bernstein(2, 1, [0, 0.5, 1])) // => [0, 0.5, 0]
 * ```
 */
export function bernstein(n: number, i: number, t: TensorLike): Value {
  if (!(Number.isInteger(n) && Number.isInteger(i) && i >= 0 && i <= n)) {
    throw new MdmlError('value', 'bernstein: need integers 0 <= i <= n')
  }
  const [tv] = promoteFloat(t) as [Value]
  let coef = 1
  for (let k = 1; k <= i; k++) coef = (coef * (n - i + k)) / k
  let out: Value = mul(onesLike(tv), coef)
  for (let k = 0; k < i; k++) out = mul(out, tv)
  for (let k = 0; k < n - i; k++) out = mul(out, sub(1, tv))
  return out
}

/** The steps of de Casteljau's algorithm at one parameter. */
export interface DeCasteljau {
  readonly point: number[]
  /** `levels[0]` is the control polygon; each next level interpolates the previous at t; the last is the point. */
  readonly levels: number[][][]
}

/**
 * De Casteljau's algorithm with its steps: repeatedly interpolate neighbouring control points at `t` until one point
 * remains. The first and last points of the levels are the control points of the two halves (see {@link bezierSplit}).
 *
 * @param control - Control points `[k, d]`.
 * @param t - The parameter.
 * @returns The point and every level of the construction.
 * @concept bezier-curves
 * @example
 * ```ts
 * import { deCasteljau } from 'mdml-engine/foundations'
 * deCasteljau([[0, 0], [1, 2], [2, 0]], 0.5).point // => [1, 1]
 * ```
 */
export function deCasteljau(control: readonly (readonly number[])[], t: number): DeCasteljau {
  if (control.length === 0) throw new MdmlError('value', 'deCasteljau: no control points')
  const levels: number[][][] = [control.map((p) => [...p])]
  let current = levels[0] as number[][]
  while (current.length > 1) {
    const next: number[][] = []
    for (let i = 0; i + 1 < current.length; i++) {
      const a = current[i] as number[]
      const b = current[i + 1] as number[]
      next.push(a.map((x, j) => (1 - t) * x + t * (b[j] ?? 0)))
    }
    levels.push(next)
    current = next
  }
  return { point: current[0] ?? [], levels }
}

/**
 * Points of a Bezier curve at parameters, by de Casteljau's algorithm on values (differentiable in the control points
 * and the parameters).
 *
 * @param control - Control points `[k, d]`.
 * @param t - Parameters `[n]`.
 * @returns Points `[n, d]`.
 * @concept bezier-curves
 * @example
 * ```ts
 * import { bezier } from 'mdml-engine/foundations'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(bezier([[0, 0], [1, 2], [2, 0]], [0, 0.5, 1])) // => [[0, 0], [1, 1], [2, 0]]
 * ```
 */
export function bezier(control: TensorLike, t: TensorLike): Value {
  const [c, tv0] = promoteFloat(control, t) as [Value, Value]
  const k = c.shape[0] ?? 0
  if (c.shape.length !== 2 || k === 0) throw new MdmlError('shape', 'bezier: control points [k, d]')
  const tv = keep(params(tv0))
  let level = Array.from({ length: k }, (_, i) => squeeze(sliceAxis(c, 0, i, i + 1), 0))
  while (level.length > 1) {
    const next: Value[] = []
    for (let i = 0; i + 1 < level.length; i++) {
      next.push(add(mul(sub(1, tv), level[i] as Value), mul(tv, level[i + 1] as Value)))
    }
    level = next
  }
  return add(mul(tv, 0), level[0] as Value)
}

/**
 * Split a Bezier curve at `t` into two Bezier curves of the same degree that trace the two pieces (the outer points
 * of de Casteljau's levels).
 *
 * @param control - Control points `[k][d]`.
 * @param t - Where to split.
 * @returns The control points of the pieces on `[0, t]` and `[t, 1]`.
 * @concept bezier-curves
 * @example
 * ```ts
 * import { bezierSplit } from 'mdml-engine/foundations'
 * bezierSplit([[0, 0], [2, 2], [4, 0]], 0.5).left // => [[0, 0], [1, 1], [2, 1]]
 * ```
 */
export function bezierSplit(
  control: readonly (readonly number[])[],
  t: number,
): { readonly left: number[][]; readonly right: number[][] } {
  const { levels } = deCasteljau(control, t)
  return {
    left: levels.map((l) => l[0] as number[]),
    right: levels.map((l) => l[l.length - 1] as number[]).reverse(),
  }
}

/**
 * The control points of the derivative (hodograph) of a Bezier curve of degree n: `n (P_{i+1} - P_i)`, a Bezier curve
 * of degree n - 1.
 *
 * @param control - Control points `[k][d]`, k >= 2.
 * @returns The derivative's control points `[k - 1][d]`.
 * @concept bezier-curves
 * @example
 * ```ts
 * import { bezierDerivative } from 'mdml-engine/foundations'
 * bezierDerivative([[0, 0], [1, 2], [2, 0]]) // => [[2, 4], [2, -4]]
 * ```
 */
export function bezierDerivative(control: readonly (readonly number[])[]): number[][] {
  const n = control.length - 1
  if (n < 1) throw new MdmlError('value', 'bezierDerivative: need at least two control points')
  return control.slice(1).map((p, i) => p.map((x, j) => n * (x - (control[i]?.[j] ?? 0))))
}

/**
 * Raise the degree of a Bezier curve by one without changing its shape:
 * `Q_i = (i / (n + 1)) P_{i-1} + (1 - i / (n + 1)) P_i`.
 *
 * @param control - Control points `[k][d]`.
 * @returns The `k + 1` control points of the same curve.
 * @concept bezier-curves
 * @example
 * ```ts
 * import { bezierElevate } from 'mdml-engine/foundations'
 * bezierElevate([[0, 0], [3, 3]]) // => [[0, 0], [1.5, 1.5], [3, 3]]
 * ```
 */
export function bezierElevate(control: readonly (readonly number[])[]): number[][] {
  const n = control.length - 1
  const out: number[][] = []
  for (let i = 0; i <= n + 1; i++) {
    const a = i / (n + 1)
    const prev = control[i - 1]
    const cur = control[i]
    const d = (cur ?? prev ?? []).length
    out.push(Array.from({ length: d }, (_, j) => a * (prev?.[j] ?? 0) + (1 - a) * (cur?.[j] ?? 0)))
  }
  return out
}
