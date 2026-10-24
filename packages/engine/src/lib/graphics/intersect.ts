import { WhiteboxError } from '../base/index.ts'
import { crossProduct, dotProduct } from '../foundations/index.ts'
import type { Vec3 } from '../scene/index.ts'
import {
  div,
  greater,
  maximum,
  minimum,
  mul,
  neg,
  promoteFloat,
  sliceAxis,
  sqrt,
  square,
  squeeze,
  stack,
  sub,
  type TensorLike,
  type Value,
  where,
} from '../tensor/index.ts'
import { basis3, check3, cross3, dot3, madd3, normalise3, scale3, sub3 } from './_vec3.ts'
import type { Ray } from './camera.ts'
import { sphereUv } from './textures.ts'

/**
 * A surface a ray can hit: a sphere, an infinite plane (through `point`, facing `normal`), a triangle (optionally with
 * per-vertex normals for smooth shading and texture coordinates), or an axis-aligned box.
 *
 * @concept ray-intersection
 */
export type Shape =
  | { readonly kind: 'sphere'; readonly centre: Vec3; readonly radius: number }
  | { readonly kind: 'plane'; readonly point: Vec3; readonly normal: Vec3 }
  | {
      readonly kind: 'triangle'
      readonly a: Vec3
      readonly b: Vec3
      readonly c: Vec3
      readonly normals?: readonly [Vec3, Vec3, Vec3]
      readonly uvs?: readonly [readonly [number, number], readonly [number, number], readonly [number, number]]
    }
  | { readonly kind: 'box'; readonly min: Vec3; readonly max: Vec3 }

/**
 * A sphere.
 *
 * @param centre - Its centre.
 * @param radius - Its radius (positive).
 * @returns The shape.
 * @throws WhiteboxError with code `value` for a non-positive radius.
 * @concept ray-intersection
 * @example
 * ```ts
 * import { sphere } from 'whitebox-engine/graphics'
 * sphere([0, 0, 0], 2).radius // => 2
 * ```
 */
export function sphere(centre: Vec3, radius: number): Shape {
  if (!(radius > 0)) throw new WhiteboxError('value', 'sphere: the radius must be positive')
  return { kind: 'sphere', centre: check3(centre, 'centre'), radius }
}

/**
 * An infinite plane.
 *
 * @param point - A point on it.
 * @param normal - Its normal (normalised here).
 * @returns The shape.
 * @concept ray-intersection
 * @example
 * ```ts
 * import { plane } from 'whitebox-engine/graphics'
 * plane([0, 0, 0], [0, 3, 0]).normal // => [0, 1, 0]
 * ```
 */
export function plane(point: Vec3, normal: Vec3): Shape {
  return { kind: 'plane', point: check3(point, 'point'), normal: normalise3(check3(normal, 'normal'), 'normal') }
}

/**
 * A triangle with vertices in counter-clockwise order seen from its front (its normal is `(b - a) x (c - a)`).
 *
 * @param a - First vertex.
 * @param b - Second vertex.
 * @param c - Third vertex.
 * @param options - Per-vertex normals (interpolated for smooth shading) and texture coordinates.
 * @returns The shape.
 * @throws WhiteboxError with code `value` for a degenerate triangle.
 * @concept ray-intersection
 * @example
 * ```ts
 * import { triangle } from 'whitebox-engine/graphics'
 * triangle([0, 0, 0], [1, 0, 0], [0, 1, 0]).kind // => 'triangle'
 * ```
 */
export function triangle(
  a: Vec3,
  b: Vec3,
  c: Vec3,
  options: {
    readonly normals?: readonly [Vec3, Vec3, Vec3]
    readonly uvs?: readonly [readonly [number, number], readonly [number, number], readonly [number, number]]
  } = {},
): Shape {
  const [pa, pb, pc] = [check3(a, 'a'), check3(b, 'b'), check3(c, 'c')]
  normalise3(cross3(sub3(pb, pa), sub3(pc, pa)), 'the triangle normal (the vertices must not be collinear)')
  return { kind: 'triangle', a: pa, b: pb, c: pc, ...options }
}

/**
 * An axis-aligned box.
 *
 * @param min - The minimum corner.
 * @param max - The maximum corner (at least `min` on every axis).
 * @returns The shape.
 * @throws WhiteboxError with code `value` when `max < min` on an axis.
 * @concept ray-intersection
 * @example
 * ```ts
 * import { box } from 'whitebox-engine/graphics'
 * box([0, 0, 0], [1, 2, 3]).max // => [1, 2, 3]
 * ```
 */
export function box(min: Vec3, max: Vec3): Shape {
  const lo = check3(min, 'min')
  const hi = check3(max, 'max')
  if (!(hi[0] >= lo[0] && hi[1] >= lo[1] && hi[2] >= lo[2])) throw new WhiteboxError('value', 'box: need max >= min')
  return { kind: 'box', min: lo, max: hi }
}

/** Where a ray meets a surface. */
export interface Hit {
  /** The ray parameter: the point is `origin + t direction`. */
  readonly t: number
  readonly point: Vec3
  /** The outward unit normal (interpolated for triangles with vertex normals). */
  readonly normal: Vec3
  /** Whether the ray arrives from the outside (against the geometric normal). */
  readonly frontFace: boolean
  /** Surface coordinates for texturing. */
  readonly uv: readonly [number, number]
}

/** The default smallest ray parameter, which keeps secondary rays from hitting the surface they leave. */
export const RAY_EPSILON = 1e-9

/**
 * Intersect a ray with a sphere: the nearest root of `|o + t d - c|^2 = r^2` in `[tMin, tMax]` (the far root when the
 * origin is inside). Texture coordinates are spherical.
 *
 * @param ray - The ray.
 * @param s - A sphere.
 * @param tMin - Smallest accepted parameter (default {@link RAY_EPSILON}).
 * @param tMax - Largest accepted parameter (default Infinity).
 * @returns The hit, or null.
 * @concept ray-intersection
 * @example
 * ```ts
 * import { intersectSphere, sphere } from 'whitebox-engine/graphics'
 * const hit = intersectSphere({ origin: [0, 0, 5], direction: [0, 0, -1] }, sphere([0, 0, 0], 1))
 * [hit?.t, hit?.normal, hit?.frontFace] // => [4, [0, 0, 1], true]
 * ```
 */
export function intersectSphere(
  ray: Ray,
  s: Shape,
  tMin: number = RAY_EPSILON,
  tMax: number = Number.POSITIVE_INFINITY,
): Hit | null {
  if (s.kind !== 'sphere') throw new WhiteboxError('value', 'intersectSphere: not a sphere')
  const oc = sub3(ray.origin, s.centre)
  const a = dot3(ray.direction, ray.direction)
  const b = dot3(oc, ray.direction)
  const c = dot3(oc, oc) - s.radius * s.radius
  const disc = b * b - a * c
  if (disc < 0) return null
  const sq = Math.sqrt(disc)
  // The stable quadratic: q avoids cancellation, and the roots are q / a and c / q.
  const q = b >= 0 ? -b - sq : -b + sq
  const r1 = q / a
  const r2 = q !== 0 ? c / q : r1
  const near = Math.min(r1, r2)
  const far = Math.max(r1, r2)
  const t = near >= tMin ? near : far
  if (!(t >= tMin && t <= tMax)) return null
  const point = madd3(ray.origin, ray.direction, t)
  const normal = scale3(sub3(point, s.centre), 1 / s.radius)
  return { t, point, normal, frontFace: dot3(ray.direction, normal) < 0, uv: sphereUv(normal) }
}

/**
 * Intersect a ray with an infinite plane: `t = (p - o) . n / (d . n)`; rays parallel to the plane miss. Texture
 * coordinates are the point's coordinates in a tangent frame of the plane, from its `point`.
 *
 * @param ray - The ray.
 * @param s - A plane.
 * @param tMin - Smallest accepted parameter.
 * @param tMax - Largest accepted parameter.
 * @returns The hit, or null.
 * @concept ray-intersection
 * @example
 * ```ts
 * import { intersectPlane, plane } from 'whitebox-engine/graphics'
 * intersectPlane({ origin: [0, 2, 0], direction: [0, -1, 0] }, plane([0, 0, 0], [0, 1, 0]))?.t // => 2
 * ```
 */
export function intersectPlane(
  ray: Ray,
  s: Shape,
  tMin: number = RAY_EPSILON,
  tMax: number = Number.POSITIVE_INFINITY,
): Hit | null {
  if (s.kind !== 'plane') throw new WhiteboxError('value', 'intersectPlane: not a plane')
  const denom = dot3(s.normal, ray.direction)
  if (Math.abs(denom) < 1e-15) return null
  const t = dot3(sub3(s.point, ray.origin), s.normal) / denom
  if (!(t >= tMin && t <= tMax)) return null
  const point = madd3(ray.origin, ray.direction, t)
  const [tu, tv] = basis3(s.normal)
  const rel = sub3(point, s.point)
  return { t, point, normal: s.normal, frontFace: denom < 0, uv: [dot3(rel, tu), dot3(rel, tv)] }
}

/**
 * Intersect a ray with a triangle by the Moller-Trumbore algorithm: solve `o + t d = a + u (b - a) + v (c - a)` with
 * Cramer's rule, accepting `u, v >= 0` and `u + v <= 1`. With `cull`, triangles seen from behind are missed.
 * Normals are interpolated from vertex normals when given; texture coordinates likewise (else `(u, v)`).
 *
 * @param ray - The ray.
 * @param s - A triangle.
 * @param tMin - Smallest accepted parameter.
 * @param tMax - Largest accepted parameter.
 * @param cull - Miss back faces (default false).
 * @returns The hit, or null.
 * @concept ray-intersection
 * @example
 * ```ts
 * import { intersectTriangle, triangle } from 'whitebox-engine/graphics'
 * const hit = intersectTriangle({ origin: [0.25, 0.25, 1], direction: [0, 0, -1] }, triangle([0, 0, 0], [1, 0, 0], [0, 1, 0]))
 * [hit?.t, hit?.uv] // => [1, [0.25, 0.25]]
 * ```
 */
export function intersectTriangle(
  ray: Ray,
  s: Shape,
  tMin: number = RAY_EPSILON,
  tMax: number = Number.POSITIVE_INFINITY,
  cull: boolean = false,
): Hit | null {
  if (s.kind !== 'triangle') throw new WhiteboxError('value', 'intersectTriangle: not a triangle')
  const e1 = sub3(s.b, s.a)
  const e2 = sub3(s.c, s.a)
  const p = cross3(ray.direction, e2)
  const det = dot3(e1, p)
  if (cull ? det < 1e-15 : Math.abs(det) < 1e-15) return null
  const inv = 1 / det
  const so = sub3(ray.origin, s.a)
  const u = dot3(so, p) * inv
  if (u < 0 || u > 1) return null
  const q = cross3(so, e1)
  const v = dot3(ray.direction, q) * inv
  if (v < 0 || u + v > 1) return null
  const t = dot3(e2, q) * inv
  if (!(t >= tMin && t <= tMax)) return null
  const geometric = normalise3(cross3(e1, e2))
  const w = 1 - u - v
  const normal =
    s.normals === undefined
      ? geometric
      : normalise3(madd3(madd3(scale3(s.normals[0], w), s.normals[1], u), s.normals[2], v))
  const uv: [number, number] =
    s.uvs === undefined
      ? [u, v]
      : [w * s.uvs[0][0] + u * s.uvs[1][0] + v * s.uvs[2][0], w * s.uvs[0][1] + u * s.uvs[1][1] + v * s.uvs[2][1]]
  return { t, point: madd3(ray.origin, ray.direction, t), normal, frontFace: det > 0, uv }
}

/**
 * The parameter interval where a ray is inside an axis-aligned box (the slab method), as `[tNear, tFar, axis]` with
 * `axis` the slab entered last, or null when the ray misses.
 *
 * @param origin - Ray origin.
 * @param direction - Ray direction.
 * @param min - Box minimum.
 * @param max - Box maximum.
 * @returns The interval and entering axis, or null.
 * @concept ray-intersection
 * @example
 * ```ts
 * import { slabs } from 'whitebox-engine/graphics'
 * slabs([-2, 0.5, 0.5], [1, 0, 0], [0, 0, 0], [1, 1, 1]) // => [2, 3, 0]
 * ```
 */
export function slabs(origin: Vec3, direction: Vec3, min: Vec3, max: Vec3): [number, number, number] | null {
  let near = Number.NEGATIVE_INFINITY
  let far = Number.POSITIVE_INFINITY
  let axis = 0
  for (let k = 0; k < 3; k++) {
    const o = origin[k] ?? 0
    const d = direction[k] ?? 0
    const lo = min[k] ?? 0
    const hi = max[k] ?? 0
    if (d === 0) {
      if (o < lo || o > hi) return null
      continue
    }
    const t1 = (lo - o) / d
    const t2 = (hi - o) / d
    const a = Math.min(t1, t2)
    const b = Math.max(t1, t2)
    if (a > near) {
      near = a
      axis = k
    }
    if (b < far) far = b
    if (near > far) return null
  }
  return [near, far, axis]
}

/**
 * Intersect a ray with an axis-aligned box by the slab method; from inside, the exit point is the hit (back face).
 * Texture coordinates are planar on the face hit.
 *
 * @param ray - The ray.
 * @param s - A box.
 * @param tMin - Smallest accepted parameter.
 * @param tMax - Largest accepted parameter.
 * @returns The hit, or null.
 * @concept ray-intersection
 * @example
 * ```ts
 * import { box, intersectBox } from 'whitebox-engine/graphics'
 * const hit = intersectBox({ origin: [-2, 0.5, 0.5], direction: [1, 0, 0] }, box([0, 0, 0], [1, 1, 1]))
 * [hit?.t, hit?.normal] // => [2, [-1, 0, 0]]
 * ```
 */
export function intersectBox(
  ray: Ray,
  s: Shape,
  tMin: number = RAY_EPSILON,
  tMax: number = Number.POSITIVE_INFINITY,
): Hit | null {
  if (s.kind !== 'box') throw new WhiteboxError('value', 'intersectBox: not a box')
  const span = slabs(ray.origin, ray.direction, s.min, s.max)
  if (span === null) return null
  const [near, far, nearAxis] = span
  const inside = near < tMin
  const t = inside ? far : near
  if (!(t >= tMin && t <= tMax && Number.isFinite(t))) return null
  const point = madd3(ray.origin, ray.direction, t)
  // The face hit: the entering slab from outside; from inside, the axis whose face the exit point lies on.
  let axis = nearAxis
  if (inside) {
    let best = Number.POSITIVE_INFINITY
    for (let k = 0; k < 3; k++) {
      const gap = Math.min(Math.abs((point[k] ?? 0) - (s.min[k] ?? 0)), Math.abs((point[k] ?? 0) - (s.max[k] ?? 0)))
      if (gap < best) {
        best = gap
        axis = k
      }
    }
  }
  const mid = ((s.min[axis] ?? 0) + (s.max[axis] ?? 0)) / 2
  const sign = (point[axis] ?? 0) >= mid ? 1 : -1
  const normal: Vec3 = [axis === 0 ? sign : 0, axis === 1 ? sign : 0, axis === 2 ? sign : 0]
  const [i, j] = axis === 0 ? [1, 2] : axis === 1 ? [0, 2] : [0, 1]
  const size = (k: number): number => (s.max[k] ?? 0) - (s.min[k] ?? 0) || 1
  const uv: [number, number] = [
    ((point[i] ?? 0) - (s.min[i] ?? 0)) / size(i),
    ((point[j] ?? 0) - (s.min[j] ?? 0)) / size(j),
  ]
  return { t, point, normal, frontFace: !inside, uv }
}

/**
 * Intersect a ray with any shape.
 *
 * @param ray - The ray.
 * @param s - The shape.
 * @param tMin - Smallest accepted parameter.
 * @param tMax - Largest accepted parameter.
 * @returns The hit, or null.
 * @concept ray-intersection
 * @example
 * ```ts
 * import { intersect, sphere } from 'whitebox-engine/graphics'
 * intersect({ origin: [0, 0, 0], direction: [1, 0, 0] }, sphere([0, 0, 0], 2))?.frontFace // => false
 * ```
 */
export function intersect(
  ray: Ray,
  s: Shape,
  tMin: number = RAY_EPSILON,
  tMax: number = Number.POSITIVE_INFINITY,
): Hit | null {
  switch (s.kind) {
    case 'sphere':
      return intersectSphere(ray, s, tMin, tMax)
    case 'plane':
      return intersectPlane(ray, s, tMin, tMax)
    case 'triangle':
      return intersectTriangle(ray, s, tMin, tMax)
    case 'box':
      return intersectBox(ray, s, tMin, tMax)
  }
}

/**
 * The nearest hit of a ray among shapes by testing every one (the brute-force reference for bounding-volume
 * hierarchies), with the index of the shape hit; of shapes hit at the same parameter, the lowest index.
 *
 * @param ray - The ray.
 * @param shapes - The shapes.
 * @param tMin - Smallest accepted parameter.
 * @param tMax - Largest accepted parameter.
 * @returns The nearest hit and its shape index, or null.
 * @concept ray-intersection
 * @example
 * ```ts
 * import { nearestHit, sphere } from 'whitebox-engine/graphics'
 * nearestHit({ origin: [0, 0, 5], direction: [0, 0, -1] }, [sphere([0, 0, 0], 1), sphere([0, 0, 2], 1)])?.index // => 1
 * ```
 */
export function nearestHit(
  ray: Ray,
  shapes: readonly Shape[],
  tMin: number = RAY_EPSILON,
  tMax: number = Number.POSITIVE_INFINITY,
): (Hit & { readonly index: number }) | null {
  let best: (Hit & { readonly index: number }) | null = null
  let limit = tMax
  for (const [index, s] of shapes.entries()) {
    const h = intersect(ray, s, tMin, limit)
    if (h !== null && (best === null || h.t < best.t)) {
      best = { ...h, index }
      limit = h.t
    }
  }
  return best
}

/**
 * The axis-aligned bounding box of a bounded shape.
 *
 * @param s - A sphere, triangle or box.
 * @returns `{ min, max }`.
 * @throws WhiteboxError with code `value` for a plane (unbounded).
 * @concept bounding-volume-hierarchy
 * @example
 * ```ts
 * import { bounds, sphere } from 'whitebox-engine/graphics'
 * bounds(sphere([1, 1, 1], 1)) // => { min: [0, 0, 0], max: [2, 2, 2] }
 * ```
 */
export function bounds(s: Shape): { readonly min: Vec3; readonly max: Vec3 } {
  switch (s.kind) {
    case 'sphere':
      return {
        min: [s.centre[0] - s.radius, s.centre[1] - s.radius, s.centre[2] - s.radius],
        max: [s.centre[0] + s.radius, s.centre[1] + s.radius, s.centre[2] + s.radius],
      }
    case 'triangle':
      return {
        min: [Math.min(s.a[0], s.b[0], s.c[0]), Math.min(s.a[1], s.b[1], s.c[1]), Math.min(s.a[2], s.b[2], s.c[2])],
        max: [Math.max(s.a[0], s.b[0], s.c[0]), Math.max(s.a[1], s.b[1], s.c[1]), Math.max(s.a[2], s.b[2], s.c[2])],
      }
    case 'box':
      return { min: s.min, max: s.max }
    case 'plane':
      throw new WhiteboxError('value', 'bounds: a plane is unbounded')
  }
}

// ------------------------------------------------------------------------------- differentiable intersection distances

const col = (v: Value, k: number): Value => squeeze(sliceAxis(v, -1, k, k + 1), -1)

/**
 * The distance along rays to a sphere, as a differentiable function of the ray and the sphere: the smaller root
 * `t = (-b - sqrt(b^2 - a c)) / a` of the quadratic, or the larger one when the smaller is not positive (the origin is
 * inside). NaN where the ray misses; negative where the sphere is behind.
 *
 * @param origin - Ray origins `[..., 3]`.
 * @param direction - Ray directions `[..., 3]` (any length; `t` is in their units).
 * @param centre - Sphere centres `[..., 3]`.
 * @param radius - Radii `[...]`.
 * @returns Distances `[...]`.
 * @concept ray-intersection
 * @example
 * ```ts
 * import { sphereDistance } from 'whitebox-engine/graphics'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(sphereDistance([[0, 0, 5], [0, 0, 0]], [0, 0, -1], [0, 0, 0], 1)) // => [4, 1]
 * ```
 */
export function sphereDistance(
  origin: TensorLike,
  direction: TensorLike,
  centre: TensorLike,
  radius: TensorLike,
): Value {
  const [o, d, c, r] = promoteFloat(origin, direction, centre, radius) as [Value, Value, Value, Value]
  const oc = sub(o, c)
  const a = dotProduct(d, d)
  const b = dotProduct(oc, d)
  const cc = sub(dotProduct(oc, oc), square(r))
  const sq = sqrt(sub(square(b), mul(a, cc)))
  const t1 = div(sub(neg(b), sq), a)
  const t2 = div(sub(sq, b), a)
  return where(greater(t1, 0), t1, t2)
}

/**
 * The distance along rays to planes, `t = (p - o) . n / (d . n)`, differentiable in every argument (infinite or NaN
 * for rays parallel to the plane).
 *
 * @param origin - Ray origins `[..., 3]`.
 * @param direction - Ray directions `[..., 3]`.
 * @param point - Points on the planes `[..., 3]`.
 * @param normal - Plane normals `[..., 3]`.
 * @returns Distances `[...]`.
 * @concept ray-intersection
 * @example
 * ```ts
 * import { planeDistance } from 'whitebox-engine/graphics'
 * import { item } from 'whitebox-engine/tensor'
 * item(planeDistance([0, 3, 0], [0, -2, 0], [0, 1, 0], [0, 1, 0])) // => 1
 * ```
 */
export function planeDistance(origin: TensorLike, direction: TensorLike, point: TensorLike, normal: TensorLike): Value {
  const [o, d, p, n] = promoteFloat(origin, direction, point, normal) as [Value, Value, Value, Value]
  return div(dotProduct(sub(p, o), n), dotProduct(d, n))
}

/**
 * Moller-Trumbore as a differentiable function: the ray parameter and barycentric coordinates `[t, u, v]` of the
 * point where rays meet the planes of triangles, `o + t d = a + u (b - a) + v (c - a)`. The ray hits the triangle
 * itself where `u, v >= 0`, `u + v <= 1` and `t > 0`.
 *
 * @param origin - Ray origins `[..., 3]`.
 * @param direction - Ray directions `[..., 3]`.
 * @param a - First vertices `[..., 3]`.
 * @param b - Second vertices `[..., 3]`.
 * @param c - Third vertices `[..., 3]`.
 * @returns `[t, u, v]` per ray `[..., 3]`.
 * @concept ray-intersection
 * @example
 * ```ts
 * import { triangleIntersection } from 'whitebox-engine/graphics'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(triangleIntersection([0.25, 0.5, 2], [0, 0, -1], [0, 0, 0], [1, 0, 0], [0, 1, 0])) // => [2, 0.25, 0.5]
 * ```
 */
export function triangleIntersection(
  origin: TensorLike,
  direction: TensorLike,
  a: TensorLike,
  b: TensorLike,
  c: TensorLike,
): Value {
  const [o, d, pa, pb, pc] = promoteFloat(origin, direction, a, b, c) as [Value, Value, Value, Value, Value]
  const e1 = sub(pb, pa)
  const e2 = sub(pc, pa)
  const p = crossProduct(d, e2)
  const det = dotProduct(e1, p)
  const s = sub(o, pa)
  const q = crossProduct(s, e1)
  return stack([div(dotProduct(e2, q), det), div(dotProduct(s, p), det), div(dotProduct(d, q), det)], -1)
}

/**
 * The slab interval of rays through axis-aligned boxes, `[tNear, tFar]` (the ray hits when `tNear <= tFar` and
 * `tFar >= 0`), differentiable almost everywhere. Directions must have no zero component.
 *
 * @param origin - Ray origins `[..., 3]`.
 * @param direction - Ray directions `[..., 3]`, every component non-zero.
 * @param min - Box minima `[..., 3]`.
 * @param max - Box maxima `[..., 3]`.
 * @returns `[tNear, tFar]` per ray `[..., 2]`.
 * @concept ray-intersection
 * @example
 * ```ts
 * import { boxDistances } from 'whitebox-engine/graphics'
 * import { toFlat } from 'whitebox-engine/tensor'
 * toFlat(boxDistances([-1, -1, -1], [1, 1, 1], [0, 0, 0], [1, 2, 3])) // => [1, 2]
 * ```
 */
export function boxDistances(origin: TensorLike, direction: TensorLike, min: TensorLike, max: TensorLike): Value {
  const [o, d, lo, hi] = promoteFloat(origin, direction, min, max) as [Value, Value, Value, Value]
  const t1 = div(sub(lo, o), d)
  const t2 = div(sub(hi, o), d)
  const near = minimum(t1, t2)
  const far = maximum(t1, t2)
  const tNear = maximum(maximum(col(near, 0), col(near, 1)), col(near, 2))
  const tFar = minimum(minimum(col(far, 0), col(far, 1)), col(far, 2))
  return stack([tNear, tFar], -1)
}
