import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import type { Vec3 } from '../base/index.ts'
import { item, toFlat, type Value } from '../tensor/index.ts'
import {
  bounds,
  box,
  boxDistances,
  intersect,
  intersectBox,
  intersectPlane,
  intersectSphere,
  intersectTriangle,
  nearestHit,
  plane,
  planeDistance,
  sphere,
  sphereDistance,
  triangle,
  triangleIntersection,
} from './index.ts'

const coord = fc.double({ min: -4, max: 4, noNaN: true })
const vec = fc.tuple(coord, coord, coord) as fc.Arbitrary<Vec3>
const direction = vec
  .filter((d) => Math.hypot(d[0], d[1], d[2]) > 0.1)
  .map((d): Vec3 => {
    const n = Math.hypot(d[0], d[1], d[2])
    return [d[0] / n, d[1] / n, d[2] / n]
  })

describe('ray-primitive intersections', () => {
  it('puts sphere hits on the sphere with the outward normal, and never misses a ray through the centre', () => {
    fc.assert(
      fc.property(vec, direction, vec, fc.double({ min: 0.2, max: 3, noNaN: true }), (o, d, c, r) => {
        const s = sphere(c, r)
        const h = intersectSphere({ origin: o, direction: d }, s)
        if (h === null) return
        const rel = [h.point[0] - c[0], h.point[1] - c[1], h.point[2] - c[2]]
        expect(Math.hypot(rel[0] ?? 0, rel[1] ?? 0, rel[2] ?? 0)).toBeCloseTo(r, 9)
        expect(h.normal[0]).toBeCloseTo((rel[0] ?? 0) / r, 9)
        expect(h.t).toBeGreaterThan(0)
        const inside = Math.hypot(o[0] - c[0], o[1] - c[1], o[2] - c[2]) < r
        expect(h.frontFace).toBe(!inside)
      }),
    )
    fc.assert(
      fc.property(vec, vec, (o, c) => {
        const d: Vec3 = [c[0] - o[0], c[1] - o[1], c[2] - o[2]]
        if (Math.hypot(d[0], d[1], d[2]) < 1) return
        expect(intersectSphere({ origin: o, direction: d }, sphere(c, 0.5))).not.toBeNull()
      }),
    )
    expect(intersectSphere({ origin: [0, 0, 5], direction: [0, 1, 0] }, sphere([0, 0, 0], 1))).toBeNull()
    expect(intersectSphere({ origin: [0, 0, 5], direction: [0, 0, 1] }, sphere([0, 0, 0], 1))).toBeNull()
    expect(intersectSphere({ origin: [0, 0, 5], direction: [0, 0, -1] }, sphere([0, 0, 0], 1), 0, 3)).toBeNull()
  })

  it('hits planes, triangles and boxes where the closed forms say', () => {
    const p = intersectPlane({ origin: [1, 2, 3], direction: [0, -1, 0] }, plane([0, -1, 0], [0, 1, 0]))
    expect(p?.t).toBe(3)
    expect(p?.frontFace).toBe(true)
    expect(intersectPlane({ origin: [0, 1, 0], direction: [1, 0, 0] }, plane([0, 0, 0], [0, 1, 0]))).toBeNull()
    fc.assert(
      fc.property(vec, vec, vec, vec, direction, (a, b, c, o, d) => {
        let t: ReturnType<typeof triangle>
        try {
          t = triangle(a, b, c)
        } catch {
          return
        }
        const h = intersectTriangle({ origin: o, direction: d }, t)
        if (h === null) return
        const [u, v] = h.uv
        // The hit is the barycentric combination it reports, and lies in the triangle.
        for (let k = 0; k < 3; k++) {
          const want = (1 - u - v) * (a[k] ?? 0) + u * (b[k] ?? 0) + v * (c[k] ?? 0)
          expect(h.point[k] ?? 0).toBeCloseTo(want, 6)
        }
        expect(u).toBeGreaterThanOrEqual(0)
        expect(v).toBeGreaterThanOrEqual(0)
        expect(u + v).toBeLessThanOrEqual(1 + 1e-12)
      }),
    )
    const tri = triangle([0, 0, 0], [1, 0, 0], [0, 1, 0])
    expect(intersectTriangle({ origin: [0.2, 0.2, -1], direction: [0, 0, 1] }, tri, 1e-9, Infinity, true)).toBeNull()
    expect(intersectTriangle({ origin: [0.2, 0.2, -1], direction: [0, 0, 1] }, tri)?.frontFace).toBe(false)
    expect(intersectTriangle({ origin: [0.8, 0.8, 1], direction: [0, 0, -1] }, tri)).toBeNull()
    const smooth = triangle([0, 0, 0], [1, 0, 0], [0, 1, 0], {
      normals: [
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
      ],
      uvs: [
        [0, 0],
        [2, 0],
        [0, 2],
      ],
    })
    const s = intersectTriangle({ origin: [0.5, 0, 1], direction: [0, 0, -1] }, smooth)
    expect(s?.uv).toEqual([1, 0])
    expect(s?.normal[0]).toBeCloseTo(0.5 / Math.hypot(0.5, 1), 12)
    const b = box([0, 0, 0], [2, 1, 1])
    const inside = intersectBox({ origin: [1, 0.5, 0.5], direction: [0, 1, 0] }, b)
    expect([inside?.t, inside?.normal, inside?.frontFace]).toEqual([0.5, [0, 1, 0], false])
    expect(intersectBox({ origin: [5, 5, 5], direction: [0, 1, 0] }, b)).toBeNull()
    expect(intersectBox({ origin: [1, 0.5, 0.5], direction: [0, 0, 0] }, b)).toBeNull()
    expect(intersectBox({ origin: [3, 0.5, 0.5], direction: [-1, 0, 0] }, b)?.uv).toEqual([0.5, 0.5])
  })

  it('agrees with brute force over mixed shapes and reports bounds', () => {
    const shapes = [sphere([0, 0, 0], 1), box([2, -1, -1], [3, 1, 1]), triangle([-3, -1, 0], [-2, -1, 0], [-2.5, 1, 0])]
    fc.assert(
      fc.property(vec, direction, (o, d) => {
        const best = nearestHit({ origin: o, direction: d }, shapes)
        const all = shapes.map((s) => intersect({ origin: o, direction: d }, s)?.t ?? Infinity)
        const min = all.reduce((m, t) => Math.min(m, t), Infinity)
        if (best === null) expect(min).toBe(Infinity)
        else expect(best.t).toBe(min)
      }),
    )
    expect(bounds(shapes[2] as never)).toEqual({ min: [-3, -1, 0], max: [-2, 1, 0] })
    expect(() => bounds(plane([0, 0, 0], [0, 1, 0]))).toThrow()
    expect(() => sphere([0, 0, 0], 0)).toThrow()
    expect(() => box([1, 0, 0], [0, 1, 1])).toThrow()
    expect(() => triangle([0, 0, 0], [1, 1, 1], [2, 2, 2])).toThrow()
    expect(() => intersectSphere({ origin: [0, 0, 0], direction: [1, 0, 0] }, plane([0, 0, 0], [0, 1, 0]))).toThrow()
  })

  it('matches the number-level intersections with the differentiable distances', () => {
    fc.assert(
      fc.property(vec, direction, vec, fc.double({ min: 0.3, max: 3, noNaN: true }), (o, d, c, r) => {
        const h = intersectSphere({ origin: o, direction: d }, sphere(c, r), 0)
        const t = item(sphereDistance(o, d, c, r))
        if (h !== null) expect(t).toBeCloseTo(h.t, 7)
        else expect(Number.isNaN(t) || t < 0).toBe(true)
      }),
    )
    expect(item(planeDistance([0, 5, 0], [0, -1, 0], [0, 1, 0], [0, 1, 0]))).toBe(4)
    expect(toFlat(triangleIntersection([0.1, 0.2, 3], [0, 0, -1], [0, 0, 0], [1, 0, 0], [0, 1, 0]))).toEqual([
      3, 0.1, 0.2,
    ])
    const span = toFlat(boxDistances([-1, 0.5, 0.25], [1, 0.1, 0.1], [0, 0, 0], [2, 1, 1]))
    const want = intersectBox({ origin: [-1, 0.5, 0.25], direction: [1, 0.1, 0.1] }, box([0, 0, 0], [2, 1, 1]))
    expect(span[0]).toBeCloseTo(want?.t ?? 0, 12)
  })

  for (const [name, f, inputs] of [
    [
      'sphereDistance',
      (o: Value, d: Value, c: Value, r: Value) => sphereDistance(o, d, c, r),
      [[0.1, 0.2, 5], [0.05, -0.02, -1], [0, 0.1, 0], [1.2]],
    ],
    [
      'planeDistance',
      (o: Value, d: Value, p: Value, n: Value) => planeDistance(o, d, p, n),
      [
        [0.1, 2, 0.3],
        [0.2, -1, 0.1],
        [0, 0.5, 0],
        [0.1, 1, 0.2],
      ],
    ],
    [
      'triangleIntersection',
      (o: Value, d: Value, a: Value, b: Value, c: Value) => triangleIntersection(o, d, a, b, c),
      [
        [0.2, 0.3, 2],
        [0.05, 0.02, -1],
        [0, 0, 0.1],
        [1, 0.1, 0],
        [0.1, 1, -0.1],
      ],
    ],
    [
      'boxDistances',
      (o: Value, d: Value, lo: Value, hi: Value) => boxDistances(o, d, lo, hi),
      [
        [-1, 0.3, 0.4],
        [1, 0.2, 0.15],
        [0, 0, 0],
        [2, 1.5, 1.2],
      ],
    ],
  ] as const) {
    it(`${name} passes checkPrimitive`, () => {
      const xs = (inputs as readonly (readonly number[])[]).map((x) => (x.length === 1 ? (x[0] ?? 0) : [...x]))
      expect(checkPrimitive(f as (...v: Value[]) => Value, xs).failures).toEqual([])
    })
  }
})
