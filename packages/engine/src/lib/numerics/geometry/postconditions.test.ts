import { describe, expect, it } from 'vitest'
import { item } from '../../tensor/index.ts'
import * as G from './index.ts'

// The Delaunay triangulation's exact output properties, on the inputs that break robust predicates: regular polygons and
// lattices (every point cocircular or four to a circle), points on a circle with 1e-9 noise, random clouds, clusters.
type P = [number, number]
const lcg = (seed: number): (() => number) => {
  let s = seed
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return s / 2 ** 32
  }
}
const polygon = (n: number): P[] =>
  Array.from({ length: n }, (_, i) => [Math.cos((2 * Math.PI * i) / n), Math.sin((2 * Math.PI * i) / n)])
const lattice = (w: number, h: number): P[] => Array.from({ length: w * h }, (_, i) => [i % w, Math.floor(i / w)])
const noisyCircle = (n: number, seed: number): P[] => {
  const u = lcg(seed)
  return polygon(n).map(([x, y]) => [x + 1e-9 * (u() - 0.5), y + 1e-9 * (u() - 0.5)])
}
const cloud = (n: number, seed: number): P[] => {
  const u = lcg(seed)
  return Array.from({ length: n }, () => [u() * 10, u() * 10])
}

const cases: readonly (readonly [string, P[]])[] = [
  ...[3, 4, 5, 8, 9, 12, 16, 25, 40].map((n) => [`regular ${n}-gon`, polygon(n)] as const),
  ['5 x 5 lattice', lattice(5, 5)],
  ['3 x 11 lattice', lattice(3, 11)],
  ['12 points on a circle with 1e-9 noise', noisyCircle(12, 4)],
  ['40 points on a circle with 1e-9 noise', noisyCircle(40, 5)],
  ['20 random points', cloud(20, 1)],
  ['200 random points', cloud(200, 2)],
  [
    'two clusters',
    [...cloud(30, 3).map(([x, y]): P => [x / 100, y / 100]), ...cloud(30, 4).map(([x, y]): P => [x + 50, y + 50])],
  ],
]

describe('delaunay output post-conditions', () => {
  it('has 2n - 2 - h counter-clockwise triangles that tile the hull, empty circumcircles and consistent neighbours', () => {
    const failures: string[] = []
    for (const [name, points] of cases) {
      const d = G.delaunay(points)
      const n = points.length
      const hull = G.convexHull(points)
      // Points on the hull boundary, collinear ones included (the hull edges of the triangulation), not just the corners.
      const boundary = d.hull.length
      const area = (t: readonly number[]): number =>
        G.orient(points[t[0] ?? 0] as P, points[t[1] ?? 0] as P, points[t[2] ?? 0] as P) / 2
      // Euler: a triangulation of n points with h on the hull boundary has 2 n - 2 - h triangles (no point is dropped).
      if (d.simplices.length !== 2 * n - 2 - boundary)
        failures.push(`${name}: ${d.simplices.length} triangles, expected ${2 * n - 2 - boundary}`)
      const used = new Set(d.simplices.flat())
      if (used.size !== n) failures.push(`${name}: ${used.size} of ${n} points are vertices`)
      // They tile the convex hull exactly: the signed areas are positive and sum to the hull's polygon area.
      const total = d.simplices.reduce((s, t) => s + area(t), 0)
      const hullArea = item(G.polygonArea(hull.map((i) => points[i] as P)))
      if (!d.simplices.every((t) => area(t) > 0)) failures.push(`${name}: a triangle is not counter-clockwise`)
      if (!(Math.abs(total - Math.abs(hullArea)) <= 1e-9 * Math.max(1, Math.abs(hullArea))))
        failures.push(`${name}: triangle areas ${total}, hull ${hullArea}`)
      // Delaunay: no point lies strictly inside the circumcircle of a triangle (to rounding).
      for (const t of d.simplices) {
        const [a, b, c] = t.map((i) => points[i] as P) as [P, P, P]
        const centre = G.circumcentre(a, b, c)
        const radius = Math.hypot(a[0] - centre[0], a[1] - centre[1])
        points.forEach((p, i) => {
          if (!t.includes(i) && Math.hypot(p[0] - centre[0], p[1] - centre[1]) < radius * (1 - 1e-7))
            failures.push(`${name}: point ${i} inside the circle of ${t.join(',')}`)
        })
      }
      // neighbors[t][v] is the triangle across the edge opposite vertex v, and the relation is symmetric.
      d.neighbors.forEach((row, t) => {
        row.forEach((other) => {
          if (other >= 0 && !d.neighbors[other]?.includes(t))
            failures.push(`${name}: triangle ${t} lists ${other}, which does not list it back`)
        })
      })
      // findSimplex locates the centroid of every triangle in that triangle, and a far point outside (-1).
      d.simplices.forEach((t, k) => {
        const cx = t.reduce((s, i) => s + (points[i]?.[0] ?? 0), 0) / 3
        const cy = t.reduce((s, i) => s + (points[i]?.[1] ?? 0), 0) / 3
        if (d.findSimplex([cx, cy]) !== k)
          failures.push(`${name}: the centroid of triangle ${k} is found in ${d.findSimplex([cx, cy])}`)
      })
      if (d.findSimplex([1e6, 1e6]) !== -1) failures.push(`${name}: a far point is inside`)
    }
    expect(failures).toEqual([])
    {
      // isolines of x^2 + y^2 on a grid: the level-1 contour is the unit circle to the interpolation error, in one path.
      const n = 41
      const axis = Array.from({ length: n }, (_, i) => (i - 20) / 10)
      const z = axis.map((x) => axis.map((y) => x * x + y * y))
      const [contour] = G.isolines(z, [1], { x: axis, y: axis })
      expect(contour?.level).toBe(1)
      expect(contour?.paths).toHaveLength(1)
      const path = contour?.paths[0] ?? []
      expect(path.every(([px = 0, py = 0]) => Math.abs(Math.hypot(px, py) - 1) < 0.012)).toBe(true)
      expect(path[0]).toEqual(path.at(-1))
      expect(path.length).toBeGreaterThan(30)
    }
  })
})
