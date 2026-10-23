import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { type Tensor, tensor, toFlat } from '../../tensor/index.ts'
import * as G from './index.ts'

const canonical = (simplices: readonly (readonly number[])[]): number[][] =>
  simplices
    .map((t) => [...t].sort((a, b) => a - b))
    .sort((a, b) => (a[0] ?? 0) - (b[0] ?? 0) || (a[1] ?? 0) - (b[1] ?? 0) || (a[2] ?? 0) - (b[2] ?? 0))

describe('Delaunay triangulation against scipy.spatial', () => {
  for (const c of loadFixture('numerics/geometry/triangulation').cases) {
    it(c.name, () => {
      const pts = decode(c.inputs[0]) as Tensor
      const tri = G.delaunay(pts)
      if (c.fn === 'delaunay') expectMatches(tensor(canonical(tri.simplices)), c.output, 0, 0, c.name)
      else if (c.fn === 'inside') {
        const q = G.toPoints(decode(c.inputs[1]) as Tensor)
        expectMatches(tensor(q.map((p) => (tri.findSimplex(p) >= 0 ? 1 : 0))), c.output, 0, 0, c.name)
      } else expectMatches(tensor([...new Set(tri.hull.flat())].sort((a, b) => a - b)), c.output, 0, 0, c.name)
    })
  }
})

describe('triangulation properties', () => {
  const pts = Array.from(
    { length: 60 },
    (_, i): G.Point2 => [(Math.sin(i * 12.9898) * 43758.5453) % 1, (Math.sin(i * 78.233) * 12345.678) % 1],
  )

  it('no point lies inside any circumcircle, and neighbours are mutual', () => {
    const tri = G.delaunay(pts)
    for (const [a, b, c] of tri.simplices) {
      const pa = pts[a] as G.Point2
      const pb = pts[b] as G.Point2
      const pc = pts[c] as G.Point2
      expect(G.orient(pa, pb, pc)).toBeGreaterThan(0)
      for (let i = 0; i < pts.length; i++)
        if (i !== a && i !== b && i !== c) expect(G.inCircle(pa, pb, pc, pts[i] as G.Point2)).toBeLessThanOrEqual(1e-12)
    }
    tri.neighbors.forEach((ns, t) => {
      for (const n of ns) if (n >= 0) expect(tri.neighbors[n]).toContain(t)
    })
    // Euler: triangles = 2n - 2 - h for points in general position.
    expect(tri.simplices.length).toBe(2 * pts.length - 2 - tri.hull.length)
  })

  it('Voronoi vertices are equidistant from their three sites', () => {
    const tri = G.delaunay(pts)
    const v = G.voronoi(pts)
    tri.simplices.forEach(([a, b, c], i) => {
      const centre = v.vertices[i] as G.Point2
      const d = (p: number): number => Math.hypot(centre[0] - (pts[p]?.[0] ?? 0), centre[1] - (pts[p]?.[1] ?? 0))
      expect(d(a)).toBeCloseTo(d(b), 10)
      expect(d(a)).toBeCloseTo(d(c), 10)
    })
  })

  it('barycentric coordinates reconstruct the point, and alpha shapes shrink to concavities', () => {
    const l = G.barycentricCoordinates(
      [
        [0, 0],
        [2, 0],
        [0, 3],
      ],
      [0.5, 1],
    )
    expect(l.reduce((a, b) => a + b)).toBeCloseTo(1, 14)
    expect(l[1] * 2).toBeCloseTo(0.5, 14)
    // An L-shaped cloud: a small alpha drops the triangles spanning the notch.
    const lshape: G.Point2[] = []
    for (let i = 0; i <= 4; i++)
      for (let j = 0; j <= 4; j++) if (i <= 1 || j <= 1) lshape.push([i + 0.01 * j, j + 0.01 * i])
    const big = G.alphaShape(lshape, 100)
    const small = G.alphaShape(lshape, 0.9)
    expect(small.triangles.length).toBeLessThan(big.triangles.length)
    expect(toFlat(G.simplexTensor(G.delaunay(lshape))).length % 3).toBe(0)
  })
})
