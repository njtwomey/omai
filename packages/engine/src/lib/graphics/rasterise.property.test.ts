import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import type { Vec3 } from '../base/index.ts'
import { barycentric } from '../foundations/index.ts'
import { toArray } from '../tensor/index.ts'
import {
  clipPolygon,
  intersectTriangle,
  mesh,
  pinholeCamera,
  pixelRay,
  projectToPixel,
  rasteriseMesh,
  rasteriseTriangle,
  SUBPIXEL,
  triangle,
} from './index.ts'

const W = 16
const H = 12
const sc = fc.double({ min: -4, max: 20, noNaN: true })
const pt = fc.tuple(sc, sc) as fc.Arbitrary<[number, number]>

describe('triangle rasterisation', () => {
  it('covers exactly the pixels whose centres lie inside, with the barycentric weights of the centre', () => {
    fc.assert(
      fc.property(pt, pt, pt, (a, b, c) => {
        const area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if (Math.abs(area) < 1e-3) return
        const got = new Map(rasteriseTriangle(a, b, c, W, H).map((f) => [f.y * W + f.x, f.weights]))
        // The reference: barycentric coordinates of every pixel centre (from foundations). A pixel is inside when they
        // are all positive in the triangle with its vertices snapped to the rasteriser's subpixel grid; its weights are
        // those in the given triangle.
        const snap = (v: [number, number]): [number, number] => [
          Math.round(v[0] * SUBPIXEL) / SUBPIXEL,
          Math.round(v[1] * SUBPIXEL) / SUBPIXEL,
        ]
        const centres: number[][] = []
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) centres.push([x + 0.5, y + 0.5])
        const [sa, sb, sc2] = [snap(a), snap(b), snap(c)]
        // A sliver that snaps to a line covers nothing; the reference cannot say where its inside is.
        if (Math.abs((sb[0] - sa[0]) * (sc2[1] - sa[1]) - (sb[1] - sa[1]) * (sc2[0] - sa[0])) < 1e-3) return
        const snapped = toArray(barycentric(centres, [sa, sb, sc2])) as number[][]
        const exact = toArray(barycentric(centres, [a, b, c])) as number[][]
        for (const [i, w] of snapped.entries()) {
          const lo = Math.min(w[0] ?? 0, w[1] ?? 0, w[2] ?? 0)
          if (Math.abs(lo) < 1e-9) continue
          expect(got.has(i), `pixel ${i}`).toBe(lo > 0)
          const g = got.get(i)
          const e = exact[i] ?? []
          if (g !== undefined) for (let k = 0; k < 3; k++) expect(g[k]).toBeCloseTo(e[k] ?? 0, 9)
        }
      }),
    )
  })

  it('covers every pixel of a shared edge exactly once (the top-left rule), even with centres on the edges', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.integer({ min: 0, max: W }), fc.integer({ min: 0, max: H })), {
          minLength: 4,
          maxLength: 4,
        }),
        (corners) => {
          // A fan of triangles around one integer point covers the pixels of the fan once each, whatever the winding.
          const [p0, p1, p2, p3] = corners as [[number, number], [number, number], [number, number], [number, number]]
          const count = new Map<number, number>()
          for (const [a, b, c] of [
            [p0, p1, p2],
            [p0, p2, p3],
          ] as const) {
            for (const f of rasteriseTriangle(a, b, c, W, H))
              count.set(f.y * W + f.x, (count.get(f.y * W + f.x) ?? 0) + 1)
          }
          const area = (a: readonly number[], b: readonly number[], c: readonly number[]): number =>
            ((b[0] ?? 0) - (a[0] ?? 0)) * ((c[1] ?? 0) - (a[1] ?? 0)) -
            ((b[1] ?? 0) - (a[1] ?? 0)) * ((c[0] ?? 0) - (a[0] ?? 0))
          // Only a convex, consistently wound quad tiles without overlap.
          const signs = [area(p0, p1, p2), area(p0, p2, p3), area(p1, p2, p3), area(p0, p1, p3)]
          if (!(signs.every((s) => s > 0) || signs.every((s) => s < 0))) return
          for (const v of count.values()) expect(v).toBe(1)
        },
      ),
    )
    // A square split along its diagonal: every pixel once.
    const all = [...rasteriseTriangle([0, 0], [8, 0], [8, 8], 8, 8), ...rasteriseTriangle([0, 0], [8, 8], [0, 8], 8, 8)]
    expect(all).toHaveLength(64)
    expect(new Set(all.map((f) => f.y * 8 + f.x)).size).toBe(64)
  })

  it('clips polygons to the view volume', () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.tuple(
            fc.double({ min: -3, max: 3, noNaN: true }),
            fc.double({ min: -3, max: 3, noNaN: true }),
            fc.double({ min: -3, max: 3, noNaN: true }),
            fc.double({ min: -1, max: 3, noNaN: true }),
          ),
          { minLength: 3, maxLength: 3 },
        ),
        (vs) => {
          const out = clipPolygon(vs.map((c) => ({ clip: c, attributes: [c[0]] })))
          for (const v of out) {
            const [x, y, z, w] = v.clip
            const tol = 1e-9 * (1 + Math.abs(w))
            for (const q of [x, y, z]) expect(Math.abs(q)).toBeLessThanOrEqual(w + tol)
            // Attributes interpolate with the position (here the attribute is x itself).
            expect(v.attributes[0]).toBeCloseTo(x, 9)
          }
          if (vs.every(([x, y, z, w]) => Math.abs(x) <= w && Math.abs(y) <= w && Math.abs(z) <= w))
            expect(out).toHaveLength(3)
        },
      ),
    )
    expect(
      clipPolygon([
        { clip: [5, 0, 0, 1], attributes: [] },
        { clip: [6, 0, 0, 1], attributes: [] },
        { clip: [5, 1, 0, 1], attributes: [] },
      ]),
    ).toEqual([])
  })
})

describe('the rasterisation pipeline against ray casting', () => {
  const cam = pinholeCamera({ eye: [0.3, 0.5, 4], target: [0, 0, 0], fov: 60, near: 0.5 })
  const coord = fc.double({ min: -2, max: 2, noNaN: true })
  const vert = fc.tuple(coord, coord, fc.double({ min: -2, max: 4.4, noNaN: true })) as fc.Arbitrary<Vec3>

  it('sees the same nearest triangle and depth at every pixel centre as primary rays do, through near-plane clipping', () => {
    fc.assert(
      fc.property(fc.array(fc.tuple(vert, vert, vert), { minLength: 1, maxLength: 5 }), (tris) => {
        const positions = tris.flat()
        const m = mesh(
          positions,
          tris.map((_, i) => [3 * i, 3 * i + 1, 3 * i + 2] as const),
        )
        const r = rasteriseMesh(m, cam, { width: W, height: H, cull: 'none' })
        const shapes = tris.map(([a, b, c]) => {
          try {
            return triangle(a, b, c)
          } catch {
            return undefined
          }
        })
        let disagreements = 0
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const ray = pixelRay(cam, W, H, x + 0.5, y + 0.5)
            let best = -1
            let bestT = Infinity
            for (const [i, s] of shapes.entries()) {
              if (s === undefined) continue
              const h = intersectTriangle(ray, s, 0, bestT)
              if (h === null) continue
              // Only the part beyond the near plane is visible to the rasteriser.
              const p = projectToPixel(cam, h.point, W, H)
              if (p === undefined || p.ndcZ < -1 || p.ndcZ > 1) continue
              best = i
              bestT = h.t
            }
            const got = r.triangleIds[y * W + x] ?? -1
            if (got !== best) {
              // Two surfaces at the same depth (coplanar overlaps) may be resolved either way.
              const other = got >= 0 ? shapes[got] : undefined
              const tie =
                other !== undefined &&
                best >= 0 &&
                Math.abs((intersectTriangle(ray, other, 0)?.t ?? Infinity) - bestT) < 1e-9
              if (!tie) disagreements++
            }
            if (got === best && best >= 0) {
              const hit = intersectTriangle(ray, shapes[best] as never, 0)
              const want = hit === null ? undefined : projectToPixel(cam, hit.point, W, H)
              expect(r.depth[y * W + x]).toBeCloseTo(want?.ndcZ ?? 0, 6)
            }
          }
        }
        // Pixel centres within rounding of an edge, or of two surfaces at equal depth, may go either way.
        expect(disagreements).toBeLessThanOrEqual(2)
      }),
      { numRuns: 40 },
    )
  })

  it('interpolates attributes perspective-correctly, matching the ray-traced barycentrics', () => {
    const a: Vec3 = [-2, -1, -3]
    const b: Vec3 = [2, -1, 1]
    const c: Vec3 = [0, 2, -1]
    const m = mesh([a, b, c], [[0, 1, 2]], {
      colours: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    })
    const correct = rasteriseMesh(m, cam, { width: W, height: H, cull: 'none' })
    const affine = rasteriseMesh(m, cam, { width: W, height: H, cull: 'none', perspectiveCorrect: false })
    const s = triangle(a, b, c)
    let maxAffineError = 0
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if ((correct.triangleIds[y * W + x] ?? -1) < 0) continue
        const h = intersectTriangle(pixelRay(cam, W, H, x + 0.5, y + 0.5), s, 0)
        if (h === null) continue
        const [u, v] = h.uv
        const want = [1 - u - v, u, v]
        for (let k = 0; k < 3; k++) {
          expect(correct.image.rgb[3 * (y * W + x) + k]).toBeCloseTo(want[k] ?? 0, 9)
          maxAffineError = Math.max(
            maxAffineError,
            Math.abs((affine.image.rgb[3 * (y * W + x) + k] ?? 0) - (want[k] ?? 0)),
          )
        }
      }
    }
    expect(maxAffineError).toBeGreaterThan(0.01)
  })
})
