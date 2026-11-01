import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { toFlat } from '../tensor/index.ts'
import * as V from './index.ts'

const rectImage = (h: number, w: number, x0: number, y0: number, rw: number, rh: number): number[][] =>
  Array.from({ length: h }, (_, y) =>
    Array.from({ length: w }, (_, x) => (x >= x0 && x < x0 + rw && y >= y0 && y < y0 + rh ? 1 : 0)),
  )

const rect = fc.record({
  x0: fc.integer({ min: 0, max: 8 }),
  y0: fc.integer({ min: 0, max: 8 }),
  rw: fc.integer({ min: 2, max: 10 }),
  rh: fc.integer({ min: 2, max: 10 }),
})

const cloud = fc.array(fc.tuple(fc.integer({ min: 0, max: 30 }), fc.integer({ min: 0, max: 30 })), {
  minLength: 4,
  maxLength: 25,
})

describe('contours of drawn rectangles', () => {
  it('measure the rectangle exactly', () => {
    fc.assert(
      fc.property(rect, ({ x0, y0, rw, rh }) => {
        const img = rectImage(20, 20, x0, y0, rw, rh)
        for (const method of ['none', 'simple'] as const) {
          const { contours, hierarchy } = V.findContours(img, { method })
          expect(contours.length).toBe(1)
          expect(hierarchy).toEqual([[-1, -1, -1, -1]])
          const c = contours[0] as V.Contour
          // The border runs through pixel centres, so the polygon is one pixel smaller each way.
          expect(V.contourArea(c)).toBe((rw - 1) * (rh - 1))
          expect(V.arcLength(c, true)).toBe(2 * (rw - 1 + rh - 1))
          expect(V.boundingRect(c)).toEqual([x0, y0, rw, rh])
          expect(V.approxPolyDP(c, 0.5, true).length).toBe(4)
          expect(V.moments(c).m00).toBe((rw - 1) * (rh - 1))
        }
        const simple = V.findContours(img).contours[0] as V.Contour
        expect(simple.length).toBe(4)
        expect(V.isContourConvex(simple)).toBe(true)
        expect(V.pointPolygonTest(simple, [x0 + 0.5, y0 + 0.5])).toBe(1)
        expect(V.pointPolygonTest(simple, [x0 - 1, y0])).toBe(-1)
        expect(V.pointPolygonTest(simple, [x0, y0])).toBe(0)
      }),
      { numRuns: RUNS },
    )
  })

  it('nests holes under their outer border', () => {
    const img = rectImage(12, 12, 1, 1, 10, 10)
    for (let y = 4; y < 8; y++) for (let x = 4; x < 8; x++) (img[y] as number[])[x] = 0
    const tree = V.findContours(img, { mode: 'tree' })
    expect(tree.hierarchy).toEqual([
      [-1, -1, 1, -1],
      [-1, -1, -1, 0],
    ])
    expect(V.contourArea(tree.contours[1] as V.Contour, true)).toBeGreaterThan(0)
    expect(V.contourArea(tree.contours[0] as V.Contour, true)).toBeLessThan(0)
    expect(V.findContours(img, { mode: 'external' }).contours.length).toBe(1)
    expect(() => V.findContours(img, { mode: 'nope' as V.RetrievalMode })).toThrow()
  })
})

describe('shape descriptors', () => {
  it('hulls contain every point and are convex', () => {
    fc.assert(
      fc.property(cloud, (pts) => {
        const hull = V.convexHull(pts).map((i) => pts[i] as [number, number])
        if (hull.length < 3 || Math.abs(V.contourArea(hull)) < 1e-9) return
        expect(V.isContourConvex(hull)).toBe(true)
        for (const p of pts) expect(V.pointPolygonTest(hull, p)).toBeGreaterThanOrEqual(0)
        // The same vertices either way round; equal points are interchangeable, so compare coordinates, not indices.
        const at = (idx: readonly number[]): Set<string> =>
          new Set(idx.map((i) => (pts[i] as [number, number]).join(',')))
        expect(at(V.convexHull(pts, { clockwise: true }))).toEqual(at(V.convexHull(pts)))
      }),
      { numRuns: RUNS },
    )
  })

  it('fits enclosing shapes around every point', () => {
    fc.assert(
      fc.property(cloud, (pts) => {
        const { center, radius } = V.minEnclosingCircle(pts)
        for (const [x, y] of pts) expect(Math.hypot(x - center[0], y - center[1])).toBeLessThanOrEqual(radius)
        const r = V.minAreaRect(pts)
        const [, , bw, bh] = V.boundingRect(pts)
        // The rotated rectangle is no larger than the upright one through the extreme points.
        expect(r.size[0] * r.size[1]).toBeLessThanOrEqual((bw - 1) * (bh - 1) + 1e-3)
        expect(r.angle).toBeGreaterThanOrEqual(-90)
        expect(r.angle).toBeLessThan(0)
        const corners = V.boxPoints(r)
        for (const p of pts) expect(V.pointPolygonTest(corners, p, true)).toBeGreaterThan(-1e-3)
      }),
      { numRuns: RUNS },
    )
  })

  it('Hu invariants survive translation, scaling and rotation', () => {
    const tri: [number, number][] = [
      [0, 0],
      [7, 1],
      [2, 5],
    ]
    const moved = tri.map(([x, y]): [number, number] => [3 * -y + 10, 3 * x + 4])
    const a = V.huMoments(V.moments(tri))
    const b = V.huMoments(V.moments(moved))
    for (let i = 0; i < 6; i++) expect(b[i]).toBeCloseTo(a[i] ?? 0, 10)
    expect(V.matchShapes(tri, moved, 2)).toBeLessThan(1e-8)
    expect(V.imageMoments([[1]]).m00).toBe(1)
    const line = V.fitLine(
      [
        [0, 0],
        [1, 2],
        [2, 4],
        [3, 6.5],
      ],
      { distance: 'huber' },
    )
    expect(Math.abs(line[1] / line[0] - 2)).toBeLessThan(0.2)
  })

  it('finds convexity defects and component statistics', () => {
    const c: [number, number][] = [
      [0, 0],
      [6, 0],
      [6, 6],
      [3, 2],
      [0, 6],
    ]
    const d = V.convexityDefects(c, V.convexHull(c))
    expect(d.map((x) => x.farthest)).toEqual([3])
    const img = rectImage(6, 7, 1, 1, 2, 2)
    ;(img[4] as number[])[5] = 1
    const r = V.connectedComponentsWithStats(img)
    expect(r.count).toBe(3)
    expect(r.stats.reduce((s, row) => s + row[4], 0)).toBe(42)
    const f = V.floodFill(img, [0, 0], 0.5)
    expect(f.area).toBe(42 - 5)
    expect(toFlat(f.mask).reduce((a, b) => a + b, 0)).toBe(37)
    expect(() => V.floodFill(img, [9, 0], 1)).toThrow()
  })
})
