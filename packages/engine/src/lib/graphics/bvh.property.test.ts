import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import type { Vec3 } from '../scene/index.ts'
import {
  box,
  buildBvh,
  bvhStructure,
  intersectBvh,
  nearestHit,
  occludedBvh,
  type Shape,
  sphere,
  triangle,
} from './index.ts'

const coord = fc.double({ min: -5, max: 5, noNaN: true })
const vec = fc.tuple(coord, coord, coord) as fc.Arbitrary<Vec3>
const unit = vec
  .filter((d) => Math.hypot(d[0], d[1], d[2]) > 0.1)
  .map((d): Vec3 => {
    const n = Math.hypot(d[0], d[1], d[2])
    return [d[0] / n, d[1] / n, d[2] / n]
  })
const shape: fc.Arbitrary<Shape> = fc.oneof(
  fc.tuple(vec, fc.double({ min: 0.1, max: 1.5, noNaN: true })).map(([c, r]) => sphere(c, r)),
  fc
    .tuple(vec, vec)
    .map(([a, b]) =>
      box(
        [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.min(a[2], b[2])],
        [Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.max(a[2], b[2])],
      ),
    ),
  fc
    .tuple(vec, vec, vec)
    .filter(([a, b, c]) => {
      const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
      const v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]]
      const n = [
        (u[1] ?? 0) * (v[2] ?? 0) - (u[2] ?? 0) * (v[1] ?? 0),
        (u[2] ?? 0) * (v[0] ?? 0) - (u[0] ?? 0) * (v[2] ?? 0),
        (u[0] ?? 0) * (v[1] ?? 0) - (u[1] ?? 0) * (v[0] ?? 0),
      ]
      return Math.hypot(n[0] ?? 0, n[1] ?? 0, n[2] ?? 0) > 1e-3
    })
    .map(([a, b, c]) => triangle(a, b, c)),
)

describe('bounding-volume hierarchies', () => {
  it('finds exactly the brute-force nearest hit, for both split rules and any leaf size', () => {
    fc.assert(
      fc.property(
        fc.array(shape, { minLength: 1, maxLength: 24 }),
        fc.array(fc.tuple(vec, unit), { minLength: 1, maxLength: 8 }),
        fc.constantFrom('sah' as const, 'median' as const),
        fc.integer({ min: 1, max: 4 }),
        (shapes, rays, split, maxLeaf) => {
          const bvh = buildBvh(shapes, { split, maxLeaf })
          // Every shape appears exactly once among the leaves, and children nest inside their parents.
          expect([...bvh.order].sort((a, b) => a - b)).toEqual(shapes.map((_, i) => i))
          for (const nd of bvh.nodes) {
            if (nd.count > 0) continue
            for (const child of [bvh.nodes[nd.left], bvh.nodes[nd.right]]) {
              for (let k = 0; k < 3; k++) {
                expect(child?.min[k] ?? 0).toBeGreaterThanOrEqual(nd.min[k] ?? 0)
                expect(child?.max[k] ?? 0).toBeLessThanOrEqual(nd.max[k] ?? 0)
              }
            }
          }
          for (const [origin, direction] of rays) {
            const ray = { origin, direction }
            const want = nearestHit(ray, shapes)
            const got = intersectBvh(bvh, ray).hit
            expect(got?.t).toBe(want?.t)
            if (want !== null) expect(shapes[got?.index ?? -1]).toBe(shapes[want.index])
            expect(occludedBvh(bvh, ray, Infinity)).toBe(want !== null)
            if (want !== null)
              expect(occludedBvh(bvh, ray, want.t * 0.999)).toBe(nearestHit(ray, shapes, 1e-9, want.t * 0.999) !== null)
          }
        },
      ),
      { numRuns: 60 },
    )
  })

  it('tests far fewer shapes than brute force on a spread-out scene', () => {
    const shapes: Shape[] = []
    for (let i = 0; i < 16; i++) for (let j = 0; j < 16; j++) shapes.push(sphere([i * 3, j * 3, 0], 1))
    const bvh = buildBvh(shapes)
    const r = intersectBvh(bvh, { origin: [21, 21, 10], direction: [0, 0, -1] })
    expect(r.hit?.index).toBe(7 * 16 + 7)
    expect(r.stats.shapeTests).toBeLessThan(8)
    expect(r.stats.nodesVisited).toBeLessThan(40)
    expect(bvh.nodes.reduce((m, n) => Math.max(m, n.depth), 0)).toBeLessThan(16)
    const s = bvhStructure(bvh)
    expect(s.nodes.filter((n) => n.role === 'leaf').length + s.edges.length / 2).toBe(s.nodes.length)
  })

  it('keeps coincident shapes in one leaf and refuses bad input', () => {
    const same = Array.from({ length: 10 }, () => sphere([1, 1, 1], 1))
    const bvh = buildBvh(same, { maxLeaf: 1 })
    expect(bvh.nodes).toHaveLength(1)
    expect(intersectBvh(bvh, { origin: [1, 1, 5], direction: [0, 0, -1] }).hit?.t).toBe(3)
    expect(() => buildBvh([])).toThrow()
    expect(() => buildBvh(same, { maxLeaf: 0 })).toThrow()
  })
})
