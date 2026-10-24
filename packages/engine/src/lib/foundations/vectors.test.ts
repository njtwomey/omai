import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { item, tensor, toArray, toFlat } from '../tensor/index.ts'
import * as F from './index.ts'

const vec3 = fc.tuple(
  fc.double({ min: -3, max: 3, noNaN: true }),
  fc.double({ min: -3, max: 3, noNaN: true }),
  fc.double({ min: -3, max: 3, noNaN: true }),
)
const away = vec3.filter((v) => Math.hypot(...v) > 0.1)

describe('vector algebra laws', () => {
  it('the cross product is anticommutative and orthogonal to its factors', () => {
    fc.assert(
      fc.property(vec3, vec3, (a, b) => {
        const c = toFlat(F.crossProduct(a, b))
        const d = toFlat(F.crossProduct(b, a))
        for (const [i, x] of c.entries()) expect(x).toBeCloseTo(-(d[i] ?? 0), 12)
        expect(item(F.dotProduct(c, a))).toBeCloseTo(0, 10)
        expect(item(F.dotProduct(c, b))).toBeCloseTo(0, 10)
      }),
    )
  })

  it('BAC-CAB, Lagrange identity and the cyclic triple product', () => {
    fc.assert(
      fc.property(vec3, vec3, vec3, (a, b, c) => {
        const lhs = toFlat(F.vectorTripleProduct(a, b, c))
        const ac = item(F.dotProduct(a, c))
        const ab = item(F.dotProduct(a, b))
        for (const [i, x] of lhs.entries()) expect(x).toBeCloseTo((b[i] ?? 0) * ac - (c[i] ?? 0) * ab, 9)
        const cross = item(F.magnitude(F.crossProduct(a, b))) ** 2
        expect(cross).toBeCloseTo(item(F.dotProduct(a, a)) * item(F.dotProduct(b, b)) - ab ** 2, 8)
        expect(item(F.tripleProduct(a, b, c))).toBeCloseTo(item(F.tripleProduct(b, c, a)), 9)
      }),
    )
  })

  it('projection plus rejection rebuilds the vector, at right angles', () => {
    fc.assert(
      fc.property(vec3, away, (a, b) => {
        const p = toFlat(F.projection(a, b))
        const r = toFlat(F.rejection(a, b))
        for (const [i, x] of p.entries()) expect(x + (r[i] ?? 0)).toBeCloseTo(a[i] ?? 0, 10)
        expect(item(F.dotProduct(r, b))).toBeCloseTo(0, 9)
        expect(item(F.scalarProjection(a, b))).toBeCloseTo(item(F.dotProduct(a, F.unitVector(b))), 10)
      }),
    )
  })

  it('angles are symmetric, in [0, pi], and exact for nearly parallel vectors', () => {
    fc.assert(
      fc.property(away, away, (a, b) => {
        const t = item(F.angleBetween(a, b))
        expect(t).toBeGreaterThanOrEqual(0)
        expect(t).toBeLessThanOrEqual(Math.PI)
        expect(item(F.angleBetween(b, a))).toBeCloseTo(t, 12)
      }),
    )
    expect(item(F.angleBetween([1, 0], [1, 1e-9]))).toBeCloseTo(1e-9, 20)
    expect(item(F.angleBetween([1, 0], [-1, 0]))).toBeCloseTo(Math.PI, 15)
  })
})

describe('independence and span', () => {
  it('orthonormalises to an orthonormal basis of the span', () => {
    const r = F.orthonormalise([
      [1, 1, 0],
      [2, 2, 0],
      [0, 1, 1],
      [1, 2, 1],
    ])
    expect(r.independent).toEqual([0, 2])
    expect(r.residuals[1]).toBe(0)
    for (const [i, p] of r.basis.entries())
      for (const [j, q] of r.basis.entries()) expect(item(F.dotProduct(p, q))).toBeCloseTo(i === j ? 1 : 0, 14)
    expect(
      F.isLinearlyIndependent([
        [1, 0],
        [0, 1],
        [1, 1],
      ]),
    ).toBe(false)
    expect(F.spanDimension([[0, 0, 0]])).toBe(0)
  })

  it('expresses members of the span and measures the distance of non-members', () => {
    const r = F.inSpan(
      [
        [1, 1, 0],
        [2, 2, 0],
        [0, 1, 1],
      ],
      [1, 3, 2],
    )
    expect(r.inSpan).toBe(true)
    expect(r.coefficients[0]).toBeCloseTo(1, 12)
    expect(r.coefficients[1]).toBe(0)
    expect(r.coefficients[2]).toBeCloseTo(2, 12)
    expect(
      toArray(
        F.linearCombination(
          [
            [1, 1, 0],
            [2, 2, 0],
            [0, 1, 1],
          ],
          r.coefficients,
        ),
      ),
    ).toEqual([1, 3, 2])
    const no = F.inSpan([[1, 0, 0]], [3, 4, 0])
    expect(no.inSpan).toBe(false)
    expect(no.residual).toBeCloseTo(4, 14)
  })
})

describe('lines and planes', () => {
  it('constructs planes three ways that agree', () => {
    const p = F.planeThrough([1, 0, 0], [0, 1, 0], [0, 0, 1])
    const [a, b, c, d] = F.planeEquation(p)
    expect([a, b, c, d]).toEqual([1, 1, 1, 1])
    const q = F.planeFromEquation([a, b, c, d])
    expect(item(F.pointPlaneDistance(q.point, p))).toBeCloseTo(0, 15)
    expect(() => F.planeThrough([0, 0, 0], [1, 1, 1], [2, 2, 2])).toThrow()
    expect(() => F.lineThrough([1, 2], [1, 2])).toThrow()
    expect(() => F.planeFromEquation([0, 0, 0, 1])).toThrow()
  })

  it('finds closest points of skew, parallel and meeting lines', () => {
    const skew = F.closestPoints({ point: [0, 0, 0], direction: [1, 0, 0] }, { point: [1, 2, 3], direction: [0, 1, 0] })
    expect(skew.onFirst).toEqual([1, 0, 0])
    expect(skew.onSecond).toEqual([1, 0, 3])
    expect(skew.parallel).toBe(false)
    const parallel = F.closestPoints({ point: [0, 0], direction: [1, 0] }, { point: [5, 2], direction: [2, 0] })
    expect(parallel.parallel).toBe(true)
    expect(parallel.distance).toBeCloseTo(2, 15)
    expect(F.lineIntersection({ point: [0, 0], direction: [1, 0] }, { point: [5, 2], direction: [2, 0] })).toBeNull()
    expect(
      F.lineIntersection({ point: [0, 0, 0], direction: [1, 0, 0] }, { point: [1, 2, 3], direction: [0, 1, 0] }),
    ).toBeNull()
  })

  it('intersects lines with planes and planes with planes', () => {
    expect(
      F.linePlaneIntersection({ point: [0, 0, 0], direction: [1, 0, 0] }, { point: [0, 0, 1], normal: [0, 0, 1] }),
    ).toBeNull()
    expect(
      F.planeIntersection({ point: [0, 0, 0], normal: [0, 0, 1] }, { point: [0, 0, 1], normal: [0, 0, 2] }),
    ).toBeNull()
    fc.assert(
      fc.property(away, away, vec3, vec3, (n1, n2, p1, p2) => {
        const a = { point: p1, normal: n1 }
        const b = { point: p2, normal: n2 }
        const l = F.planeIntersection(a, b)
        // Nearly parallel planes meet far away and ill-conditioned; keep to a clear angle.
        if (l === null || Math.hypot(...l.direction) < 0.1 * Math.hypot(...n1) * Math.hypot(...n2)) return
        for (const s of [-1, 0, 2]) {
          const x = l.point.map((v, i) => v + s * (l.direction[i] ?? 0))
          const scale = Math.max(1, Math.hypot(...x))
          expect(Math.abs(item(F.pointPlaneDistance(x, a))) / scale).toBeLessThan(1e-8)
          expect(Math.abs(item(F.pointPlaneDistance(x, b))) / scale).toBeLessThan(1e-8)
        }
      }),
    )
  })
})

describe('derivatives', () => {
  it('passes the conformance check', () => {
    const a = tensor([
      [0.3, -1, 2],
      [1, 0.5, -0.7],
    ])
    const b = tensor([
      [1.2, 0.4, -0.3],
      [-0.6, 1.1, 0.9],
    ])
    const c = tensor([
      [0.2, 0.8, 1.5],
      [1, -1, 0.4],
    ])
    expectConformant(F.dotProduct, [a, b])
    expectConformant(F.crossProduct, [a, b])
    expectConformant(F.crossProduct, [tensor([1, 2]), tensor([-0.5, 3])])
    expectConformant(F.tripleProduct, [a, b, c])
    expectConformant(F.vectorTripleProduct, [a, b, c])
    expectConformant(F.magnitude, [a])
    expectConformant(F.unitVector, [a])
    expectConformant(F.projection, [a, b])
    expectConformant(F.scalarProjection, [a, b])
    expectConformant(F.rejection, [a, b])
    expectConformant(F.angleBetween, [a, b])
    expectConformant((x) => F.pointLineDistance(x, { point: [1, 0, 0], direction: [1, 2, 2] }), [a])
    expectConformant((x) => F.pointPlaneDistance(x, { point: [1, 0, 0], normal: [1, 2, 2] }), [a])
    expectConformant(F.linearCombination, [a, tensor([0.5, -2])])
  })
})
