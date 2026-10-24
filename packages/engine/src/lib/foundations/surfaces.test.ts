import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import {
  add,
  cos,
  cosh,
  mul,
  sin,
  sliceAxis,
  squeeze,
  stack,
  sum,
  tensor,
  toArray,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import * as F from './index.ts'

const parts = (p: Value): [Value, Value] => [squeeze(sliceAxis(p, -1, 0, 1), -1), squeeze(sliceAxis(p, -1, 1, 2), -1)]
const catenoid: F.ParametricSurface = (p) => {
  const [a, b] = parts(p)
  return stack([mul(cosh(a), cos(b)), mul(cosh(a), sin(b)), a], -1)
}
const dot = (a: readonly number[], b: readonly number[]): number => a.reduce((s, x, i) => s + x * (b[i] ?? 0), 0)
const pts = tensor([
  [0.4, 0.3],
  [1.1, -0.8],
  [2, 2.5],
])

describe('curvature of classical surfaces', () => {
  it('spheres have constant K = 1/R^2 and |H| = 1/R', () => {
    for (const k of toFlat(F.gaussianCurvature(F.sphere(3), pts))) expect(k).toBeCloseTo(1 / 9, 13)
    for (const h of toFlat(F.meanCurvature(F.sphere(3), pts))) expect(Math.abs(h)).toBeCloseTo(1 / 3, 13)
  })

  it('the catenoid is minimal (H = 0) with K = -1/cosh^4 u', () => {
    for (const h of toFlat(F.meanCurvature(catenoid, pts))) expect(h).toBeCloseTo(0, 13)
    const ks = toFlat(F.gaussianCurvature(catenoid, pts))
    for (const [i, k] of ks.entries()) expect(k).toBeCloseTo(-1 / Math.cosh(toFlat(pts)[2 * i] ?? 0) ** 4, 12)
  })

  it('a torus built as a surface of revolution matches the torus', () => {
    const revolved = F.surfaceOfRevolution((t) => stack([add(cos(t), 3), sin(t)], -1))
    const a = toFlat(F.gaussianCurvature(revolved, pts))
    const b = toFlat(F.gaussianCurvature(F.torus(3, 1), pts))
    for (const [i, v] of a.entries()) expect(v).toBeCloseTo(b[i] ?? 0, 12)
  })

  it('principal curvatures multiply to K, average to H, along orthogonal directions', () => {
    const s = F.torus(2, 0.7)
    for (const p of toArray(pts) as [number, number][]) {
      const pc = F.principalCurvatures(s, p)
      const [k1, k2] = pc.curvatures
      expect(k1 * k2).toBeCloseTo(toFlat(F.gaussianCurvature(s, [p]))[0] ?? 0, 12)
      expect((k1 + k2) / 2).toBeCloseTo(toFlat(F.meanCurvature(s, [p]))[0] ?? 0, 12)
      expect(dot(pc.directions[0], pc.directions[1])).toBeCloseTo(0, 10)
    }
    // At an umbilic every direction is principal.
    const umbilic = F.principalCurvatures(F.sphere(1), [1, 1])
    expect(umbilic.curvatures[0]).toBeCloseTo(umbilic.curvatures[1], 12)
  })

  it('the tangent plane is orthogonal to the coordinate directions', () => {
    const s = F.graphSurface((p) => sum(mul(p, p), -1))
    const plane = F.tangentPlane(s, [1, 2])
    const { ru, rv } = F.surfaceDerivatives(s, [1, 2])
    expect(dot(plane.normal, toFlat(ru))).toBeCloseTo(0, 14)
    expect(dot(plane.normal, toFlat(rv))).toBeCloseTo(0, 14)
    expect(plane.point).toEqual([1, 2, 5])
    expect(() => F.surfaceNormal(s, [1, 2, 3])).toThrow()
  })
})

describe('areas, geodesics and faces', () => {
  it('measures the area of a catenoid band', () => {
    // Area of the catenoid over u in [0, 1]: pi (u + sinh u cosh u) from 0 to 1.
    const want = Math.PI * (1 + Math.sinh(1) * Math.cosh(1))
    expect(F.surfaceArea(catenoid, [0, 1], [0, 2 * Math.PI])).toBeCloseTo(want, 10)
  })

  it('geodesics on a sphere are great circles', () => {
    const g = F.geodesic(F.sphere(1), [1, 0.3], [0.4, 1], { length: 2, steps: 200 })
    const [p0, p1] = [g.points[0] ?? [], g.points[1] ?? []]
    // The great circle lies in the plane through the centre spanned by the first point and the first step.
    const n = [
      (p0[1] ?? 0) * (p1[2] ?? 0) - (p0[2] ?? 0) * (p1[1] ?? 0),
      (p0[2] ?? 0) * (p1[0] ?? 0) - (p0[0] ?? 0) * (p1[2] ?? 0),
      (p0[0] ?? 0) * (p1[1] ?? 0) - (p0[1] ?? 0) * (p1[0] ?? 0),
    ]
    const nn = Math.hypot(...n)
    for (const p of g.points) {
      expect(Math.hypot(...p)).toBeCloseTo(1, 12)
      expect(dot(p, n) / nn).toBeCloseTo(0, 8)
    }
    // Unit speed: the chord between consecutive points is the step length.
    expect(Math.hypot(...p1.map((x, i) => x - (p0[i] ?? 0)))).toBeCloseTo(0.01, 6)
    expect(() => F.geodesic(F.sphere(1), [1, 0], [0, 0])).toThrow()
  })

  it('builds scene faces, coloured by a function of the parameters', () => {
    const faces = F.surfaceFaces(F.torus(2, 1), [0, 2 * Math.PI], [0, 2 * Math.PI], [3, 5], {
      fill: (u) => (u < Math.PI ? 'tab1' : 'tab2'),
    })
    expect(faces).toHaveLength(15)
    expect(faces[0]?.kind).toBe('face')
    expect(faces[0]?.points).toHaveLength(4)
    expect(new Set(faces.map((f) => f.fill))).toEqual(new Set(['tab1', 'tab2']))
  })
})

describe('derivatives', () => {
  it('passes the conformance check', () => {
    expectConformant(F.sphere(2), [pts])
    expectConformant(F.torus(2, 0.5), [pts])
    expectConformant(catenoid, [pts])
    expectConformant((p) => F.surfaceNormal(F.torus(2, 0.5), p), [pts])
    expectConformant((p) => F.firstFundamentalForm(catenoid, p).uv, [pts])
    expectConformant((p) => F.gaussianCurvature(F.torus(2, 0.5), p), [pts])
    expectConformant((p) => F.meanCurvature(catenoid, p), [pts])
  })
})
