import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { item, mul, sin, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import * as F from './index.ts'

describe('angles and the unit circle', () => {
  it('converts degrees and radians both ways', () => {
    expect(toFlat(F.degrees(F.radians([0, 45, 360])))).toEqual([0, 45, 360])
  })

  it('classifies quadrants and reference angles', () => {
    expect([0.5, 2, 4, 5.5].map(F.quadrant)).toEqual([1, 2, 3, 4])
    expect(F.quadrant(0)).toBe(0)
    expect(F.quadrant(-Math.PI / 2)).toBe(0)
    expect(F.referenceAngle((7 * Math.PI) / 6)).toBeCloseTo(Math.PI / 6, 14)
    expect(F.referenceAngle(-Math.PI / 3)).toBeCloseTo(Math.PI / 3, 14)
  })

  it('inverts in a chosen quadrant and refuses impossible ones', () => {
    for (const q of [1, 2, 3, 4] as const) {
      const t = F.inverseInQuadrant('cos', q === 1 || q === 4 ? 0.3 : -0.3, q)
      expect(F.quadrant(t)).toBe(q)
      expect(Math.cos(t)).toBeCloseTo(q === 1 || q === 4 ? 0.3 : -0.3, 14)
    }
    expect(() => F.inverseInQuadrant('sin', 0.5, 3)).toThrow(/wrong sign/)
    expect(() => F.inverseInQuadrant('sin', 2, 1)).toThrow(/cannot be/)
  })

  it('lists every solution in an interval', () => {
    const ts = F.trigSolutions('tan', 1, -2 * Math.PI, 2 * Math.PI)
    expect(ts.length).toBe(4)
    for (const t of ts) expect(Math.tan(t)).toBeCloseTo(1, 12)
    expect(F.trigSolutions('sin', 1)).toHaveLength(1)
    expect(F.trigSolutions('sin', 0, 0, 2 * Math.PI)).toHaveLength(3)
  })

  it('wraps into one turn for any angle', () => {
    fc.assert(
      fc.property(
        fc.double({ min: -100, max: 100, noNaN: true }).filter((t) => Math.abs(t) > 1e-9),
        (t) => {
          const s = item(F.wrapAngle(t))
          const p = item(F.wrapAngle(t, 'positive'))
          expect(s).toBeGreaterThanOrEqual(-Math.PI)
          expect(s).toBeLessThan(Math.PI)
          expect(p).toBeGreaterThanOrEqual(0)
          expect(p).toBeLessThan(2 * Math.PI)
          expect(Math.cos(s)).toBeCloseTo(Math.cos(t), 10)
          expect(Math.sin(p)).toBeCloseTo(Math.sin(t), 10)
        },
      ),
    )
  })
})

describe('identities', () => {
  it('every catalogued identity holds on a grid', () => {
    for (const identity of F.trigIdentities) {
      const r = F.checkIdentity(identity)
      expect(r.holds, identity.name).toBe(true)
      expect(r.samples).toBeGreaterThan(100)
    }
  })

  it('finds a counterexample to a false identity', () => {
    const r = F.checkIdentity({ lhs: (a, b) => sin(mul(a, b)), rhs: (a, b) => mul(sin(a), sin(b)) })
    expect(r.holds).toBe(false)
    expect(r.counterexample).not.toBeNull()
  })
})

describe('triangles', () => {
  const close = (t: F.Triangle | undefined, want: Partial<F.Triangle>): void => {
    expect(t).toBeDefined()
    for (const [k, v] of Object.entries(want)) expect(t?.[k as keyof F.Triangle] as number).toBeCloseTo(v as number, 12)
  }

  it('classifies all five cases', () => {
    expect(F.triangleCase({ a: 1, b: 1, c: 1 })).toBe('SSS')
    expect(F.triangleCase({ b: 1, c: 1, A: 1 })).toBe('SAS')
    expect(F.triangleCase({ B: 1, C: 1, a: 1 })).toBe('ASA')
    expect(F.triangleCase({ A: 1, B: 1, a: 1 })).toBe('AAS')
    expect(F.triangleCase({ a: 1, c: 1, C: 1 })).toBe('SSA')
    expect(() => F.triangleCase({ A: 1, B: 1, C: 1 })).toThrow()
    expect(() => F.triangleCase({ a: 1, b: 1 })).toThrow()
  })

  it('solves every case consistently with the 3-4-5 triangle', () => {
    const [ref] = F.solveTriangle({ a: 3, b: 4, c: 5 })
    if (ref === undefined) throw new Error('no triangle')
    close(ref, { area: 6, perimeter: 12, C: Math.PI / 2 })
    const { a, b, c, A, B, C } = ref
    close(F.solveTriangle({ a, b, C })[0], { c, A, B })
    close(F.solveTriangle({ a, c, B })[0], { b, A, C })
    close(F.solveTriangle({ A, B, c })[0], { a, b, C })
    close(F.solveTriangle({ B, C, a })[0], { b, c, A })
    close(F.solveTriangle({ A, B, a })[0], { b, c, C })
    close(F.solveTriangle({ A, C, b })[0], { a, c, B })
    // SSA with the right angle opposite the known longer side: one solution.
    const ssa = F.solveTriangle({ b, c, C })
    expect(ssa).toHaveLength(1)
    close(ssa[0], { a, A, B })
  })

  it('returns both triangles of the ambiguous case', () => {
    const two = F.solveTriangle({ a: 6, b: 8, A: Math.PI / 6 })
    expect(two).toHaveLength(2)
    for (const t of two) {
      expect(t.a / Math.sin(t.A)).toBeCloseTo(t.b / Math.sin(t.B), 12)
      expect(t.A + t.B + t.C).toBeCloseTo(Math.PI, 12)
    }
    expect(two[0]?.B ?? 0).toBeCloseTo(Math.PI - (two[1]?.B ?? 0), 12)
    // Right-angled tangent case: exactly one.
    expect(F.solveTriangle({ a: 4, b: 8, A: Math.PI / 6 })).toHaveLength(1)
    expect(F.solveTriangle({ a: 1, b: 2, c: 5 })).toEqual([])
  })

  it('places the vertices so the sides have the right lengths', () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0.5, max: 3, noNaN: true }),
        fc.double({ min: 0.5, max: 3, noNaN: true }),
        fc.double({ min: 0.1, max: 3, noNaN: true }),
        (b, c, A) => {
          const [t] = F.solveTriangle({ b, c, A })
          if (t === undefined) throw new Error('no triangle')
          const [p, q, r] = t.vertices
          expect(Math.hypot(q[0] - r[0], q[1] - r[1])).toBeCloseTo(t.a, 10)
          expect(Math.hypot(p[0] - r[0], p[1] - r[1])).toBeCloseTo(t.b, 10)
          expect(Math.hypot(p[0] - q[0], p[1] - q[1])).toBeCloseTo(t.c, 10)
          expect(item(F.heronArea(t.a, t.b, t.c))).toBeCloseTo(t.area, 9)
          expect(item(F.lawOfCosines(b, c, A))).toBeCloseTo(t.a, 10)
          expect(item(F.lawOfCosinesAngle(b, c, t.a))).toBeCloseTo(A, 8)
          expect(item(F.lawOfSines(t.a, A, t.B))).toBeCloseTo(t.b, 10)
        },
      ),
    )
  })

  it("keeps Heron's digits on a needle", () => {
    // Kahan's example: a = 1e5, b = 99999.99979, c = 0.00029 has area about 10.
    expect(item(F.heronArea(100000, 99999.99979, 0.00029))).toBeCloseTo(9.999999999, 3)
    expect(Number.isNaN(item(F.heronArea(1, 1, 3)))).toBe(true)
  })
})

describe('derivatives', () => {
  it('passes the conformance check', () => {
    expectConformant(F.radians, [tensor([10, -30])])
    expectConformant((t: Value) => F.wrapAngle(t), [tensor([0.4, 4])])
    expectConformant(F.unitCirclePoint, [tensor([0.3, 2])])
    expectConformant(F.directionAngle, [tensor([0.3, -1]), tensor([1, 0.5])])
    expectConformant(F.lawOfCosines, [tensor([2, 3]), tensor([3, 2]), tensor([0.7, 1.2])])
    expectConformant(F.lawOfCosinesAngle, [tensor([3, 4]), tensor([4, 4]), tensor([5, 3])])
    expectConformant(F.lawOfSines, [tensor([2]), tensor([0.6]), tensor([1.1])])
    expectConformant(F.heronArea, [tensor([3, 5]), tensor([4, 6.5]), tensor([5.5, 7])])
  })

  it('unit-circle points have norm one', () => {
    const p = toArray(F.unitCirclePoint([0.1, 2, -4])) as number[][]
    for (const [x, y] of p) expect(Math.hypot(x ?? 0, y ?? 0)).toBeCloseTo(1, 15)
  })
})
