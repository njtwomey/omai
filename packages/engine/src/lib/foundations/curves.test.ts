import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { add, cos, mul, sin, stack, sub, sum, tensor, toArray, toFlat, type Value } from '../tensor/index.ts'
import * as F from './index.ts'

const helix: F.ParametricCurve = (t) => stack([mul(cos(t), 2), mul(sin(t), 2), mul(t, 0.5)], -1)
const ellipse: F.ParametricCurve = (t) => stack([mul(cos(t), 3), sin(t)], -1)
const dot = (a: readonly number[], b: readonly number[]): number => a.reduce((s, x, i) => s + x * (b[i] ?? 0), 0)

describe('parametric curves', () => {
  it('a helix has constant curvature and torsion a/(a^2+b^2), b/(a^2+b^2)', () => {
    const f = F.frenetFrame(helix, [0.1, 1, 2.5])
    for (const k of f.curvature) expect(k).toBeCloseTo(2 / 4.25, 13)
    for (const tau of f.torsion ?? []) expect(tau).toBeCloseTo(0.5 / 4.25, 13)
    for (let i = 0; i < 3; i++) {
      const [tt, nn, bb] = [f.tangent[i] ?? [], f.normal[i] ?? [], f.binormal?.[i] ?? []]
      expect(dot(tt, tt)).toBeCloseTo(1, 14)
      expect(dot(nn, nn)).toBeCloseTo(1, 14)
      expect(dot(tt, nn)).toBeCloseTo(0, 14)
      expect(dot(bb, tt)).toBeCloseTo(0, 14)
      // The normal of a helix points at its axis.
      const p = f.point[i] ?? []
      expect(dot(nn, [-(p[0] ?? 0), -(p[1] ?? 0), 0]) / 2).toBeCloseTo(1, 13)
    }
    expect(F.arcLength(helix, 0, 2 * Math.PI)).toBeCloseTo(2 * Math.PI * Math.sqrt(4.25), 10)
  })

  it('obeys the Frenet-Serret equations: dT/ds = k N', () => {
    const h = 1e-5
    const f0 = F.frenetFrame(helix, [1 - h, 1 + h])
    const speed = toFlat(F.curveSpeed(helix, [1]))[0] ?? 1
    const f = F.frenetFrame(helix, [1])
    const dT = (f0.tangent[1] ?? []).map((v, j) => (v - (f0.tangent[0]?.[j] ?? 0)) / (2 * h * speed))
    for (const [j, v] of dT.entries()) expect(v).toBeCloseTo((f.curvature[0] ?? 0) * (f.normal[0]?.[j] ?? 0), 8)
  })

  it('the osculating circle of an ellipse and its evolute (an astroid)', () => {
    const c = F.osculatingCircle(ellipse, [0, Math.PI / 2])
    expect(c.radii[0]).toBeCloseTo(1 / 3, 13)
    expect(c.radii[1]).toBeCloseTo(9, 12)
    // The evolute of x = a cos t, y = b sin t is ((a^2 - b^2)/a cos^3 t, (b^2 - a^2)/b sin^3 t).
    const ev = F.evolute(ellipse, 0.3, 1.2, 4)
    const ts = toFlat(tensor([0.3, 0.6, 0.9, 1.2]))
    for (const [i, p] of ev.entries()) {
      const t = ts[i] ?? 0
      expect(p[0]).toBeCloseTo((8 / 3) * Math.cos(t) ** 3, 12)
      expect(p[1]).toBeCloseTo(-8 * Math.sin(t) ** 3, 11)
    }
    expect(F.sampleCurve(ellipse, 0, 1, 5).points).toHaveLength(5)
    expect(F.cumulativeArcLength(ellipse, [0, Math.PI / 2, Math.PI])[2]).toBeCloseTo(
      F.arcLength(ellipse, 0, Math.PI),
      10,
    )
  })

  it('signed curvature changes sign with orientation', () => {
    const reversed: F.ParametricCurve = (t) => ellipse(mul(t, -1))
    const k = toFlat(F.curvature(ellipse, [0.4]))[0] ?? 0
    expect(k).toBeGreaterThan(0)
    expect(toFlat(F.curvature(reversed, [-0.4]))[0]).toBeCloseTo(-k, 13)
    expect(() => F.frenetFrame((t: Value) => stack([t, t, t, t], -1), [0])).toThrow()
  })
})

describe('implicit curves', () => {
  const unit = (p: Value): Value => sub(sum(mul(p, p), -1), 1)
  const ell = (p: Value): Value => {
    const q = mul(p, tensor([1 / 3, 1]))
    return sub(sum(mul(q, q), -1), 1)
  }

  it('traces a closed curve whose points satisfy the equation', () => {
    const [loop] = F.traceImplicitCurve(unit, [
      [-1.5, 1.5],
      [-1.5, 1.5],
    ])
    if (loop === undefined) throw new Error('no curve')
    expect(loop[0]).toEqual(loop[loop.length - 1])
    for (const [x, y] of loop) expect(Math.hypot(x, y)).toBeCloseTo(1, 3)
  })

  it('agrees with the parametric curvature', () => {
    const g = F.implicitCurveGeometry(ell, [3 * Math.cos(0.7), Math.sin(0.7)])
    const k = toFlat(F.curvature(ellipse, [0.7]))[0] ?? 0
    expect(Math.abs(g.curvature)).toBeCloseTo(k, 12)
    expect(dot(g.tangent, g.normal)).toBeCloseTo(0, 15)
    expect(() => F.implicitCurveGeometry((p: Value) => sum(mul(p, p), -1), [0, 0])).toThrow()
  })
})

describe('conic sections', () => {
  it('the points of a classified conic satisfy its equation', () => {
    for (const coef of [
      [2, 1, 3, -1, 2, -5],
      [1, 4, -2, 0, 3, -1],
      [1, 2, 1, -3, 4, 1],
      [1, 0, 1, -2, 4, 1],
    ] as const) {
      const c = F.classifyConic(coef)
      const [A, B, C, D, E, G] = coef
      for (const branch of F.conicPoints(c, 20)) {
        for (const [x, y] of branch) {
          const value = A * x * x + B * x * y + C * y * y + D * x + E * y + G
          expect(Math.abs(value) / (1 + x * x + y * y)).toBeLessThan(1e-10)
        }
      }
    }
  })

  it('focus-directrix property: distance to focus = e * distance to directrix', () => {
    for (const coef of [
      [2, 1, 3, -1, 2, -5],
      [1, 4, -2, 0, 3, -1],
      [1, 2, 1, -3, 4, 1],
    ] as const) {
      const c = F.classifyConic(coef)
      const branch = F.conicPoints(c, 9)[0] ?? []
      const focus = c.foci[c.foci.length - 1] ?? [0, 0]
      const [a, b, d] = c.directrices[c.directrices.length - 1] ?? [0, 0, 0]
      for (const [x, y] of branch) {
        const toFocus = Math.hypot(x - focus[0], y - focus[1])
        const toLine = Math.abs(a * x + b * y - d) / Math.hypot(a, b)
        // Only the branch on the focus's side satisfies it with this focus; others use the mirror pair.
        if (c.type !== 'hyperbola' || x * (focus[0] - (c.centre?.[0] ?? 0)) > 0) {
          expect(toFocus).toBeCloseTo(c.eccentricity * toLine, 9)
        }
      }
    }
  })

  it('names degenerate cases and rejects non-conics', () => {
    expect(F.classifyConic([1, 0, 0, 0, 0, 0]).type).toBe('coincident lines')
    expect(F.classifyConic([1, 0, 0, 0, 0, 1]).type).toBe('empty')
    expect(() => F.classifyConic([0, 0, 0, 1, 1, 1])).toThrow()
  })
})

describe('Bezier curves', () => {
  const control = [
    [0, 0],
    [1, 3],
    [3, 3],
    [4, 0],
  ]

  it('Bernstein polynomials partition unity and match the curve', () => {
    fc.assert(
      fc.property(fc.double({ min: 0, max: 1, noNaN: true }), (t) => {
        const b = [0, 1, 2, 3].map((i) => toFlat(F.bernstein(3, i, [t]))[0] ?? 0)
        expect(b.reduce((s, x) => s + x, 0)).toBeCloseTo(1, 14)
        const p = toFlat(F.bezier(control, [t]))
        const viaBasis = [0, 1].map((j) => b.reduce((s, w, i) => s + w * (control[i]?.[j] ?? 0), 0))
        for (const [j, v] of p.entries()) expect(v).toBeCloseTo(viaBasis[j] ?? 0, 13)
        expect(F.deCasteljau(control, t).point[0]).toBeCloseTo(p[0] ?? 0, 13)
      }),
    )
    expect(() => F.bernstein(2, 3, 0.5)).toThrow()
  })

  it('splitting and degree elevation keep the curve', () => {
    const { left, right } = F.bezierSplit(control, 0.3)
    const elevated = F.bezierElevate(control)
    expect(elevated).toHaveLength(5)
    for (const s of [0, 0.25, 0.5, 1]) {
      const onLeft = toFlat(F.bezier(left, [s]))
      const whole = toFlat(F.bezier(control, [0.3 * s]))
      for (const [j, v] of onLeft.entries()) expect(v).toBeCloseTo(whole[j] ?? 0, 13)
      const onRight = toFlat(F.bezier(right, [s]))
      const whole2 = toFlat(F.bezier(control, [0.3 + 0.7 * s]))
      for (const [j, v] of onRight.entries()) expect(v).toBeCloseTo(whole2[j] ?? 0, 13)
      const el = toFlat(F.bezier(elevated, [s]))
      const orig = toFlat(F.bezier(control, [s]))
      for (const [j, v] of el.entries()) expect(v).toBeCloseTo(orig[j] ?? 0, 13)
    }
    expect(F.deCasteljau(control, 0.5).levels.map((l) => l.length)).toEqual([4, 3, 2, 1])
    expect(() => F.deCasteljau([], 0.5)).toThrow()
    expect(() => F.bezierDerivative([[1, 1]])).toThrow()
  })

  it('the hodograph is the derivative', () => {
    const curve: F.ParametricCurve = (t) => F.bezier(control, t)
    const d = toArray(F.curveDerivatives(curve, [0.4], 1)[1] as Value) as number[][]
    const h = toFlat(F.bezier(F.bezierDerivative(control), [0.4]))
    for (const [j, v] of h.entries()) expect(v).toBeCloseTo(d[0]?.[j] ?? 0, 13)
  })
})

describe('derivatives', () => {
  it('passes the conformance check', () => {
    expectConformant(F.bezier, [
      tensor([
        [0, 0],
        [1, 2],
        [3, 1],
      ]),
      tensor([0.2, 0.7]),
    ])
    expectConformant((t) => F.bernstein(4, 2, t), [tensor([0.1, 0.6])])
    expectConformant((t) => F.curveSpeed(helix, t), [tensor([0.3, 1.2])])
    expectConformant((t) => F.curvature(ellipse, t), [tensor([0.3, 1.2])])
    expectConformant(
      (t) => F.torsion((s: Value) => stack([s, mul(s, s), mul(s, mul(s, s))], -1), t),
      [tensor([0.5, 1])],
    )
    expectConformant((t) => add(F.curveDerivatives(ellipse, t, 2)[2] as Value, 0), [tensor([0.3, 2])])
  })
})
