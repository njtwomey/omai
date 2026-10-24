import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import {
  type ComplexValue,
  cDiv,
  cExp,
  cMul,
  complex,
  mul,
  neg,
  onesLike,
  stack,
  sum,
  tensor,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import * as F from './index.ts'

const one = (w: ComplexValue): ComplexValue => complex(onesLike(w.re))
const flat = (w: ComplexValue): [number[], number[]] => [toFlat(w.re), toFlat(w.im)]

describe('complex scalars', () => {
  it('roots of unity sum to zero and their powers cycle', () => {
    for (const n of [2, 3, 5, 8]) {
      const [re, im] = flat(F.rootsOfUnity(n))
      expect(re.reduce((s, v) => s + v, 0)).toBeCloseTo(0, 14)
      expect(im.reduce((s, v) => s + v, 0)).toBeCloseTo(0, 14)
      const w = F.complexPow(F.rootsOfUnity(n), n)
      for (const v of toFlat(w.re)) expect(v).toBeCloseTo(1, 12)
    }
    expect(() => F.nthRoots(1, 0)).toThrow()
  })

  it('n-th roots raised to the n-th power give the number back', () => {
    fc.assert(
      fc.property(
        fc.double({ min: -5, max: 5, noNaN: true }),
        fc.double({ min: -5, max: 5, noNaN: true }),
        fc.integer({ min: 1, max: 7 }),
        (re, im, n) => {
          if (Math.hypot(re, im) < 1e-3) return
          const roots = F.nthRoots({ re, im }, n)
          let p = roots
          for (let k = 1; k < n; k++) p = cMul(p, roots)
          for (const v of toFlat(p.re)) expect(v).toBeCloseTo(re, 9)
          for (const v of toFlat(p.im)) expect(v).toBeCloseTo(im, 9)
        },
      ),
    )
  })

  it('exp and log are inverse, and sqrt squares back', () => {
    const w = complex([0.3, -2, 1.5], [1.2, 0.4, -2.5])
    const back = cExp(F.complexLog(w))
    const sq = F.complexSqrt(w)
    const s2 = cMul(sq, sq)
    for (const [i, v] of toFlat(w.re).entries()) {
      expect(toFlat(back.re)[i]).toBeCloseTo(v, 14)
      expect(toFlat(s2.re)[i]).toBeCloseTo(v, 14)
    }
    for (const v of toFlat(sq.re)) expect(v).toBeGreaterThanOrEqual(0)
    expect(toFlat(F.complexPow(complex([0], [0]), complex([2], [0])).re)).toEqual([0])
    expect(toFlat(F.complexSqrt(complex([0], [0])).re)).toEqual([0])
    // sin^2 + cos^2 = 1 for complex arguments too.
    const s = F.complexSin(w)
    const c = F.complexCos(w)
    const total = toFlat(sum(stack([cMul(s, s).re, cMul(c, c).re]), 0))
    for (const v of total) expect(v).toBeCloseTo(1, 12)
  })
})

describe('Mobius transformations', () => {
  const m = F.mobius({ re: 1, im: 1 }, 2, { re: 0, im: -1 }, 3)

  it('compose with their inverse to the identity and fix their fixed points', () => {
    const id = F.mobiusCompose(m, F.mobiusInverse(m))
    expect(F.classifyMobius(id).kind).toBe('identity')
    const { points } = F.mobiusFixedPoints(m)
    for (const p of points) {
      const w = F.mobiusApply(m, complex([p.re], [p.im]))
      expect(toFlat(w.re)[0]).toBeCloseTo(p.re, 12)
      expect(toFlat(w.im)[0]).toBeCloseTo(p.im, 12)
    }
    expect(() => F.mobius(1, 2, 2, 4)).toThrow()
  })

  it('classifies by the trace', () => {
    expect(F.classifyMobius(F.mobius({ re: Math.cos(0.4), im: Math.sin(0.4) }, 0, 0, 1)).kind).toBe('elliptic')
    expect(F.classifyMobius(F.mobius({ re: 2, im: 1 }, 0, 0, 1)).kind).toBe('loxodromic')
    expect(F.mobiusFixedPoints(F.mobius(1, 1, 0, 1))).toEqual({ points: [], infinite: true })
    expect(F.mobiusFixedPoints(F.mobius(2, 1, 0, 1)).points).toEqual([{ re: -1, im: 0 }])
  })

  it('preserves cross-ratios: three points determine the map', () => {
    const src: [number, number, number] = [0, 1, -1]
    const dst: [F.Mobius['a'], F.Mobius['a'], F.Mobius['a']] = [
      { re: 0, im: 1 },
      { re: 2, im: 0 },
      { re: 1, im: -1 },
    ]
    const fit = F.mobiusFromPoints(src, dst)
    for (const [i, s] of src.entries()) {
      const w = F.mobiusApply(fit, [s])
      expect(toFlat(w.re)[0]).toBeCloseTo(dst[i]?.re ?? 0, 12)
      expect(toFlat(w.im)[0]).toBeCloseTo(dst[i]?.im ?? 0, 12)
    }
  })
})

describe('conformal maps and pictures', () => {
  it('grid images meet at right angles under an analytic map', () => {
    const g = F.conformalGrid(
      F.conformalMaps.exp,
      [
        [0, 1],
        [0, 1],
      ],
      { lines: 3, samples: 41 },
    )
    // The middle horizontal line and the middle vertical line cross at their middle samples.
    const h = g.after[1] ?? []
    const v = g.after[4] ?? []
    const [hx, hy] = [(h[21]?.[0] ?? 0) - (h[19]?.[0] ?? 0), (h[21]?.[1] ?? 0) - (h[19]?.[1] ?? 0)]
    const [vx, vy] = [(v[21]?.[0] ?? 0) - (v[19]?.[0] ?? 0), (v[21]?.[1] ?? 0) - (v[19]?.[1] ?? 0)]
    const cos = (hx * vx + hy * vy) / (Math.hypot(hx, hy) * Math.hypot(vx, vy))
    expect(cos).toBeCloseTo(0, 6)
    expect(g.before).toHaveLength(6)
  })

  it('detects failures of the Cauchy-Riemann equations and critical points', () => {
    const conj = F.cauchyRiemann((w) => ({ re: w.re, im: neg(w.im) }), { re: 0.5, im: 0.5 })
    expect(conj.residual).toBeCloseTo(2, 12)
    expect(F.cauchyRiemann(F.conformalMaps.square, 0).conformal).toBe(false)
    expect(F.cauchyRiemann(F.conformalMaps.inversion, { re: 1, im: 1 }).conformal).toBe(true)
  })

  it('colours a zero of order k with every hue k times', () => {
    const d = F.domainColouring(
      (w) => cMul(w, cMul(w, w)),
      [
        [-1, 1],
        [-1, 1],
      ],
      { size: [9, 9] },
    )
    expect(d.pixels.width).toBe(9)
    expect(d.modulus[4]?.[4]).toBe(0)
    // Going once round the centre the argument of z^3 winds three times.
    const ring = [
      [0, 4],
      [0, 8],
      [4, 8],
      [8, 8],
      [8, 4],
      [8, 0],
      [4, 0],
      [0, 0],
    ]
    let turns = 0
    for (let i = 0; i < ring.length; i++) {
      const [r0 = 0, c0 = 0] = ring[i] ?? []
      const [r1 = 0, c1 = 0] = ring[(i + 1) % ring.length] ?? []
      let step = (d.argument[r1]?.[c1] ?? 0) - (d.argument[r0]?.[c0] ?? 0)
      while (step > Math.PI) step -= 2 * Math.PI
      while (step < -Math.PI) step += 2 * Math.PI
      turns += step
    }
    expect(Math.abs(turns / (2 * Math.PI))).toBeCloseTo(3, 6)
    expect(
      F.domainColouring(
        (w) => cDiv(one(w), w),
        [
          [-1, 1],
          [-1, 1],
        ],
        { size: [3, 3], bands: false },
      ).pixels.rgba[16],
    ).toBe(255)
  })
})

describe('contour integrals and residues', () => {
  it("Cauchy's theorem, the residue theorem and the essential singularity of exp(1/z)", () => {
    const analytic = F.contourIntegral(cExp, F.circleContour({ re: 0.3, im: -0.2 }, 2))
    expect(Math.hypot(analytic.re, analytic.im)).toBeLessThan(1e-12)
    const essential = F.residueAt((w) => cExp(cDiv(one(w), w)), 0, { radius: 1 })
    expect(essential.re).toBeCloseTo(1, 12)
    // The integral round a circle enclosing both poles of 1/(z(z-1)) is 2 pi i times (-1 + 1) = 0.
    const f: F.ComplexFunction = (w) => cDiv(one(w), cMul(w, { re: w.re, im: w.im }))
    expect(F.contourIntegral(f, F.circleContour(0, 3)).im).toBeCloseTo(0, 12)
  })

  it('winding numbers and the argument principle count with multiplicity and sign', () => {
    const twice: F.ComplexContour = (t) => F.circleContour(0, 1)(mul(t, 2))
    expect(F.windingNumber(twice, 0.2).winding).toBe(2)
    const meromorphic: F.ComplexFunction = (w) => cDiv(cMul(w, w), { re: w.re, im: w.im })
    expect(F.argumentPrinciple(meromorphic, F.circleContour(0, 1)).count).toBe(1)
    const pole: F.ComplexFunction = (w) => cDiv(one(w), cMul(w, w))
    expect(F.argumentPrinciple(pole, F.circleContour(0, 1)).count).toBe(-2)
  })
})

describe('derivatives', () => {
  it('passes the conformance check', () => {
    const re = tensor([0.5, -1.2, 2])
    const im = tensor([1.1, 0.4, -0.7])
    const asParts =
      (f: (w: ComplexValue) => ComplexValue) =>
      (a: Value, b: Value): Value => {
        const w = f({ re: a, im: b })
        return stack([w.re, w.im])
      }
    expectConformant(asParts(F.complexLog), [re, im])
    expectConformant(asParts(F.complexSqrt), [re, im])
    expectConformant(asParts(F.complexSin), [re, im])
    expectConformant(asParts(F.complexCos), [re, im])
    expectConformant(
      asParts((w) => F.complexPow(w, complex([0.5], [0.3]))),
      [re, im],
    )
    expectConformant(
      asParts((w) => F.mobiusApply(F.mobius({ re: 1, im: 1 }, 2, { re: 0, im: -1 }, 3), w)),
      [re, im],
    )
    expectConformant(asParts(F.conformalMaps.joukowski), [re, im])
    expectConformant(
      (a: Value, b: Value) => stack([F.toPolarForm({ re: a, im: b }).r, F.toPolarForm({ re: a, im: b }).theta]),
      [re, im],
    )
  })
})
