import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { polyFromRoots } from '../numerics/polynomial/index.ts'
import * as S from './index.ts'

// Polynomials built from chosen roots, kept away from the stability boundary so the verdicts are unambiguous.
const realRoot = fc.oneof(fc.double({ min: -3, max: -0.2, noNaN: true }), fc.double({ min: 0.2, max: 3, noNaN: true }))
const pair = fc.record({
  re: fc.oneof(fc.double({ min: -2, max: -0.2, noNaN: true }), fc.double({ min: 0.2, max: 2, noNaN: true })),
  im: fc.double({ min: 0.3, max: 2, noNaN: true }),
})
const rootsArb = fc
  .record({ reals: fc.array(realRoot, { maxLength: 3 }), pairs: fc.array(pair, { maxLength: 2 }) })
  .filter((r) => r.reals.length + 2 * r.pairs.length >= 1)

const polyOf = (r: { reals: number[]; pairs: { re: number; im: number }[] }): number[] => {
  const re = [...r.reals, ...r.pairs.flatMap((q) => [q.re, q.re])]
  const im = [...r.reals.map(() => 0), ...r.pairs.flatMap((q) => [q.im, -q.im])]
  return polyFromRoots({ re, im }).reverse()
}

describe('stability tests agree with the pole locations', () => {
  it('Routh counts the right half-plane roots', () => {
    fc.assert(
      fc.property(rootsArb, (r) => {
        const unstable = r.reals.filter((x) => x > 0).length + 2 * r.pairs.filter((q) => q.re > 0).length
        const result = S.routh(polyOf(r))
        expect(result.signChanges).toBe(unstable)
        expect(result.stable).toBe(unstable === 0)
        expect(S.isStable(S.tf([1], polyOf(r)))).toBe(unstable === 0)
      }),
      { numRuns: 60 },
    )
  })

  it('Jury says stable exactly when every root is inside the unit circle', () => {
    fc.assert(
      fc.property(rootsArb, fc.double({ min: 0.3, max: 1.6, noNaN: true }), (r, scale) => {
        // Scale the roots into and out of the unit disc.
        const scaled = {
          reals: r.reals.map((x) => (x * scale) / 3),
          pairs: r.pairs.map((q) => ({ re: (q.re * scale) / 3, im: (q.im * scale) / 3 })),
        }
        const moduli = [...scaled.reals.map(Math.abs), ...scaled.pairs.map((q) => Math.hypot(q.re, q.im))]
        fc.pre(moduli.every((m) => Math.abs(m - 1) > 0.02))
        const inside = moduli.every((m) => m < 1)
        expect(S.jury(polyOf(scaled)).stable).toBe(inside)
        expect(S.isStable(S.tf([1], polyOf(scaled), { dt: 1 }))).toBe(inside)
      }),
      { numRuns: 60 },
    )
  })

  it('poles of a realisation are the roots of its denominator', () => {
    fc.assert(
      fc.property(rootsArb, (r) => {
        const den = polyOf(r)
        const fromSs = S.poles(S.toSs(S.tf([1], den)))
        const fromTf = S.poles(S.tf([1], den))
        for (const [i, v] of fromSs.re.entries()) expect(v).toBeCloseTo(fromTf.re[i] ?? Number.NaN, 6)
      }),
      { numRuns: 30 },
    )
  })
})
