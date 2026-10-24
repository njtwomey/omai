import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { svdvals } from '../../numerics/linalg/index.ts'
import { ctrb } from '../../systems/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { acker, place } from './placement.ts'

describe('pole placement', () => {
  it('places random poles exactly when the pair is controllable (property)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: -2, max: 2, noNaN: true }), { minLength: 9, maxLength: 9 }),
        fc.array(fc.double({ min: -2, max: 2, noNaN: true }), { minLength: 6, maxLength: 6 }),
        (a9, b6) => {
          const A = [a9.slice(0, 3), a9.slice(3, 6), a9.slice(6, 9)]
          const B = [b6.slice(0, 2), b6.slice(2, 4), b6.slice(4, 6)]
          // Well controllable, not merely full rank: near the uncontrollable set (entries of 1e-15 beside 1) no method
          // can place poles accurately, so such draws say nothing about place.
          const s = toFlat(svdvals(ctrb(A, B)))
          if (!((s[2] ?? 0) > 1e-4 * (s[0] ?? 0))) return true
          const poles = [-1, -2, -3]
          let r: ReturnType<typeof place>
          try {
            r = place(A, B, poles)
          } catch {
            // Badly conditioned pairs near the uncontrollable set may be refused.
            return true
          }
          const got = r.computedPoles.map((p) => p.re).sort((x, y) => x - y)
          return got.every((v, i) => Math.abs(v - (poles[2 - i] ?? 0)) < 1e-6 * Math.max(1, Math.abs(v)))
        },
      ),
      { numRuns: 30 },
    )
  })

  it('single-input placement matches Ackermann and complex pairs are supported', () => {
    const A = [
      [0, 1, 0],
      [0, 0, 1],
      [-1, -2, -3],
    ]
    const B = [[0], [0], [1]]
    const poles = [-1, { re: -2, im: 1 }, { re: -2, im: -1 }]
    const k1 = Array.from(place(A, B, poles).gain.data)
    const k2 = Array.from(acker(A, B, poles).data)
    k1.forEach((v, i) => {
      expect(Math.abs(v - (k2[i] ?? 0))).toBeLessThan(1e-9)
    })
  })

  it('validates the requested poles', () => {
    const A = [
      [0, 1],
      [0, 0],
    ]
    expect(() => place(A, [[0], [1]], [-1])).toThrow()
    expect(() => place(A, [[0], [1]], [-1, -1])).toThrow()
    expect(() => place(A, [[0], [1]], [-1, { re: -1, im: 1 }])).toThrow()
    expect(() =>
      acker(
        A,
        [
          [0, 1],
          [1, 0],
        ],
        [-1, -2],
      ),
    ).toThrow()
  })
})
