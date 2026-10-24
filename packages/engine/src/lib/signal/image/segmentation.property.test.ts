import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../../testing/arbitrary.ts'
import { toFlat } from '../../tensor/index.ts'
import * as I from './index.ts'

describe('segmentation', () => {
  it('graph cuts reach the brute-force minimum energy', () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: 0, max: 1, noNaN: true }), { minLength: 6, maxLength: 6 }),
        fc.double({ min: 0.01, max: 0.5, noNaN: true }),
        (xs, lambda) => {
          const img = [xs.slice(0, 3), xs.slice(3, 6)]
          const opts = { foreground: 0.8, background: 0.2, lambda, sigma: 0.3 }
          const r = I.graphCut(img, opts)
          const energy = (labels: number[]): number => {
            let e = 0
            for (const [p, l] of labels.entries()) e += ((xs[p] ?? 0) - (l === 1 ? 0.8 : 0.2)) ** 2
            const pairs = [
              [0, 1],
              [1, 2],
              [3, 4],
              [4, 5],
              [0, 3],
              [1, 4],
              [2, 5],
            ]
            for (const [p = 0, q = 0] of pairs)
              if (labels[p] !== labels[q]) e += lambda * Math.exp(-(((xs[p] ?? 0) - (xs[q] ?? 0)) ** 2) / (2 * 0.09))
            return e
          }
          let best = Number.POSITIVE_INFINITY
          for (let m = 0; m < 64; m++) best = Math.min(best, energy(Array.from({ length: 6 }, (_, i) => (m >> i) & 1)))
          expect(r.energy).toBeCloseTo(best, 9)
          expect(energy(toFlat(r.mask))).toBeCloseTo(best, 9)
        },
      ),
      { numRuns: RUNS },
    )
  })

  it('watershed labels every pixel with a marker label', () => {
    fc.assert(
      fc.property(fc.array(fc.double({ min: 0, max: 1, noNaN: true }), { minLength: 20, maxLength: 20 }), (xs) => {
        const img = [xs.slice(0, 5), xs.slice(5, 10), xs.slice(10, 15), xs.slice(15, 20)]
        const markers = [
          [1, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 2],
        ]
        const labels = toFlat(I.watershed(img, markers))
        expect(labels.every((l) => l === 1 || l === 2)).toBe(true)
      }),
      { numRuns: RUNS },
    )
  })
})
