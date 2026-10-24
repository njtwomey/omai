import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../../testing/arbitrary.ts'
import { expectConformant } from '../../../testing/check.ts'
import { tensor, toFlat } from '../../tensor/index.ts'
import * as I from './index.ts'

const image = (h: number, w: number): fc.Arbitrary<number[][]> =>
  fc.array(fc.array(fc.double({ min: -2, max: 2, noNaN: true }), { minLength: w, maxLength: w }), {
    minLength: h,
    maxLength: h,
  })

describe('image filters', () => {
  it('differentiates the linear filters in the image', () => {
    const x = tensor([
      [0.3, -1.2, 0.8, 2.1],
      [-0.4, 0.9, 1.5, -0.7],
      [0.2, 0.1, 0.6, -0.3],
    ])
    expectConformant((v) => I.gaussianFilter(v, [1, 0.7], { order: [0, 1], mode: 'mirror' }), [x])
    expectConformant((v) => I.uniformFilter(v, 2, { mode: 'wrap' }), [x])
    expectConformant((v) => I.sobel(v), [x])
    expectConformant((v) => I.gaussianLaplace(v, 0.8), [x])
    expectConformant((v) => I.gabor(v, 0.3, { theta: 0.5 }).imag, [x])
  })

  it('smoothing preserves constants and the mean under wrap borders', () => {
    fc.assert(
      fc.property(image(5, 6), fc.double({ min: 0.3, max: 2, noNaN: true }), (rows, sigma) => {
        const c = I.gaussianFilter(tensor(rows.map((r) => r.map(() => 1.5))), sigma)
        for (const v of toFlat(c)) expect(v).toBeCloseTo(1.5, 10)
        const x = tensor(rows)
        const m = (a: number[]): number => a.reduce((s, v) => s + v, 0) / a.length
        expect(m(toFlat(I.uniformFilter(x, 3, { mode: 'wrap' })))).toBeCloseTo(m(toFlat(x)), 10)
      }),
      { numRuns: RUNS },
    )
  })

  it('rank filters are ordered: minimum <= median <= maximum', () => {
    fc.assert(
      fc.property(image(4, 5), (rows) => {
        const lo = toFlat(I.minimumFilter(rows, 3))
        const mid = toFlat(I.medianFilter(rows, 3))
        const hi = toFlat(I.maximumFilter(rows, 3))
        for (let i = 0; i < lo.length; i++) {
          expect(lo[i] ?? 0).toBeLessThanOrEqual(mid[i] ?? 0)
          expect(mid[i] ?? 0).toBeLessThanOrEqual(hi[i] ?? 0)
        }
      }),
      { numRuns: RUNS },
    )
  })
})
