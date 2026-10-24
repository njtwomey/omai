import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../../testing/arbitrary.ts'
import { toFlat } from '../../tensor/index.ts'
import * as I from './index.ts'

const binaryImage = (h: number, w: number): fc.Arbitrary<number[][]> =>
  fc.array(fc.array(fc.integer({ min: 0, max: 1 }), { minLength: w, maxLength: w }), { minLength: h, maxLength: h })

const count = (x: ArrayLike<number>): number => Array.from(x).reduce((a, b) => a + b, 0)

describe('morphology', () => {
  it('opening is anti-extensive and closing extensive away from the border', () => {
    fc.assert(
      fc.property(binaryImage(6, 7), (rows) => {
        const x = rows.flat()
        const open = toFlat(I.binaryOpening(rows))
        const close = toFlat(I.binaryClosing(rows))
        for (const [i, v] of x.entries()) {
          expect(open[i] ?? 0).toBeLessThanOrEqual(v)
          const r = Math.floor(i / 7)
          const c = i % 7
          // With border value 0 the erosion of a closing eats into sets touching the edge (as scipy).
          if (r > 1 && r < 4 && c > 1 && c < 5) expect(close[i] ?? 0).toBeGreaterThanOrEqual(v)
        }
      }),
      { numRuns: RUNS },
    )
  })

  it('thinning keeps the connected components; skeletons lie inside the set', () => {
    fc.assert(
      fc.property(binaryImage(8, 8), (rows) => {
        const full = I.generateBinaryStructure(2, 2)
        const before = I.label(rows, full).count
        for (const s of [I.skeletonize(rows), I.thin(rows)]) {
          const v = toFlat(s)
          const x = rows.flat()
          for (const [i, b] of v.entries()) expect(b).toBeLessThanOrEqual(x[i] ?? 0)
          expect(count(v)).toBeLessThanOrEqual(count(x))
        }
        // Zhang and Suen's conditions erase 2 x 2 blocks entirely, so only thinning preserves every component.
        expect(I.label(I.thin(rows), full).count).toBe(before)
      }),
      { numRuns: RUNS },
    )
  })

  it('the distance transform is 0 on the background and 1-Lipschitz', () => {
    fc.assert(
      fc.property(binaryImage(6, 6), (rows) => {
        const d = toFlat(I.distanceTransformEdt(rows))
        const x = rows.flat()
        for (const [i, v] of x.entries()) if (v === 0) expect(d[i]).toBe(0)
        for (let r = 0; r < 6; r++)
          for (let c = 0; c + 1 < 6; c++) {
            const a = d[r * 6 + c] ?? 0
            const b = d[r * 6 + c + 1] ?? 0
            if (Number.isFinite(a) && Number.isFinite(b)) expect(Math.abs(a - b)).toBeLessThanOrEqual(1 + 1e-12)
          }
      }),
      { numRuns: RUNS },
    )
  })
})
