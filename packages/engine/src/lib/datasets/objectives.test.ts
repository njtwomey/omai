import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { item, tensor, toFlat } from '../tensor/index.ts'
import { dampedSine, forrester, gramacyLee } from './objectives.ts'

describe('one-dimensional test functions', () => {
  const functions = [forrester(), gramacyLee(), dampedSine()]

  it('reach their minimum at the minimiser, where the gradient vanishes', () => {
    for (const f of functions) {
      const m = f.minimizers[0] as number[]
      expect(item(f.f(tensor(m))), f.name).toBeCloseTo(f.minimum, 10)
      expect(Math.abs(toFlat(f.gradient(m))[0] ?? Number.NaN), f.name).toBeLessThan(1e-6)
      expect(toFlat(f.hessian(m))[0] ?? Number.NaN, f.name).toBeGreaterThan(0)
    }
  })

  it('have no lower value in their box than the minimum', () => {
    for (const f of functions) {
      const [lo, hi] = f.bounds[0] as readonly [number, number]
      const grid = Array.from({ length: 4001 }, (_, i) => [lo + ((hi - lo) * i) / 4000])
      const values = toFlat(f.f(tensor(grid)))
      expect(Math.min(...values), f.name).toBeGreaterThanOrEqual(f.minimum - 1e-9)
    }
  })

  it('evaluate a batch as the points one by one', () => {
    fc.assert(
      fc.property(fc.array(fc.double({ min: 0.5, max: 1, noNaN: true }), { minLength: 1, maxLength: 8 }), (xs) => {
        for (const f of functions) {
          const [lo, hi] = f.bounds[0] as readonly [number, number]
          const points = xs.map((u) => [lo + u * (hi - lo)])
          const batch = toFlat(f.f(tensor(points)))
          points.forEach((p, i) => {
            expect(batch[i]).toBeCloseTo(item(f.f(tensor(p))), 12)
          })
        }
      }),
      { numRuns: 25 },
    )
  })
})
