import { describe, expect, it } from 'vitest'
import { toArray } from '../../tensor/index.ts'
import * as L from './index.ts'

// Singular values of [[1, 2], [3, 4], [5, 6]] (numpy.linalg.svd).
const S1 = 9.525518091565107
const S2 = 0.5143005806586446

describe('linalg at the edges of the double range', () => {
  const scales = [1e77, 1e100, 1e200, 1e300, 1e-162, 1e-170, 1e-300, 1e-310]
  const methods = ['jacobi', 'golub-kahan'] as const
  it.each(scales.flatMap((scale) => methods.map((method) => [scale, method] as const)))(
    'svd of a matrix scaled by %s (%s) has the singular values, cond and rank of the unit matrix',
    (scale, method) => {
      const a = [
        [1 * scale, 2 * scale],
        [3 * scale, 4 * scale],
        [5 * scale, 6 * scale],
      ]
      const s = toArray(L.svd(a, { method }).s) as number[]
      // A denormal scale loses digits in the input itself (1e-310 has about 14 of them).
      const rtol = scale < 1e-300 ? 1e-9 : 1e-12
      expect(Math.abs(s[0] / scale / S1 - 1)).toBeLessThan(rtol)
      expect(Math.abs(s[1] / scale / S2 - 1)).toBeLessThan(rtol * 100)
      if (method === 'jacobi') {
        expect(Math.abs((toArray(L.cond(a)) as unknown as number) / (S1 / S2) - 1)).toBeLessThan(rtol * 100)
        expect(L.matrixRank(a)).toBe(2)
        expect(L.matrixRank([[scale, 2 * scale], [2 * scale, 4 * scale]])).toBe(1)
      }
    },
  )
  it.each([
    [[[1e200, 0], [0, 1e200]], [1e200, 1e200]],
    [[[1e-310, 0], [0, 1e-310]], [1e-310, 1e-310]],
  ])('svd of the scaled identity %j', (a, expected) => {
    expect(toArray(L.svd(a).s)).toEqual(expected)
  })
})
