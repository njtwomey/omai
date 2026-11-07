import { describe, expect, it } from 'vitest'
import { toArray } from '../../tensor/index.ts'
import * as P from './index.ts'

describe('polyfit with fewer distinct abscissae than coefficients is the minimum-norm fit (numpy.polynomial.polyfit)', () => {
  it.each([
    ['a repeated abscissa, degree 1', [1, 1, 1], [1, 2, 3], 1, [1, 1]],
    ['two abscissae, degree 2', [0, 0, 1, 1], [1, 3, 2, 2], 2, [2, 0, 0]],
    ['one abscissa, degree 2', [2, 2, 2, 2], [1, 2, 3, 4], 2, [0.8333333333333334, 0.41666666666666663, 0.20833333333333331]],
  ])('%s', (_name, x, y, deg, expected) => {
    const c = toArray(P.polyfit(x, y, deg)) as number[]
    expect(c).toEqual(expected.map((e) => expect.closeTo(e, 12)))
  })
})
