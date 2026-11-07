import { describe, expect, it } from 'vitest'
import { grad } from '../../autodiff/index.ts'
import { sum, tensor, toArray, type Value, zeros } from '../../tensor/index.ts'
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

describe('lstsq of a rank-deficient system is the minimum-norm least-squares solution (numpy.linalg.lstsq)', () => {
  const third = 1 / 3
  const cases: readonly (readonly [string, number[][], number[] | number[][], number[] | number[][]])[] = [
    ['three equal columns', [[1, 1, 1], [1, 1, 1], [1, 1, 1]], [1, 1, 1], [third, third, third]],
    ['the zero matrix', [[0, 0], [0, 0]], [1, 1], [0, 0]],
    ['a column twice the other', [[1, 2], [2, 4], [3, 6]], [1, 2, 3], [0.2, 0.4]],
    ['the same, with a right-hand side outside the range', [[1, 2], [2, 4], [3, 6]], [1, 0, 0], [1 / 70, 2 / 70]],
    ['a zero column', [[1, 0], [2, 0], [3, 0]], [1, 0, 5], [8 / 7, 0]],
    ['wide, a repeated column', [[1, 1, 0], [2, 2, 0]], [1, 2], [0.5, 0.5, 0]],
    ['square of rank one', [[1, 2, 3], [2, 4, 6], [3, 6, 9]], [1, 2, 3], [1 / 14, 2 / 14, 3 / 14]],
    ['several right-hand sides', [[1, 1], [1, 1]], [[1, 2], [3, 0]], [[1, 0.5], [1, 0.5]]],
  ]
  it.each(cases)('%s', (_name, a, b, expected) => {
    const x = toArray(L.lstsq(a, b)) as number[] | number[][]
    expect(x.flat()).toEqual(expected.flat().map((e) => expect.closeTo(e, 12)))
  })

  it('keeps the QR solution (and its bits) for a full-rank matrix, and its gradient through pinv when deficient', () => {
    expect(toArray(L.lstsq([[1, 0], [0, 1], [0, 0]], [1, 2, 3]))).toEqual([1, 2])
    const g = toArray(grad((b: Value) => sum(L.lstsq([[1, 2], [2, 4], [3, 6]], b)))(tensor([1, 2, 3]))) as number[]
    expect(g).toEqual([3 / 70, 6 / 70, 9 / 70].map((e) => expect.closeTo(e, 12)))
  })
})

describe('singular systems and the matrix functions that sit on them', () => {
  const zeroDiagonal = [
    [0, 0],
    [1, 1],
  ]
  it.each([
    ['solveTriangular with a zero on the diagonal', () => L.solveTriangular(zeroDiagonal, [1, 1], { lower: true }), /singular/],
    ['solveTriangular, transposed', () => L.solveTriangular(zeroDiagonal, [1, 1], { transpose: true }), /singular/],
    ['choSolve of a zero Cholesky factor', () => L.choSolve(zeroDiagonal, [1, 1]), /singular/],
    ['luSolve of the factors of a singular matrix', () => L.luSolve(L.luFactor([[1, 2], [2, 4]]), [1, 2]), /singular/],
    ['sqrtm of a matrix with a negative eigenvalue', () => L.sqrtm([[-1, 0], [0, 1]]), /^sqrtm:/],
    ['logm of a matrix with a negative eigenvalue', () => L.logm([[-1, 0], [0, 1]]), /^logm:/],
    ['signm of a singular matrix', () => L.signm([[0, 0], [0, 1]]), /^signm:/],
  ])('%s throws a value error that names the cause', (_name, call, message) => {
    expect(call).toThrow(expect.objectContaining({ code: 'value', message: expect.stringMatching(message) }))
  })

  it('takes a unit diagonal as it is, the square root of zero as zero, and the determinant of 0 x 0 as 1', () => {
    expect(toArray(L.solveTriangular(zeroDiagonal, [1, 1], { lower: true, unitDiagonal: true }))).toEqual([1, 0])
    expect(toArray(L.sqrtm([[0, 0], [0, 0]]))).toEqual([[0, 0], [0, 0]])
    expect(toArray(L.det(zeros([0, 0])))).toBe(1)
    expect(toArray(L.slogdet(zeros([2, 0, 0])).sign)).toEqual([1, 1])
  })
})
