import { describe, expect, it } from 'vitest'
import { distanceMatrix } from './evaluate.ts'

describe('distanceMatrix', () => {
  it('is the flat pairwise matrix of a named metric, symmetric with a zero diagonal', () => {
    const { n, data } = distanceMatrix(
      [
        [0, 0],
        [3, 4],
        [6, 8],
      ],
      'manhattan',
    )
    expect(n).toBe(3)
    expect(Array.from(data)).toEqual([0, 7, 14, 7, 0, 7, 14, 7, 0])
  })

  it('takes a precomputed matrix as it is, and refuses one that is not square', () => {
    expect(
      Array.from(
        distanceMatrix(
          [
            [0, 2],
            [2, 0],
          ],
          'precomputed',
        ).data,
      ),
    ).toEqual([0, 2, 2, 0])
    expect(() => distanceMatrix([[0, 1, 2]], 'precomputed')).toThrow(/square/)
  })
})
