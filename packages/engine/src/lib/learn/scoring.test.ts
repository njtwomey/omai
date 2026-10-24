import { describe, expect, it } from 'vitest'
import { gaussianNB, leastSquares } from '../../testing/learn.ts'
import { getScorer, scorerFor, scorersOf } from './scoring.ts'

const x = [[0], [1], [2], [3], [4], [5]]
const y = [0, 0, 0, 1, 1, 1]

describe('scorers', () => {
  it('negate losses and use the right verb', () => {
    const nb = gaussianNB().fit({ x, y })
    expect(getScorer('accuracy').score(nb, { x, y })).toBe(1)
    expect(getScorer('neg_log_loss').score(nb, { x, y })).toBeLessThan(0)
    expect(getScorer('roc_auc').score(nb, { x, y })).toBe(1)
    const lr = leastSquares().fit({ x, y: [1, 2, 3, 4, 5, 6] })
    expect(getScorer('neg_mean_squared_error').score(lr, { x, y: [1, 2, 3, 4, 5, 6] })).toBeCloseTo(0, 12)
    expect(() => getScorer('accuracy').score(lr, { x })).toThrow(/targets/)
  })

  it('default to the estimator score and keep names unique', () => {
    const lr = leastSquares().fit({ x, y: [1, 2, 3, 4, 5, 6] })
    expect(scorerFor(undefined).score(lr, { x, y: [1, 2, 3, 4, 5, 6] })).toBeCloseTo(1, 12)
    expect(Object.keys(scorersOf(['r2', 'r2']))).toEqual(['r2'])
  })
})
