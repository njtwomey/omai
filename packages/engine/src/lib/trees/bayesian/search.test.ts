import { describe, expect, it } from 'vitest'
import { normalGammaLeaf } from './leaf.ts'
import { searchBayesSplits } from './search.ts'
import { bayesData } from './tree.ts'

describe('searchBayesSplits', () => {
  it('finds the step', () => {
    const model = normalGammaLeaf({ m0: 0, kappa0: 1, alpha0: 2, beta0: 1 })
    const data = bayesData([[0], [1], [2], [3], [4], [5]], [0, 0.1, 0, 5, 5.1, 5], model)
    const found = searchBayesSplits(data, undefined, 2)
    expect(found.best?.threshold).toBe(2.5)
    expect(found.available).toBe(3)
  })
})
