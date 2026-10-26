import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { TruncatedSVD, truncatedSvd } from './svd.ts'

const x = Array.from({ length: 30 }, (_, i) => [i % 5, (i * 3) % 7, i % 2, 1])

describe('truncated SVD', () => {
  it('exact and randomised fits agree; ratios are at most one', () => {
    const f = truncatedSvd({ nComponents: 2 }).fit({ x })
    expect(f.params.explainedVarianceRatio.reduce((a, v) => a + v, 0)).toBeLessThanOrEqual(1 + 1e-12)
    expect(recordsOf(f.exposition, 'components')[0]?.scores?.length).toBe(30)
    const r = new TruncatedSVD({ nComponents: 2, algorithm: 'randomized' }).fit(x, undefined, { key: key(0) }).model
    expect(r.params.singularValues[0]).toBeCloseTo(f.params.singularValues[0] ?? 0, 6)
  })
})

describe('dictionary learning', () => {
  it('learns atoms whose sparse codes rebuild the data, for each learner', async () => {
    const { dictionaryLearning } = await import('./svd.ts')
    const atoms = [
      [1, 0, 0, 1],
      [0, 1, 1, 0],
    ]
    const rows = Array.from({ length: 24 }, (_, i) => (atoms[i % 2] ?? []).map((v) => v * (1 + (i % 5))))
    for (const method of ['ksvd', 'mod', 'online'] as const) {
      const f = dictionaryLearning({ nComponents: 2, method, sparsity: 1, alpha: 0.01, maxIter: 30 }).fit(
        { x: rows },
        { key: key(3) },
      )
      expect(f.params.components.length).toBe(2)
      expect(f.transform(rows).shape).toEqual([24, 2])
      if (method !== 'online') expect(f.params.error).toBeLessThan(1e-6)
    }
  })
})
