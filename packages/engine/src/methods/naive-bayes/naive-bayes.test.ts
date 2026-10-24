import { describe, expect, it } from 'vitest'
import { iris } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { classificationSet } from '../../testing/supervised.ts'
import { BernoulliNB, gaussianNB, MultinomialNB } from './naive-bayes.ts'

describe('naive Bayes', () => {
  it('explains each class by per-feature contributions', () => {
    const flowers = iris()
    const f = gaussianNB().fit(flowers)
    expect(f.score?.(flowers)).toBeGreaterThan(0.9)
    const table = recordsOf(f.exposition, 'likelihoodTable')[0]
    expect(table?.classes).toEqual(['setosa', 'versicolor', 'virginica'])
    expect(table?.contributions[0]).toHaveLength(4)
    expect(f.explain([5, 3.4, 1.5, 0.2]).contributions).toHaveLength(3)
    expect(f.describe().edges).toHaveLength(4)
    const blobs = classificationSet('blobs', { n: 90 }, 2)
    expect(recordsOf(gaussianNB().fit(blobs).exposition, 'decisionSurface')).toHaveLength(1)
    const counts = [
      [3, 0, 1],
      [2, 0, 0],
      [0, 4, 1],
      [0, 3, 2],
    ]
    expect(new MultinomialNB().fit(counts, [0, 0, 1, 1]).predict([[4, 0, 0]]).data[0]).toBe(0)
    expect(new BernoulliNB().fit(counts, [0, 0, 1, 1]).predict([[0, 1, 1]]).data[0]).toBe(1)
  })
})
