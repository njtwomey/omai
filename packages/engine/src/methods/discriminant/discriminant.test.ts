import { describe, expect, it } from 'vitest'
import { iris } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { classificationSet } from '../../testing/supervised.ts'
import {
  LinearDiscriminantAnalysis,
  linearDiscriminantAnalysis,
  quadraticDiscriminantAnalysis,
} from './discriminant.ts'

describe('discriminant analysis', () => {
  it('LDA classifies, projects and explains; QDA bends its boundaries', () => {
    const flowers = iris()
    const f = linearDiscriminantAnalysis().fit(flowers)
    expect(f.score?.(flowers)).toBeGreaterThan(0.95)
    const comps = recordsOf(f.exposition, 'components')[0]
    expect(comps?.components).toHaveLength(2)
    expect(comps?.scores).toHaveLength(150)
    expect((comps?.explainedVarianceRatio?.[0] ?? 0) > 0.9).toBe(true)
    expect(f.transform(flowers.x).shape).toEqual([150, 2])
    expect(
      new LinearDiscriminantAnalysis({ solver: 'lsqr', shrinkage: 'auto' })
        .fit(flowers.x, flowers.y)
        .score(flowers.x, flowers.y),
    ).toBeGreaterThan(0.95)
    const circles = classificationSet('circles', { n: 120, noise: 0.05 }, 3)
    const q = quadraticDiscriminantAnalysis().fit(circles)
    expect(q.score?.(circles)).toBeGreaterThan(0.9)
    expect(recordsOf(q.exposition, 'decisionSurface')).toHaveLength(1)
    expect(q.predictDistribution(circles.x).batchShape).toEqual([120])
  })
})
