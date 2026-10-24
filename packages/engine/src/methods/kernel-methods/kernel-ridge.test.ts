import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { regressionSet } from '../../testing/supervised.ts'
import { KernelRidge, kernelRidge } from './kernel-ridge.ts'

describe('kernel ridge', () => {
  it('fits, explains its leave-one-out curve and predicts distributions', () => {
    const sine = regressionSet('sineCurve', { n: 50 }, 1)
    const f = kernelRidge({ kernel: 'rbf', alpha: 0.1, gamma: 1 }).fit(sine)
    expect(f.score?.(sine)).toBeGreaterThan(0.6)
    const curve = recordsOf(f.exposition, 'crossValidationCurve')[0]
    expect(curve?.values).toHaveLength(25)
    expect(recordsOf(f.exposition, 'posteriorGrid')[0]?.mean).toHaveLength(60)
    expect(f.predictDistribution(sine.x).batchShape).toEqual([50])
    expect(
      new KernelRidge({ kernel: 'polynomial', degree: 2 }).fit(sine.x, sine.y).score(sine.x, sine.y),
    ).toBeGreaterThan(0)
  })
})
