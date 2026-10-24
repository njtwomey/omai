import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { regressionSet } from '../../testing/supervised.ts'
import { ArdRegression, bayesianRidge } from './bayesian.ts'
import { IsotonicRegression } from './isotonic.ts'

const curve = regressionSet('sineCurve', { n: 60 }, 1)

describe('Bayesian linear models and isotonic regression', () => {
  it('evidence maximisation returns its surface, path and posterior grid', () => {
    const f = bayesianRidge().fit(curve)
    expect(f.score?.(curve)).toBeGreaterThan(0)
    const surface = recordsOf(f.exposition, 'hyperparameterSurface')[0]
    expect(surface?.values).toHaveLength(25)
    expect(surface?.path?.length).toBe(f.params.iterations)
    expect(recordsOf(f.exposition, 'posteriorGrid')[0]?.mean).toHaveLength(60)
    expect(f.predictDistribution(curve.x).batchShape).toEqual([60])
    const ard = new ArdRegression().fit(curve.x, curve.y)
    expect(recordsOf(ard.exposition, 'coefficientTable')[0]?.standardErrors).toHaveLength(1)
  })

  it('isotonic regression is monotone', () => {
    const f = new IsotonicRegression().fit(curve.x, curve.y)
    const p = Array.from(f.predict(curve.x).data)
    const order = Array.from(curve.x.data.keys()).sort((a, b) => (curve.x.data[a] ?? 0) - (curve.x.data[b] ?? 0))
    for (let i = 1; i < order.length; i++)
      expect(p[order[i] ?? 0]).toBeGreaterThanOrEqual((p[order[i - 1] ?? 0] ?? 0) - 1e-12)
    expect(recordsOf(f.exposition, 'posteriorGrid')).toHaveLength(1)
  })
})
