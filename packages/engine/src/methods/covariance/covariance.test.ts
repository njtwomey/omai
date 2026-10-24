import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key, normal } from '../../lib/random/index.ts'
import { toArray, toFlat } from '../../lib/tensor/index.ts'
import { GraphicalLasso } from './graphical-lasso.ts'
import { MinCovDet } from './robust.ts'
import { CovarianceEstimator, covarianceEstimator } from './shrinkage.ts'

const clean = toArray(normal(key(0), [60, 3])) as number[][]
const dirty = clean.map((r, i) => (i < 8 ? r.map((v) => v + 12) : r))

describe('covariance estimators', () => {
  it('empirical and shrinkage estimators give Mahalanobis scores and a likelihood', () => {
    for (const method of ['empirical', 'shrunk', 'ledoit-wolf', 'oas'] as const) {
      const f = covarianceEstimator({ method }).fit({ x: clean })
      const [rec] = recordsOf(f.exposition, 'anomalyScores')
      expect(rec?.scores.length).toBe(60)
      expect(Number.isFinite(f.score({ x: clean }))).toBe(true)
      if (method !== 'empirical') expect(f.params.shrinkage).toBeGreaterThan(0)
    }
    expect(new CovarianceEstimator().fit(clean).model.mahalanobis(clean).shape).toEqual([60])
  })

  it('the MCD ignores gross outliers that pull the empirical mean', () => {
    const emp = covarianceEstimator().fit({ x: dirty }).params.location
    const mcd = new MinCovDet().fit(dirty, undefined, { key: key(1) }).model
    expect(Math.hypot(...mcd.params.location)).toBeLessThan(0.6)
    expect(Math.hypot(...emp)).toBeGreaterThan(2)
    expect(mcd.params.support.slice(0, 8).every((s) => !s)).toBe(true)
    const scores = toFlat(mcd.mahalanobis(dirty))
    expect(Math.min(...scores.slice(0, 8))).toBeGreaterThan(Math.max(...scores.slice(8)))
  })

  it('the graphical lasso traces a path of partial correlations that shrinks with the penalty', () => {
    const m = new GraphicalLasso({ alpha: 0.05 }).fit(clean).model
    const [path] = recordsOf(m.exposition, 'regularisationPath')
    expect(path?.features.length).toBe(3)
    expect(path?.active?.[0]?.length ?? 1).toBeLessThanOrEqual(path?.active?.at(-1)?.length ?? 0)
  })
})
