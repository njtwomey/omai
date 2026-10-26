import { describe, expect, it } from 'vitest'
import { diabetes } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { standardScaler } from '../../lib/learn/preprocessing/index.ts'
import { olsSummary } from './inference.ts'
import { enetPath, ridgePath } from './paths.ts'
import {
  ElasticNet,
  elasticNetCV,
  Lasso,
  LinearRegression,
  lassoCV,
  linearRegression,
  orthogonalMatchingPursuit,
  ridge,
  ridgeCV,
} from './regression.ts'

const raw = diabetes()
const data = { ...raw, x: standardScaler().fit(raw).transform(raw.x) }

describe('linear regression methods', () => {
  it('least squares, ridge, lasso and the elastic net fit and explain themselves', () => {
    for (const est of [
      linearRegression(),
      ridge({ alpha: 10 }),
      new Lasso({ alpha: 1 }).estimator,
      new ElasticNet({ alpha: 0.5 }).estimator,
    ]) {
      const f = est.fit(data)
      expect(f.score?.(data), est.name).toBeGreaterThan(0.4)
      const table = recordsOf(f.exposition, 'coefficientTable')[0]
      expect(table?.names).toHaveLength(11)
      expect(est.hyperSpace?.kind).toBe('dict')
    }
    expect(recordsOf(linearRegression().fit(data).exposition, 'coefficientTable')[0]?.standardErrors).toHaveLength(11)
    const path = recordsOf(ridge({ alpha: 10 }).fit(data).exposition, 'regularisationPath')[0]
    expect(path?.chosen).toBe(10)
    const lassoRecord = recordsOf(new Lasso({ alpha: 1 }).fit(data.x, data.y).exposition, 'regularisationPath')[0]
    expect(lassoRecord?.active?.[0]).toEqual([])
    expect(lassoRecord?.gaps).toHaveLength(30)
    expect(new LinearRegression().fit(data.x, data.y).score(data.x, data.y)).toBeGreaterThan(0.5)
  })

  it('cross-validated forms choose a penalty and return their curves', () => {
    const l = lassoCV({ nAlphas: 15 }).fit(data)
    expect(l.alpha).toBeGreaterThan(0)
    const curve = recordsOf(l.exposition, 'crossValidationCurve')[0]
    expect(curve?.values).toHaveLength(15)
    expect(curve?.best).toBe(l.alpha)
    expect(elasticNetCV({ nAlphas: 8 }).fit(data).score?.(data)).toBeGreaterThan(0.4)
    const r = ridgeCV({ alphas: [0.1, 1, 10, 100] }).fit(data)
    expect([0.1, 1, 10, 100]).toContain(r.alpha)
    expect(recordsOf(r.exposition, 'crossValidationCurve')[0]?.meanScores).toHaveLength(4)
  })

  it('greedy pursuit, paths and the OLS summary', () => {
    const o = orthogonalMatchingPursuit({ nNonzeroCoefs: 3 }).fit(data)
    expect(o.params.coef.filter((v) => v !== 0)).toHaveLength(3)
    expect(recordsOf(o.exposition, 'regularisationPath')[0]?.penalties).toEqual([1, 2, 3])
    const path = enetPath(data, { nAlphas: 10 })
    expect(path.coefs[0]?.every((v) => v === 0)).toBe(true)
    expect(ridgePath(data, [1, 10]).coefs).toHaveLength(2)
    const s = olsSummary(data)
    expect(s.rSquared).toBeGreaterThan(0.5)
    expect(s.exposition[0]?.kind).toBe('coefficientTable')
  })

  it('predicts distributions', () => {
    const f = linearRegression().fit(data)
    expect(f.predictDistribution(data.x).batchShape).toEqual([442])
  })
})
