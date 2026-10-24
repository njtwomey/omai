import { describe, expect, it } from 'vitest'
import { recordsOf } from '../../lib/exposition/index.ts'
import { foldIn, key, normal, poisson } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import {
  gammaRegressor,
  inverseGaussianRegressor,
  negativeBinomialRegressor,
  PoissonRegressor,
  tweedieRegressor,
} from './glm.ts'

const x = Array.from(toFlat(normal(key(1), [100])), (v) => [v * 0.5])
const mu = x.map((r) => Math.exp(0.5 + 0.8 * (r[0] ?? 0)))
const counts = mu.map((m, i) => toFlat(poisson(foldIn(key(2), i), m, []))[0] ?? 0)
const positive = mu.map((m, i) => m * (0.5 + ((i * 37) % 10) / 10))

describe('GLM regressors', () => {
  it('fit counts and positive responses, with family distributions and coefficient tables', () => {
    const p = new PoissonRegressor({ alpha: 0 }).fit(x, counts)
    expect(p.model.params.coef[0]).toBeGreaterThan(0.4)
    expect(recordsOf(p.exposition, 'coefficientTable')[0]?.standardErrors).toHaveLength(2)
    expect(p.model.predictDistribution(x).batchShape).toEqual([100])
    for (const est of [
      gammaRegressor({ alpha: 0.01 }),
      tweedieRegressor({ alpha: 0.01 }),
      inverseGaussianRegressor({ alpha: 0.01 }),
      negativeBinomialRegressor({ alpha: 0.01 }),
    ]) {
      const f = est.fit({ x, y: est.name.startsWith('negative') ? counts : positive })
      expect(f.params.coef[0], est.name).toBeGreaterThan(0.3)
      expect(f.exposition[0]?.kind).toBe('coefficientTable')
    }
  })
})
