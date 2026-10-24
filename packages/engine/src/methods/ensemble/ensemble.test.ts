import { describe, expect, it } from 'vitest'
import { iris } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { classificationSet, regressionSet } from '../../testing/supervised.ts'
import { adaBoostClassifier, adaBoostRegressor } from './adaboost.ts'
import { baggingClassifier, baggingRegressor } from './bagging.ts'
import { ExtraTreesClassifier, extraTreesRegressor, randomForestClassifier, randomForestRegressor } from './forest.ts'
import { gradientBoostingClassifier, gradientBoostingRegressor } from './gradient-boosting.ts'

const moons = classificationSet('moons', { n: 120, noise: 0.25 }, 1)
const sine = regressionSet('sineCurve', { n: 80 }, 2)
const flowers = iris()
const k = { key: key(5) }

describe('ensembles', () => {
  it('forests and extra trees record their stages, a tree and the surface', () => {
    const f = randomForestClassifier({ nEstimators: 15, maxDepth: 6 }).fit(moons, k)
    expect(f.score?.(moons)).toBeGreaterThan(0.9)
    const stages = recordsOf(f.exposition, 'ensembleStages')[0]
    expect(stages?.trainScores).toHaveLength(15)
    expect(stages?.validationScores).toHaveLength(15)
    expect(stages?.bootstraps).toHaveLength(15)
    expect(recordsOf(f.exposition, 'tree')[0]?.nodes.length).toBeGreaterThan(3)
    expect(recordsOf(f.exposition, 'decisionSurface')).toHaveLength(1)
    expect(f.params.oobScore).toBeGreaterThan(0.75)
    expect(
      new ExtraTreesClassifier({ nEstimators: 10 }).fit(flowers.x, flowers.y).score(flowers.x, flowers.y),
    ).toBeGreaterThan(0.95)
    const r = randomForestRegressor({ nEstimators: 10 }).fit(sine, k)
    expect(r.score?.(sine)).toBeGreaterThan(0.7)
    expect(r.predictDistribution(sine.x).batchShape).toEqual([80])
    expect(extraTreesRegressor({ nEstimators: 10 }).fit(sine, k).score?.(sine)).toBeGreaterThan(0.7)
  })

  it('gradient boosting and AdaBoost record residuals, weights and losses', () => {
    const g = gradientBoostingClassifier({ nEstimators: 20 }).fit(moons, k)
    expect(g.score?.(moons)).toBeGreaterThan(0.9)
    const stages = recordsOf(g.exposition, 'ensembleStages')[0]
    expect(stages?.trainScores[19]).toBeLessThan(stages?.trainScores[0] ?? 0)
    expect(stages?.targets?.[0]).toHaveLength(120)
    expect(gradientBoostingClassifier({ nEstimators: 10 }).fit(flowers).score?.(flowers)).toBeGreaterThan(0.95)
    const sub = gradientBoostingRegressor({ nEstimators: 20, subsample: 0.7 }).fit(sine, k)
    expect(sub.score?.(sine)).toBeGreaterThan(0.7)
    expect(recordsOf(sub.exposition, 'ensembleStages')[0]?.validationScores).toHaveLength(20)
    for (const loss of ['absolute_error', 'huber', 'quantile'] as const)
      expect(
        gradientBoostingRegressor({ nEstimators: 20, loss, alpha: 0.5 }).fit(sine).score?.(sine),
        loss,
      ).toBeGreaterThan(0.5)
    const a = adaBoostClassifier({ nEstimators: 20 }).fit(moons, k)
    expect(a.score?.(moons)).toBeGreaterThan(0.85)
    expect(recordsOf(a.exposition, 'ensembleStages')[0]?.weights.length).toBeGreaterThan(1)
    expect(a.predictProba?.(moons.x).shape).toEqual([120, 2])
    expect(adaBoostRegressor({ nEstimators: 10 }).fit(sine, k).score?.(sine)).toBeGreaterThan(0.6)
  })

  it('bagging wraps any estimator', () => {
    const b = baggingClassifier({ nEstimators: 8 }).fit(moons, k)
    expect(b.score?.(moons)).toBeGreaterThan(0.9)
    expect(recordsOf(b.exposition, 'ensembleStages')[0]?.validationScores).toHaveLength(8)
    expect(baggingRegressor({ nEstimators: 5 }).fit(sine, k).score?.(sine)).toBeGreaterThan(0.7)
    expect(baggingClassifier().with({ 'estimator.maxDepth': 2 }).hyper.estimator.hyper).toMatchObject({ maxDepth: 2 })
  })
})
