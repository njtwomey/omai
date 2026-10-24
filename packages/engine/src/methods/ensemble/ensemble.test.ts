import { describe, expect, it } from 'vitest'
import { iris } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { classificationSet, regressionSet } from '../../testing/supervised.ts'
import { logisticRegression, ridge } from '../linear/index.ts'
import { decisionTreeClassifier, decisionTreeRegressor } from '../trees/index.ts'
import { adaBoostClassifier, adaBoostRegressor } from './adaboost.ts'
import { baggingClassifier, baggingRegressor } from './bagging.ts'
import { ExtraTreesClassifier, extraTreesRegressor, randomForestClassifier, randomForestRegressor } from './forest.ts'
import { gradientBoostingClassifier, gradientBoostingRegressor } from './gradient-boosting.ts'
import { histGradientBoostingClassifier, histGradientBoostingRegressor } from './histogram-boosting.ts'
import { stackingClassifier, stackingRegressor, votingClassifier, votingRegressor } from './stacking.ts'

const moons = classificationSet('moons', { n: 120, noise: 0.25 }, 1)
const sine = regressionSet('sineCurve', { n: 80 }, 2)
const flowers = iris()
const k = { key: key(5) }

describe('ensembles', () => {
  it('record their decision surface stage by stage, with named scores', () => {
    for (const est of [
      adaBoostClassifier({ nEstimators: 12 }),
      gradientBoostingClassifier({ nEstimators: 12 }),
      randomForestClassifier({ nEstimators: 12 }),
      histGradientBoostingClassifier({ maxIter: 12 }),
    ]) {
      const fitted = est.fit(moons, k)
      const stages = recordsOf(fitted.exposition, 'ensembleStages')[0]
      const surface = recordsOf(fitted.exposition, 'decisionSurface')[0]
      expect(stages?.scoreName).toBeTypeOf('string')
      expect(stages?.higherIsBetter).toBeTypeOf('boolean')
      expect(stages?.surface).toBe(surface)
      expect(surface?.steps?.[0]?.iteration).toBe(1)
      expect(surface?.steps?.at(-1)?.labels).toEqual(surface?.labels)
      expect(surface?.rows?.labels).toHaveLength(120)
    }
  })

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

describe('histogram boosting, voting and stacking', () => {
  it('histogram boosting with early stopping and missing values', () => {
    const h = histGradientBoostingClassifier({ maxIter: 30, minSamplesLeaf: 5 }).fit(moons, k)
    expect(h.score?.(moons)).toBeGreaterThan(0.9)
    expect(recordsOf(h.exposition, 'ensembleStages')[0]?.trainScores).toHaveLength(30)
    const e = histGradientBoostingRegressor({ maxIter: 200, earlyStopping: true, minSamplesLeaf: 5 }).fit(sine, k)
    expect(recordsOf(e.exposition, 'ensembleStages')[0]?.validationScores?.length).toBeGreaterThan(0)
    expect(e.score?.(sine)).toBeGreaterThan(0.6)
    const x = [[Number.NaN], [1], [2], [3], [Number.NaN], [5], [6], [7]]
    const y = [0, 0, 0, 0, 1, 1, 1, 1]
    expect(
      histGradientBoostingRegressor({ maxIter: 5, minSamplesLeaf: 1 })
        .fit({ x, y })
        .predict([[Number.NaN]]).shape,
    ).toEqual([1])
    expect(histGradientBoostingClassifier({ maxIter: 10 }).fit(flowers).score?.(flowers)).toBeGreaterThan(0.9)
  })

  it('voting and stacking combine any members', () => {
    const members = [
      ['tree', decisionTreeClassifier({ maxDepth: 3 })],
      ['logistic', logisticRegression()],
    ] as const
    const v = votingClassifier({ estimators: members }).fit(moons, k)
    expect(v.score?.(moons)).toBeGreaterThan(0.85)
    expect(recordsOf(v.exposition, 'ensembleStages')[0]?.trainScores).toHaveLength(2)
    const s = stackingClassifier({ estimators: members }).fit(moons, k)
    expect(s.score?.(moons)).toBeGreaterThan(0.85)
    expect(recordsOf(s.exposition, 'coefficientTable')).toHaveLength(1)
    const r = stackingRegressor({
      estimators: [
        ['tree', decisionTreeRegressor({ maxDepth: 4 })],
        ['ridge', ridge()],
      ],
    }).fit(sine, k)
    expect(r.score?.(sine)).toBeGreaterThan(0.6)
    expect(
      votingRegressor({ estimators: [['tree', decisionTreeRegressor({ maxDepth: 4 })]] })
        .fit(sine)
        .score?.(sine),
    ).toBeGreaterThan(0.6)
  })
})
