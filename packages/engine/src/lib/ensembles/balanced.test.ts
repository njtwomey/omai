import { describe, expect, it } from 'vitest'
import { recordsOf } from '../exposition/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { Log, metricSeries, tensorSeries } from '../trace/index.ts'
import { decisionTreeClassifier } from '../trees/decision-tree.ts'
import { adaBoostClassifier } from './adaboost.ts'
import { baggingClassifier } from './bagging.ts'
import {
  balancedBaggingClassifier,
  balancedRandomForestClassifier,
  easyEnsembleClassifier,
  rusBoostClassifier,
} from './balanced.ts'
import { randomForestClassifier } from './forest.ts'

/** 120 majority samples and 12 minority samples that overlap them: two noisy features on a deterministic grid. */
function imbalanced(n0: number, n1: number, phase: number): { x: number[][]; y: number[] } {
  const wobble = (i: number, f: number): number => ((i * 37 + f * 101 + phase * 17) % 97) / 97 - 0.5
  const x = [
    ...Array.from({ length: n0 }, (_, i) => [wobble(i, 1) * 2, wobble(i, 2) * 2]),
    ...Array.from({ length: n1 }, (_, i) => [1 + wobble(i, 3), 1 + wobble(i, 4)]),
  ]
  return { x, y: x.map((_, i) => (i < n0 ? 0 : 1)) }
}

const train = imbalanced(120, 12, 1)
const test = imbalanced(200, 40, 5)

const recall = (predicted: ArrayLike<number>, y: readonly number[]): number => {
  let hit = 0
  let total = 0
  y.forEach((v, i) => {
    if (v !== 1) return
    total++
    if (predicted[i] === 1) hit++
  })
  return hit / total
}

/** The class totals of each member's multiplicities, from a session's `ensemble/bootstrap` tensors. */
function classSums(
  bags: ReturnType<typeof tensorSeries>,
  y: readonly number[],
): { minority: number; majority: number }[] {
  return bags.map((b) => {
    const v = b.values ?? []
    return {
      minority: v.reduce((s, c, i) => s + (y[i] === 1 ? c : 0), 0),
      majority: v.reduce((s, c, i) => s + (y[i] === 0 ? c : 0), 0),
    }
  })
}

describe('balanced ensembles', () => {
  it('balanced bagging fits every member on equal class sizes and finds more of the minority than plain bagging', () => {
    const logs = { 'ensemble/bootstrap': Log({ detail: 'full' as const }) }
    const s = balancedBaggingClassifier({
      nEstimators: 8,
      estimator: decisionTreeClassifier({ maxDepth: 3, record: 0 }),
    }).session(train, { key: key(1), logs })
    s.advance()
    for (const { minority, majority } of classSums(tensorSeries(s.run(), 'ensemble/bootstrap'), train.y)) {
      expect(minority).toBe(majority)
      expect(minority).toBeGreaterThan(3)
    }
    const balanced = balancedBaggingClassifier({
      nEstimators: 15,
      estimator: decisionTreeClassifier({ maxDepth: 3, record: 0 }),
    }).fit(train, {
      key: key(2),
    })
    const plain = baggingClassifier({
      nEstimators: 15,
      estimator: decisionTreeClassifier({ maxDepth: 3, record: 0 }),
    }).fit(train, {
      key: key(2),
    })
    expect(recall(toFlat(balanced.predict(test.x)), test.y)).toBeGreaterThanOrEqual(
      recall(toFlat(plain.predict(test.x)), test.y),
    )
    expect(balanced.params.oobScore).not.toBeNull()
    const proba = toFlat(balanced.predictProba?.(test.x) as never)
    expect(proba).toHaveLength(240 * 2)
    expect((proba[0] ?? 0) + (proba[1] ?? 0)).toBeCloseTo(1, 12)
  })

  it('balanced bagging with a sampling strategy and replacement draws what it asks for', () => {
    const logs = { 'ensemble/bootstrap': Log({ detail: 'full' as const }) }
    const s = balancedBaggingClassifier({
      nEstimators: 4,
      bootstrap: false,
      balance: { samplingStrategy: { 0: 24 }, replacement: true },
    }).session(train, { key: key(3), logs })
    s.advance()
    for (const { minority, majority } of classSums(tensorSeries(s.run(), 'ensemble/bootstrap'), train.y)) {
      expect(majority).toBe(24)
      expect(minority).toBe(12)
    }
  })

  it('the balanced random forest grows each tree on a balanced bootstrap and reports its stages', () => {
    const logs = { 'ensemble/bootstrap': Log({ detail: 'full' as const }), 'ensemble/oobError': Log() }
    const forest = balancedRandomForestClassifier({ nEstimators: 12 })
    const s = forest.session(train, { key: key(4), logs })
    s.advance()
    for (const { minority, majority } of classSums(tensorSeries(s.run(), 'ensemble/bootstrap'), train.y)) {
      expect(minority).toBe(12)
      expect(majority).toBe(12)
    }
    expect(metricSeries(s.run(), 'ensemble/oobError').values.length).toBeGreaterThan(0)
    const fitted = forest.fit(train, { key: key(4) })
    expect(fitted.params.trees).toHaveLength(12)
    expect(fitted.params.oobScore).not.toBeNull()
    expect(recordsOf(fitted.exposition, 'ensembleStages')[0]?.trainScores).toHaveLength(12)
    const plain = randomForestClassifier({ nEstimators: 12 }).fit(train, { key: key(4) })
    expect(recall(toFlat(fitted.predict(test.x)), test.y)).toBeGreaterThanOrEqual(
      recall(toFlat(plain.predict(test.x)), test.y),
    )
  })

  it('EasyEnsemble averages AdaBoost members, each on its own balanced sample without replacement', () => {
    const logs = { 'ensemble/bootstrap': Log({ detail: 'full' as const }) }
    const estimator = easyEnsembleClassifier({ nEstimators: 5, estimator: adaBoostClassifier({ nEstimators: 8 }) })
    const s = estimator.session(train, { key: key(6), logs })
    s.advance()
    const sums = classSums(tensorSeries(s.run(), 'ensemble/bootstrap'), train.y)
    expect(sums).toHaveLength(5)
    for (const { minority, majority } of sums) {
      expect(minority).toBe(12)
      expect(majority).toBe(12)
    }
    const fitted = estimator.fit(train, { key: key(6) })
    expect(fitted.params.members).toHaveLength(5)
    expect(recall(toFlat(fitted.predict(test.x)), test.y)).toBeGreaterThan(0.4)
  })

  it('RUSBoost boosts on random under-samples, keeping errors and weights over all the rows', () => {
    const estimator = rusBoostClassifier({ nEstimators: 12 })
    const s = estimator.session(train, {
      key: key(7),
      logs: { 'boost/error': Log(), 'boost/alpha': Log(), 'ensemble/target': Log({ detail: 'full' as const }) },
    })
    s.advance()
    expect(metricSeries(s.run(), 'boost/error').values.length).toBeGreaterThan(2)
    expect(tensorSeries(s.run(), 'ensemble/target')[0]?.shape).toEqual([132])
    const fitted = estimator.fit(train, { key: key(7) })
    const plain = adaBoostClassifier({ nEstimators: 12 }).fit(train, { key: key(7) })
    expect(recall(toFlat(fitted.predict(test.x)), test.y)).toBeGreaterThanOrEqual(
      recall(toFlat(plain.predict(test.x)), test.y),
    )
    const again = estimator.fit(train, { key: key(7) })
    expect(again.params).toEqual(fitted.params)
  })
})
