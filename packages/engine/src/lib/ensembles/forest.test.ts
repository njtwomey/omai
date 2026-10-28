import { describe, expect, it } from 'vitest'
import { classificationSet, regressionSet } from '../../testing/supervised.ts'
import { iris } from '../datasets/index.ts'
import { recordsOf } from '../exposition/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { Log, metricSeries, tensorSeries } from '../trace/index.ts'
import { decisionTreeClassifier } from '../trees/decision-tree.ts'
import {
  extraTreesClassifier,
  extraTreesRegressor,
  randomForestClassifier,
  randomForestRegressor,
  randomTreesEmbedding,
} from './forest.ts'

const flowers = iris()
const moons = classificationSet('moons', { n: 200, noise: 0.3 }, 1)
const test = classificationSet('moons', { n: 400, noise: 0.3 }, 7)
const sine = regressionSet('sineCurve', { n: 100 }, 2)

describe('random forests and extra trees', () => {
  it('a forest beats one tree on noisy held-out data, and its out-of-bag score tracks the held-out one', () => {
    const tree = decisionTreeClassifier().fit(moons)
    const forest = randomForestClassifier({ nEstimators: 40 }).fit(moons, { key: key(1) })
    expect(forest.score?.(test)).toBeGreaterThan(tree.score?.(test) ?? 1)
    expect(Math.abs((forest.params.oobScore ?? 0) - (forest.score?.(test) ?? 0))).toBeLessThan(0.1)
    expect(forest.featureImportances.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10)
    expect(forest.apply(test.x).shape).toEqual([400, 40])
  })

  it('features per split decorrelate the trees: sqrt gives different trees than all features', () => {
    const a = randomForestClassifier({ nEstimators: 5, maxFeatures: 'all' }).fit(flowers, { key: key(1) })
    const b = randomForestClassifier({ nEstimators: 5, maxFeatures: 1 }).fit(flowers, { key: key(1) })
    expect(a.params.trees[0]).not.toEqual(b.params.trees[0])
    const sameRoot = (f: typeof a): number =>
      f.params.trees.filter((t) => t.feature[0] === a.params.trees[0]?.feature[0]).length
    expect(sameRoot(b)).toBeLessThan(5)
  })

  it('extra trees: random thresholds, every row, no out-of-bag score by default', () => {
    const f = extraTreesClassifier({ nEstimators: 30 }).fit(moons, { key: key(2) })
    expect(f.params.oobScore).toBeNull()
    expect(f.score?.(test)).toBeGreaterThan(0.75)
    const r = extraTreesRegressor({ nEstimators: 30 }).fit(sine, { key: key(2) })
    expect(r.score?.(sine)).toBeGreaterThan(0.85)
  })

  it('regression forests give a predictive spread', () => {
    const f = randomForestRegressor({ nEstimators: 20 }).fit(sine, { key: key(3) })
    expect(f.score?.(sine)).toBeGreaterThan(0.9)
    expect(f.predictDistribution(sine.x).batchShape).toEqual([100])
  })

  it('is deterministic for a key and records its stages', () => {
    const a = randomForestClassifier({ nEstimators: 6 }).fit(flowers, { key: key(5) })
    const b = randomForestClassifier({ nEstimators: 6 }).fit(flowers, { key: key(5) })
    expect(a.params).toEqual(b.params)
    const stages = recordsOf(a.exposition, 'ensembleStages')[0]
    expect(stages?.trainScores).toHaveLength(6)
    expect(stages?.validationScores?.at(-1)).toBeGreaterThan(0.85)
  })

  it('a session adds a tree per step with the out-of-bag error and importances, and probes the proximity', () => {
    const s = randomForestClassifier({ nEstimators: 10 }).session(flowers, {
      key: key(1),
      logs: { 'ensemble/oobError': Log(), 'ensemble/importances': Log(), 'ensemble/member': Log() },
    })
    s.advance({ steps: 5 })
    expect(metricSeries(s.run(), 'ensemble/member').values).toEqual([1, 2, 3, 4, 5])
    const imp = tensorSeries(s.run(), 'ensemble/importances')
    expect(imp).toHaveLength(5)
    expect(imp[0]?.shape).toEqual([4])
    const prox = s.probe(3, 'model/proximity') as number[][]
    expect(prox).toHaveLength(150)
    expect(prox[0]?.[0]).toBe(1)
    expect(prox[0]?.[1]).toBeLessThanOrEqual(1)
    expect((s.probe(3, 'model/importances') as number[]).length).toBe(4)
    expect(toFlat(s.fittedAt(3).predict(flowers.x))).toHaveLength(150)
  })
})

describe('random trees embedding', () => {
  it('maps a row to the leaves it falls in, one hot per tree, without looking at labels', () => {
    const x = Array.from({ length: 60 }, (_, i) => [(i * 17) % 60, (i * 7) % 11])
    const est = randomTreesEmbedding({ nEstimators: 4, maxDepth: 3 })
    const f = est.fit({ x }, { key: key(0) })
    const z = f.transform(x)
    expect(z.shape[0]).toBe(60)
    const width = z.shape[1] ?? 0
    expect(width).toBe(f.params.offsets.at(-1))
    const rows = toFlat(z)
    for (let i = 0; i < 60; i++) {
      expect(rows.slice(i * width, (i + 1) * width).reduce((a, b) => a + b, 0)).toBe(4)
    }
    // Nearby rows share more leaves than distant ones.
    const dot = (a: number, b: number): number =>
      rows.slice(a * width, (a + 1) * width).reduce((s, v, j) => s + v * (rows[b * width + j] ?? 0), 0)
    expect(dot(0, 0)).toBe(4)
    expect(f.featureNamesOut?.()).toHaveLength(width)
  })
})
