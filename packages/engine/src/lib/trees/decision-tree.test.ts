import { describe, expect, it } from 'vitest'
import { classificationSet, regressionSet } from '../../testing/supervised.ts'
import { iris } from '../datasets/index.ts'
import { recordsOf } from '../exposition/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { Log, messages, metricSeries, tensorSeries } from '../trace/index.ts'
import {
  decisionTreeClassifier,
  decisionTreeRegressor,
  extraTreeClassifier,
  extraTreeRegressor,
} from './decision-tree.ts'
import { applyTree, decisionPath, leafCount, nodeRegions, treeDepth } from './tree.ts'

const flowers = iris()
const moons = classificationSet('moons', { n: 150, noise: 0.2 }, 1)
const sine = regressionSet('sineCurve', { n: 80 }, 2)

describe('decision tree estimators', () => {
  it('a classifier fits iris and records its growth with every candidate split', () => {
    const f = decisionTreeClassifier({ maxDepth: 3 }).fit(flowers)
    expect(f.score?.(flowers)).toBeGreaterThan(0.95)
    const record = recordsOf(f.exposition, 'tree')[0]
    expect(record?.nodes[0]?.rows).toHaveLength(150)
    expect(record?.nodes[0]?.candidates.length).toBeGreaterThan(20)
    expect(record?.nodes[0]?.split?.feature).toBeDefined()
    expect(record?.classes).toEqual(['setosa', 'versicolor', 'virginica'])
    expect(recordsOf(f.exposition, 'pruningPath')[0]?.leaves.at(-1)).toBe(1)
    expect(f.exportText()).toContain('|---')
    expect(f.describe().nodes).toHaveLength(f.params.feature.length)
    expect(f.predictDistribution(flowers.x).batchShape).toEqual([150])
    expect(f.featureImportances.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
  })

  it('exposes decision paths, leaves and the box each node covers', () => {
    const f = decisionTreeClassifier({ maxDepth: 2 }).fit(flowers)
    const leaves = Array.from(f.apply(flowers.x))
    const paths = decisionPath(f.params, flowers.x)
    expect(paths.map((p) => p.at(-1))).toEqual(leaves)
    expect(paths.every((p) => p[0] === 0)).toBe(true)
    const regions = nodeRegions(f.params)
    // Every training row lies in the box of its leaf.
    const d = flowers.x.shape[1] ?? 4
    leaves.forEach((leaf, i) => {
      const r = regions[leaf]
      for (let j = 0; j < d; j++) {
        const v = toFlat(flowers.x)[i * d + j] ?? 0
        expect(v).toBeGreaterThan(r?.lower[j] ?? Number.NEGATIVE_INFINITY)
        expect(v).toBeLessThanOrEqual(r?.upper[j] ?? Number.POSITIVE_INFINITY)
      }
    })
  })

  it('grows depth-first, breadth-first and best-first to the same partition with different numbering', () => {
    const depth = decisionTreeClassifier({ maxDepth: 4 }).fit(moons)
    const breadth = decisionTreeClassifier({ maxDepth: 4, growth: 'breadth' }).fit(moons)
    expect(leafCount(breadth.params)).toBe(leafCount(depth.params))
    expect(toFlat(breadth.predict(moons.x))).toEqual(toFlat(depth.predict(moons.x)))
    // Preorder: the left child follows its parent; level order: siblings are adjacent.
    expect(depth.params.left[0]).toBe(1)
    expect(breadth.params.right[0]).toBe(2)
    const best = decisionTreeClassifier({ maxLeafNodes: 4 }).fit(moons)
    expect(leafCount(best.params)).toBe(4)
    expect(treeDepth(decisionTreeClassifier({ ccpAlpha: 0.02 }).fit(moons).params)).toBeLessThan(
      treeDepth(decisionTreeClassifier().fit(moons).params),
    )
  })

  it('regression trees fit a curve, and extra trees are keyed', () => {
    const r = decisionTreeRegressor({ maxDepth: 4 }).fit(sine)
    expect(r.score?.(sine)).toBeGreaterThan(0.7)
    const a = extraTreeClassifier().fit(moons, { key: key(1) })
    const b = extraTreeClassifier().fit(moons, { key: key(1) })
    expect(a.params).toEqual(b.params)
    expect(a.score?.(moons)).toBeGreaterThan(0.9)
    expect(
      extraTreeRegressor({ maxDepth: 5 })
        .fit(sine, { key: key(1) })
        .score?.(sine),
    ).toBeGreaterThan(0.5)
  })

  it('splits categorical features by subsets of their codes and routes missing values', () => {
    // Codes 0, 2 and 4 are class 1, whatever their order.
    const x = Array.from({ length: 60 }, (_, i) => [i % 5, (i * 7) % 11])
    const y = x.map(([c = 0]) => (c % 2 === 0 ? 1 : 0))
    const f = decisionTreeClassifier({ categorical: [0] }).fit({ x, y })
    expect(f.score?.({ x, y })).toBe(1)
    expect(f.params.categories[0]?.length).toBeGreaterThan(0)
    expect(leafCount(f.params)).toBe(2)
    const withMissing = x.map(([a = 0, b = 0], i) => [i % 10 === 0 ? Number.NaN : a, b])
    const m = decisionTreeClassifier({ categorical: [0], maxDepth: 2 }).fit({ x: withMissing, y })
    expect(toFlat(m.predict([[Number.NaN, 3]]))).toHaveLength(1)
    expect(Array.from(applyTree(m.params, [[Number.NaN, 3]]))).toHaveLength(1)
  })

  it('uses sample weights, and rejects a criterion that does not suit the task', () => {
    const x = [[0], [1], [2], [3]]
    const y = [0, 1, 0, 1]
    const heavy = decisionTreeClassifier({ maxDepth: 1 }).fit({ x, y, weights: [1, 1, 1, 100] })
    expect(heavy.params.weightedNSamples[0]).toBe(103)
    expect(() => decisionTreeClassifier({ criterion: 'poisson' as never }).fit({ x, y })).toThrow(/does not suit/)
  })

  it('serialises: parameters survive JSON and revive into the same predictions', () => {
    const est = decisionTreeClassifier({ maxDepth: 3 })
    const f = est.fit(flowers)
    const back = est.revive(JSON.parse(JSON.stringify(f.params)))
    expect(toFlat(back.predict(flowers.x))).toEqual(toFlat(f.predict(flowers.x)))
  })
})

describe('the tree session', () => {
  it('runs live, one node per step, with the paths of the supervised-learning table', () => {
    const est = decisionTreeClassifier({ maxDepth: 3 })
    const s = est.session(flowers, {
      key: key(0),
      logs: {
        'tree/split': Log(),
        'tree/candidates': Log({ detail: 'full' }),
        'tree/leaves': Log(),
        'tree/depth': Log(),
        'tree/impurity': Log(),
        'eval/accuracy': Log(),
      },
    })
    s.advance({ steps: 3 })
    expect(s.t).toBe(3)
    const splits = messages(s.run(), 'tree/split')
    expect(splits).toHaveLength(3)
    expect(metricSeries(s.run(), 'tree/leaves').values).toEqual([2, 2, 3])
    expect(metricSeries(s.run(), 'tree/impurity').values.at(-1)).toBeLessThan(
      metricSeries(s.run(), 'tree/impurity').values[0] ?? 0,
    )
    expect(metricSeries(s.run(), 'eval/accuracy').values.at(-1)).toBeGreaterThan(0.6)
    const candidates = tensorSeries(s.run(), 'tree/candidates')
    expect(candidates.map((e) => e.step)).toEqual([0, 2]) // step 1 decided a pure leaf: no candidates
    expect(candidates[0]?.shape[1]).toBe(5)
  })

  it('probes the structure and the model at an early step, and the partial tree is a valid classifier', () => {
    const s = decisionTreeClassifier({ maxDepth: 3 }).session(flowers, { key: key(0), logs: 'all' })
    s.advance({ steps: 4 })
    const early = s.probe(2, 'model/structure') as { readonly kind: string; readonly nodes: readonly unknown[] }
    expect(early.kind).toBe('tree')
    const late = s.probe(4, 'model/structure') as { readonly nodes: readonly unknown[] }
    expect(late.nodes.length).toBeGreaterThanOrEqual(early.nodes.length)
    expect(s.probe(2, 'model/importances')).toHaveLength(4)
    const partial = s.fittedAt(2)
    expect(toFlat(partial.predict(flowers.x))).toHaveLength(150)
    const surface = s.probe(2, 'model/decision-surface', { resolution: 8 }) as { readonly kind: string }
    expect(surface.kind).toBe('decisionSurface')
    s.advance()
    expect(s.fittedAt(s.t).params).toEqual(decisionTreeClassifier({ maxDepth: 3 }).fit(flowers, { key: key(0) }).params)
    // Seeking back re-runs from a checkpoint to the same state.
    expect(s.fittedAt(3).params.left.length).toBeLessThanOrEqual(s.fittedAt(s.t).params.left.length)
  })

  it('a regression session logs the R² of the tree so far', () => {
    const s = decisionTreeRegressor({ maxDepth: 3 }).session(sine, { key: key(0), logs: { 'eval/r2': Log() } })
    s.advance()
    const r2 = metricSeries(s.run(), 'eval/r2').values
    expect(r2.at(-1)).toBeGreaterThan(r2[0] ?? 1)
  })
})
