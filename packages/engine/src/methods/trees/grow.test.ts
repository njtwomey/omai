import { describe, expect, it } from 'vitest'
import { iris } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { classificationSet, regressionSet } from '../../testing/supervised.ts'
import {
  DecisionTreeClassifier,
  decisionTreeClassifier,
  decisionTreeRegressor,
  extraTreeClassifier,
  extraTreeRegressor,
} from './grow.ts'
import { nLeaves, treeDepth } from './tree.ts'

const flowers = iris()
const moons = classificationSet('moons', { n: 150, noise: 0.2 }, 1)
const sine = regressionSet('sineCurve', { n: 80 }, 2)

describe('decision trees', () => {
  it('a classifier records its growth, pruning path and surface', () => {
    const f = new DecisionTreeClassifier({ maxDepth: 3 }).fit(flowers.x, flowers.y)
    expect(f.score(flowers.x, flowers.y)).toBeGreaterThan(0.95)
    const record = recordsOf(decisionTreeClassifier({ maxDepth: 3 }).fit(flowers).exposition, 'tree')[0]
    expect(record?.nodes[0]?.rows).toHaveLength(150)
    expect(record?.nodes[0]?.candidates.length).toBeGreaterThan(20)
    expect(record?.nodes[0]?.split?.feature).toBeDefined()
    expect(record?.classes).toEqual(['setosa', 'versicolor', 'virginica'])
    expect(recordsOf(f.exposition, 'pruningPath')[0]?.leaves.at(-1)).toBe(1)
    expect(f.model.describe().nodes).toHaveLength(f.model.params.feature.length)
    expect(f.model.exportText()).toContain('|---')
    expect(f.model.predictDistribution(flowers.x).batchShape).toEqual([150])
    const m = decisionTreeClassifier({ maxDepth: 4 }).fit(moons)
    expect(recordsOf(m.exposition, 'decisionSurface')).toHaveLength(1)
  })

  it('best-first growth, pruning, regression and extra trees', () => {
    const b = decisionTreeClassifier({ maxLeafNodes: 4 }).fit(moons)
    expect(nLeaves(b.params)).toBe(4)
    expect(treeDepth(decisionTreeClassifier({ ccpAlpha: 0.02 }).fit(moons).params)).toBeLessThan(
      treeDepth(decisionTreeClassifier().fit(moons).params),
    )
    const r = decisionTreeRegressor({ maxDepth: 4 }).fit(sine)
    expect(r.score?.(sine)).toBeGreaterThan(0.7)
    expect(
      extraTreeClassifier()
        .fit(moons, { key: key(1) })
        .score?.(moons),
    ).toBeGreaterThan(0.9)
    expect(
      extraTreeRegressor({ maxDepth: 5 })
        .fit(sine, { key: key(1) })
        .score?.(sine),
    ).toBeGreaterThan(0.5)
  })
})
