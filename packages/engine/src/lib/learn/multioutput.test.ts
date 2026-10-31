import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { metricSeries } from '../trace/index.ts'
import { revive, serialiseFitted } from './estimator.ts'
import { binaryRelevance, classifierChain, labelPowerset } from './multilabel.ts'
import { multiOutputClassifier, multiOutputRegressor, regressorChain } from './multioutput.ts'
import { naiveBayes } from './naive-bayes.ts'
import { kNeighboursClassifier, kNeighboursRegressor } from './neighbours.ts'

const x = [[0], [1], [2], [10], [11], [12]]
const labels = [
  [1, 0, 1],
  [1, 0, 1],
  [1, 0, 1],
  [0, 1, 1],
  [0, 1, 1],
  [0, 1, 1],
]

describe('multi-output regression', () => {
  const y = [
    [0, 5],
    [1, 4],
    [2, 3],
    [3, 2],
    [4, 1],
    [5, 0],
  ]

  it('fits one regressor per target, or a chain, and scores by the mean R squared', () => {
    const base = kNeighboursRegressor({ nNeighbors: 1 })
    const independent = multiOutputRegressor(base).fit({ x: [[0], [1], [2], [3], [4], [5]], y })
    expect(toArray(independent.predict([[2.2]]))).toEqual([[2, 3]])
    expect(independent.score({ x: [[0], [1], [2], [3], [4], [5]], y })).toBe(1)
    const chain = regressorChain(base, { order: [1, 0] }).fit({ x: [[0], [1], [2], [3], [4], [5]], y })
    expect(chain.params.plan.order).toEqual([1, 0])
    expect(toArray(chain.predict([[2.2]]))).toEqual([[2, 3]])
    expect(() => regressorChain(base, { order: [0, 0] }).fit({ x, y })).toThrow(/permutation/)
  })

  it('draws a random order from the key, reproducibly', () => {
    const base = kNeighboursRegressor({ nNeighbors: 1 })
    const a = regressorChain(base, { order: 'random' }).fit({ x, y }, { key: key(3) }).params.plan.order
    const b = regressorChain(base, { order: 'random' }).fit({ x, y }, { key: key(3) }).params.plan.order
    expect(a).toEqual(b)
    expect([...a].sort()).toEqual([0, 1])
  })
})

describe('multilabel', () => {
  it('relevance predicts each label, a constant label as the constant, and gives the marginals', () => {
    const f = binaryRelevance(kNeighboursClassifier({ nNeighbors: 1 })).fit({ x, y: labels })
    expect(toArray(f.predict([[1.5], [11.5]]))).toEqual([
      [1, 0, 1],
      [0, 1, 1],
    ])
    expect(f.params.models[2]).toEqual({ constant: 1 })
    expect(toArray((f.predictProba as NonNullable<typeof f.predictProba>)([[1.5]]))).toEqual([[1, 0, 1]])
    expect(f.score({ x, y: labels })).toBe(1)
  })

  it('solves one label per step in a session, with the chain fitted in its order', () => {
    const est = classifierChain(kNeighboursClassifier({ nNeighbors: 1 }), { order: [1, 0, 2] })
    const s = est.session({ x, y: labels }, { key: key(0), logs: 'all' })
    s.advance()
    expect(metricSeries(s.run(), 'solver/problems').values).toEqual([1, 2, 3])
    expect(s.fittedAt(0).nProblems).toBe(1)
    const full = revive(est, JSON.parse(JSON.stringify(serialiseFitted(s.fittedAt(s.t)))))
    expect(toArray(full.predict([[0.4], [11.9]]))).toEqual([
      [1, 0, 1],
      [0, 1, 1],
    ])
  })

  it('the label powerset predicts a combination seen in training and marginalises its probabilities', () => {
    const f = labelPowerset(naiveBayes()).fit({ x, y: labels })
    expect(toArray(f.predict([[1], [11]]))).toEqual([
      [1, 0, 1],
      [0, 1, 1],
    ])
    const marginal = toFlat(f.predictProba([[1]]))
    expect(marginal[2]).toBeCloseTo(1, 12)
    expect(toFlat(f.predictCombinations([[1]])).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12)
    const s = labelPowerset(naiveBayes()).session({ x, y: labels }, { key: key(0), logs: 'all' })
    s.advance()
    expect(s.t).toBe(0)
    expect(toArray(s.fittedAt(0).predict([[11]]))).toEqual([[0, 1, 1]])
  })
})

describe('multi-output classification', () => {
  it('fits each output as a class label of its own, exact match scoring', () => {
    const y = [
      [0, 3],
      [0, 3],
      [0, 3],
      [1, 4],
      [1, 4],
      [1, 5],
    ]
    const f = multiOutputClassifier(kNeighboursClassifier({ nNeighbors: 1 })).fit({ x, y })
    expect(toArray(f.predict([[1], [11.9]]))).toEqual([
      [0, 3],
      [1, 5],
    ])
    expect(f.score({ x, y })).toBe(1)
    expect(f.predictProbaEach?.([[1]])).toHaveLength(2)
  })
})
