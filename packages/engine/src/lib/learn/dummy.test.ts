import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { dummyClassifier, dummyRegressor, weightedQuantile } from './dummy.ts'
import { crossValidate } from './model-selection/index.ts'

const x = Array.from({ length: 12 }, (_, i) => [i])
const y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2]

describe('dummyClassifier', () => {
  it('prior predicts the most frequent class with the class frequencies as probabilities', () => {
    const f = dummyClassifier().fit({ x, y })
    expect(f.classes).toEqual([0, 1, 2])
    expect(toFlat(f.predict([[1], [2]]))).toEqual([0, 0])
    expect(toArray(f.predictProba([[1]]))).toEqual([[0.5, 1 / 3, 1 / 6]])
    expect(f.score({ x, y })).toBeCloseTo(0.5, 12)
  })

  it('most_frequent gives one-hot probabilities, constant gives its label, and an unseen constant throws', () => {
    expect(
      toArray(
        dummyClassifier({ strategy: 'most_frequent' })
          .fit({ x, y })
          .predictProba([[0]]),
      ),
    ).toEqual([[1, 0, 0]])
    const c = dummyClassifier({ strategy: 'constant', constant: 2 }).fit({ x, y })
    expect(toFlat(c.predict([[0], [1]]))).toEqual([2, 2])
    expect(toArray(c.predictProba([[0]]))).toEqual([[0, 0, 1]])
    expect(() => dummyClassifier({ strategy: 'constant', constant: 7 }).fit({ x, y })).toThrow()
    expect(() => dummyClassifier({ strategy: 'constant' }).fit({ x, y })).toThrow()
  })

  it('sample weights change the prior and hence the prediction', () => {
    const weights = y.map((c) => (c === 2 ? 10 : 1))
    const f = dummyClassifier().fit({ x, y, weights })
    expect(toFlat(f.predict([[0]]))).toEqual([2])
  })

  it('stratified and uniform draw from the key of the fit, the same for the same rows', () => {
    const rows = Array.from({ length: 3000 }, (_, i) => [i])
    for (const strategy of ['stratified', 'uniform'] as const) {
      const f = dummyClassifier({ strategy }).fit({ x, y }, { key: key(4) })
      const a = toFlat(f.predict(rows))
      expect(toFlat(f.predict(rows))).toEqual(a)
      const share = (c: number): number => a.filter((v) => v === c).length / a.length
      if (strategy === 'stratified') {
        expect(share(0)).toBeCloseTo(0.5, 1)
        expect(share(2)).toBeCloseTo(1 / 6, 1)
        // The probabilities are the one-hot of the same draws.
        const proba = toArray(f.predictProba(rows)) as number[][]
        expect(proba.map((r) => r.indexOf(1))).toEqual(a)
      } else {
        for (const c of [0, 1, 2]) expect(share(c)).toBeCloseTo(1 / 3, 1)
        expect(toArray(f.predictProba([[0]]))[0]?.[0]).toBeCloseTo(1 / 3, 12)
      }
    }
  })

  it('serialises: revive rebuilds the same predictions', () => {
    const est = dummyClassifier({ strategy: 'stratified' })
    const f = est.fit({ x, y }, { key: key(9) })
    expect(toFlat(est.revive(f.params).predict(x))).toEqual(toFlat(f.predict(x)))
  })

  it('is a baseline under cross-validation', () => {
    const r = crossValidate(dummyClassifier(), { x, y }, { cv: 3, key: key(0) })
    expect(Object.values(r.testScores)[0]?.length).toBe(3)
  })
})

describe('dummyRegressor', () => {
  const yr = [1, 2, 3, 4, 100]
  const xr = yr.map((_, i) => [i])
  it('mean, median, quantile and constant', () => {
    expect(
      toFlat(
        dummyRegressor()
          .fit({ x: xr, y: yr })
          .predict([[0]]),
      ),
    ).toEqual([22])
    expect(
      toFlat(
        dummyRegressor({ strategy: 'median' })
          .fit({ x: xr, y: yr })
          .predict([[0]]),
      ),
    ).toEqual([3])
    expect(
      toFlat(
        dummyRegressor({ strategy: 'quantile', quantile: 0.25 })
          .fit({ x: xr, y: yr })
          .predict([[0]]),
      ),
    ).toEqual([2])
    expect(
      toFlat(
        dummyRegressor({ strategy: 'constant', constant: 5 })
          .fit({ x: xr, y: yr })
          .predict([[0], [1]]),
      ),
    ).toEqual([5, 5])
    expect(() => dummyRegressor({ strategy: 'quantile' }).fit({ x: xr, y: yr })).toThrow()
    expect(() => dummyRegressor({ strategy: 'quantile', quantile: 1.5 }).fit({ x: xr, y: yr })).toThrow()
    expect(() => dummyRegressor({ strategy: 'constant' }).fit({ x: xr, y: yr })).toThrow()
  })

  it('weights enter the mean and the median', () => {
    const f = dummyRegressor().fit({ x: xr, y: yr, weights: [1, 1, 1, 1, 0] })
    expect(f.params.constant).toBe(2.5)
    expect(weightedQuantile([1, 2, 3, 4], [1, 1, 1, 5], 0.5)).toBe(4)
  })

  it('scores zero R^2 on its own training targets (mean)', () => {
    expect(dummyRegressor().fit({ x: xr, y: yr }).score({ x: xr, y: yr })).toBeCloseTo(0, 12)
  })
})
