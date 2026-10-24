import { describe, expect, it } from 'vitest'
import { leastSquares, nearestCentroid, rows } from '../../testing/learn.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { revive, serialiseFitted } from './estimator.ts'
import { fRegression, selectFromModel, selectKBest } from './feature-selection.ts'
import { rfe, stabilitySelection } from './feature-selection-wrappers.ts'

const x = Array.from({ length: 40 }, (_, i) => [Math.sin(i), Math.cos(3 * i), ((i * 7) % 11) / 11, Math.sin(5 * i)])
const y = x.map(([a = 0, b = 0]) => 3 * a - 2 * b)

describe('feature selection', () => {
  it('stability selection keeps the features a sparse selector picks consistently', () => {
    const f = stabilitySelection({ selector: selectKBest({ scoreFunc: fRegression, k: 2 }), nSubsamples: 30 }).fit(
      { x, y },
      { key: key(5) },
    )
    expect(f.support).toEqual([true, true, false, false])
    expect(f.params.frequencies[0]).toBe(1)
  })

  it('wrappers predict with the final estimator on the kept features and revive', () => {
    const est = rfe({ estimator: leastSquares(), nFeaturesToSelect: 2 })
    const f = est.fit({ x, y })
    expect(f.support).toEqual([true, true, false, false])
    const pred = toFlat(f.predict?.(x) ?? f.transform(x))
    pred.forEach((v, i) => {
      expect(v).toBeCloseTo(y[i] ?? 0, 9)
    })
    const again = revive(est, JSON.parse(JSON.stringify(serialiseFitted(f))))
    expect(rows(again.transform(x))).toEqual(rows(f.transform(x)))
    expect(rows(f.inverseTransform(f.transform(x)))[0]?.slice(2)).toEqual([0, 0])
  })

  it('selection from a model needs importances', () => {
    expect(() =>
      selectFromModel({ estimator: nearestCentroid() }).fit({ x, y: y.map((v) => (v > 0 ? 1 : 0)) }),
    ).toThrow(/featureImportances/)
    expect(() => selectFromModel({ estimator: leastSquares(), threshold: 'twice' }).fit({ x, y })).toThrow(/threshold/)
  })
})
