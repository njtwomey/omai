import { describe, expect, it } from 'vitest'
import { gaussianNB, leastSquares, nearestCentroid, ridge, rows } from '../../testing/learn.ts'
import { isOmaiError } from '../base/index.ts'
import { toFlat } from '../tensor/index.ts'
import {
  can,
  cloneEstimator,
  type Estimator,
  EstimatorClass,
  isClassifier,
  isEstimator,
  requireTargets,
  requireVerb,
  revive,
  serialiseFitted,
} from './estimator.ts'
import { standardScaler } from './preprocessing/index.ts'
import { asFloatMatrix, codesFor, consistentLength, encodeLabels, matrixView, sampleWeightsOf } from './validation.ts'

const x = [
  [0, 1],
  [1, 0],
  [2, 2],
  [3, 1],
  [4, 5],
]
const y = [1, 2, 4, 5, 9]

describe('the estimator protocol', () => {
  it('fit returns a new value and never mutates the inputs or the estimator', () => {
    const est = leastSquares()
    const before = JSON.stringify({ x, y, hyper: est.hyper })
    const a = est.fit({ x, y })
    const b = est.fit({ x, y: y.map((v) => 2 * v) })
    expect(JSON.stringify({ x, y, hyper: est.hyper })).toBe(before)
    expect(a.params.coef).not.toEqual(b.params.coef)
    expect(toFlat(a.predict(x)).every((v, i) => Math.abs(v - (toFlat(b.predict(x))[i] ?? 0) / 2) < 1e-9)).toBe(true)
  })

  it('fitted parameters are plain JSON and revive into working verbs', () => {
    const all: readonly Estimator[] = [leastSquares(), ridge({ alpha: 0.5 }), nearestCentroid(), gaussianNB()]
    for (const est of all) {
      const fitted = est.fit({ x, y: [0, 0, 1, 1, 1] })
      const json = JSON.parse(JSON.stringify(serialiseFitted(fitted)))
      const again = revive(est, json)
      expect(toFlat(requireVerb(again, 'predict', 'test')(x))).toEqual(
        toFlat(requireVerb(fitted, 'predict', 'test')(x)),
      )
      expect(again.estimator).toBe(est.name)
    }
    expect(() => revive(ridge(), { estimator: 'least-squares', params: {} })).toThrow()
  })

  it('with and cloneEstimator change hyperparameters without touching the original', () => {
    const r = ridge({ alpha: 1 })
    const r2 = cloneEstimator(r, { alpha: 10 })
    expect(r.hyper.alpha).toBe(1)
    expect(r2.hyper.alpha).toBe(10)
    expect(r2.fit({ x, y }).params.coef[0] ?? 0).toBeLessThan(r.fit({ x, y }).params.coef[0] ?? 0)
  })

  it('default scores are R^2 for regressors and accuracy for classifiers', () => {
    const fitted = leastSquares().fit({ x, y })
    const score = requireVerb(fitted, 'score', 'test')
    expect(score({ x, y })).toBeGreaterThan(0.9)
    const nc = nearestCentroid().fit({ x, y: [0, 0, 0, 0, 1] })
    expect(requireVerb(nc, 'score', 'test')({ x, y: [0, 0, 0, 0, 1] })).toBe(1)
    expect(isClassifier(nc)).toBe(true)
    expect(isClassifier(fitted)).toBe(false)
  })

  it('capability guards narrow and requireVerb names the estimator', () => {
    const nb = gaussianNB().fit({ x, y: [0, 0, 1, 1, 1] })
    expect(can(nb, 'predictProba')).toBe(true)
    expect(can(nb, 'transform')).toBe(false)
    try {
      requireVerb(nb, 'transform', 'pipeline')
    } catch (e) {
      expect(isOmaiError(e, 'value')).toBe(true)
      expect(String(e)).toContain('gaussian-nb')
    }
    expect(isEstimator(nb)).toBe(false)
    expect(isEstimator(gaussianNB())).toBe(true)
    expect(() => requireTargets({ x }, 'fit')).toThrow()
  })

  it('EstimatorClass wraps the functional form', () => {
    const c = new EstimatorClass(standardScaler())
    expect(() => c.transform(x)).toThrow()
    const t = rows(c.fitTransform(x))
    expect(t[0]?.[0]).toBeCloseTo(-Math.SQRT2, 12)
    expect(rows(c.inverseTransform(t))).toEqual(x.map((r) => r.map((v) => expect.closeTo(v, 12))))
    c.setParams({ withMean: false })
    expect(c.fitted).toBeUndefined()
    expect(c.getParams().withMean).toBe(false)
    const lr = new EstimatorClass(leastSquares()).fit(x, y)
    expect(lr.score(x, y)).toBeGreaterThan(0.9)
    expect(toFlat(lr.predict([[1, 1]])).length).toBe(1)
  })
})

describe('validation', () => {
  it('reads matrices with clear shape and value errors', () => {
    expect(asFloatMatrix([[1, 2]], 'demo').shape).toEqual([1, 2])
    expect(() => matrixView([[1, 2]], 'demo', { nFeatures: 3 })).toThrow(/expected 3 features/)
    expect(() => matrixView([], 'demo')).toThrow()
    expect(() => matrixView([[Number.POSITIVE_INFINITY]], 'demo')).toThrow(/infinite/)
    expect(matrixView([[Number.NaN]], 'demo', { allowNaN: true }).n).toBe(1)
  })

  it('encodes labels and checks weights and lengths', () => {
    expect(encodeLabels([3, 1, 3, 2], 'demo').classes).toEqual([1, 2, 3])
    expect(Array.from(codesFor([2, 3], [1, 2, 3], 'demo'))).toEqual([1, 2])
    expect(() => sampleWeightsOf([1, 2], 3, 'demo')).toThrow()
    expect(() => consistentLength('demo', { x: 2, y: 3 })).toThrow(/inconsistent/)
  })
})
