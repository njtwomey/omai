import { describe, expect, it } from 'vitest'
import { rows } from '../../../testing/learn.ts'
import { key } from '../../random/index.ts'
import { exp, log } from '../../tensor/index.ts'
import { functionTransformer, kBinsDiscretizer, polynomialFeatures, splineFeatures } from './features.ts'
import { iterativeImputer, missingIndicator, simpleImputer } from './imputation.ts'
import { nearMiss, samplingTargets } from './resampling.ts'

describe('feature construction edge cases', () => {
  it('validates its options', () => {
    expect(() => polynomialFeatures({ degree: [3, 2] }).fit({ x: [[1]] })).toThrow()
    expect(() => functionTransformer({ func: log, inverseFunc: log }).fit({ x: [[2]] })).toThrow(/invert/)
    expect(() => functionTransformer({ func: log, inverseFunc: exp }).fit({ x: [[2]] })).not.toThrow()
    expect(() =>
      splineFeatures({ extrapolation: 'error' })
        .fit({ x: [[0], [1]] })
        .transform([[2]]),
    ).toThrow()
    expect(() => splineFeatures({ extrapolation: 'periodic', nKnots: 3, degree: 3 }).fit({ x: [[0], [1]] })).toThrow()
  })

  it('bins a constant feature into one bin', () => {
    const b = kBinsDiscretizer({ nBins: 3, encode: 'onehot' }).fit({
      x: [
        [1, 0],
        [1, 1],
        [1, 2],
      ],
    })
    expect(b.params.edges[0]).toEqual([Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY])
    expect(rows(b.transform([[1, 2]]))[0]?.[0]).toBe(1)
  })

  it('periodic splines repeat with the period', () => {
    const s = splineFeatures({ extrapolation: 'periodic', nKnots: 5, degree: 3 }).fit({ x: [[0], [4]] })
    const [a = [], b = []] = rows(s.transform([[1.3], [5.3]]))
    a.forEach((v, i) => {
      expect(v).toBeCloseTo(b[i] ?? 0, 12)
    })
    expect(a.reduce((p, q) => p + q, 0)).toBeCloseTo(1, 12)
  })
})

describe('imputation edge cases', () => {
  it('errors on new missing features and on an inverse without indicators', () => {
    expect(() =>
      missingIndicator()
        .fit({ x: [[1, 2]] })
        .transform([[Number.NaN, 2]]),
    ).toThrow()
    expect(() =>
      simpleImputer()
        .fit({ x: [[1]] })
        .inverseTransform([[1]]),
    ).toThrow(/addIndicator/)
  })

  it('iterative imputation with a predictor subset is reproducible from the key', () => {
    const x = Array.from({ length: 20 }, (_, i) => [i, 2 * i + (i % 3), i % 5 === 0 ? Number.NaN : 3 - i, i * i])
    const est = iterativeImputer({ nNearestFeatures: 2, imputationOrder: 'random' })
    const a = rows(est.fit({ x }, { key: key(3) }).transform(x))
    const b = rows(est.fit({ x }, { key: key(3) }).transform(x))
    expect(a).toEqual(b)
    expect(a.flat().some(Number.isNaN)).toBe(false)
  })
})

describe('sampling strategies', () => {
  it('compute targets from names, ratios and counts', () => {
    expect(samplingTargets([10, 4], [0, 1], 'all', 'under')).toEqual([4, 4])
    expect(samplingTargets([10, 4], [0, 1], { 1: 8 }, 'over')).toEqual([undefined, 8])
    expect(() => samplingTargets([10, 4], [0, 1], { 1: 2 }, 'over')).toThrow()
    expect(() => samplingTargets([1, 2, 3], [0, 1, 2], 0.5, 'over')).toThrow()
    expect(
      nearMiss({ version: 3 }).fitResample({ x: [[0], [1], [2], [8], [9]], y: [0, 0, 0, 1, 1] }).indices.length,
    ).toBe(4)
  })
})
