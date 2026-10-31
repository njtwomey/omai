import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { asTensor, toFlat } from '../tensor/index.ts'
import {
  aomScores,
  averageScores,
  combineScores,
  contaminationThreshold,
  flagOutliers,
  iqrThreshold,
  madThreshold,
  maximizationScores,
  minmaxNormalise,
  moaScores,
  normaliseScores,
  outlierVerbs,
  potThreshold,
  quantileThreshold,
  rankNormalise,
  unifyScores,
  zscoreNormalise,
} from './combine.ts'

const integerVector = fc.array(fc.integer({ min: -50, max: 50 }), { minLength: 2, maxLength: 30 })
const scoreVector = fc.array(fc.double({ min: -100, max: 100, noNaN: true }), { minLength: 2, maxLength: 30 })

describe('thresholds', () => {
  it('flags the contamination fraction of the scores', () => {
    const scores = Array.from({ length: 100 }, (_, i) => i)
    for (const c of [0.05, 0.1, 0.25]) {
      const flagged = flagOutliers(scores, contaminationThreshold(scores, c)).filter(Boolean).length
      expect(flagged).toBe(Math.floor(100 * c))
    }
    expect(() => contaminationThreshold(scores, 0)).toThrow()
    expect(() => contaminationThreshold(scores, 0.6)).toThrow()
  })

  it('has Tukey fences, MAD limits and quantiles that agree with hand values', () => {
    expect(iqrThreshold([1, 2, 3, 4, 5, 6, 7, 8], 1.5)).toBeCloseTo(6.25 + 1.5 * 3.5, 12)
    expect(madThreshold([1, 2, 3, 4, 100], 3)).toBeCloseTo(3 + 3 * 1.4826 * 1, 12)
    expect(quantileThreshold([0, 10], 0.3)).toBeCloseTo(3, 12)
    expect(() => quantileThreshold([1], 2)).toThrow()
  })

  it('puts the extreme-value threshold in the tail, beyond the empirical level for a rarer risk', () => {
    const scores = Array.from({ length: 400 }, (_, i) => -Math.log(1 - (i + 0.5) / 400))
    const rare = potThreshold(scores, { risk: 0.001 })
    expect(rare).toBeGreaterThan(potThreshold(scores, { risk: 0.01 }))
    // The exponential tail has quantile -log(risk).
    expect(rare).toBeGreaterThan(5.5)
    expect(rare).toBeLessThan(8.5)
    expect(() => potThreshold([], {})).toThrow()
    expect(() => potThreshold([1, 2, 3], { risk: 2 })).toThrow()
  })
})

describe('score normalisation', () => {
  it('has hand values', () => {
    expect(zscoreNormalise([1, 2, 3])[1]).toBe(0)
    expect(minmaxNormalise([2, 4, 6])).toEqual([0, 0.5, 1])
    expect(rankNormalise([30, 10, 20, 20])).toEqual([1, 0, 0.5, 0.5])
    expect(unifyScores([5, 5, 5])).toEqual([0, 0, 0])
    expect(unifyScores([0, 0, 0, 0, 0, 100])[0]).toBe(0)
    expect(normaliseScores([1, 2, 3], 'rank')).toEqual([0, 0.5, 1])
    expect(zscoreNormalise([])).toEqual([])
  })

  it('property: z-scores have mean 0 and deviation 1, min-max spans [0, 1], ranks and unified scores are monotone', () => {
    fc.assert(
      fc.property(scoreVector, (v) => {
        const spread = Math.max(...v) - Math.min(...v)
        const z = zscoreNormalise(v)
        const m = minmaxNormalise(v)
        const r = rankNormalise(v)
        const u = unifyScores(v)
        if (spread > 1e-6) {
          expect(z.reduce((a, b) => a + b, 0) / v.length).toBeCloseTo(0, 8)
          expect(Math.sqrt(z.reduce((a, b) => a + b * b, 0) / v.length)).toBeCloseTo(1, 8)
          expect(Math.min(...m)).toBe(0)
          expect(Math.max(...m)).toBe(1)
        }
        for (let i = 0; i < v.length; i++)
          for (let j = 0; j < v.length; j++) {
            if ((v[i] ?? 0) < (v[j] ?? 0)) {
              expect(r[i] ?? 0).toBeLessThan(r[j] ?? 0)
              expect(u[i] ?? 0).toBeLessThanOrEqual((u[j] ?? 0) + 1e-12)
            }
          }
        expect(u.every((s) => s >= 0 && s <= 1)).toBe(true)
      }),
      { numRuns: 60 },
    )
  })

  it('property: z, min-max and rank do not change under a positive affine map of the scores', () => {
    fc.assert(
      fc.property(integerVector, fc.integer({ min: 1, max: 4 }), fc.integer({ min: -5, max: 5 }), (v, a, b) => {
        const w = v.map((s) => a * s + b)
        for (const how of ['zscore', 'minmax', 'rank'] as const) {
          const p = normaliseScores(v, how)
          const q = normaliseScores(w, how)
          // Constant vectors normalise to zero either way; the others agree.
          p.forEach((s, i) => {
            expect(q[i] ?? 0).toBeCloseTo(s, 6)
          })
        }
      }),
      { numRuns: 60 },
    )
  })
})

describe('score combination', () => {
  const k = 4
  const scores = [
    [1, 2, 9],
    [3, 8, 1],
    [2, 4, 5],
    [4, 2, 3],
  ]

  it('has hand values', () => {
    expect(averageScores(scores)).toEqual([2.5, 4, 4.5])
    expect(maximizationScores(scores)).toEqual([4, 8, 9])
    expect(aomScores(scores, { nBuckets: 2 })).toEqual([3, 6, 6])
    expect(moaScores(scores, { nBuckets: 2 })).toEqual([3.5, 5, 7])
    expect(() => averageScores([])).toThrow()
    expect(() => averageScores([[1], [1, 2]])).toThrow()
  })

  it('has the buckets interpolate between the mean and the maximum', () => {
    expect(aomScores(scores, { nBuckets: k })).toEqual(averageScores(scores))
    expect(moaScores(scores, { nBuckets: k })).toEqual(maximizationScores(scores))
    expect(aomScores(scores, { nBuckets: 1 })).toEqual(maximizationScores(scores))
    expect(moaScores(scores, { nBuckets: 1 })).toEqual(averageScores(scores))
    const shuffled = aomScores(scores, { nBuckets: 2, key: key(3) })
    expect(shuffled.length).toBe(3)
  })

  it('property: the mean lies between the minimum and the maximum, the bucketed combinations stay in range and the maximum of bucket means is at least the mean', () => {
    fc.assert(
      fc.property(
        fc.array(fc.array(fc.integer({ min: -10, max: 10 }), { minLength: 3, maxLength: 3 }), {
          minLength: 1,
          maxLength: 7,
        }),
        fc.integer({ min: 1, max: 7 }),
        (s, b) => {
          const mean = averageScores(s)
          const max = maximizationScores(s)
          const aom = aomScores(s, { nBuckets: b })
          const moa = moaScores(s, { nBuckets: b })
          for (let i = 0; i < 3; i++) {
            const col = s.map((r) => r[i] ?? 0)
            expect(mean[i] ?? 0).toBeGreaterThanOrEqual(Math.min(...col) - 1e-6)
            expect(mean[i] ?? 0).toBeLessThanOrEqual((max[i] ?? 0) + 1e-6)
            expect(aom[i] ?? 0).toBeLessThanOrEqual((max[i] ?? 0) + 1e-6)
            expect(aom[i] ?? 0).toBeGreaterThanOrEqual(Math.min(...col) - 1e-6)
            expect(moa[i] ?? 0).toBeLessThanOrEqual((max[i] ?? 0) + 1e-6)
            expect(moa[i] ?? 0).toBeGreaterThanOrEqual((mean[i] ?? 0) - 1e-6)
          }
        },
      ),
      { numRuns: 60 },
    )
  })

  it('combines normalised detectors into a record with the strongest detector as evidence', () => {
    const informative = [0.1, 0.2, 0.15, 0.12, 0.18, 0.95]
    const noisy = [5, 1, 4, 2, 6, 3]
    const rec = combineScores([informative, noisy], { combination: 'mean', contamination: 0.2 })
    expect(rec.scores.indexOf(Math.max(...rec.scores))).toBe(5)
    expect(rec.scores.filter((s) => s > (rec.threshold ?? Number.POSITIVE_INFINITY)).length).toBe(1)
    expect(rec.evidence?.values[5]).toBe(0)
    expect(combineScores([informative, noisy], { normalisation: 'rank', combination: 'median' }).scores.length).toBe(6)
  })
})

describe('the outlier verbs', () => {
  it('labels with the sign convention that negative is an outlier', () => {
    const v = outlierVerbs((x) => Array.from(toFlat(asTensor(x)), (a) => a * 2), 3, 1)
    expect(toFlat(v.decisionFunction([1, 2]))).toEqual([1, -1])
    expect(toFlat(v.predict([1, 2]))).toEqual([1, -1])
    expect(v.nFeaturesIn).toBe(1)
  })
})
