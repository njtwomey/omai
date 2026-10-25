import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { expectedCalibrationError } from '../metrics/index.ts'
import * as M from './index.ts'

const v = (r: { readonly estimate: { readonly value: number } }): number => r.estimate.value

describe('fairness metrics', () => {
  it('measures calibration by group as the largest gap between group calibration errors', () => {
    const yTrue = [1, 0, 1, 0, 1, 1, 0, 0]
    const yProb = [0.9, 0.1, 0.4, 0.3, 0.8, 0.6, 0.2, 0.7]
    const groups = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
    const r = M.calibrationByGroup(yTrue, yProb, groups, { nBins: 2 })
    const ece = (idx: number[]): number =>
      v(
        expectedCalibrationError(
          idx.map((i) => yTrue[i] ?? 0),
          idx.map((i) => yProb[i] ?? 0),
          { nBins: 2 },
        ),
      )
    expect(v(r)).toBeCloseTo(Math.abs(ece([0, 1, 2, 3]) - ece([4, 5, 6, 7])), 15)
  })

  it('compares with a reference group (AIF360 style) and rejects a missing one', () => {
    const y = [1, 0, 1, 0, 1, 0]
    const p = [1, 1, 1, 0, 0, 0]
    const g = ['a', 'a', 'b', 'b', 'c', 'c']
    // Selection rates: a 1, b 0.5, c 0; the largest gap to a is 1, to b 0.5.
    expect(v(M.demographicParityDifference(y, p, g, { method: 'to_reference', reference: 'a' }))).toBe(1)
    expect(v(M.demographicParityDifference(y, p, g, { method: 'to_reference', reference: 'b' }))).toBe(0.5)
    expect(() => M.demographicParityDifference(y, p, g, { method: 'to_reference', reference: 'z' })).toThrow(
      /reference/,
    )
  })

  it('names every rate of a group, with FN / FP undefined without false positives', () => {
    const r = M.groupRates([1, 1, 0, 0], [1, 0, 0, 0], ['a', 'a', 'a', 'a']).groups['a']
    expect(r?.negativePredictiveValue).toBeCloseTo(2 / 3)
    expect(r?.falseOmissionRate).toBeCloseTo(1 / 3)
    expect(r?.falseDiscoveryRate).toBe(0)
    expect(r?.errorRatio).toBeNaN()
  })

  it('conditional parity is zero when each stratum is fair, and its worst case is the largest stratum gap', () => {
    const p = [1, 0, 1, 0, 1, 1, 0, 0]
    const g = ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'b']
    const s = [0, 0, 0, 0, 1, 1, 1, 1]
    const r = M.conditionalDemographicParityDifference(p, g, s)
    expect(r.breakdown).toEqual({ '0': 0, '1': 0 })
    const worst = M.conditionalDemographicParityDifference([1, 0, 0, 0], ['a', 'b', 'a', 'b'], [0, 0, 1, 1], {
      agg: 'worst_case',
    })
    expect(v(worst)).toBe(1)
  })

  it('multicalibration bounds calibration within each group and is zero for perfectly calibrated cells', () => {
    const y = [1, 0, 1, 0, 1, 1, 0, 0]
    const prob = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    const g = ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b']
    expect(v(M.multicalibration(y, prob, [g]))).toBe(0)
    const h = ['x', 'y', 'x', 'y', 'x', 'y', 'x', 'y']
    // a∩x holds two positives at 0.5.
    const r = M.multicalibration(y, prob, [g, h])
    expect(v(r)).toBe(0.5)
    expect(r.breakdown?.['0:a & 1:x']).toBe(0.5)
    expect(M.multicalibration(y, prob, [g, h], { intersections: false }).breakdown?.['0:a & 1:x']).toBeUndefined()
  })

  it('a metric frame agrees with the parity metrics and its worst group is its minimum (property)', () => {
    const accuracy = (t: unknown[], p: unknown[]): number => t.filter((x, i) => x === p[i]).length / t.length
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.boolean(), fc.boolean(), fc.integer({ min: 0, max: 2 })), { minLength: 1, maxLength: 40 }),
        (rows) => {
          const y = rows.map(([a]) => (a ? 1 : 0))
          const p = rows.map(([, b]) => (b ? 1 : 0))
          const g = rows.map(([, , c]) => c)
          const f = M.metricFrame(y, p, g, accuracy)
          return (
            Math.abs(f.difference - v(M.accuracyDifference(y, p, g))) < 1e-12 &&
            f.worst.value === f.groupMin &&
            f.toReference[f.reference] === 0 &&
            v(M.multicalibration(y, p, [g])) >= 0
          )
        },
      ),
      { numRuns: 100 },
    )
  })
})
