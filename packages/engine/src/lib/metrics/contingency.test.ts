import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { accuracyScore, confusionMatrix } from './classification.ts'
import { confusionMargins, thresholdConfusion } from './contingency.ts'

describe('contingency tables', () => {
  it('margins of a binary table name every rate', () => {
    // [[TN, FP], [FN, TP]] = [[50, 10], [5, 35]]
    const m = confusionMargins({
      labels: [0, 1],
      matrix: [
        [50, 10],
        [5, 35],
      ],
    })
    expect(m.n).toBe(100)
    expect(m.actual).toEqual([60, 40])
    expect(m.predicted).toEqual([55, 45])
    expect(m.rowRate[1]).toBeCloseTo(35 / 40) // TPR
    expect(m.rowMiss[1]).toBeCloseTo(5 / 40) // FNR
    expect(m.rowRate[0]).toBeCloseTo(50 / 60) // TNR
    expect(m.rowMiss[0]).toBeCloseTo(10 / 60) // FPR
    expect(m.columnRate[1]).toBeCloseTo(35 / 45) // PPV
    expect(m.columnMiss[1]).toBeCloseTo(10 / 45) // FDR
    expect(m.columnRate[0]).toBeCloseTo(50 / 55) // NPV
    expect(m.columnMiss[0]).toBeCloseTo(5 / 55) // FOR
    expect(m.prevalence[1]).toBeCloseTo(0.4)
    expect(m.accuracy).toBeCloseTo(0.85)
  })

  it('empty rows and columns give NaN rates, not errors', () => {
    const m = confusionMargins({
      labels: [0, 1],
      matrix: [
        [4, 0],
        [0, 0],
      ],
    })
    expect(m.rowRate[1]).toBeNaN()
    expect(m.columnRate[1]).toBeNaN()
    expect(m.accuracy).toBe(1)
    expect(
      confusionMargins({
        labels: [0, 1],
        matrix: [
          [0, 0],
          [0, 0],
        ],
      }).accuracy,
    ).toBeNaN()
  })

  it('rejects matrices that do not match their labels', () => {
    expect(() =>
      confusionMargins({
        labels: [0, 1, 2],
        matrix: [
          [1, 0],
          [0, 1],
        ],
      }),
    ).toThrow(/3 by 3/)
  })

  it('thresholding orders negative then positive and honours posLabel and weights', () => {
    expect(thresholdConfusion([1, 1, 2, 2], [0.1, 0.9, 0.2, 0.8], 0.5, { posLabel: 1 })).toEqual({
      labels: [2, 1],
      matrix: [
        [1, 1],
        [1, 1],
      ],
    })
    expect(thresholdConfusion([0, 1, 1], [0.9, 0.9, 0.1], 0.5, { sampleWeight: [2, 3, 5] }).matrix).toEqual([
      [0, 2],
      [5, 3],
    ])
    expect(thresholdConfusion([1, 1], [0.2, 0.7], 0.5).labels).toEqual([0, 1])
    expect(() => thresholdConfusion([0, 1, 2], [0, 0, 0], 0.5)).toThrow(/two classes/)
    expect(() => thresholdConfusion([0, 1], [0], 0.5)).toThrow(/different lengths/)
  })

  it('agrees with confusionMatrix of the thresholded predictions, and margins with accuracy (property)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.boolean(), fc.double({ min: 0, max: 1, noNaN: true })), { minLength: 2, maxLength: 40 }),
        fc.double({ min: 0, max: 1, noNaN: true }),
        (rows, threshold) => {
          const y = [0, 1, ...rows.map(([b]) => (b ? 1 : 0))]
          const s = [0.5, 0.5, ...rows.map(([, v]) => v)]
          const cut = thresholdConfusion(y, s, threshold)
          const direct = confusionMatrix(
            y,
            s.map((v) => (v >= threshold ? 1 : 0)),
            { labels: [0, 1] },
          )
          const m = confusionMargins(cut)
          const rowsSumToOne = m.rowRate.every(
            (r, i) => Number.isNaN(r) || Math.abs(r + (m.rowMiss[i] ?? 0) - 1) < 1e-12,
          )
          const colsSumToOne = m.columnRate.every(
            (r, j) => Number.isNaN(r) || Math.abs(r + (m.columnMiss[j] ?? 0) - 1) < 1e-12,
          )
          const acc = accuracyScore(
            y,
            s.map((v) => (v >= threshold ? 1 : 0)),
          ).estimate.value
          return (
            JSON.stringify(cut.matrix) === JSON.stringify(direct.matrix) &&
            m.n === y.length &&
            Math.abs(m.accuracy - acc) < 1e-12 &&
            rowsSumToOne &&
            colsSumToOne
          )
        },
      ),
      { numRuns: 200 },
    )
  })
})
