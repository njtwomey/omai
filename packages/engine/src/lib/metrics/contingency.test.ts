import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { sweepAt } from '../exposition/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { accuracyScore, confusionMatrix } from './classification.ts'
import { analyseThresholds, confusionMargins, tableCost, thresholdConfusion, thresholdCosts } from './contingency.ts'
import { rocAucScore } from './curves.ts'
import { averagePrecisionScore } from './precision-recall.ts'

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
    const [actual, predicted, rowRate, rowMiss, columnRate, columnMiss, prevalence] = [
      m.actual,
      m.predicted,
      m.rowRate,
      m.rowMiss,
      m.columnRate,
      m.columnMiss,
      m.prevalence,
    ].map(toFlat)
    expect(actual).toEqual([60, 40])
    expect(predicted).toEqual([55, 45])
    expect(rowRate?.[1]).toBeCloseTo(35 / 40) // TPR
    expect(rowMiss?.[1]).toBeCloseTo(5 / 40) // FNR
    expect(rowRate?.[0]).toBeCloseTo(50 / 60) // TNR
    expect(rowMiss?.[0]).toBeCloseTo(10 / 60) // FPR
    expect(columnRate?.[1]).toBeCloseTo(35 / 45) // PPV
    expect(columnMiss?.[1]).toBeCloseTo(10 / 45) // FDR
    expect(columnRate?.[0]).toBeCloseTo(50 / 55) // NPV
    expect(columnMiss?.[0]).toBeCloseTo(5 / 55) // FOR
    expect(prevalence?.[1]).toBeCloseTo(0.4)
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
    expect(toFlat(m.rowRate)[1]).toBeNaN()
    expect(toFlat(m.columnRate)[1]).toBeNaN()
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
    const tied = thresholdConfusion([1, 1, 2, 2], [0.1, 0.9, 0.2, 0.8], 0.5, { posLabel: 1 })
    expect(tied.labels).toEqual([2, 1])
    expect(toArray(tied.matrix)).toEqual([
      [1, 1],
      [1, 1],
    ])
    expect(toArray(thresholdConfusion([0, 1, 1], [0.9, 0.9, 0.1], 0.5, { sampleWeight: [2, 3, 5] }).matrix)).toEqual([
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
          const rowMiss = toFlat(m.rowMiss)
          const columnMiss = toFlat(m.columnMiss)
          const rowsSumToOne = toFlat(m.rowRate).every(
            (r, i) => Number.isNaN(r) || Math.abs(r + (rowMiss[i] ?? 0) - 1) < 1e-12,
          )
          const colsSumToOne = toFlat(m.columnRate).every(
            (r, j) => Number.isNaN(r) || Math.abs(r + (columnMiss[j] ?? 0) - 1) < 1e-12,
          )
          const acc = accuracyScore(
            y,
            s.map((v) => (v >= threshold ? 1 : 0)),
          ).estimate.value
          return (
            JSON.stringify(toArray(cut.matrix)) === JSON.stringify(toArray(direct.matrix)) &&
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

describe('misclassification costs', () => {
  it('cost a table cell by cell', () => {
    const c = tableCost(
      {
        labels: [0, 1],
        matrix: [
          [880, 70],
          [8, 42],
        ],
      },
      [
        [0, 1],
        [50, -2],
      ],
    )
    expect(toArray(c.cells)).toEqual([
      [0, 70],
      [400, -84],
    ])
    expect(c.total).toBe(386)
    expect(c.perCase).toBeCloseTo(0.386)
    expect(() =>
      tableCost(
        {
          labels: [0, 1],
          matrix: [
            [1, 0],
            [0, 1],
          ],
        },
        [[0, 1]],
      ),
    ).toThrow(/2 by 2/)
  })

  it('agree with tableCost of thresholdConfusion at every threshold (property)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.boolean(), fc.integer({ min: 0, max: 20 })), { minLength: 1, maxLength: 30 }),
        fc.tuple(fc.integer({ min: 0, max: 9 }), fc.integer({ min: 0, max: 9 }), fc.integer({ min: 0, max: 9 })),
        (rows, [fp, fn, tp]) => {
          const y = [0, 1, ...rows.map(([b]) => (b ? 1 : 0))]
          const s = [3, 7, ...rows.map(([, v]) => v / 2)]
          const costs = [
            [0, fp],
            [fn, -tp],
          ]
          const sweep = thresholdCosts(y, s, costs)
          const direct = toFlat(sweep.thresholds).map((th) => tableCost(thresholdConfusion(y, s, th), costs).total)
          const min = Math.min(...direct)
          return (
            toFlat(sweep.totals).every((v, i) => Math.abs(v - (direct[i] ?? Number.NaN)) < 1e-9) &&
            sweep.best.total === min &&
            sweep.thresholds.shape[0] === new Set(s).size + 1
          )
        },
      ),
      { numRuns: 200 },
    )
  })
})

describe('threshold analysis', () => {
  it('names the classes, keeps the rows and reads any threshold back', () => {
    const r = analyseThresholds(['ham', 'spam', 'spam', 'ham'], [0.2, 0.7, 0.5, 0.5], { posLabel: 'spam' })
    expect(r.classes).toEqual(['ham', 'spam'])
    expect(r.thresholds).toEqual([0.2, 0.5, 0.7, Number.POSITIVE_INFINITY])
    expect(r.scored?.labels).toEqual([0, 1, 1, 0])
    expect(sweepAt(r, 0.6).table).toEqual([
      [2, 0],
      [1, 1],
    ])
    expect(sweepAt(r, 9).selectionRate).toBe(0)
    expect(Number.isNaN(sweepAt(r, 9).precision)).toBe(true)
    expect(r.cost).toBeUndefined()
    expect(analyseThresholds([0, 1], [0, 1], { keepRows: false }).scored).toBeUndefined()
    expect(() => analyseThresholds([0, 1], [0, Number.NaN])).toThrow(/NaN/)
    expect(() => analyseThresholds([0, 1], [0, 1], { costs: [[0, 1]] })).toThrow(/2 by 2/)
  })

  it('weights rows and leaves the areas undefined with one class', () => {
    const r = analyseThresholds([0, 1, 1], [0.9, 0.9, 0.1], { sampleWeight: [2, 3, 5] })
    expect(sweepAt(r, 0.5).table).toEqual([
      [0, 2],
      [5, 3],
    ])
    expect(r.positives).toBe(8)
    expect(Number.isNaN(analyseThresholds([1, 1], [0.2, 0.4]).rocAuc)).toBe(true)
  })

  it('agrees with thresholdConfusion, thresholdCosts and the curve metrics at every threshold (property)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.boolean(), fc.integer({ min: 0, max: 20 }), fc.integer({ min: 1, max: 4 })), {
          minLength: 1,
          maxLength: 30,
        }),
        fc.tuple(fc.integer({ min: 0, max: 9 }), fc.integer({ min: 0, max: 9 })),
        (rows, [fp, fn]) => {
          const y = [0, 1, ...rows.map(([b]) => (b ? 1 : 0))]
          const s = [3, 7, ...rows.map(([, v]) => v / 2)]
          const w = [1, 1, ...rows.map(([, , k]) => k)]
          const costs = [
            [0, fp],
            [fn, 0],
          ]
          const r = analyseThresholds(y, s, { costs, sampleWeight: w })
          const cut = thresholdCosts(y, s, costs, { sampleWeight: w })
          const tablesAgree = r.thresholds.every(
            (t, i) =>
              JSON.stringify(sweepAt(r, t).table) ===
                JSON.stringify(toArray(thresholdConfusion(y, s, t, { sampleWeight: w }).matrix)) && sweepAt(r, t).index === i,
          )
          const close = (a: number, b: number): boolean => Math.abs(a - b) < 1e-9
          return (
            tablesAgree &&
            r.cost?.every((c, i) => close(c, toFlat(cut.totals)[i] ?? Number.NaN)) === true &&
            r.thresholds[r.cheapest ?? -1] === cut.best.threshold &&
            close(r.rocAuc, rocAucScore(y, s, { sampleWeight: w }).estimate.value) &&
            close(r.averagePrecision, averagePrecisionScore(y, s, { sampleWeight: w }).estimate.value)
          )
        },
      ),
      { numRuns: 150 },
    )
  })
})
