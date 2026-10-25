import { describe, expect, it } from 'vitest'
import { groupDisparities, groupReport, groupReportAt, groupReportFromTables, tableRates } from './index.ts'

const tables = [
  [
    [40, 10],
    [5, 45],
  ],
  [
    [30, 5],
    [10, 5],
  ],
]

describe('group reports', () => {
  it('derive rates, counts, the overall rates and the reference from tables', () => {
    const r = groupReport(groupReportFromTables({ attribute: 'sex', groups: ['m', 'f'], tables }))
    expect(r.counts).toEqual([100, 50])
    expect(r.reference).toBe(0)
    expect(r.thresholds).toEqual([0.5, 0.5])
    expect(r.rates[0]).toEqual({
      selectionRate: 0.55,
      truePositiveRate: 0.9,
      falsePositiveRate: 0.2,
      precision: 45 / 55,
      accuracy: 0.85,
      baseRate: 0.5,
    })
    expect(r.overall.selectionRate).toBeCloseTo(65 / 150, 12)
    expect(r.overall.truePositiveRate).toBeCloseTo(50 / 65, 12)
  })

  it('summarise the disparities between groups and against the reference', () => {
    const d = Object.fromEntries(groupDisparities(tables.map(tableRates), 0).map((x) => [x.name, x.value]))
    expect(d['demographic parity difference']).toBeCloseTo(0.55 - 0.2, 12)
    expect(d['equal opportunity difference']).toBeCloseTo(0.9 - 1 / 3, 12)
    expect(d['equalised odds difference']).toBeCloseTo(Math.max(0.9 - 1 / 3, 0.2 - 5 / 35), 12)
    expect(d['disparate impact']).toBeCloseTo(0.2 / 0.55, 12)
    expect(
      tableRates([
        [0, 0],
        [0, 0],
      ]).precision,
    ).toBe(0)
  })

  it('re-threshold from the scored rows', () => {
    const scored = { scores: [0.1, 0.4, 0.6, 0.9, 0.3, 0.7], labels: [0, 1, 0, 1, 0, 1], groups: [0, 0, 0, 1, 1, 1] }
    const base = groupReport(
      groupReportFromTables({
        attribute: 'a',
        groups: ['x', 'y'],
        tables: [
          [
            [1, 0],
            [0, 1],
          ],
          [
            [1, 0],
            [0, 1],
          ],
        ],
        scored,
      }),
    )
    const at = groupReportAt(base, [0.5, 0.2])
    expect(at.tables).toEqual([
      [
        [1, 1],
        [1, 0],
      ],
      [
        [0, 1],
        [0, 2],
      ],
    ])
    expect(at.thresholds).toEqual([0.5, 0.2])
    expect(() => groupReportAt(base, [0.5])).toThrow(/thresholds/)
    const { scored: _, ...bare } = base
    expect(() => groupReportAt(groupReport(bare), [0.5, 0.5])).toThrow(/no scored rows/)
  })

  it('reject mismatched shapes', () => {
    const ok = groupReportFromTables({ attribute: 'a', groups: ['x', 'y'], tables })
    expect(() => groupReport({ ...ok, thresholds: [0.5] })).toThrow(/thresholds/)
    expect(() =>
      groupReport({
        ...ok,
        tables: [
          tables[0] ?? [],
          [
            [1, 2, 3],
            [4, 5, 6],
          ],
        ],
      }),
    ).toThrow(/tables\[1\]/)
    expect(() => groupReport({ ...ok, reference: 2 })).toThrow(/reference/)
    expect(() =>
      groupReport({
        ...ok,
        roc: [{ fpr: [0, 1], tpr: [0], thresholds: [1, 0] }, ok.roc?.[0] ?? { fpr: [], tpr: [], thresholds: [] }],
      }),
    ).toThrow(/roc\[0\]\.tpr/)
    expect(() => groupReport({ ...ok, scored: { scores: [1, 2], labels: [1], groups: [0, 1] } })).toThrow(
      /scored\.labels/,
    )
  })
})
