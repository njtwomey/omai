import { describe, expect, it } from 'vitest'
import { groupReport, groupReportFromTables } from 'whitebox-engine/exposition'
import { groupReportView, groupSweepView } from './fairness.ts'
import { stillOf, viewFor } from './registry.ts'

type Opt = {
  grid: unknown[]
  series: { name?: string; data?: unknown[] }[]
  title?: { text?: string }[] | { text?: string }
}

const scored = { scores: [0.1, 0.4, 0.6, 0.9, 0.3, 0.7], labels: [0, 1, 0, 1, 0, 1], groups: [0, 0, 0, 1, 1, 1] }
const record = groupReport(
  groupReportFromTables({
    attribute: 'group',
    groups: ['majority', 'minority'],
    tables: [
      [
        [1, 1],
        [1, 0],
      ],
      [
        [1, 0],
        [0, 2],
      ],
    ],
    roc: [
      { fpr: [0, 0.5, 1], tpr: [0, 0.5, 1], thresholds: [1, 0.5, 0] },
      { fpr: [0, 1], tpr: [0, 1], thresholds: [1, 0] },
    ],
    scored,
    sweep: {
      thresholds: [0, 1],
      selectionRate: [
        [1, 0],
        [1, 0],
      ],
      truePositiveRate: [
        [1, 0],
        [1, 0],
      ],
      falsePositiveRate: [
        [1, 0],
        [1, 0],
      ],
    },
  }),
)

describe('group report views', () => {
  it('draws rates, gaps and ROC curves in one figure', () => {
    const o = groupReportView(record) as unknown as Opt
    expect(o.grid).toHaveLength(3)
    expect(o.series.filter((s) => s.name === 'majority' || s.name === 'minority')).toHaveLength(2)
    expect(o.series.filter((s) => s.name?.startsWith('__roc'))).toHaveLength(2)
    expect(viewFor('groupReport').component).toBe('GroupReportView')
    expect((stillOf(record) as unknown as Opt).grid).toHaveLength(3)
  })

  it('moves the operating points with the thresholds', () => {
    const point = (o: Opt, g: number): unknown => o.series.find((s) => s.name === `__point${g}`)?.data?.[0]
    const before = groupReportView(record) as unknown as Opt
    const after = groupReportView(record, { thresholds: [0.5, 0.95] }) as unknown as Opt
    expect(point(after, 0)).toEqual(point(before, 0))
    expect(point(after, 1)).not.toEqual(point(before, 1))
  })

  it('draws the sweep, one line per group and rate', () => {
    expect((groupSweepView(record) as unknown as Opt).series).toHaveLength(6)
  })

  it('draws the interval of each rate when the report has them', () => {
    const withIntervals = {
      ...record,
      intervals: {
        level: 0.95,
        method: 'bootstrap',
        rates: [{ selectionRate: [0.2, 0.6] as const }, { selectionRate: [0.5, 0.9] as const }],
      },
    }
    const o = groupReportView(withIntervals) as unknown as Opt
    expect(o.series.filter((s) => s.name?.startsWith('__errors'))).toHaveLength(2)
    expect((groupReportView(record) as unknown as Opt).series.some((s) => s.name?.startsWith('__errors'))).toBe(false)
  })
})
