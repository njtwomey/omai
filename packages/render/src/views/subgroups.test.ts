import { dataBias, paretoFront, subgroupGrid, subgroupReport, tableRates } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { dataBiasPanels, dataBiasView } from './data-bias.ts'
import { paretoFrontView } from './frontiers.ts'
import { stillOf, viewFor } from './registry.ts'
import { subgroupBarsPanel, subgroupHeatmapPanel, subgroupView } from './subgroups.ts'

type Opt = { grid: unknown[]; series: { name?: string; data?: unknown[] }[] }

const tables = [
  [
    [1, 1],
    [0, 2],
  ],
  [
    [2, 0],
    [1, 1],
  ],
  [
    [3, 0],
    [0, 1],
  ],
]
const at = { subgroup: 1, gap: 0.3 }
const worst = {
  selectionRate: at,
  truePositiveRate: at,
  falsePositiveRate: at,
  precision: at,
  accuracy: at,
  baseRate: at,
}
const report = subgroupReport({
  attributes: ['sex', 'age'],
  values: [
    ['f', 'm'],
    ['old', 'young'],
  ],
  subgroups: [
    ['f', 'old'],
    ['f', 'young'],
    ['m', 'old'],
  ],
  counts: [4, 4, 4],
  tables,
  rates: tables.map(tableRates),
  overall: tableRates([
    [6, 1],
    [1, 4],
  ]),
  minSize: 3,
  gaps: { selectionRate: 0.5, truePositiveRate: 0.5, falsePositiveRate: 0.5, precision: 0, accuracy: 0, baseRate: 0 },
  worst,
})

describe('subgroup and data-bias views', () => {
  it('lays a rate out over the attributes, leaving empty subgroups blank', () => {
    const g = subgroupGrid(report, 'selectionRate')
    expect(g.rows).toEqual(['f', 'm'])
    expect(g.columns).toEqual(['old', 'young'])
    expect(g.values[1]?.[1]).toBeNaN()
    expect(g.counts[1]).toEqual([4, 0])
    expect(subgroupHeatmapPanel(report, 'light', 'selectionRate').layers).toHaveLength(1)
    expect(subgroupBarsPanel(report, 'light', 'accuracy').options?.title).toContain('worst f × young')
  })

  it('draws the heatmap and the bars in one figure, and is the registered view', () => {
    expect((subgroupView(report) as unknown as Opt).grid).toHaveLength(2)
    expect(viewFor('subgroupReport').component).toBe('SubgroupView')
    expect((stillOf(report) as unknown as Opt).grid).toHaveLength(2)
  })

  it('draws only the data-bias panels that have data', () => {
    const base = {
      attribute: 'sex',
      groups: ['f', 'm'],
      reference: 1,
      counts: [40, 60],
      shares: [0.4, 0.6],
      baseRates: [0.3, 0.5],
      overallBaseRate: 0.42,
      disparities: [],
    }
    expect(Object.keys(dataBiasPanels(dataBias(base), 'light'))).toEqual(['shares', 'rates'])
    const full = dataBias({
      ...base,
      populationShares: [0.5, 0.5],
      labelFlips: { positiveToNegative: [0.3, 0], negativeToPositive: [0, 0] },
      proxies: { features: ['postcode', 'income'], auc: [0.9, 0.55], dependence: [0.4, 0.01], combinedAuc: 0.92 },
      amplification: { selectionRates: [0.2, 0.55], value: 0.1 },
    })
    expect((dataBiasView(full) as unknown as Opt).grid).toHaveLength(4)
    expect(viewFor('dataBias').component).toBe('DataBiasView')
  })
})

describe('Pareto front view', () => {
  it('draws every family, its front, the marks and the choice', () => {
    const r = paretoFront({
      violationName: 'gap',
      families: [
        { name: 'grid', violation: [0.1, 0.2, 0.3], accuracy: [0.7, 0.8, 0.75], front: [0, 1] },
        { name: 'thresholds', violation: [0.05, 0.15], accuracy: [0.72, 0.79], front: [0, 1] },
      ],
      marks: [{ violation: 0, accuracy: 0.7, label: 'exact' }],
      chosen: { family: 1, index: 1, rule: 'best within 0.2' },
    })
    const o = paretoFrontView(r) as unknown as Opt
    expect(o.series.some((s) => s.name === 'grid')).toBe(true)
    expect(o.series.some((s) => s.name === 'exact')).toBe(true)
    expect(o.series.some((s) => s.name?.startsWith('chosen'))).toBe(true)
    expect(viewFor('paretoFront').component).toBe('ParetoFrontView')
    expect(() =>
      paretoFront({ violationName: 'g', families: [{ name: 'a', violation: [0], accuracy: [1], front: [3] }] }),
    ).toThrow(/candidate/)
  })
})
