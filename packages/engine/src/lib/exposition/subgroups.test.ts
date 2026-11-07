import { describe, expect, it } from 'vitest'
import { type GroupRateName, tableRates } from './fairness.ts'
import { type SubgroupReport, subgroupGrid, subgroupReport } from './subgroups.ts'

const RATES: GroupRateName[] = [
  'selectionRate',
  'truePositiveRate',
  'falsePositiveRate',
  'precision',
  'accuracy',
  'baseRate',
]
const zero = Object.fromEntries(RATES.map((r) => [r, 0])) as Record<GroupRateName, number>
const spot = { subgroup: 0, gap: 0 }
const worst = Object.fromEntries(RATES.map((r) => [r, spot])) as Record<GroupRateName, typeof spot>
const tableA = [
  [30, 10],
  [5, 55],
]
const tableB = [
  [20, 20],
  [10, 10],
]
const tableC = [
  [8, 2],
  [1, 9],
]

/** Three subgroups of sex (f, m) by age (old, young, mid): `m, mid` has no rows, so there is no subgroup for it. */
function report(over: Partial<Omit<SubgroupReport, 'kind'>> = {}): SubgroupReport {
  const tables = [tableA, tableB, tableC]
  return subgroupReport({
    attributes: ['sex', 'age'],
    values: [
      ['f', 'm'],
      ['old', 'young', 'mid'],
    ],
    subgroups: [
      ['f', 'old'],
      ['f', 'young'],
      ['m', 'young'],
    ],
    counts: [100, 60, 20],
    tables,
    rates: tables.map(tableRates),
    overall: zero,
    minSize: 1,
    gaps: zero,
    worst,
    ...over,
  })
}

describe('subgroupGrid', () => {
  it('lays a rate out with the first attribute as rows and the others as columns (one named column for one attribute, joined combinations for three), NaN and 0 where a cell has no rows', () => {
    const grid = subgroupGrid(report(), 'selectionRate')
    expect(grid.rows).toEqual(['f', 'm'])
    expect(grid.columns).toEqual(['old', 'young', 'mid'])
    // selection rate = (fp + tp) / n of each table: 65 / 100, 30 / 60, 11 / 20
    expect(grid.values).toEqual([
      [0.65, 0.5, Number.NaN],
      [Number.NaN, 0.55, Number.NaN],
    ])
    expect(grid.counts).toEqual([
      [100, 60, 0],
      [0, 20, 0],
    ])
    // another rate reads the same cells
    expect(subgroupGrid(report(), 'accuracy').values[0]).toEqual([85 / 100, 0.5, Number.NaN])

    const one = subgroupReport({
      attributes: ['sex'],
      values: [['f', 'm']],
      subgroups: [['f'], ['m']],
      counts: [100, 60],
      tables: [tableA, tableB],
      rates: [tableRates(tableA), tableRates(tableB)],
      overall: zero,
      minSize: 1,
      gaps: zero,
      worst,
    })
    const g = subgroupGrid(one, 'truePositiveRate')
    expect([g.rows, g.columns, g.values]).toEqual([['f', 'm'], ['sex'], [[55 / 60], [0.5]]])
    const three = subgroupReport({
      attributes: ['sex', 'age', 'region'],
      values: [['f'], ['old', 'young'], ['n', 's']],
      subgroups: [['f', 'young', 's']],
      counts: [100],
      tables: [tableA],
      rates: [tableRates(tableA)],
      overall: zero,
      minSize: 1,
      gaps: zero,
      worst,
    })
    const h = subgroupGrid(three, 'baseRate')
    expect(h.columns).toEqual(['old × n', 'old × s', 'young × n', 'young × s'])
    expect(h.values).toEqual([[Number.NaN, Number.NaN, Number.NaN, 0.6]])
    // a subgroup naming a value that no attribute lists is left out
    const stray = report({
      subgroups: [
        ['f', 'old'],
        ['x', 'young'],
        ['m', 'young'],
      ],
    })
    expect(subgroupGrid(stray, 'selectionRate').counts).toEqual([
      [100, 0, 0],
      [0, 20, 0],
    ])
  })
})

describe('subgroupReport', () => {
  it('checks every per-subgroup field, the tables and the worst subgroups, and does not share its arrays', () => {
    const shape = expect.objectContaining({ code: 'shape' })
    const value = expect.objectContaining({ code: 'value' })
    const bad: [string, Partial<Omit<SubgroupReport, 'kind'>>, unknown][] = [
      ['counts', { counts: [1, 2] }, shape],
      ['tables', { tables: [tableA] }, shape],
      ['rates', { rates: [tableRates(tableA)] }, shape],
      ['values per attribute', { values: [['f', 'm']] }, shape],
      ['a subgroup with one value', { subgroups: [['f'], ['f', 'young'], ['m', 'young']] }, shape],
      [
        'a 3-row table',
        {
          tables: [
            tableA,
            tableB,
            [
              [1, 2],
              [3, 4],
              [5, 6],
            ],
          ],
        },
        shape,
      ],
      [
        'a table row of 3',
        {
          tables: [
            tableA,
            tableB,
            [
              [1, 2, 3],
              [4, 5, 6],
            ],
          ],
        },
        shape,
      ],
      ['a worst subgroup past the end', { worst: { ...worst, accuracy: { subgroup: 3, gap: 0 } } }, value],
      ['a negative worst subgroup', { worst: { ...worst, accuracy: { subgroup: -1, gap: 0 } } }, value],
      ['a fractional worst subgroup', { worst: { ...worst, accuracy: { subgroup: 0.5, gap: 0 } } }, value],
      [
        'an audit with fewer weights than features',
        {
          audit: {
            rate: 'falsePositive',
            weightedDisparity: 0,
            disparity: 0,
            size: 0.1,
            subgroupRate: 0,
            baseline: 0,
            features: ['sex', 'age'],
            weights: [1],
            intercept: 0,
          },
        },
        shape,
      ],
      ['a NaN gap', { gaps: { ...zero, accuracy: Number.NaN } }, value],
    ]
    for (const [name, over, error] of bad) expect(() => report(over), name).toThrow(error)
    const input = [
      [30, 10],
      [5, 55],
    ]
    const sealed = report({ tables: [input, tableB, tableC] })
    const first = input[0] as number[]
    first[0] = 99
    expect(sealed.tables[0]?.[0]).toEqual([30, 10])
    expect(sealed.kind).toBe('subgroupReport')
  })
})
