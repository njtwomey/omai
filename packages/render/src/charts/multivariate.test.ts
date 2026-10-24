import { describe, expect, it } from 'vitest'
import { figure } from '../figure/figure.ts'
import {
  brushFromAxisEvent,
  brushRows,
  parallelCoordinates,
  parallelLayout,
  radarChart,
  scatterMatrix,
  scatterMatrixCells,
  type Table,
} from './multivariate.ts'

const table: Table = {
  columns: ['a', 'b', 'c'],
  rows: [
    [0, 5, 1],
    [1, 6, 2],
    [2, 7, 2],
    [3, 9, 0],
  ],
  groups: [0, 0, 1, 1],
  groupNames: { 0: 'first', 1: 'second' },
}

type Opt = Record<string, unknown> & {
  readonly series: readonly Record<string, unknown>[]
  readonly graphic?: readonly Record<string, unknown>[]
}
const asOption = (o: unknown): Opt => o as Opt

describe('brushes', () => {
  it('keeps the rows inside every brushed interval, in either order, ignoring unknown columns', () => {
    expect(brushRows(table, {})).toEqual([0, 1, 2, 3])
    expect(brushRows(table, { a: [2.5, 0.5] })).toEqual([1, 2])
    expect(brushRows(table, { a: [0.5, 2.5], c: [2, 2] })).toEqual([1, 2])
    expect(brushRows(table, { nope: [0, 0] })).toEqual([0, 1, 2, 3])
  })

  it('updates from ECharts area-select events and ignores anything else', () => {
    const b = brushFromAxisEvent(
      {},
      {
        parallelAxisId: 'b',
        intervals: [
          [1, 2],
          [7, 5],
        ],
      },
    )
    expect(b).toEqual({ b: [5, 7] })
    expect(brushFromAxisEvent(b, { parallelAxisId: 'b', intervals: [] })).toEqual({})
    expect(brushFromAxisEvent(b, { type: 'click' })).toBe(b)
    expect(brushFromAxisEvent(b, null)).toBe(b)
  })
})

describe('parallelCoordinates', () => {
  it('draws one axis per column (ids are the column names) and one series per group', () => {
    const o = asOption(parallelCoordinates(table, { mode: 'dark' }))
    const axes = o['parallelAxis'] as { id: string; dim: number; min: number; max: number }[]
    expect(axes.map((a) => [a.id, a.dim])).toEqual([
      ['a', 0],
      ['b', 1],
      ['c', 2],
    ])
    expect(o.series.map((s) => s['name'])).toEqual(['first', 'second'])
    expect(o.series.map((s) => (s['data'] as unknown[]).length)).toEqual([2, 2])
    expect(o.graphic).toBeUndefined()
  })

  it('fades the rows outside a brush and marks the brushed interval on its axis', () => {
    const fig = figure({ width: 600, height: 300 })
    const o = asOption(parallelCoordinates(table, { figure: fig, brush: { a: [0.5, 2.5] } }))
    expect(o.series.map((s) => [s['name'], (s['data'] as unknown[]).length])).toEqual([
      ['__outside', 2],
      ['first', 1],
      ['second', 1],
    ])
    const layout = parallelLayout(table, { figure: fig })
    const rect = o.graphic?.[0]?.['shape'] as { x: number; y: number; height: number }
    expect(rect.x + 8).toBeCloseTo(layout.axisX[0] ?? 0, 9)
    // a's range is [0, 3]: the interval [0.5, 2.5] spans two thirds of the axis height.
    expect(rect.height).toBeCloseTo((layout.height * 2) / 3, 9)
  })

  it('places axes evenly across the plotting box, with nice ranges', () => {
    const l = parallelLayout(table, { figure: figure({ width: 496 }) })
    expect(l.axisX).toEqual([48, 248, 448])
    expect(l.ranges[1]).toEqual([5, 9])
  })

  it('colours rows along a column when asked', () => {
    const o = asOption(parallelCoordinates(table, { colourBy: 'c' }))
    expect(o.series).toHaveLength(1)
    const data = o.series[0]?.['data'] as { lineStyle: { color: string } }[]
    expect(new Set(data.map((d) => d.lineStyle.color)).size).toBe(3)
  })

  it('rejects ragged rows', () => {
    expect(() => parallelCoordinates({ columns: ['a', 'b'], rows: [[1]] })).toThrow(/one value per column/)
  })
})

describe('radarChart', () => {
  it('draws a polygon per series with nice maxima per spoke, fading all but the highlighted series', () => {
    const o = asOption(
      radarChart(
        [
          { name: 'x', values: [1, 7, 0.3] },
          { name: 'y', values: [2, 3, 0.4] },
        ],
        { indicators: ['p', { name: 'q', max: 10 }, 'r'], highlight: 'y' },
      ),
    )
    const radar = o['radar'] as { indicator: { name: string; max: number }[] }
    expect(radar.indicator.map((i) => i.max)).toEqual([2, 10, 0.4])
    const data = o.series[0]?.['data'] as { name: string; lineStyle: { opacity: number } }[]
    expect(data.map((d) => [d.name, d.lineStyle.opacity])).toEqual([
      ['x', 0.3],
      ['y', 1],
    ])
    expect(() => radarChart([{ name: 'x', values: [1] }], { indicators: ['p', 'q'] })).toThrow(/needs 2 values/)
  })
})

describe('scatterMatrix', () => {
  it('lays out n by n panels with histograms on the diagonal and names in them', () => {
    const o = asOption(scatterMatrix(table))
    expect((o['grid'] as unknown[]).length).toBe(9)
    const texts = (o.graphic ?? []).map((g) => (g['style'] as { text?: string }).text)
    expect(texts).toEqual(['a', 'b', 'c'])
    expect((o['legend'] as { data: string[] }).data).toEqual(['first', 'second'])
  })

  it('fades unselected rows in every panel', () => {
    const o = asOption(scatterMatrix(table, { selection: [0, 3] }))
    const faded = o.series.filter((s) => s['name'] === '__faded')
    expect(faded).toHaveLength(6)
    expect(faded.every((s) => (s['data'] as unknown[]).length === 2)).toBe(true)
  })

  it('reports each panel’s place and ranges for overlays', () => {
    const cells = scatterMatrixCells(table)
    expect(cells).toHaveLength(9)
    const c = cells.find((x) => x.row === 2 && x.col === 0)
    expect([c?.x, c?.y, c?.diagonal]).toEqual(['a', 'c', false])
    expect(c?.xRange?.[0]).toBeCloseTo(-0.18, 9)
    expect(c?.xRange?.[1]).toBeCloseTo(3.18, 9)
    const right = cells.find((x) => x.row === 0 && x.col === 1)
    expect((right?.rect.x ?? 0) - ((c?.rect.x ?? 0) + (c?.rect.width ?? 0))).toBeCloseTo(8, 9)
  })
})
