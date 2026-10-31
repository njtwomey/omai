import { normal, poisson } from 'mdml-engine/dists'
import { describe, expect, it } from 'vitest'
import { figure } from '../figure/figure.ts'
import { barChart, bars, histogram, paretoChart } from './bars.ts'
import {
  boxPlot,
  boxStats,
  density,
  densityChart,
  ecdfCurve,
  kde,
  meanBand,
  pmf,
  priorPosterior,
  qqPlot,
  violinPlot,
} from './distributions.ts'
import { band, curvesFromTensor, lineChart, lines } from './lines.ts'
import { cartesian, layer, panelParts } from './plot.ts'
import { points, rug, scatterChart } from './points.ts'

// biome-ignore lint/suspicious/noExplicitAny: tests inspect untyped ECharts option trees
type Opt = any
const opt = (o: unknown): Opt => o as Opt

// A minimal stand-in for the ECharts custom-series API: data coordinates map to pixels one to one.
const api = (values: readonly number[]) => ({
  coord: (p: readonly (number | string)[]) => p.map((v) => (typeof v === 'number' ? v : 0)),
  value: (d: number) => values[d] ?? 0,
  size: (p: readonly number[]) => p.map((v) => v * 10),
})

describe('cartesian', () => {
  it('assembles grid, axes, series and legend in the figure area', () => {
    const o = opt(
      cartesian(
        [
          lines([
            { y: [1, 2], name: 'a' },
            { y: [2, 1], name: 'b' },
          ]),
        ],
        {
          figure: figure({ width: 400, aspect: 2 }),
          title: 'T',
          x: { label: 'step', limits: [0, 1] },
          y: { scale: 'log', label: 'loss', limits: { kind: 'auto', nice: true } },
          zoom: true,
        },
      ),
    )
    expect(o.grid[0]).toMatchObject({ left: 56, top: 28, width: 324, height: 128 })
    expect(o.xAxis[0]).toMatchObject({ type: 'value', min: 0, max: 1, name: 'step' })
    expect(o.yAxis[0]).toMatchObject({ type: 'log', min: 1, max: 10 })
    expect(o.yAxis[0].axisLabel.formatter(100)).toBe('10²')
    expect(o.legend).toMatchObject({ show: true, data: ['a', 'b'] })
    expect(o.title.text).toBe('T')
    expect(o.series.map((s: Opt) => s.xAxisIndex)).toEqual([0, 0])
    expect(o.dataZoom).toHaveLength(2)
    expect(o.tooltip.valueFormatter(0.123456)).toBe('0.1235')
  })
  it('honours equal data aspect and hides axes on request', () => {
    const p = panelParts(
      [layer([], { x: [0, 1], y: [0, 1], tight: true })],
      { aspect: 'equal', x: { show: false } },
      { x: 0, y: 0, width: 200, height: 100 },
      2,
      3,
      0,
    )
    expect(p.ranges).toEqual({ x: [-0.5, 1.5], y: [0, 1] })
    expect(p.xAxis['axisLabel']).toMatchObject({ show: false })
    expect(p.xAxis['gridIndex']).toBe(2)
    expect(opt(cartesian([], { tooltip: false })).tooltip.show).toBe(false)
  })
  it('adds a colour bar beside the plot', () => {
    const l = layer(
      [{ type: 'scatter', data: [] }],
      { x: [0, 1], y: [0, 1] },
      { colourBar: { min: 0, max: 2, colours: ['#000', '#fff'], label: 'p' } },
    )
    const o = opt(cartesian([l, l], { mode: 'dark' }))
    expect(o.visualMap.map((v: Opt) => v.seriesIndex)).toEqual([0, 1])
    expect(o.visualMap[0]).toMatchObject({ type: 'continuous', min: 0, max: 2 })
    const piece = opt(
      cartesian([layer([], {}, { colourBar: { min: 0, max: 2, colours: ['#000', '#fff'], levels: [0, 1, 2] } })]),
    )
    expect(piece.visualMap[0].pieces).toEqual([
      { min: 0, max: 1, color: '#000' },
      { min: 1, max: 2, color: '#fff' },
    ])
  })
})

describe('a legend that wraps', () => {
  const curves = (names: string[]) => lines(names.map((name) => ({ y: [1, 2], name })))
  const top = (names: string[]): number =>
    opt(cartesian([curves(names)], { figure: figure({ width: 440, height: 320 }) })).grid[0].top

  it('leaves the plot area where it was while the legend fits one line', () => {
    expect(top(['a', 'b', 'c'])).toBe(28)
  })

  it('moves the plot area down a line of type for each extra row of the legend', () => {
    const seven = ['gradientDescent', 'conjugateGradient', 'newton', 'bfgs', 'lbfgs', 'coordinateDescent', 'gradient']
    expect(top(seven)).toBeGreaterThan(28 + 15)
  })
})

describe('lines and bands', () => {
  it('breaks lines at non-finite values and fills areas to zero', () => {
    const l = lines([{ x: [0, 1, 2], y: [1, Number.NaN, 3], area: true, step: 'end', dash: [2, 2] }], { mode: 'dark' })
    const s = opt(l.series[0])
    expect(s.data).toEqual([
      [0, 1],
      [1, null],
      [2, 3],
    ])
    expect(s.areaStyle.opacity).toBe(0.2)
    expect(s.lineStyle.type).toEqual([2, 2])
    expect(l.extent.y).toEqual([0, 3])
    expect(() => lines([{ x: [0], y: [1, 2] }])).toThrow()
  })
  it('draws each band as polygons split at gaps', () => {
    const b = band([{ x: [0, 1, 2, 3, 4], lower: [0, 0, Number.NaN, 0, 0], upper: [1, 1, 1, 1, 1], edges: true }])
    const s = opt(b.series[0])
    expect(s.data).toEqual([[0], [1]])
    const item = s.renderItem({ dataIndex: 1 }, api([]))
    expect(item.shape.points).toEqual([
      [3, 1],
      [4, 1],
      [4, 0],
      [3, 0],
    ])
    expect(() => band([{ x: [0], lower: [0, 1], upper: [1] }])).toThrow()
  })
  it('reads curves from tensors and makes line charts', () => {
    expect(
      curvesFromTensor([
        [1, 2],
        [3, 4],
      ]),
    ).toHaveLength(2)
    expect(() => curvesFromTensor([[[1]]])).toThrow()
    expect(opt(lineChart([{ y: [1, 2] }])).tooltip.trigger).toBe('axis')
  })
})

describe('points', () => {
  it('groups, sizes, colours and labels points', () => {
    const l = points([
      {
        x: [0, 1, 2],
        y: [0, 1, 2],
        sizes: [1, 2, 3],
        values: [0, 1, 2],
        labels: ['a', 'b', 'c'],
        shapes: true,
        groups: ['u', 'v', 'u'],
      },
    ])
    const [u, v] = l.series.map(opt)
    expect(u?.data).toHaveLength(2)
    expect(v?.symbol).toBe('rect')
    expect(u?.data[0].symbolSize).toBe(4)
    expect(u?.data[1].symbolSize).toBe(28)
    expect(u?.data[0].itemStyle.color).toBe('#440154')
    expect(u?.data[1].label.formatter).toBe('c')
    expect(() => points([{ x: [1, 2, 3] }])).toThrow()
    expect(() => points([{ x: [1], y: [1, 2] }])).toThrow()
    expect(() => points([{ x: [1], y: [1], groups: [] }])).toThrow()
  })
  it('draws rugs along either edge', () => {
    const r = opt(rug([1, 2], { axis: 'y' }).series[0])
    const item = r.renderItem({ coordSys: { x: 10, y: 0, width: 100, height: 50 } }, api([2]))
    expect(item.shape).toEqual({ x1: 10, y1: 2, x2: 18, y2: 2 })
    const x = opt(rug([1]).series[0]).renderItem({ coordSys: { x: 0, y: 0, width: 100, height: 50 } }, api([3]))
    expect(x.shape).toEqual({ x1: 3, y1: 50, x2: 3, y2: 42 })
    expect(opt(scatterChart([{ x: [0], y: [0] }])).series).toHaveLength(1)
  })
})

describe('bars', () => {
  it('places groups side by side or stacked, with error bars', () => {
    const l = bars(
      ['a', 'b'],
      [
        { values: [1, -2], errors: [0.5, 0.5] },
        { values: [2, 2], name: 'g' },
      ],
      { stacked: true, labels: true },
    )
    expect(l.extent.y).toEqual([-2, 3])
    expect(l.series.map((s) => opt(s).type)).toEqual(['bar', 'custom', 'bar'])
    const item = opt(l.series[1]).renderItem({}, api([1, 1.5, 2.5]))
    expect(item.children[1].shape).toEqual({ x1: 0, y1: 1.5, x2: 0, y2: 2.5 })
    const h = bars(['a'], [3], { horizontal: true })
    expect(h.yCategories).toEqual(['a'])
    expect(opt(h.series[0]).itemStyle.borderRadius).toEqual([0, 2, 2, 0])
    expect(() => bars(['a'], [1, 2])).toThrow()
    expect(opt(barChart(['a', 'b'], [1, 2])).yAxis[0].min).toBe(0)
  })
  it('draws histograms as adjoining rectangles, densities and cumulative counts', () => {
    const h = opt(histogram([0, 1, 1, 2, 3], { bins: [0, 1, 3], density: true }).series[0])
    expect(h.data).toEqual([
      [0, 1, 0.2],
      [1, 3, 0.4],
    ])
    const rect = h.renderItem({}, api([1, 3, 0.4]))
    expect(rect.shape).toEqual({ x: 1, y: 0.4, width: 1.5, height: 0 })
    const c = histogram([0, 1, 2, 3], { bins: 2, cumulative: true, density: true, outline: true })
    expect(opt(c.series[0]).data).toEqual([
      [0, 0.5],
      [1.5, 1],
      [3, 1],
    ])
  })
  it('sorts Pareto categories with a cumulative share on a secondary axis', () => {
    const o = opt(paretoChart(['a', 'b'], [1, 3]))
    expect(o.yAxis).toHaveLength(2)
    expect(o.series[1].yAxisIndex).toBe(1)
    expect(o.series[1].data).toEqual([0.75, 1])
  })
})

describe('distributions', () => {
  it('draws densities, KDEs, ECDFs and PMFs', () => {
    const d = density(normal(1, 2), { n: 5, fill: 0.3, dash: 'dashed', name: 'n' })
    expect(d.extent.baseline).toBe('y')
    expect(opt(d.series[0]).data).toHaveLength(5)
    expect(kde([1, 2, 3], { range: [0, 4], n: 3, bandwidth: 'silverman' }).extent.x).toEqual([0, 4])
    expect(ecdfCurve([2, 1]).extent.y).toEqual([0, 1])
    const p = pmf(poisson(2))
    expect(p.extent.x?.[0]).toBe(0)
    expect(
      opt(
        densityChart([
          { name: 'a', dist: normal(0, 1) },
          { name: 'b', dist: normal(2, 1), colour: 'tab3' },
        ]),
      ).legend.show,
    ).toBe(true)
  })
  it('summarises samples as boxes and violins', () => {
    expect(() => boxStats([])).toThrow()
    expect(boxStats([1, 2, 3]).median).toBe(2)
    const b = boxPlot([
      { name: 'a', values: [1, 2, 3] },
      { name: 'b', values: [2, 3, 50] },
    ])
    expect(opt(b.series[0]).data[1].value).toEqual([2, 2.5, 3, 26.5, 50])
    const v = violinPlot([{ name: 'a', values: [0, 1, 1, 2] }], { n: 10 })
    const item = opt(v.series[0]).renderItem({ dataIndex: 0 }, api([]))
    expect(item.children[0].shape.points).toHaveLength(20)
    expect(opt(v.series[0]).renderItem({ dataIndex: 5 }, api([])).children).toEqual([])
  })
  it('makes QQ plots, prior-posterior overlays and mean bands', () => {
    const q = qqPlot([0, 1, 2, 3])
    expect(q.series).toHaveLength(2)
    const pp = priorPosterior(normal(0, 1), normal(1, 1), { n: 10, names: ['p', 'q'] })
    expect(pp.series.map((s) => opt(s).name)).toEqual(['p', 'q'])
    expect(meanBand([0, 1], [1, 1], [0, 0], [2, 2]).extent).toEqual({ x: [0, 1], y: [0, 2] })
  })
})
