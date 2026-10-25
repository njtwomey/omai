import { describe, expect, it } from 'vitest'
import { annotations } from './annotations.ts'
import { waterfallChart } from './bars.ts'
import { hexbin } from './binning.ts'
import { ridgeline } from './distributions.ts'
import { prGainChart } from './evaluation.ts'
import { sankeyChart } from './flows.ts'
import { sunburstChart, treemapChart } from './hierarchies.ts'
import { funnelChart, gaugeChart, pieChart } from './parts.ts'
import { errorBars } from './points.ts'
import { stackedAreas } from './stacks.ts'
import { calendarHeatmap, candlestickChart } from './time.ts'

const seriesOf = (o: unknown): { type?: string; data?: unknown[] }[] =>
  (o as { series: { type?: string; data?: unknown[] }[] }).series

describe('more chart types', () => {
  it('stacks areas from zero, as shares, and about a centred baseline', () => {
    const x = [0, 1, 2]
    const s = [
      { name: 'a', y: [1, 2, 3] },
      { name: 'b', y: [3, 2, 1] },
    ]
    expect(stackedAreas(x, s).extent.y).toEqual([0, 4])
    expect(stackedAreas(x, s, { offset: 'expand' }).extent.y).toEqual([0, 1])
    const w = stackedAreas(x, s, { offset: 'wiggle' }).extent.y ?? [0, 0]
    expect(w[0]).toBeLessThan(0)
    // The streamgraph's baseline moves below zero and its top above it: the layers sit about the middle.
    expect(w[1]).toBeGreaterThan(0)
  })

  it('bins points into hexagons, every point counted once', () => {
    const x = Array.from({ length: 200 }, (_, i) => Math.cos(i) * 3)
    const y = Array.from({ length: 200 }, (_, i) => Math.sin(i * 1.7) * 3)
    const data = seriesOf(hexbin(x, y, { radius: 0.5 }))[0]?.data as [number, number, number][]
    expect(data.reduce((s, d) => s + d[2], 0)).toBe(200)
  })

  it('floats a waterfall from the running total, and ends on it', () => {
    const series = seriesOf(
      waterfallChart(
        [
          { name: 'a', value: 5 },
          { name: 'b', value: -2 },
        ],
        { start: { name: 's', value: 1 } },
      ),
    )
    const [base, rise, fall] = series.map((s) => s.data)
    expect(base).toEqual([0, 1, 4, 0])
    expect(rise).toEqual([1, 5, null, 4])
    expect(fall).toEqual([null, null, 2, null])
  })

  it('draws error bars, ridgelines, annotations and the PR-gain curve', () => {
    expect(errorBars([{ x: 1, y: 1, xErr: 0.5, yErr: 0.2 }]).extent).toEqual({ x: [0.5, 1.5], y: [0.8, 1.2] })
    expect(
      ridgeline([
        { name: 'a', values: [0, 1, 2, 3] },
        { name: 'b', values: [2, 3, 4] },
      ]).series,
    ).toHaveLength(4)
    expect(annotations({ labels: [{ at: [1, 2], text: 'x' }] }).extent).toEqual({ x: [1, 1], y: [2, 2] })
    const prg = seriesOf(
      prGainChart({
        precisionGain: [1, 0.5, 0],
        recallGain: [-0.5, 0.5, 1],
        thresholds: [0.9, 0.5, 0.1],
        isCrossing: [false, false, true],
      }),
    )
    expect(prg.at(-1)?.data).toHaveLength(2)
  })

  it('builds the native types: treemap, sunburst, Sankey, calendar, candles, pie, funnel and gauge', () => {
    const tree = [
      { name: 'a', children: [{ name: 'b', value: 1 }] },
      { name: 'c', value: 2 },
    ]
    expect(seriesOf(treemapChart(tree))[0]?.type).toBe('treemap')
    expect(seriesOf(sunburstChart(tree))[0]?.type).toBe('sunburst')
    expect(seriesOf(sankeyChart([{ source: 'x', target: 'y', value: 1 }]))[0]?.type).toBe('sankey')
    expect(seriesOf(calendarHeatmap([{ date: '2026-02-01', value: 1 }]))[0]?.type).toBe('heatmap')
    expect(
      seriesOf(candlestickChart([{ label: 'd', open: 1, high: 2, low: 0.5, close: 1.5, volume: 3 }], { average: 2 })),
    ).toHaveLength(3)
    expect(seriesOf(pieChart([{ name: 'a', value: 1 }]))[0]?.type).toBe('pie')
    expect(seriesOf(funnelChart([{ name: 'a', value: 1 }]))[0]?.type).toBe('funnel')
    expect(seriesOf(gaugeChart(5))[0]?.type).toBe('gauge')
  })
})
