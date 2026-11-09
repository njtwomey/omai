import { describe, expect, it } from 'vitest'
import { chartProjection, linearProjection, type PixelConverter } from './projection.ts'

type Axis = Record<string, unknown>
interface Chart extends PixelConverter {
  readonly calls: { to: number; from: number; option: number }
  fire(): void
  option: unknown
}

const grid = { left: 50, top: 20, width: 300, height: 200 }
const optionOf = (
  x: Axis,
  y: Axis,
  extra: Record<string, unknown> = {},
  g: Record<string, unknown> = grid,
): unknown => ({
  grid: [g],
  xAxis: [{ type: 'value', gridIndex: 0, ...x }],
  yAxis: [{ type: 'value', gridIndex: 0, ...y }],
  ...extra,
})

// A chart that counts what it is asked. Its converters map by a reference formula written here (linear or log10 on
// either axis), standing in for ECharts; `events` gives it `on`, as the real instance has.
function fakeChart(option: unknown, events: boolean, log = false): Chart {
  const calls = { to: 0, from: 0, option: 0 }
  const handlers: (() => void)[] = []
  const [x0, x1, y1] = [0, 10, 100]
  const lx = (v: number): number => (log ? Math.log10(v) : v)
  const chart = {
    calls,
    option,
    convertToPixel(_f: { gridIndex: number }, [u = 0, v = 0]: number[]): number[] {
      calls.to++
      const [a, b] = [lx(x0 || 1), lx(x1)]
      return [50 + ((lx(u || 1e-300) - a) / (b - a)) * 300, 20 + ((Math.log10(y1) - Math.log10(v)) / 2) * 200]
    },
    convertFromPixel(_f: { gridIndex: number }, [px = 0, py = 0]: number[]): number[] {
      calls.from++
      return [((px - 50) / 300) * 10, 10 ** (2 - ((py - 20) / 200) * 2)]
    },
    getOption(): unknown {
      calls.option++
      return chart.option
    },
    ...(events ? { on: (_e: string, h: () => void) => void handlers.push(h) } : {}),
    fire: () => {
      for (const h of handlers) h()
    },
  }
  return chart as Chart
}

describe('chartProjection from the option', () => {
  it('maps linear value axes with numeric limits without asking the chart, and agrees with the converter', () => {
    const chart = fakeChart(optionOf({ min: -2, max: 6 }, { min: 0, max: 5 }), true)
    const p = chartProjection(chart)
    // The reference: what a converter that is linear in both axes answers.
    const refTo = ([u, v]: [number, number]): [number, number] => [50 + ((u + 2) / 8) * 300, 20 + ((5 - v) / 5) * 200]
    for (const pt of [
      [-2, 0],
      [6, 5],
      [1.5, 3.25],
      [0, 0],
      [7, -1],
    ] as [number, number][]) {
      const [px, py] = p.toPixel(pt)
      expect(px).toBeCloseTo(refTo(pt)[0], 9)
      expect(py).toBeCloseTo(refTo(pt)[1], 9)
      const [u, v] = p.toData([px, py])
      expect(u).toBeCloseTo(pt[0], 9)
      expect(v).toBeCloseTo(pt[1], 9)
    }
    expect(p.rect).toEqual({ x: 50, y: 20, width: 300, height: 200 })
    expect(p.project(1.5, 3.25)).toEqual(p.toPixel([1.5, 3.25]))
    expect(p.unproject(200, 70)).toEqual(p.toData([200, 70]))
    expect(chart.calls.to + chart.calls.from).toBe(0)
    // Read once for all of that: the chart tells when it redraws.
    expect(chart.calls.option).toBe(1)
  })

  it('is the linear projection of the box and the ranges', () => {
    const p = chartProjection(fakeChart(optionOf({ min: 0, max: 10 }, { min: 0, max: 1 }), true))
    const l = linearProjection({ x: [0, 10], y: [0, 1], width: 350, height: 220, margins: { left: 50, top: 20 } })
    expect(p.toPixel([3, 0.25])).toEqual(l.toPixel([3, 0.25]))
    expect(p.toData([120, 90])).toEqual(l.toData([120, 90]))
  })

  it('honours inverse axes', () => {
    const chart = fakeChart(optionOf({ min: 0, max: 10, inverse: true }, { min: 0, max: 5, inverse: true }), true)
    const p = chartProjection(chart)
    expect(p.toPixel([0, 0])).toEqual([350, 20])
    expect(p.toPixel([10, 5])).toEqual([50, 220])
    expect(p.toData([50, 220])).toEqual([10, 5])
    expect(chart.calls.to + chart.calls.from).toBe(0)
  })

  it('picks the axes of its own grid, and an option written with single objects', () => {
    const option = {
      grid: [grid, { left: 0, top: 0, width: 100, height: 100 }],
      xAxis: { type: 'value', gridIndex: 1, min: 0, max: 1 },
      yAxis: [
        { type: 'value', gridIndex: 0, min: 0, max: 5 },
        { type: 'value', gridIndex: 1, min: 0, max: 2 },
      ],
    }
    const chart = fakeChart(option, true)
    expect(chartProjection(chart, 1).toPixel([1, 2])).toEqual([100, 0])
    // Grid 0 has no x axis of its own: ECharts must convert.
    chartProjection(chart, 0).toPixel([1, 2])
    expect(chart.calls.to).toBeGreaterThan(0)
  })

  it('falls back to the converters for log, category, time and auto-range axes, a zoomed or label-fitted grid', () => {
    const cases: [string, unknown][] = [
      ['log', optionOf({ min: 1, max: 100, type: 'log' }, { min: 0, max: 5 })],
      ['category', optionOf({ type: 'category' }, { min: 0, max: 5 })],
      ['time', optionOf({ type: 'time', min: 0, max: 5 }, { min: 0, max: 5 })],
      ['auto', optionOf({ min: 0, max: 5 }, {})],
      ['half', optionOf({ min: 0 }, { min: 0, max: 5 })],
      ['zoom', optionOf({ min: 0, max: 5 }, { min: 0, max: 5 }, { dataZoom: [{ type: 'inside' }] })],
      ['labels', optionOf({ min: 0, max: 5 }, { min: 0, max: 5 }, {}, { ...grid, containLabel: true })],
      [
        'percent',
        optionOf({ min: 0, max: 5 }, { min: 0, max: 5 }, {}, { left: '10%', top: 20, width: 300, height: 200 }),
      ],
    ]
    for (const [name, option] of cases) {
      const chart = fakeChart(option, true)
      const p = chartProjection(chart)
      p.toPixel([1, 2])
      p.toData([100, 100])
      expect(chart.calls.to, name).toBeGreaterThan(0)
      expect(chart.calls.from, name).toBeGreaterThan(0)
    }
  })

  it("answers a log axis with the converter's own values", () => {
    const chart = fakeChart(optionOf({ min: 1, max: 100, type: 'log' }, { min: 0, max: 1 }), true, true)
    const p = chartProjection(chart)
    const out = p.toPixel([5, 10])
    const direct = chart.convertToPixel({ gridIndex: 0 }, [5, 10]) as number[]
    const back = chart.convertFromPixel({ gridIndex: 0 }, [200, 70]) as number[]
    expect(out[0]).toBeCloseTo(direct[0] as number, 9)
    expect(out[1]).toBeCloseTo(direct[1] as number, 9)
    expect(p.toData([200, 70])[0]).toBeCloseTo(back[0] as number, 9)
    expect(p.toData([200, 70])[1]).toBeCloseTo(back[1] as number, 9)
  })

  it('reads the option again after a redraw, and tells its host only when the box or the ranges moved', () => {
    const chart = fakeChart(optionOf({ min: 0, max: 10 }, { min: 0, max: 1 }), true)
    let changes = 0
    const p = chartProjection(chart, 0, () => changes++)
    expect(p.toPixel([10, 0])).toEqual([350, 220])
    // A redraw with the same geometry (a hover, say): read again, nothing to tell.
    chart.fire()
    expect(changes).toBe(0)
    // The chart was resized and its option rebuilt: the next redraw tells, and the map follows.
    chart.option = optionOf({ min: 0, max: 10 }, { min: 0, max: 1 }, {}, { left: 50, top: 20, width: 500, height: 200 })
    expect(p.toPixel([10, 0])).toEqual([350, 220])
    chart.fire()
    expect(changes).toBe(1)
    expect(p.toPixel([10, 0])).toEqual([550, 220])
    expect(p.rect?.width).toBe(500)
    expect(chart.calls.to + chart.calls.from).toBe(0)
  })

  it('without events, asks for the option each time and rereads it when its identity changes', () => {
    const chart = fakeChart(optionOf({ min: 0, max: 10 }, { min: 0, max: 1 }), false)
    const p = chartProjection(chart)
    p.toPixel([0, 0])
    p.toPixel([1, 0])
    expect(chart.calls.option).toBe(2)
    chart.option = optionOf({ min: 0, max: 20 }, { min: 0, max: 1 })
    expect(p.toPixel([20, 0])).toEqual([350, 220])
    expect(chart.calls.to).toBe(0)
  })
})
