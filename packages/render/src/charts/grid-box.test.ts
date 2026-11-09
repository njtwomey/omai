import * as echarts from 'echarts'
import { describe, expect, it } from 'vitest'
import { figure } from '../figure/figure.ts'
import { withMarginals } from '../figure/grid.ts'
import { chartProjection } from '../react/interaction/projection.ts'
import { barChart, histogram } from './bars.ts'
import { subplots } from './compose.ts'
import { lineChart, lines } from './lines.ts'
import { cartesian } from './plot.ts'
import { points, scatterChart } from './points.ts'

// ECharts 6's default `grid.outerBoundsMode: 'auto'` moves the plotting box inside the box the option names (to keep the
// axis labels and names in the figure), so the option's numbers are not where the axes are drawn. Real instances
// (server-side) draw the option and `chartProjection` is compared with where ECharts says the axes are.
const TAU = 2 * Math.PI
const slide = figure({
  medium: 'slide',
  width: 1544,
  aspect: 16 / 9,
  fontSize: 22,
  padding: { top: 33, right: 28, bottom: 55, left: 72 },
})
const xs = Array.from({ length: 50 }, (_, i) => (i / 49) * TAU)
const sine = { x: xs, y: xs.map(Math.sin), name: 'sine' }

type Grid = { left: number; top: number; width: number; height: number }

// Runs an ECharts call and counts it.
const tally = { asked: 0 }
function counted<T>(call: () => T): T {
  tally.asked++
  return call()
}

function draw(option: echarts.EChartsCoreOption, width: number, height: number): echarts.ECharts {
  const chart = echarts.init(null, null, { renderer: 'svg', ssr: true, width, height })
  chart.setOption(option)
  return chart
}

// The rect of every grid, as ECharts lays it out.
function rects(chart: echarts.ECharts): { x: number; y: number; width: number; height: number }[] {
  const model = (chart as unknown as { getModel(): { getComponent(t: string, i: number): unknown } }).getModel()
  const out: { x: number; y: number; width: number; height: number }[] = []
  for (let i = 0; ; i++) {
    const c = model.getComponent('grid', i) as { coordinateSystem?: { getRect(): never } } | undefined
    if (c === undefined) return out
    out.push(c.coordinateSystem?.getRect() as never)
  }
}

const TICKS: [number, number][] = [
  [0, 0],
  [TAU, 0],
  [0, -2],
  [TAU, 2],
  [Math.PI, 1],
]

describe('chartProjection against where ECharts draws the axes', () => {
  const option = cartesian([lines([sine])], {
    figure: slide,
    legend: false,
    tooltip: false,
    x: { label: 'x', limits: [0, TAU] },
    y: { label: 'y', limits: [-2, 2] },
  })

  it('the cartesian builder leaves the box to ECharts, which moves it inside the one named for the axis labels', () => {
    const chart = draw(option, slide.width, slide.height)
    const [drawn] = rects(chart)
    const grid = (option as unknown as { grid: Grid | Grid[] }).grid
    const named = (Array.isArray(grid) ? grid[0] : grid) as Grid
    chart.dispose()
    expect(drawn?.x).toBeGreaterThan(named.left)
    expect(drawn?.height).toBeLessThan(named.height)
  })

  it('project, unproject and rect are where the axes are drawn (within 1e-6), whatever the grid mode did', () => {
    const chart = draw(option, slide.width, slide.height)
    const p = chartProjection({
      getOption: () => chart.getOption(),
      convertToPixel: (f, v) => chart.convertToPixel(f, v),
      convertFromPixel: (f, v) => chart.convertFromPixel(f, v),
      on: () => undefined,
    })
    for (const [a, b] of TICKS) {
      const [px, py] = p.project(a, b)
      const direct = chart.convertToPixel({ gridIndex: 0 }, [a, b]) as number[]
      expect(Math.abs(px - (direct[0] as number))).toBeLessThan(1e-6)
      expect(Math.abs(py - (direct[1] as number))).toBeLessThan(1e-6)
      const [u, v] = p.unproject(px, py)
      expect(u).toBeCloseTo(a, 6)
      expect(v).toBeCloseTo(b, 6)
    }
    const [drawn] = rects(chart)
    expect(p.rect?.x).toBeCloseTo(drawn?.x ?? Number.NaN, 6)
    expect(p.rect?.y).toBeCloseTo(drawn?.y ?? Number.NaN, 6)
    expect(p.rect?.width).toBeCloseTo(drawn?.width ?? Number.NaN, 6)
    expect(p.rect?.height).toBeCloseTo(drawn?.height ?? Number.NaN, 6)
    chart.dispose()
  })

  it('asks ECharts only once per redraw (the two corners), so a drag makes no call and logs nothing', () => {
    const chart = draw(option, slide.width, slide.height)
    tally.asked = 0
    const p = chartProjection({
      getOption: () => chart.getOption(),
      convertToPixel: (f, v) => counted(() => chart.convertToPixel(f, v)),
      convertFromPixel: (f, v) => counted(() => chart.convertFromPixel(f, v)),
      on: () => undefined,
    })
    p.project(1, 0)
    const learnt = tally.asked
    expect(learnt).toBe(2)
    for (let i = 0; i < 50; i++) {
      p.project(i / 10, 0)
      p.unproject(100 + i, 200)
    }
    expect(tally.asked).toBe(learnt)
    chart.dispose()
  })

  it('a grid pinned with outerBoundsMode none is computed from the option alone: ECharts is asked for nothing', () => {
    const pinned = {
      grid: [{ left: 72, top: 33, width: 1444, height: 773, outerBoundsMode: 'none' }],
      xAxis: [{ type: 'value', min: 0, max: TAU }],
      yAxis: [{ type: 'value', min: -2, max: 2 }],
      series: [{ type: 'line', data: xs.map((x) => [x, Math.sin(x)]) }],
    }
    const chart = draw(pinned, slide.width, slide.height)
    tally.asked = 0
    const p = chartProjection({
      getOption: () => chart.getOption(),
      convertToPixel: (f, v) => counted(() => chart.convertToPixel(f, v)),
      convertFromPixel: (f, v) => counted(() => chart.convertFromPixel(f, v)),
      on: () => undefined,
    })
    for (const [a, b] of TICKS) {
      const [px, py] = p.project(a, b)
      const direct = chart.convertToPixel({ gridIndex: 0 }, [a, b]) as number[]
      expect(Math.abs(px - (direct[0] as number))).toBeLessThan(1e-6)
      expect(Math.abs(py - (direct[1] as number))).toBeLessThan(1e-6)
    }
    expect(tally.asked).toBe(0)
    chart.dispose()
  })

  it('a chart without events is asked on every call, as in 0.1.9 (the converters, not a stored map)', () => {
    const chart = draw(option, slide.width, slide.height)
    tally.asked = 0
    const p = chartProjection({
      convertToPixel: (f, v) => counted(() => chart.convertToPixel(f, v)),
      convertFromPixel: (f, v) => counted(() => chart.convertFromPixel(f, v)),
    })
    const [px, py] = p.project(1, 0)
    const direct = chart.convertToPixel({ gridIndex: 0 }, [1, 0]) as number[]
    expect(tally.asked).toBeGreaterThan(0)
    expect([px, py]).toEqual(direct)
    chart.dispose()
  })
})
