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

// Regression for ECharts 6's default `grid.outerBoundsMode: 'auto'`, which moves the plotting box inside the box the
// option names (to keep the axis labels in the figure). Real instances (server-side) draw each builder's option and the
// box ECharts uses is compared with the numbers the builder wrote.
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

describe('the plotting box ECharts draws is the box the builders write', () => {
  const cases: [string, echarts.EChartsCoreOption, number, number][] = [
    [
      'cartesian on a slide figure',
      cartesian([lines([sine])], {
        figure: slide,
        legend: false,
        tooltip: false,
        x: { label: 'x', limits: [0, TAU] },
        y: { label: 'y', limits: [-2, 2] },
      }),
      slide.width,
      slide.height,
    ],
    ['lineChart', lineChart([sine], { title: 'sine' }), 640, 400],
    ['scatterChart', scatterChart([{ x: [0, 1, 2], y: [1, 0, 1], groups: [0, 1, 1] }]), 640, 400],
    ['barChart', barChart(['a', 'b'], [{ values: [1, 2] }], { x: { label: 'category' } }), 640, 400],
    [
      'subplots',
      subplots(withMarginals(), {
        main: {
          layers: [points([{ x: [0, 1, 2, 3], y: [1, 3, 2, 4] }])],
          options: { x: { label: 'x' }, y: { label: 'y' } },
        },
        top: { layers: [histogram([0, 1, 2, 3], { bins: 2 })] },
        right: { layers: [histogram([1, 3, 2, 4], { bins: 2 })] },
      }),
      640,
      400,
    ],
  ]

  for (const [name, option, width, height] of cases) {
    it(`${name}: every grid's rect is the option's numbers`, () => {
      const grids = (option as unknown as { grid: Grid[] }).grid
      const chart = draw(option, width, height)
      const drawn = rects(chart)
      chart.dispose()
      expect(drawn).toHaveLength(grids.length)
      for (const [i, g] of grids.entries()) {
        expect(g).toMatchObject({ outerBoundsMode: 'none', containLabel: false })
        const r = drawn[i] as { x: number; y: number; width: number; height: number }
        const round = (v: number): number => Number(v.toFixed(6))
        expect([r.x, r.y, r.width, r.height].map(round)).toEqual([g.left, g.top, g.width, g.height].map(round))
      }
    })
  }

  it('chartProjection agrees with ECharts convertToPixel and convertFromPixel on a cartesian chart', () => {
    const option = cartesian([lines([sine])], {
      figure: slide,
      legend: false,
      tooltip: false,
      x: { label: 'x', limits: [0, TAU] },
      y: { label: 'y', limits: [-2, 2] },
    })
    const chart = draw(option, slide.width, slide.height)
    // The analytic path: ECharts is asked for nothing.
    tally.asked = 0
    const counting = {
      getOption: () => chart.getOption(),
      convertToPixel: (f: { gridIndex: number }, v: number[]) => counted(() => chart.convertToPixel(f, v)),
      convertFromPixel: (f: { gridIndex: number }, v: number[]) => counted(() => chart.convertFromPixel(f, v)),
    }
    const p = chartProjection(counting)
    for (const [a, b] of [
      [0, 0],
      [TAU, 0],
      [0, -2],
      [TAU, 2],
      [Math.PI, 1],
    ] as [number, number][]) {
      const [px, py] = p.project(a, b)
      const direct = chart.convertToPixel({ gridIndex: 0 }, [a, b]) as number[]
      expect(Math.abs(px - (direct[0] as number))).toBeLessThan(1e-6)
      expect(Math.abs(py - (direct[1] as number))).toBeLessThan(1e-6)
      const [u, v] = p.unproject(px, py)
      expect(u).toBeCloseTo(a, 6)
      expect(v).toBeCloseTo(b, 6)
    }
    expect(tally.asked).toBe(0)
    chart.dispose()
  })

  it('a hand-written grid gets ECharts auto bounds, and chartProjection asks ECharts then', () => {
    const option = {
      grid: [{ left: 72, top: 33, width: 1444, height: 773 }],
      xAxis: [{ type: 'value', min: 0, max: TAU }],
      yAxis: [{ type: 'value', min: -2, max: 2 }],
      series: [{ type: 'line', data: xs.map((x) => [x, Math.sin(x)]) }],
    }
    const chart = draw(option, slide.width, slide.height)
    tally.asked = 0
    const p = chartProjection({
      getOption: () => chart.getOption(),
      convertToPixel: (f, v) => counted(() => chart.convertToPixel(f, v)),
      convertFromPixel: (f, v) => counted(() => chart.convertFromPixel(f, v)),
    })
    const [px, py] = p.project(0, 0)
    const direct = chart.convertToPixel({ gridIndex: 0 }, [0, 0]) as number[]
    expect(tally.asked).toBeGreaterThan(0)
    expect([px, py]).toEqual(direct)
    chart.dispose()
  })
})
