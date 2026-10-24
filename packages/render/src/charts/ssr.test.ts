import * as echarts from 'echarts'
import { describe, expect, it } from 'vitest'
import { normal } from 'whitebox-engine/dists'
import { barChart, histogram, paretoChart } from './bars.ts'
import { boxPlot, densityChart, violinPlot } from './distributions.ts'
import { band, lineChart, lines } from './lines.ts'
import { confusionMatrixChart, heatmapChart } from './matrices.ts'
import { cartesian } from './plot.ts'
import { scatterChart } from './points.ts'

// Every builder's option renders through ECharts' server-side SVG renderer without errors.
function render(option: echarts.EChartsCoreOption): string {
  const chart = echarts.init(null, null, { renderer: 'svg', ssr: true, width: 640, height: 400 })
  chart.setOption(option)
  const svg = chart.renderToSVGString()
  chart.dispose()
  return svg
}

describe('server-side rendering', () => {
  it('renders the cartesian builders', () => {
    const options = [
      lineChart(
        [
          { y: [1, 3, 2], name: 'a' },
          { y: [2, 2, 1], name: 'b', dash: 'dashed' },
        ],
        { title: 'lines' },
      ),
      cartesian([band([{ x: [0, 1, 2], lower: [0, 1, 0], upper: [1, 2, 2] }]), lines([{ y: [0.5, 1.5, 1] }])]),
      scatterChart([{ x: [0, 1, 2], y: [1, 0, 1], groups: [0, 1, 1] }]),
      barChart(['a', 'b'], [{ values: [1, 2], errors: [0.1, 0.2] }]),
      cartesian([histogram([0, 1, 1, 2, 3, 3, 3])]),
      paretoChart(['a', 'b', 'c'], [3, 1, 2]),
      densityChart([{ name: 'n', dist: normal(0, 1) }]),
      cartesian([boxPlot([{ name: 'a', values: [1, 2, 3, 9] }])]),
      cartesian([violinPlot([{ name: 'a', values: [1, 2, 2, 3] }])]),
      heatmapChart(
        [
          [0, 1],
          [2, 3],
        ],
        { label: 'z' },
      ),
      confusionMatrixChart(
        [
          [5, 1],
          [0, 4],
        ],
        ['x', 'y'],
      ),
    ]
    for (const o of options) expect(render(o)).toContain('<svg')
  })
})
