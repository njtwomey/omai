import * as echarts from 'echarts'
import { normal } from 'mdml-engine/dists'
import { describe, expect, it } from 'vitest'
import { withMarginals } from '../figure/grid.ts'
import { barChart, histogram, paretoChart } from './bars.ts'
import { subplots } from './compose.ts'
import { boxPlot, densityChart, violinPlot } from './distributions.ts'
import { intervalChart, rocChart } from './evaluation.ts'
import { contours, decisionRegions, fieldArrows, streamlines, vectorField } from './fields.ts'
import { fn, referenceLines, riemann } from './functions.ts'
import { band, lineChart, lines } from './lines.ts'
import { confusionMatrixChart, heatmapChart } from './matrices.ts'
import { cartesian } from './plot.ts'
import { points, scatterChart } from './points.ts'

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
      cartesian([
        decisionRegions([
          [
            [0.9, 0.2],
            [0.6, 0.1],
          ],
          [
            [0.1, 0.8],
            [0.4, 0.9],
          ],
        ]),
        contours(
          [
            [0, 1],
            [1, 2],
          ],
          { levels: [0.5, 1.5] },
        ),
      ]),
      cartesian([
        vectorField(fieldArrows((x, y) => [-y, x], { x: [-1, 1], y: [-1, 1], n: 5 })),
        streamlines([
          [
            [0, 0],
            [1, 1],
            [2, 1],
          ],
        ]),
      ]),
      cartesian([
        fn(Math.tan, [-3, 3]),
        riemann(Math.sin, [0, 3], 6),
        referenceLines({ x: [1.5], y: [{ value: 0, label: 'zero' }] }),
      ]),
      rocChart({ fpr: [0, 0.2, 1], tpr: [0, 0.7, 1], thresholds: [Infinity, 0.5, 0] }),
      intervalChart([{ name: 'a', value: { value: 1, interval: [0.5, 1.5] } }], { reference: 0 }),
      subplots(withMarginals(), {
        main: { layers: [points([{ x: [0, 1, 2], y: [1, 2, 0] }])] },
        top: { layers: [histogram([0, 1, 2], { bins: 2 })] },
        right: { layers: [histogram([1, 2, 0], { bins: 2, horizontal: true })] },
      }),
    ]
    for (const o of options) expect(render(o)).toContain('<svg')
  })
})
