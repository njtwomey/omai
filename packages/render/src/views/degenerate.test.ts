import * as echarts from 'echarts'
import { describe, expect, it } from 'vitest'
import { clusteringSteps, regularisationPath } from 'whitebox-engine/exposition'
import { regularisationPathView } from './paths.ts'
import { clusteringFrames } from './steps.ts'

// Degenerate data (a converged objective of ~1e-26, a path of one penalty) still renders through ECharts.
function render(option: echarts.EChartsCoreOption): string {
  const chart = echarts.init(null, null, { renderer: 'svg', ssr: true, width: 640, height: 400 })
  chart.setOption(option)
  const svg = chart.renderToSVGString()
  chart.dispose()
  return svg
}

describe('degenerate ranges', () => {
  it('a vanishing objective and a one-point path render', () => {
    const steps = clusteringSteps({
      method: 'spectral',
      assignments: [
        [0, 1, 1],
        [0, 1, 1],
      ],
      objective: [3e-26, 1e-26],
      rows: {
        points: [
          [0, 0],
          [1, 1],
          [1.1, 1],
        ],
        features: ['a', 'b'],
      },
    })
    const frames = clusteringFrames(steps)
    expect(render(frames.option(0))).toContain('<svg')
    const path = regularisationPath({ penalties: [0.5], features: ['a'], coefficients: [[1.2]] })
    expect(render(regularisationPathView(path))).toContain('<svg')
  })
})
