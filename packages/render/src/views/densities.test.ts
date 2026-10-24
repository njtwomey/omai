import * as echarts from 'echarts'
import { describe, expect, it } from 'vitest'
import { densityCurves } from 'whitebox-engine/exposition'
import { densityFrames } from './densities.ts'

type Opt = { series: { name?: string }[] }

describe('density views', () => {
  const x = [0, 0.25, 0.5, 0.75, 1]
  const record = densityCurves({
    parameter: 'p',
    x,
    curves: [
      { name: 'prior', density: [1, 1, 1, 1, 1], role: 'prior' },
      { name: 'posterior', density: [0, 0.5, 2, 0.5, 0], role: 'posterior' },
    ],
    interval: { lower: 0.25, upper: 0.75, level: 0.95 },
    marks: [{ name: 'posterior mean', value: 0.5 }],
    steps: [
      { observed: 0, density: [1, 1, 1, 1, 1], estimate: 0.5 },
      { observed: 4, density: [0, 0.5, 2, 0.5, 0], interval: [0.25, 0.75], estimate: 0.5 },
    ],
  })

  it('play the posterior from no data to all of it over the prior', () => {
    const frames = densityFrames(record)
    expect(frames.count).toBe(2)
    expect(frames.label(1)).toBe('4 observations · mean 0.5')
    const names = (frames.option(1) as Opt).series.map((s) => s.name)
    expect(names).toContain('prior')
    expect(names).toContain('posterior')
    expect(names).toContain('__interval')
  })

  it('render without steps, the interval shaded under the posterior', () => {
    const still = densityFrames({ ...record, steps: [] })
    expect(still.count).toBe(1)
    const chart = echarts.init(null, null, { renderer: 'svg', ssr: true, width: 640, height: 400 })
    chart.setOption(still.option(0) as echarts.EChartsCoreOption)
    expect(chart.renderToSVGString()).toContain('<svg')
    chart.dispose()
  })
})
