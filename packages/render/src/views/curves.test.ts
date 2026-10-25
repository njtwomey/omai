import * as echarts from 'echarts'
import { curveSet } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { curveSetFrames, curveSetView, curveStyles } from './curves.ts'

type Opt = { series: { name?: string; yAxisIndex?: number }[]; yAxis: { name?: string }[] }
const o = (x: unknown): Opt => x as Opt

function render(option: unknown): string {
  const chart = echarts.init(null, null, { renderer: 'svg', ssr: true, width: 640, height: 400 })
  chart.setOption(option as echarts.EChartsCoreOption)
  const svg = chart.renderToSVGString()
  chart.dispose()
  return svg
}

const race = curveSet({
  x: [1, 2, 3, 4],
  xLabel: 'epoch',
  yLabel: 'loss',
  y2Label: 'accuracy',
  series: [
    { name: 'sgd: training', y: [1, 0.7, 0.5, 0.4], group: 'sgd', spread: [0.1, 0.1, 0.1, 0.1] },
    { name: 'sgd: validation', y: [1.1, 0.8, 0.6, 0.55], group: 'sgd' },
    { name: 'adam', y: [0.8, 0.4, 0.3, 0.2], group: 'adam' },
    { name: 'accuracy', y: [0.5, 0.7, 0.8, 0.9], axis: 'right' },
  ],
  marks: [{ x: 2, label: 'drop' }],
})

describe('curve sets', () => {
  it('colour by group and dash within one', () => {
    const styles = curveStyles(race.series)
    expect(styles[0]?.colour).toBe(styles[1]?.colour)
    expect([styles[0]?.dash, styles[1]?.dash]).toEqual(['solid', 'dashed'])
    expect(styles[2]?.colour).not.toBe(styles[0]?.colour)
  })

  it('draw every series, bands and the right-hand axis, and render', () => {
    const opt = o(curveSetView(race))
    expect(opt.yAxis.map((a) => a.name)).toEqual(['loss', 'accuracy'])
    expect(opt.series.find((s) => s.name === 'accuracy')?.yAxisIndex).toBe(1)
    expect(opt.series.some((s) => s.name === '__band0')).toBe(true)
    expect(render(opt)).toContain('<svg')
  })

  it('reveal the curves along x, the last frame the whole record', () => {
    const frames = curveSetFrames(race)
    expect(frames.count).toBe(4)
    expect(frames.label(1)).toBe('epoch 2')
    expect(o(frames.option(0)).series.filter((s) => s.name?.startsWith('__now'))).toHaveLength(4)
    expect(o(frames.option(3)).series.some((s) => s.name?.startsWith('__now'))).toBe(false)
  })
})
