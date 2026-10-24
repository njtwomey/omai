import * as echarts from 'echarts'
import { describe, expect, it } from 'vitest'
import { confusionMatrix, valueGrids } from 'whitebox-engine/exposition'
import { accuracyOf, cellsLayout, confusionFrames, valueGridFrames } from './cells.ts'

function render(option: unknown): string {
  const chart = echarts.init(null, null, { renderer: 'svg', ssr: true, width: 640, height: 480 })
  chart.setOption(option as echarts.EChartsCoreOption)
  const svg = chart.renderToSVGString()
  chart.dispose()
  return svg
}

describe('value grid views', () => {
  const map = (k: number) => [
    [k, 0, -k],
    [0, k, 0],
  ]
  const record = valueGrids({
    grids: [1, 2, 3, 4, 5].map((k) => ({ name: `filter ${k}`, values: map(k) })),
    diverging: true,
    steps: [0, 1].map((t) => ({ iteration: t, grids: [1, 2, 3, 4, 5].map((k) => map(k * (t + 1))) })),
    stepName: 'epoch',
  })

  it('lay grids out in small multiples and play the steps on one colour range', () => {
    expect(cellsLayout(record)).toEqual({ columns: 3, rows: 2 })
    const frames = valueGridFrames(record)
    expect(frames.count).toBe(2)
    expect(frames.label(1)).toBe('epoch 1')
    const opt = frames.option(0) as { grid: unknown[]; visualMap: { min: number; max: number }[] }
    expect(opt.grid).toHaveLength(5)
    // One shared, symmetric bar over every grid and step.
    expect(opt.visualMap).toHaveLength(1)
    expect([opt.visualMap[0]?.min, opt.visualMap[0]?.max]).toEqual([-10, 10])
    expect(render(opt)).toContain('<svg')
  })

  it('draw one grid with named axes', () => {
    const one = valueGrids({ grids: [{ name: 'hidden', values: map(1) }], axes: ['unit', 'position'] })
    const opt = valueGridFrames(one).option(0) as { yAxis: { name?: string }[] }
    expect(opt.yAxis[0]?.name).toBe('unit')
    expect(render(opt)).toContain('<svg')
  })
})

describe('confusion matrix views', () => {
  it('count the diagonal and play the steps', () => {
    expect(
      accuracyOf([
        [3, 1],
        [0, 4],
      ]),
    ).toBe(0.875)
    const r = confusionMatrix({
      classes: ['a', 'b'],
      counts: [
        [3, 1],
        [0, 4],
      ],
      split: 'validation',
      steps: [
        {
          iteration: 1,
          counts: [
            [2, 2],
            [2, 2],
          ],
        },
        {
          iteration: 2,
          counts: [
            [3, 1],
            [0, 4],
          ],
        },
      ],
      stepName: 'epoch',
    })
    const frames = confusionFrames(r)
    expect(frames.label(0)).toBe('epoch 1 · accuracy 50.0%')
    expect(render(frames.option(1))).toContain('<svg')
    expect(render(confusionFrames(r, { normalise: true }).option(1))).toContain('<svg')
  })
})
