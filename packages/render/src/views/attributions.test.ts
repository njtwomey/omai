import * as echarts from 'echarts'
import { featureAttributions, importances } from 'omai-engine/exposition'
import { describe, expect, it } from 'vitest'
import {
  attributionBarView,
  attributionFrames,
  attributionOrder,
  beeswarmView,
  dependenceScatterView,
  forceSegments,
  forceView,
  importanceView,
  waterfallView,
} from './attributions.ts'
import { stillOf, viewFor } from './registry.ts'

function render(option: echarts.EChartsCoreOption): string {
  const chart = echarts.init(null, null, { renderer: 'svg', ssr: true, width: 640, height: 400 })
  chart.setOption(option)
  const svg = chart.renderToSVGString()
  chart.dispose()
  return svg
}

const features = ['age', 'bmi', 'bp', 's5']
const rows: [number, number, number, number][] = Array.from({ length: 30 }, (_, i) => [
  i % 7,
  (i * 3) % 11,
  i % 5,
  (i * 7) % 13,
])
const one = featureAttributions({
  method: 'tree',
  features,
  output: 'prediction',
  baseValue: 150,
  values: [[2, 30, -8, 12]],
  data: [[50, 0.06, 0.02, 0.03]],
  predictions: [186],
  convergence: {
    row: 0,
    samples: [10, 20, 30],
    estimates: [
      [5, 20, -2, 13],
      [3, 28, -6, 11],
      [2, 30, -8, 12],
    ],
  },
})
const many = featureAttributions({
  method: 'tree',
  features,
  output: 'prediction',
  baseValue: 150,
  values: rows.map((r) => [r[0] - 3, 4 * (r[1] - 5), -(r[2] - 2), r[3] - 6]),
  data: rows,
  predictions: rows.map((r) => 150 + r[0] - 3 + 4 * (r[1] - 5) - (r[2] - 2) + r[3] - 6),
  partners: [1, 0, 1, 1],
})

describe('attribution views', () => {
  it('orders features by mean |attribution|', () => {
    expect(attributionOrder(many)[0]).toBe(1)
  })

  it('draws a waterfall from the base value to the prediction, the rest summed', () => {
    const option = waterfallView(one, { maxFeatures: 3 }) as { yAxis: { data: string[] }[] }
    const labels = option.yAxis[0]?.data ?? []
    expect(labels[0]).toBe('2 other features')
    expect(labels.at(-1)).toMatch(/^bmi = /)
    expect(render(option as echarts.EChartsCoreOption)).toContain('f(x) = 186')
  })

  it('plays the sampled estimate settling, one frame per record', () => {
    const frames = attributionFrames(one)
    expect(frames.count).toBe(3)
    expect(frames.label(1)).toBe('20 samples')
    expect(render(frames.option(0))).toContain('<svg')
  })

  it('draws the beeswarm, the mean bar and a dependence scatter', () => {
    const swarm = beeswarmView(many) as { series: { data: number[][] }[]; visualMap: { text: string[] }[] }
    const dots = swarm.series.find((s) => (s as { name?: string }).name === '__swarm')
    expect(dots?.data).toHaveLength(30 * 4)
    expect(swarm.visualMap[0]?.text).toEqual(['high', 'low'])
    expect(render(swarm as echarts.EChartsCoreOption)).toContain('<svg')
    expect(render(attributionBarView(many))).toContain('bmi')
    const dep = dependenceScatterView(many, { feature: 1 }) as { visualMap: { text: string[] }[] }
    // Coloured by the partner (age) with its own bar.
    expect(dep.visualMap[0]?.text?.[0]).toMatch(/^age/)
    expect(render(dependenceScatterView(many, { feature: 0, colourBy: -1 }))).toContain('<svg')
  })

  it('draws importances with their whiskers and repeats, and both kinds through the registry', () => {
    const imp = importances({
      method: 'permutation',
      features,
      mean: [0.01, 0.2, 0.05, 0.1],
      spread: [0.01, 0.03, 0.02, 0.01],
      repeats: [
        [0, 0.02],
        [0.17, 0.23],
        [0.03, 0.07],
        [0.09, 0.11],
      ],
      metric: 'R²',
      direction: 'higher',
      baseline: 0.5,
    })
    expect(render(importanceView(imp))).toContain('decrease in R²')
    expect(viewFor('importances').component).toBe('ImportanceView')
    expect(viewFor('featureAttributions').component).toBe('AttributionView')
    expect(render(stillOf(many) as echarts.EChartsCoreOption)).toContain('<svg')
    expect(render(stillOf(one) as echarts.EChartsCoreOption)).toContain('<svg')
  })
})

describe('force plot', () => {
  it('stacks raising features up to the prediction and lowering ones beyond it', () => {
    const r = featureAttributions({
      method: 'exact',
      features: ['a', 'b', 'c'],
      output: 'y',
      baseValue: 1,
      values: [[2, -0.5, 0.25]],
      data: [[1, 2, 3]],
      predictions: [2.75],
    })
    const { segments, end } = forceSegments(r, 0)
    expect(end).toBeCloseTo(2.75, 12)
    expect(segments.filter((s) => s.up).map((s) => s.end)).toEqual([2.75, 0.75])
    expect(segments.find((s) => !s.up)?.start).toBeCloseTo(2.75, 12)
    expect((forceView(r) as { series: unknown[] }).series.length).toBeGreaterThan(0)
  })
})
