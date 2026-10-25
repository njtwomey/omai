import * as echarts from 'echarts'
import { structure } from 'mdml-engine/scene'
import type { Run, TraceEvent } from 'mdml-engine/trace'
import { describe, expect, it } from 'vitest'
import { cartesian } from '../charts/plot.ts'
import { cumulativeRegret, episodeChart, episodeCurves, rolling } from './episodes.ts'
import { messageFrames, messageTimeline } from './messages.ts'
import { metricChart, metricCurves, metricNames, runsChart, smoothEma } from './metrics.ts'
import { optimisationPaths, parameterPath } from './optimisation.ts'
import { runTable } from './tables.ts'
import { histogramsOverTime, normHeatmap, parameterTrajectories } from './tensors.ts'

const summary = (norm: number) => ({
  n: 2,
  mean: 0,
  std: 1,
  min: -1,
  max: 1,
  norm,
  zeros: 0,
  histogram: { edges: [-1, 0, 1], counts: [1, 1] },
})
const events: TraceEvent[] = [
  { type: 'run', runId: 'r', name: 'train', seed: 1, version: '1', config: {} },
  { type: 'run', runId: 'c', name: 'inner', seed: null, version: '1', config: {}, parent: 'r' },
  ...Array.from({ length: 20 }, (_, t): TraceEvent[] => [
    { type: 'metric', runId: 'r', step: t, name: 'loss', value: 1 / (t + 1) },
    { type: 'metric', runId: 'r', step: t, name: 'acc', value: t / 20 },
    { type: 'tensor', runId: 'r', step: t, name: 'w', role: 'param', shape: [3], values: [t, -t, t / 2] },
    { type: 'tensor', runId: 'r', step: t, name: 'g', role: 'grad', shape: [2], summary: summary(1 + t) },
    { type: 'episode', runId: 'r', episode: t, return: t % 5, length: 10 + t, terminated: true },
    {
      type: 'message',
      runId: 'r',
      step: t,
      kind: t % 2 === 0 ? 'up' : 'down',
      from: 'a',
      to: 'b',
      payload: { value: t },
    },
  ]).flat(),
  { type: 'end', runId: 'c', status: 'completed', steps: 3, summary: { residual: 0.01 } },
  { type: 'end', runId: 'r', status: 'converged', steps: 20, summary: { loss: 0.05 } },
]
const run: Run = { schema: 'mdml.run/1', events }

// biome-ignore lint/suspicious/noExplicitAny: tests inspect untyped ECharts option trees
type Opt = any

function render(option: object): string {
  const chart = echarts.init(null, null, { renderer: 'svg', ssr: true, width: 640, height: 400 })
  chart.setOption(option as echarts.EChartsCoreOption)
  const svg = chart.renderToSVGString()
  chart.dispose()
  return svg
}

describe('trace figures', () => {
  it('draws metric curves, smoothed and overlaid', () => {
    expect(metricNames(run)).toEqual(['loss', 'acc'])
    expect(smoothEma([Number.NaN, 2], 0.5)).toEqual([Number.NaN, 2])
    expect(metricCurves(run, ['loss'], { smoothing: 0.9 }).series).toHaveLength(2)
    const o: Opt = metricChart(run, undefined, { smoothing: 0.5 })
    expect(o.legend.data).toEqual(['loss', 'acc'])
    expect(render(o)).toContain('<svg')
    expect(render(runsChart([{ name: 'a', run }], 'loss', { smoothing: 0.3 }))).toContain('<svg')
  })
  it('draws tensors over time', () => {
    expect(parameterTrajectories(run, 'w', { indices: [2] }).extent.y).toEqual([0, 9.5])
    const h = histogramsOverTime(run, 'w', { bins: 5 })
    expect(h.extent.x).toEqual([0, 19])
    expect(histogramsOverTime(run, 'g').series.length).toBeGreaterThan(0)
    expect(() => histogramsOverTime(run, 'missing')).toThrow()
    const n = normHeatmap(run, ['g', 'w'], { log: false })
    expect(n.yCategories).toEqual(['w', 'g'])
    for (const l of [h, n]) expect(render(cartesian([l]))).toContain('<svg')
  })
  it('draws messages, episodes, optimisation paths and tables', () => {
    const m = messageTimeline(run, { kind: 'up' })
    expect(m.series).toHaveLength(1)
    const tl = messageFrames(structure('g', [], []), run)
    expect(tl.frames).toHaveLength(20)
    expect(tl.frames[3]?.overlay.badges).toEqual({ b: { message: 3 } })
    expect(rolling([1, 2, 3, 4], 2).std).toEqual([0, 0.5, 0.5, 0.5])
    expect(episodeCurves([{ name: 'a', run }], { quantity: 'length' })).toHaveLength(3)
    expect(render(episodeChart(run, { window: 5 }))).toContain('<svg')
    expect(cumulativeRegret([0, 0], [1, 2])).toEqual([1, 3])
    expect(parameterPath(run, 'w', [0, 2]).at(-1)).toEqual([19, 9.5])
    expect(() => parameterPath(run, 'g')).toThrow()
    const paths = optimisationPaths(
      (x, y) => x * x + 3 * y * y,
      [
        {
          name: 'gd',
          points: [
            [1, 1],
            [0, 0],
          ],
        },
      ],
      {
        x: [-1, 1],
        y: [-1, 1],
        n: 10,
        logScale: false,
        fill: false,
      },
    )
    expect(paths).toHaveLength(2)
    expect(render(cartesian([m, ...paths]))).toContain('<svg')
    const t = runTable(run)
    expect(t.columns).toEqual(['run', 'status', 'steps', 'seed', 'loss', 'residual'])
    expect(t.rows[1]).toEqual(['  inner', 'completed', 3, null, null, 0.01])
  })
})
