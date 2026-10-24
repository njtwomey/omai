import { describe, expect, it } from 'vitest'
import { choice, dict, real } from 'whitebox-engine/space'
import { functionObjective, medianPruner, randomSampler, tune } from 'whitebox-engine/tune'
import { contourChart, historyChart, historyLayers, importanceChart, intermediateChart, sliceChart } from './tuning.ts'

type Opt = Record<string, unknown> & { readonly series: readonly Record<string, unknown>[] }
const asOption = (o: unknown): Opt => o as Opt

const space = dict({ lr: real(1e-4, 1, { log: true }), dropout: real(0, 0.6), act: choice(['relu', 'tanh']) })
const loss = (p: { lr: number; dropout: number; act: string }): number =>
  (Math.log10(p.lr) + 2) ** 2 + 3 * (p.dropout - 0.2) ** 2 + (p.act === 'relu' ? 0 : 0.3)
const study = tune(
  functionObjective((p) => loss(p as never)),
  space,
  { sampler: randomSampler(), nTrials: 24, seed: 3 },
).study

describe('study charts', () => {
  it('draws the history as trial points and a falling running best', () => {
    const [trials, best] = historyLayers(study)
    expect(trials?.series[0]?.['data' as never]).toHaveLength(24)
    const ys = ((best?.series[0] ?? { data: [] }) as { data: number[][] }).data.map((p) => p[1] ?? 0)
    for (let i = 1; i < ys.length; i++) expect(ys[i]).toBeLessThanOrEqual(ys[i - 1] ?? Number.POSITIVE_INFINITY)
    expect(asOption(historyChart(study, { log: true }))['yAxis']).toMatchObject([{ type: 'log' }])
  })

  it('ranks importances with the largest bar on top', () => {
    const o = asOption(importanceChart({ a: 0.1, b: 0.6, c: 0.3 }))
    expect((o['yAxis'] as { data: string[] }[])[0]?.data).toEqual(['a', 'c', 'b'])
    expect(asOption(importanceChart({ a: 0.1, b: 0.6, c: 0.3 }, { top: 2 }))['yAxis']).toMatchObject([
      { data: ['c', 'b'] },
    ])
  })

  it('slices every parameter on its own axis kind, sharing the objective axis', () => {
    const o = asOption(sliceChart(study))
    const x = o['xAxis'] as { type: string; name: string }[]
    expect(x.map((a) => [a.name, a.type])).toEqual([
      ['lr', 'log'],
      ['dropout', 'value'],
      ['act', 'value'],
    ])
    const y = o['yAxis'] as { min: number; max: number }[]
    expect(new Set(y.map((a) => `${a.min},${a.max}`)).size).toBe(1)
    // A shared log axis is ranged on the logarithms and ends on 1-2-5 values.
    const logY = (asOption(sliceChart(study, { log: true }))['yAxis'] as { min: number; max: number }[])[0]
    const mantissa = (v: number): number => Math.round((v / 10 ** Math.floor(Math.log10(v) + 1e-9)) * 1e6) / 1e6
    expect([1, 2, 5]).toContain(mantissa(logY?.min ?? 0))
    expect([1, 2, 5, 10]).toContain(mantissa(logY?.max ?? 0))
  })

  it('draws a surrogate contour under the trials, on log-10 axes for log parameters', () => {
    const o = asOption(contourChart(study, 'lr', 'dropout', { gridSize: 8 }))
    expect(o.series.at(-1)?.['type']).toBe('scatter')
    const x = (o['xAxis'] as { min: number; max: number }[])[0]
    expect(x?.min).toBeCloseTo(-4, 12)
    expect(x?.max).toBeCloseTo(0, 12)
  })

  it('draws reported curves, pruned trials grey', () => {
    const curves = tune(
      functionObjective((p, trial) => {
        const final = loss(p as never)
        for (let s = 0; s < 8; s++) if (trial.report(s, final + 2 * Math.exp(-s / 2))) return final
        return final
      }),
      space,
      { sampler: randomSampler(), pruner: medianPruner({ nStartupTrials: 3 }), nTrials: 12, seed: 2 },
    ).study
    const o = asOption(intermediateChart(curves))
    const names = o.series.map((s) => s['name'])
    expect(names).toContain('complete')
    expect(o.series.length).toBe(12)
  })
})
