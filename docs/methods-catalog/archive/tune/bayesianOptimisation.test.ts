import { describe, expect, it } from 'vitest'
import { forrester } from '../../lib/datasets/index.ts'
import { metricSeries, session } from '../../lib/trace/index.ts'
import { bayesianOptimisation } from './bayesianOptimisation.ts'

describe('bayesianOptimisation', () => {
  it('finds the minimum of the Forrester function in thirty trials, with the surrogate to show at any step', () => {
    const s = session(bayesianOptimisation, { mode: 'eager', steps: 30, config: { objective: 'forrester' } })
    const run = s.run()
    const best = metricSeries(run, 'best').values
    expect(best.length).toBe(30)
    const target = forrester()
    expect(best.at(-1) ?? Number.NaN).toBeLessThan(target.minimum + 0.1)
    const found = s.probe(30, 'model/params') as { x: number; value: number }
    expect(Math.abs(found.x - (target.minimizers[0]?.[0] ?? Number.NaN))).toBeLessThan(0.05)
    // The posterior at a step before the end: a mark per trial so far, a band on the mean, the acquisition on the right.
    const posterior = s.probe(12, 'tune/posterior') as {
      kind: string
      x: number[]
      marks: unknown[]
      series: { name: string; lower?: number[]; upper?: number[]; axis?: string; y: number[] }[]
    }
    expect(posterior.kind).toBe('curves')
    expect(posterior.marks.length).toBe(12)
    expect(posterior.series.map((x) => x.name)).toEqual(['function', 'surrogate mean', 'expected-improvement'])
    expect(posterior.series[1]?.lower?.length).toBe(posterior.x.length)
    expect(posterior.series[2]?.axis).toBe('right')
  })
})
