import { describe, expect, it } from 'vitest'
import { metricSeries, session, tensorSeries } from '../../lib/trace/index.ts'
import { diffusionPaths } from './diffusionPaths.ts'

describe('diffusionPaths', () => {
  it('advances a bundle whose mean follows the exact one, with its law as a probe', () => {
    const s = session(diffusionPaths, { mode: 'eager', steps: 200, config: { paths: 64, dt: 0.01 } })
    const run = s.run()
    const x = tensorSeries(run, 'paths/x')
    expect(x).toHaveLength(200)
    expect(x[0]?.shape).toEqual([64])
    const mean = metricSeries(run, 'paths/mean').values
    const exact = metricSeries(run, 'paths/exactMean').values
    // Ornstein-Uhlenbeck from 2 towards 0: the bundle's mean decays with the exact mean.
    expect(Math.abs((mean.at(-1) ?? 0) - (exact.at(-1) ?? 0))).toBeLessThan(0.3)
    expect(exact.at(-1)).toBeLessThan(0.5)
    const law = s.probe(100, 'paths/distribution') as { readonly series: { name: string }[] }
    expect(law.series.map((q) => q.name)).toEqual(['the bundle', 'the exact law at t = 1.00'])
  })

  it('keeps the square-root process non-negative', () => {
    const s = session(diffusionPaths, { mode: 'eager', steps: 300, config: { process: 'cir', sigma: 2, x0: 0.1 } })
    expect(s.state.x.every((v) => v >= 0)).toBe(true)
  })
})
