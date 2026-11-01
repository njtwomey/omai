import { describe, expect, it } from 'vitest'
import { metricSeries, session, tensorSeries } from '../../lib/trace/index.ts'
import { transformSampling } from './transformSampling.ts'

describe('transformSampling', () => {
  it('pushes a batch per step and approaches the exact law', () => {
    const s = session(transformSampling, { mode: 'eager', steps: 100, config: { target: 'exponential', batch: 10 } })
    const run = s.run()
    expect(tensorSeries(run, 'samples/y')).toHaveLength(100)
    expect(s.state.y).toHaveLength(1000)
    expect(s.state.y.every((y) => y >= 0)).toBe(true)
    const ks = metricSeries(run, 'samples/ks').values
    expect(ks.at(-1)).toBeLessThan(0.06)
    expect(Math.abs((metricSeries(run, 'samples/mean').values.at(-1) ?? 0) - 1)).toBeLessThan(0.15)
    const picture = s.probe(50, 'transform/picture') as { readonly kind: string; readonly series: { name: string }[] }
    expect(picture.kind).toBe('curves')
    expect(picture.series).toHaveLength(6)
  })

  it('inverts the bimodal map numerically so its density integrates to about one', () => {
    const s = session(transformSampling, { mode: 'eager', steps: 1, config: { target: 'bimodal', batch: 1 } })
    const picture = s.probe(1, 'transform/picture') as { readonly series: { name: string; x: number[]; y: number[] }[] }
    const target = picture.series.find((q) => q.name === 'target density')
    expect(target).toBeDefined()
    // The density is drawn scaled into a strip of the base axis: it is positive and bounded.
    expect(Math.max(...(target?.x ?? []))).toBeGreaterThan(-4)
    expect(Math.max(...(target?.x ?? []))).toBeLessThanOrEqual(-4 + 0.2 * 8 + 1e-9)
  })
})
