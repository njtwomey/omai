import type { DecisionSurface } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { metricSeries, session, tensorSeries } from '../../lib/trace/index.ts'
import { slicSuperpixels } from './slicSuperpixels.ts'

describe('slicSuperpixels', () => {
  it('assigns every pixel to a superpixel, with the painted scene as a probe', () => {
    const s = session(slicSuperpixels, { mode: 'eager', steps: 8, config: { size: 32, segments: 16 } })
    const run = s.run()
    const labels = tensorSeries(run, 'segments/labels')
    expect(labels.length).toBeGreaterThan(0)
    expect(labels[0]?.shape).toEqual([32, 32])
    expect(labels.at(-1)?.values?.every((v) => v >= 0 && v < 16)).toBe(true)
    const leak = metricSeries(run, 'segments/leak').values
    expect(leak.at(-1)).toBeLessThan(0.15)
    expect(leak.at(-1)).toBeLessThanOrEqual(leak[0] ?? 0)
    const p = s.probe(2, 'segments/picture') as DecisionSurface
    expect(p.values?.length).toBe(32)
    expect(p.rows?.points.length).toBe(16)
  })
})
