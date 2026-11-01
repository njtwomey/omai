import type { DecisionSurface } from 'mdml-engine/exposition'
import { describe, expect, it } from 'vitest'
import { metricSeries, session, tensorSeries } from '../../lib/trace/index.ts'
import { kltTracking } from './kltTracking.ts'

describe('kltTracking', () => {
  it('follows the corners of a drifting scene to within a pixel or two', () => {
    const s = session(kltTracking, { mode: 'eager', steps: 100, config: { size: 48, frames: 12, amplitude: 3 } })
    const run = s.run()
    const pts = tensorSeries(run, 'track/points')
    // One step per frame pair, and the run ends with the sequence.
    expect(pts).toHaveLength(11)
    const alive = metricSeries(run, 'track/alive').values
    expect(alive.at(-1)).toBeGreaterThan(0)
    const error = metricSeries(run, 'track/error').values
    expect(error.at(-1)).toBeLessThan(2)
    const p = s.probe(3, 'track/picture') as DecisionSurface
    expect(p.values?.length).toBe(48)
    expect(p.rows?.classes).toEqual(['lost', 'tracked'])
  })
})
