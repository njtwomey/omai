import { describe, expect, it } from 'vitest'
import { toFlat } from '../../lib/tensor/index.ts'
import { session, tensorSeries } from '../../lib/trace/index.ts'
import { kalmanTracking } from './kalmanTracking.ts'

const mean = (xs: readonly number[]): number => xs.reduce((a, v) => a + v, 0) / xs.length

describe('kalmanTracking', () => {
  it('filters 300 readings, beating the raw ones, and seeks to a past step', () => {
    const s = session(kalmanTracking, { mode: 'eager', steps: 300, seed: 3 })
    expect(s.status).toBe('completed')
    const run = s.run()
    const estimates = tensorSeries(run, 'estimate')
    const truth = tensorSeries(run, 'truth')
    const observed = tensorSeries(run, 'observation')
    expect(estimates.length).toBe(300)
    expect(tensorSeries(run, 'covariance')[0]?.shape).toEqual([2, 2])
    const dist = (a: readonly number[] | undefined, b: readonly number[] | undefined): number =>
      Math.hypot((a?.[0] ?? 0) - (b?.[0] ?? 0), (a?.[1] ?? 0) - (b?.[1] ?? 0))
    const late = estimates.slice(100).map((_, i) => i + 100)
    const raw = mean(late.map((t) => dist(observed[t]?.values, truth[t]?.values)))
    const filtered = mean(late.map((t) => dist(estimates[t]?.values, truth[t]?.values)))
    expect(filtered).toBeLessThan(0.8 * raw)
    // Seeking to a past step goes through a checkpoint: the state survives a round trip through JSON.
    const alg = s.algorithm
    const at = s.stateAt(120)
    const back = alg.deserialise?.(alg.serialise?.(at) ?? null)
    expect(Array.from(toFlat(back?.x ?? at.x))).toEqual(Array.from(toFlat(at.x)))
  })
})
