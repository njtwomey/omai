// The AMP page's compute: the problem, the state evolution it is to follow, and the run read as its noise level and error.
import { describe, expect, it } from 'vitest'
import { snapshotOf } from '../../../shared/smoke.ts'
import { problemOf } from './compute.ts'

describe('the AMP run', () => {
  const made = problemOf({ seed: 1, columns: 300, delta: 0.5, sparsity: 0.1, noiseVariance: 0.01 })
  const kept = snapshotOf(made.run, {}, 100)

  it('has the truth and the state evolution to compare the run with', () => {
    expect(made.truth).toHaveLength(300)
    expect(made.predicted).toHaveLength(61)
  })

  it('reads the noise level and the error of every iteration, and the estimate of every entry', () => {
    expect(kept.tau2.length).toBeGreaterThan(5)
    expect(kept.mse.length).toBe(kept.tau2.length)
    expect(kept.estimate).toHaveLength(300)
    // The run follows the prediction: the noise level of the last iteration is close to the predicted fixed point.
    const last = kept.tau2.at(-1) ?? Number.NaN
    expect(Math.abs(Math.log(last / (made.predicted[kept.tau2.length - 1] ?? 1)))).toBeLessThan(0.5)
  })
})
