// The kernel methods on the benchmark models: parallel tempering finds the mode weights of the bimodal target where NUTS stays in
// one mode, and the kernels agree with the truths of the conjugate and regression benchmarks.
import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { session } from '../../trace/index.ts'
import { nutsMethod, temperingMethod } from '../inference/index.ts'
import { runPlan } from './agreement.ts'
import type { MethodPlan } from './benchmark.ts'
import { bimodal } from './geometry.ts'

const fractions = (draws: Float64Array, chains: number): number[] => {
  const n = draws.length / chains / 2
  return Array.from({ length: chains }, (_, c) => {
    let up = 0
    for (let i = 0; i < n; i++) if ((draws[(c * n + i) * 2] ?? 0) > 0) up += 1
    return up / n
  })
}

describe('parallel tempering on the bimodal benchmark', () => {
  it('recovers the mode weights, which NUTS cannot: each NUTS chain stays in one mode', { timeout: 120_000 }, () => {
    const nuts = nutsMethod(bimodal.methodModel())
    const sn = session(nuts, {
      config: { numChains: 4, numWarmup: 200, numDraws: 300, inferDiscrete: false },
      steps: 100_000,
      mode: 'eager',
      seed: 1,
    })
    const xn = nuts.posterior?.(sn.state, sn.config).draws?.['x']?.data as Float64Array
    // Every NUTS chain sits in one mode, so its fraction in the upper mode is 0 or 1 (not the one half of the truth).
    expect(fractions(xn, 4).every((f) => f < 0.05 || f > 0.95)).toBe(true)

    const plan: MethodPlan = {
      kind: 'parallel-tempering',
      config: {
        kernel: 'rwm',
        numChains: 3,
        numWarmup: 300,
        numDraws: 1500,
        numTemperatures: 6,
        minBeta: 0.02,
        inferDiscrete: false,
      },
      steps: 100_000,
      expect: 'agree',
      note: 'tempered replicas cross the barrier',
    }
    const run = runPlan(bimodal, plan, { key: key(2) })
    const x = run.posterior.draws?.['x']?.data as Float64Array
    const f = fractions(x, 3)
    const pooled = f.reduce((a, v) => a + v, 0) / f.length
    expect(Math.abs(pooled - 0.5)).toBeLessThan(0.12)
    // Every chain visits both modes.
    expect(f.every((v) => v > 0.2 && v < 0.8)).toBe(true)
    void temperingMethod
  })
})
