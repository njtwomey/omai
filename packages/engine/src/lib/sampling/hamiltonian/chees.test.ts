import { describe, expect, it } from 'vitest'
import { expectInvariant, standardNormal } from '../../../testing/sampling.ts'
import { key, normal } from '../../random/index.ts'
import { runAlgorithm, stateToJson } from '../../trace/index.ts'
import { cheesAdaptationAlgorithm, cheesAdaptationResult, cheesKernel } from './chees.ts'

describe('cheesKernel', () => {
  it('leaves the product of its chains invariant, jittered or not', () => {
    const chains = 4
    const draws = normal(key(1), [1500 * chains]).data
    const truth = { mean: new Array<number>(chains).fill(0), sd: new Array<number>(chains).fill(1) }
    for (const jitter of [true, false])
      expectInvariant(
        cheesKernel(standardNormal(1), { numChains: chains, stepSize: 0.4, trajectoryLength: 1.6, jitter }),
        draws,
        truth,
        {
          key: key(2),
          steps: 2,
        },
      )
  })

  it('draws the same number of steps in every chain from a quasi-random jitter, and reports the trajectories for the adaptation', () => {
    const k = cheesKernel(standardNormal(2), { numChains: 3, stepSize: 0.2, trajectoryLength: 2 })
    let s = k.init(normal(key(3), [6]).data)
    const steps: number[] = []
    for (let i = 0; i < 8; i++) {
      s = k.step(key(10 + i), s).state
      steps.push(s.numSteps)
    }
    // The Halton fractions 1/2, 1/4, 3/4, ... of 10 steps: ceil(h T / eps).
    expect(steps.slice(0, 3)).toEqual([5, 3, 8])
    expect(s.proposals.length).toBe(6)
    expect(s.acceptProbs.every((a) => a >= 0 && a <= 1)).toBe(true)
  })

  it('validates its settings', () => {
    expect(() => cheesKernel(standardNormal(1), { numChains: 0, stepSize: 0.1, trajectoryLength: 1 })).toThrow()
    expect(() => cheesKernel(standardNormal(1), { numChains: 2, stepSize: 0, trajectoryLength: 1 })).toThrow()
    expect(() =>
      cheesKernel(standardNormal(1), { numChains: 2, stepSize: 0.1, trajectoryLength: 1 }).init([0]),
    ).toThrow()
  })
})

describe('cheesAdaptationAlgorithm', () => {
  it('grows the trajectory from one step to about a quarter period, at the acceptance it aims for', () => {
    const chains = 24
    const alg = cheesAdaptationAlgorithm(standardNormal(3), normal(key(4), [chains * 3]).data, {
      numChains: chains,
      numWarmup: 500,
      learningRate: 0.05,
    })
    const run = runAlgorithm(alg, { key: key(5), maxSteps: 500 })
    const p = cheesAdaptationResult(run.state)
    expect(p.trajectoryLength).toBeGreaterThanOrEqual(p.stepSize)
    // For a standard normal the best trajectory is about a quarter of the period, pi / 2.
    expect(p.trajectoryLength).toBeGreaterThan(0.7)
    expect(p.trajectoryLength).toBeLessThan(4)
    expect(p.stepSize).toBeGreaterThan(0.3)
    const accept = run.state.chain.acceptProbs.reduce((a, v) => a + v, 0) / chains
    expect(accept).toBeGreaterThan(0.4)
  })

  it('serialises its state, and needs a population of chains', () => {
    const alg = cheesAdaptationAlgorithm(standardNormal(2), normal(key(6), [8]).data, { numChains: 4, numWarmup: 30 })
    const whole = runAlgorithm(alg, { key: key(7), maxSteps: 30 }).state
    const part = runAlgorithm(alg, { key: key(7), maxSteps: 12 }).state
    const back = alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(part) ?? null)))
    expect(back?.adamStep).toBe(part.adamStep)
    expect(stateToJson(Array.from(back?.chain.position ?? []))).toEqual(stateToJson(Array.from(part.chain.position)))
    expect(Number.isFinite(whole.trajectoryLength)).toBe(true)
    expect(() =>
      cheesAdaptationAlgorithm(standardNormal(1), Float64Array.of(0), { numChains: 1, numWarmup: 10 }),
    ).toThrow()
  })
})
