import { describe, expect, it } from 'vitest'
import { moments, standardNormal } from '../../testing/sampling.ts'
import { foldIn, type Key, key, normal } from '../random/index.ts'
import { sampleChain } from './chains.ts'
import { polynomialStepSize } from './langevin.ts'
import { sghmcKernel } from './sghmc.ts'

describe('sghmcKernel', () => {
  it('samples a Gaussian from its exact gradient with friction, up to a discretisation bias', () => {
    const k = sghmcKernel((_, x) => x.map((v) => -v), 2, {
      stepSize: 0.05,
      numSteps: 10,
      friction: 0.5,
      target: standardNormal(2),
    })
    const r = sampleChain(k, [2, -2], { key: key(1), numDraws: 4000, numWarmup: 300 })
    const x = Float64Array.from({ length: 4000 }, (_, i) => r.draws[2 * i] ?? 0)
    const { mean, sd } = moments(x)
    expect(Math.abs(mean)).toBeLessThan(0.2)
    expect(Math.abs(sd - 1)).toBeLessThan(0.2)
    expect(r.state.t).toBe(4300)
  })

  it('tolerates gradient noise, with the estimate of its variance subtracted from the injected noise', () => {
    // A noisy gradient (variance 1 per coordinate): the injected noise is reduced by the estimate beta.
    const noise = (kk: Key, x: Float64Array): number[] =>
      Array.from(x, (v, i) => -v + 0.5 * (normal(foldIn(kk, 7), [x.length]).data[i] ?? 0))
    const k = sghmcKernel(noise, 1, { stepSize: 0.05, numSteps: 5, friction: 0.5, noiseEstimate: 0.25 })
    const r = sampleChain(k, [1], { key: key(2), numDraws: 4000, numWarmup: 200 })
    expect(Math.abs(moments(r.draws).sd - 1)).toBeLessThan(0.3)
    // A decaying schedule is accepted and advances with the iteration.
    const decaying = sghmcKernel((_, x) => x.map((v) => -v), 1, { stepSize: polynomialStepSize(0.1, 1, 0.55) })
    expect(decaying.step(key(3), { position: Float64Array.of(0), logDensity: 0, t: 10 }).info.stepSize).toBeCloseTo(
      0.1 * 11 ** -0.55,
      12,
    )
  })

  it('traces the step size, and validates its settings', () => {
    const k = sghmcKernel((_, x) => x, 1, { stepSize: 0.1 })
    expect(k.step(key(0), k.init([1])).info.stepSize).toBe(0.1)
    expect(() => sghmcKernel((_, x) => x, 1, { stepSize: 0 })).toThrow()
    expect(() => sghmcKernel((_, x) => x, 1, { stepSize: 0.1, numSteps: 0 })).toThrow()
    expect(() => sghmcKernel((_, x) => x, 1, { stepSize: 0.1, friction: 0 })).toThrow()
    expect(() =>
      sghmcKernel((_, x) => x, 1, { stepSize: 0.1, noiseEstimate: 100 }).step(key(0), {
        position: Float64Array.of(0),
        logDensity: 0,
        t: 0,
      }),
    ).toThrow()
  })
})
