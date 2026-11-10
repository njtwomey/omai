import { describe, expect, it } from 'vitest'
import { expectInvariant, gaussian2, moments, standardNormal } from '../../testing/sampling.ts'
import { key, normal } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { sampleChain } from './chains.ts'
import { mclmcAdaptationAlgorithm, mclmcAdaptationResult, mclmcKernel } from './mclmc.ts'
import { numericTarget } from './target.ts'

describe('mclmcKernel', () => {
  it('keeps the direction a unit vector, draws one at the first step, and reports the energy change', () => {
    const k = mclmcKernel(standardNormal(3), { stepSize: 0.3, momentumDecoherenceLength: 2 })
    const s0 = k.init([0.2, -0.1, 0.4])
    expect(s0.momentum.every((v) => v === 0)).toBe(true)
    const s1 = k.step(key(0), s0).state
    expect(Math.abs(Math.hypot(...s1.momentum) - 1)).toBeLessThan(1e-12)
    expect(Number.isFinite(s1.energyChange)).toBe(true)
    // The next step keeps the direction (up to the partial refresh).
    const s2 = k.step(key(1), s1)
    expect(Math.abs(Math.hypot(...s2.state.momentum) - 1)).toBeLessThan(1e-12)
  })

  it('has a small energy error for a small step: it shrinks as the sixth power of the step size', () => {
    const energy = (eps: number): number => {
      const k = mclmcKernel(standardNormal(4), {
        stepSize: eps,
        momentumDecoherenceLength: 1e9,
        integrator: 'mcLachlan',
      })
      let s = k.step(key(0), k.init([0.5, -0.5, 0.2, 0.1])).state
      let v = 0
      for (let i = 0; i < 200; i++) {
        s = k.step(key(i + 1), s).state
        v += s.energyChange ** 2
      }
      return v / 200
    }
    expect(energy(0.1)).toBeLessThan(energy(0.4) / 10)
  })

  it('samples a correlated Gaussian to within its discretisation bias', () => {
    const t = gaussian2([1, -1], [1, 2], 0.7)
    const k = mclmcKernel(t, { stepSize: 0.25, momentumDecoherenceLength: 3, integrator: 'mcLachlan' })
    const r = sampleChain(k, [0, 0], { key: key(1), numDraws: 6000, numWarmup: 300 })
    const a = moments(Float64Array.from({ length: 6000 }, (_, i) => r.draws[2 * i] ?? 0))
    const b = moments(Float64Array.from({ length: 6000 }, (_, i) => r.draws[2 * i + 1] ?? 0))
    expect(Math.abs(a.mean - 1)).toBeLessThan(0.2)
    expect(Math.abs(b.mean + 1)).toBeLessThan(0.4)
    expect(Math.abs(a.sd - 1)).toBeLessThan(0.15)
    expect(Math.abs(b.sd - 2)).toBeLessThan(0.3)
  })

  it('leaves a Gaussian approximately invariant in one small step from exact draws', () => {
    const draws = normal(key(2), [3000, 2]).data
    expectInvariant(
      mclmcKernel(standardNormal(2), { stepSize: 0.15, momentumDecoherenceLength: 2 }),
      draws,
      { mean: [0, 0], sd: [1, 1] },
      { key: key(3), steps: 2 },
    )
  })

  it('undoes a step that leaves the support and validates its settings', () => {
    const half = numericTarget(
      (x) => ((x[0] ?? 0) > 0 ? -(x[0] ?? 0) - 0.5 * (x[1] ?? 0) ** 2 : Number.NEGATIVE_INFINITY),
      2,
      {
        grad: (x) => [-1, -(x[1] ?? 0)],
      },
    )
    const k = mclmcKernel(half, { stepSize: 5, momentumDecoherenceLength: 1 })
    const s = k.init([0.1, 0])
    const r = k.step(key(4), s)
    expect(r.state.position[0]).toBeGreaterThan(0)
    expect(() => mclmcKernel(standardNormal(1), { stepSize: 1, momentumDecoherenceLength: 1 })).toThrow()
    expect(() => mclmcKernel(standardNormal(2), { stepSize: 0, momentumDecoherenceLength: 1 })).toThrow()
    expect(() => mclmcKernel(standardNormal(2), { stepSize: 1, momentumDecoherenceLength: 0 })).toThrow()
  })
})

describe('mclmcAdaptationAlgorithm', () => {
  it('tunes the step size to the energy variance and learns the scale of a badly scaled Gaussian', () => {
    const t = gaussian2([0, 0], [0.1, 3], 0)
    const alg = mclmcAdaptationAlgorithm(t, Float64Array.of(0.05, 1), { numWarmup: 600 })
    const p = mclmcAdaptationResult(runAlgorithm(alg, { key: key(3), maxSteps: 600 }).state)
    // The inverse mass matrix approximates the variances (0.01, 9), and the step size is positive and finite.
    expect(p.stepSize).toBeGreaterThan(0)
    expect(Number.isFinite(p.stepSize)).toBe(true)
    expect(p.inverseMassMatrix[1] as number).toBeGreaterThan((p.inverseMassMatrix[0] as number) * 50)
    expect(p.momentumDecoherenceLength).toBeGreaterThan(0)
  })

  it('needs at least ten warm-up steps and serialises its state', () => {
    const t = standardNormal(2)
    expect(() => mclmcAdaptationAlgorithm(t, Float64Array.of(0, 0), { numWarmup: 5 })).toThrow()
    const alg = mclmcAdaptationAlgorithm(t, Float64Array.of(0.1, 0.2), { numWarmup: 30 })
    const s0 = alg.init(key(0))
    const j = alg.serialise?.(s0)
    const back = alg.deserialise?.(JSON.parse(JSON.stringify(j)))
    expect(Array.from((back as { inverseMassMatrix: Float64Array }).inverseMassMatrix)).toEqual([1, 1])
  })
})
