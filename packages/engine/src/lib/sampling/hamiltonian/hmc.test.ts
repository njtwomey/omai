import { describe, expect, it } from 'vitest'
import { expectInvariant, expectMoments, gaussian2, standardNormal } from '../../../testing/sampling.ts'
import { key, normal } from '../../random/index.ts'
import { numericTarget } from '../target.ts'
import { hmc, hmcAdaptation, hmcKernel, hmcState } from './hmc.ts'

function correlatedDraws(n: number, rho: number): Float64Array {
  const z = normal(key(7), [n, 2]).data
  return Float64Array.from({ length: 2 * n }, (_, k) => {
    const i = Math.floor(k / 2)
    const a = z[2 * i] ?? 0
    return k % 2 === 0 ? a : rho * a + Math.sqrt(1 - rho * rho) * (z[2 * i + 1] ?? 0)
  })
}

describe('hmcKernel', () => {
  const target = gaussian2([0, 0], [1, 1], 0.8)

  it('leaves a correlated Gaussian invariant', () => {
    const k = hmcKernel(target, { stepSize: 0.25, numIntegrationSteps: 8 })
    expectInvariant(k, correlatedDraws(1500, 0.8), { mean: [0, 0], sd: [1, 1] }, { key: key(1) })
  })

  it('stays invariant with the Yoshida integrator and a diagonal mass matrix', () => {
    const k = hmcKernel(target, {
      stepSize: 0.4,
      numIntegrationSteps: 5,
      integrator: 'yoshida',
      inverseMassMatrix: [0.8, 1.2],
    })
    expectInvariant(k, correlatedDraws(1500, 0.8), { mean: [0, 0], sd: [1, 1] }, { key: key(2) })
  })

  it('flags divergences when the energy error explodes, and rejects', () => {
    const narrow = numericTarget((x) => (-0.5 * (x[0] ?? 0) ** 2) / 1e-4, 1, { grad: (x) => [-(x[0] ?? 0) / 1e-4] })
    const k = hmcKernel(narrow, { stepSize: 1, numIntegrationSteps: 20 })
    const t = k.step(key(3), k.init([0.01]))
    expect(t.info.divergent).toBe(true)
    expect(t.info.accepted).toBe(false)
    expect(t.info.acceptanceRate).toBe(0)
  })

  it('stops integrating once the log density leaves the support', () => {
    const halfLine = numericTarget((x) => ((x[0] ?? 0) > 0 ? -(x[0] ?? 0) : Number.NEGATIVE_INFINITY), 1, {
      grad: () => [-1],
    })
    const k = hmcKernel(halfLine, { stepSize: 0.5, numIntegrationSteps: 50 })
    const t = k.step(key(4), { position: Float64Array.from([0.01]), logDensity: -0.01, grad: Float64Array.from([-1]) })
    expect(t.info.numSteps).toBeLessThan(50)
    expect(t.state.position[0]).toBeGreaterThan(0)
  })

  it('needs a gradient, a positive step size and a sensible number of steps', () => {
    expect(() =>
      hmcKernel(
        numericTarget(() => 0, 1),
        { stepSize: 0.1, numIntegrationSteps: 1 },
      ),
    ).toThrow()
    expect(() => hmcKernel(target, { stepSize: -1, numIntegrationSteps: 1 })).toThrow()
    expect(() => hmcKernel(target, { stepSize: 0.1, numIntegrationSteps: 0 })).toThrow()
    expect(() => hmcState(target, [1])).toThrow()
  })
})

describe('hmc (window adaptation and chains)', () => {
  it('samples a 4-D standard normal with tuned step size', () => {
    const c = hmc(standardNormal(4), {
      key: key(5),
      numIntegrationSteps: 8,
      numChains: 2,
      numWarmup: 200,
      numDraws: 300,
    })
    expectMoments(c, { mean: [0, 0, 0, 0], sd: [1, 1, 1, 1] })
    expect(c.stats.energy).toBeDefined()
  })

  it('adapts a scaled Gaussian: the inverse mass matrix finds the variances', () => {
    const t = gaussian2([2, -3], [0.1, 5], 0)
    const adapt = hmcAdaptation(t, { numIntegrationSteps: 6 })
    const w = adapt.run(Float64Array.from([2, -3]), { key: key(6), numWarmup: 400 })
    const m = (w.parameters as { inverseMassMatrix: number[] }).inverseMassMatrix
    expect((m[0] ?? 0) / 0.01).toBeGreaterThan(0.4)
    expect((m[0] ?? 0) / 0.01).toBeLessThan(2.5)
    expect((m[1] ?? 0) / 25).toBeGreaterThan(0.4)
    expect((m[1] ?? 0) / 25).toBeLessThan(2.5)
  })
})
