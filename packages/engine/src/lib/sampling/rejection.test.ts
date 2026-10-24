import { describe, expect, it } from 'vitest'
import { bimodal, expectDistribution, normalCdf } from '../../testing/sampling.ts'
import { gamma, normal } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { item } from '../tensor/index.ts'
import { adaptiveRejectionSample, ratioOfUniforms, rejectionSample } from './rejection.ts'
import { numericTarget, proposalFromDistribution } from './target.ts'

const gaussian = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2, 1, { grad: (x) => [-(x[0] ?? 0)] })

describe('rejectionSample', () => {
  it('draws exact samples under a valid envelope', () => {
    const r = rejectionSample(gaussian, proposalFromDistribution(normal(0, 2)), {
      key: key(1),
      n: 2000,
      logEnvelope: Math.log(2 * Math.sqrt(2 * Math.PI)),
    })
    expect(r.complete).toBe(true)
    expect(r.envelopeViolations).toBe(0)
    expectDistribution(r.samples.data, normalCdf)
  })

  it('reports envelope violations and stops at the proposal budget', () => {
    const r = rejectionSample(gaussian, proposalFromDistribution(normal(0, 2)), { key: key(2), n: 100, logEnvelope: 0 })
    expect(r.envelopeViolations).toBeGreaterThan(0)
    const capped = rejectionSample(gaussian, proposalFromDistribution(normal(0, 2)), {
      key: key(3),
      n: 100,
      logEnvelope: 40,
      maxProposals: 50,
    })
    expect(capped.complete).toBe(false)
    expect(capped.proposals).toBe(50)
    expect(() =>
      rejectionSample(
        numericTarget(() => 0, 2),
        proposalFromDistribution(normal(0, 1)),
        { key: key(0), n: 1, logEnvelope: 0 },
      ),
    ).toThrow()
  })
})

describe('adaptiveRejectionSample', () => {
  it('samples a normal exactly with a tightening envelope', () => {
    const r = adaptiveRejectionSample(gaussian, { key: key(4), n: 2000 })
    expectDistribution(r.samples.data, normalCdf)
    expect(r.acceptanceRate).toBeGreaterThan(0.95)
    expect(r.abscissae.length).toBeGreaterThan(2)
    expect(r.abscissae.length).toBeLessThanOrEqual(50)
  })

  it('samples a gamma on a half-line', () => {
    const t = numericTarget((x) => 2 * Math.log(x[0] ?? 0) - (x[0] ?? 0), 1, { grad: (x) => [2 / (x[0] ?? 1) - 1] })
    const r = adaptiveRejectionSample(t, {
      key: key(5),
      n: 2000,
      domain: [0, Number.POSITIVE_INFINITY],
      initialPoints: [1, 5],
    })
    const g = gamma(3, 1)
    expectDistribution(r.samples.data, (x) => item(g.cdf(x)))
  })

  it('refuses densities that are not log-concave, and bad starts', () => {
    expect(() =>
      adaptiveRejectionSample(bimodal(3), { key: key(6), n: 200, initialPoints: [-4, -0.5, 0.5, 4] }),
    ).toThrow()
    expect(() => adaptiveRejectionSample(gaussian, { key: key(6), n: 10, initialPoints: [1, 2] })).toThrow()
    expect(() => adaptiveRejectionSample(gaussian, { key: key(6), n: 10, initialPoints: [0] })).toThrow()
    expect(() =>
      adaptiveRejectionSample(
        numericTarget(() => 0, 1),
        { key: key(6), n: 10 },
      ),
    ).toThrow()
  })
})

describe('ratioOfUniforms', () => {
  it('samples a normal and a Cauchy (bounded regions) exactly', () => {
    const r = ratioOfUniforms((x) => Math.exp(-0.5 * x * x), { key: key(7), n: 2000 })
    expectDistribution(r.samples.data, normalCdf)
    const c = ratioOfUniforms((x) => 1 / (1 + x * x), { key: key(8), n: 2000 })
    expectDistribution(c.samples.data, (x) => 0.5 + Math.atan(x) / Math.PI)
  })

  it('uses given bounds, and refuses a density that vanishes', () => {
    const r = ratioOfUniforms((x) => (x >= 0 ? Math.exp(-x) : 0), {
      key: key(9),
      n: 1000,
      bounds: { uMax: 1, vMin: 0, vMax: 2 / Math.E },
    })
    expectDistribution(r.samples.data, (x) => (x > 0 ? 1 - Math.exp(-x) : 0))
    expect(() => ratioOfUniforms(() => 0, { key: key(0), n: 1 })).toThrow()
  })
})
