import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { foldIn, key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, recorder } from '../trace/index.ts'
import {
  advancedComposition,
  analyticGaussianScale,
  basicComposition,
  exponentialMechanism,
  gaussianMechanism,
  gaussianScale,
  heterogeneousComposition,
  laplaceMechanism,
  randomisedResponse,
  randomisedResponseEstimate,
  rdpAccountant,
  subsampledGaussianRdp,
} from './privacy.ts'

describe('privacy mechanisms', () => {
  it('add noise of the calibrated scale', () => {
    const zeros = new Array<number>(4000).fill(0)
    const lap = toFlat(laplaceMechanism(key(1), zeros, { sensitivity: 2, epsilon: 0.5 }))
    const meanAbs = lap.reduce((a, b) => a + Math.abs(b), 0) / lap.length
    expect(Math.abs(meanAbs - 4)).toBeLessThan(0.3)
    const g = toFlat(gaussianMechanism(key(2), zeros, { sensitivity: 1, epsilon: 1, delta: 1e-5 }))
    const sd = Math.sqrt(g.reduce((a, b) => a + b * b, 0) / g.length)
    expect(Math.abs(sd - analyticGaussianScale({ sensitivity: 1, epsilon: 1, delta: 1e-5 }))).toBeLessThan(0.15)
  })

  it('the analytic calibration never needs more noise than the classic one', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 99 }), fc.integer({ min: 3, max: 9 }), (e, d) => {
        const options = { sensitivity: 1, epsilon: e / 100, delta: 10 ** -d }
        expect(analyticGaussianScale(options)).toBeLessThanOrEqual(gaussianScale(options) + 1e-12)
      }),
    )
  })

  it('randomised response is unbiased after correction', () => {
    const truth = Array.from({ length: 6000 }, (_, i) => (i % 10 < 3 ? 1 : 0))
    let total = 0
    for (let r = 0; r < 5; r++) total += randomisedResponseEstimate(randomisedResponse(foldIn(key(3), r), truth, 1), 1)
    expect(Math.abs(total / 5 - 0.3)).toBeLessThan(0.03)
    expect(() => randomisedResponse(key(0), [2], 1)).toThrow()
  })

  it('the exponential mechanism prefers high utility', () => {
    const counts = [0, 0, 0]
    for (let r = 0; r < 300; r++) {
      const i = exponentialMechanism(foldIn(key(4), r), [0, 1, 3], { epsilon: 2, sensitivity: 1 })
      counts[i] = (counts[i] ?? 0) + 1
    }
    expect((counts[2] ?? 0) > (counts[1] ?? 0) && (counts[1] ?? 0) > (counts[0] ?? 0)).toBe(true)
  })
})

describe('accounting', () => {
  it('composition bounds are ordered and the advanced bound wins for many small steps', () => {
    const many = new Array(200).fill({ epsilon: 0.05, delta: 0 })
    const naive = basicComposition(many)
    const het = heterogeneousComposition(many, 1e-5)
    expect(het.epsilon).toBeLessThan(naive.epsilon)
    expect(advancedComposition({ epsilon: 0.05, delta: 0 }, 200, 1e-5).epsilon).toBeLessThan(naive.epsilon)
  })

  it('RDP grows with steps and sampling rate, shrinks with noise; the accountant is traced', () => {
    const a = subsampledGaussianRdp(0.01, 1, 10, [2, 8])
    const b = subsampledGaussianRdp(0.01, 1, 20, [2, 8])
    const c = subsampledGaussianRdp(0.01, 2, 10, [2, 8])
    const d = subsampledGaussianRdp(0.05, 1, 10, [2, 8])
    a.forEach((v, i) => {
      expect(b[i] ?? 0).toBeCloseTo(2 * v, 12)
      expect(c[i] ?? 0).toBeLessThan(v)
      expect(d[i] ?? 0).toBeGreaterThan(v)
    })
    const rec = recorder()
    const acc = rdpAccountant(
      [
        { sampleRate: 0.01, noiseMultiplier: 1.1, steps: 500 },
        { sampleRate: 0.02, noiseMultiplier: 1.3, steps: 300 },
      ],
      { delta: 1e-5, tracer: rec },
    )
    const eps = metricSeries(rec.run(), 'epsilon').values
    expect(eps.length).toBe(2)
    expect((eps[1] ?? 0) > (eps[0] ?? 0)).toBe(true)
    expect(acc.epsilon).toBeCloseTo(eps[1] ?? 0, 12)
  })
})
