import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { normal as normalDist } from '../dists/index.ts'
import { key } from '../random/index.ts'
import { mul, neg, square, sub, sum, toFlat, type Value } from '../tensor/index.ts'
import { fitDistribution, newtonMle, observedInformation } from './fit.ts'

describe('maximum likelihood fitting', () => {
  it('recovers the parameters of simulated data with consistent standard errors', () => {
    const x = toFlat(normalDist(3, 2).sample(key(11), [4000]))
    const f = fitDistribution(x, 'normal')
    expect(f.params['loc'] ?? 0).toBeCloseTo(3, 1)
    expect(f.params['scale'] ?? 0).toBeCloseTo(2, 1)
    expect(f.standardErrors['loc'] ?? 0).toBeCloseTo(2 / Math.sqrt(4000), 2)
    expect(f.aic).toBeCloseTo(-2 * f.logLikelihood + 4, 10)
  })

  it('prefers the true family by AIC', () => {
    const y = toFlat(normalDist(0, 1).sample(key(13), [300])).map((v) => Math.exp(v))
    const ln = fitDistribution(y, 'logNormal')
    expect(ln.converged).toBe(true)
    expect(ln.aic).toBeLessThan(fitDistribution(y, 'exponential').aic)
    expect(ln.aic).toBeLessThan(fitDistribution(y, 'gamma').aic)
  })

  it('Newton finds the maximum of a concave quadratic and its information is the negative Hessian', () => {
    fc.assert(
      fc.property(fc.double({ min: -3, max: 3, noNaN: true }), fc.double({ min: 0.5, max: 4, noNaN: true }), (m, c) => {
        const ll = (t: Value): Value => neg(sum(mul(square(sub(t, m)), c)))
        expect(newtonMle(ll, [0]).eta[0] ?? 0).toBeCloseTo(m, 8)
        expect(toFlat(observedInformation(ll, [m]))[0]).toBeCloseTo(2 * c, 10)
      }),
      { numRuns: 20 },
    )
  })
})
