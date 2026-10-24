import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../autodiff/index.ts'
import { key } from '../random/index.ts'
import { item, tensor, type Value } from '../tensor/index.ts'
import * as C from './compose/index.ts'
import * as D from './index.ts'

const mc = (p: D.Distribution, q: D.Distribution): number => item(D.klMonteCarlo(p, q, key(3), 10000))

describe('KL divergences without a torch reference', () => {
  const pairs: readonly (readonly [string, D.Distribution, D.Distribution])[] = [
    ['weibull', D.weibull(1.7, 2), D.weibull(2.5, 1.5)],
    ['vonMises', D.vonMises(0.3, 2), D.vonMises(-0.4, 4)],
    ['inverseGamma', D.inverseGamma(3, 2), D.inverseGamma(4, 1.5)],
    ['inverseGaussian (exponential-family identity)', D.inverseGaussian(1.5, 2), D.inverseGaussian(1, 3)],
    [
      'wishart (exponential-family identity)',
      D.wishart(5, {
        covariance: [
          [1, 0.2],
          [0.2, 0.6],
        ],
      }),
      D.wishart(6, {
        covariance: [
          [0.8, 0],
          [0, 0.9],
        ],
      }),
    ],
  ]
  for (const [name, p, q] of pairs) {
    it(`${name} agrees with Monte Carlo`, () => {
      const exact = item(D.klDivergence(p, q))
      expect(Math.abs(exact - mc(p, q))).toBeLessThan(0.05 * Math.max(1, exact))
    })
  }

  it('is invariant under a shared bijector and refuses unknown pairs without a key', () => {
    const b = D.bijectors.exp()
    const kl = item(D.klDivergence(C.transformed(D.normal(0, 1), b), C.transformed(D.normal(1, 2), b)))
    expect(kl).toBeCloseTo(item(D.klDivergence(D.normal(0, 1), D.normal(1, 2))), 14)
    expect(() => D.klDivergence(D.studentT(3), D.normal(0, 1))).toThrow(/Monte Carlo/)
    expect(Number.isFinite(item(D.klDivergence(D.studentT(3), D.normal(0, 1), { key: key(1) })))).toBe(true)
  })

  it('differentiates in the parameters', () => {
    const f = (m: Value, s: Value): Value => D.klDivergence(D.normal(m, s), D.normal(0.3, 1.2))
    expect(checkPrimitive(f, [tensor(0.4), tensor(0.8)], { rtol: 1e-6, atol: 1e-9 }).failures).toEqual([])
  })
})

describe('Bhattacharyya, Hellinger and Wasserstein', () => {
  it('closed forms agree with Monte Carlo', () => {
    for (const [p, q] of [
      [D.normal(0, 1), D.normal(1, 2)],
      [D.gamma(2, 1), D.gamma(3, 1.5)],
      [
        D.multivariateNormal([0, 0], {
          covariance: [
            [1, 0.3],
            [0.3, 1],
          ],
        }),
        D.multivariateNormal([1, -0.5], {
          covariance: [
            [2, 0],
            [0, 0.5],
          ],
        }),
      ],
      [D.poisson(3), D.poisson(5)],
    ] as const) {
      const exact = item(D.bhattacharyyaCoefficient(p, q))
      const estimate = item(D.bhattacharyyaCoefficient(wrap(p), q, { key: key(2), samples: 12000 }))
      expect(Math.abs(exact - estimate)).toBeLessThan(0.03)
    }
    expect(item(D.hellinger(D.normal(0, 1), D.normal(0, 1)))).toBe(0)
    expect(
      item(D.bhattacharyyaDistance(D.categorical({ probs: [0.5, 0.5] }), D.categorical({ probs: [0.5, 0.5] }))),
    ).toBeCloseTo(0, 14)
  })

  it('the Gaussian Wasserstein distance reduces to its closed forms', () => {
    expect(item(D.wasserstein2Gaussian(D.normal(0, 1), D.normal(3, 5)))).toBeCloseTo(5, 14)
    // Commuting (diagonal) covariances: |dmu|^2 + |sqrt(Sp) - sqrt(Sq)|_F^2.
    const p = D.multivariateNormal([0, 0], {
      covariance: [
        [4, 0],
        [0, 1],
      ],
    })
    const q = D.multivariateNormal([1, 1], {
      covariance: [
        [1, 0],
        [0, 9],
      ],
    })
    expect(item(D.wasserstein2Gaussian(p, q))).toBeCloseTo(Math.sqrt(2 + 1 + 4), 10)
    expect(() => D.wasserstein2Gaussian(D.gamma(1, 1), D.normal(0, 1))).toThrow(/normals/)
  })
})

/** The same distribution without a recognised name, so only the Monte Carlo route applies. */
function wrap(d: D.Distribution): D.Distribution {
  return { ...d, name: 'opaque' }
}
