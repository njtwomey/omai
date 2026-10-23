import { describe, expect, it } from 'vitest'
import { jvp } from '../../autodiff/index.ts'
import { key } from '../../random/index.ts'
import { item, mean, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as D from '../index.ts'

const N = 3000

const empiricalTau = (u: Tensor): number => toFlat(D.kendallTauMatrix(u))[1] ?? Number.NaN

describe('copula sampling', () => {
  const families: readonly (readonly [string, D.Copula])[] = [
    [
      'gaussian',
      D.gaussianCopula([
        [1, 0.6],
        [0.6, 1],
      ]),
    ],
    [
      'studentT',
      D.studentTCopula(
        [
          [1, -0.4],
          [-0.4, 1],
        ],
        4,
      ),
    ],
    ['clayton', D.claytonCopula(2)],
    ['gumbel', D.gumbelCopula(1.8)],
    ['frank', D.frankCopula(3)],
    ['frank (negative)', D.frankCopula(-2.5)],
    ['clayton (3-d)', D.claytonCopula(1.5, 3)],
    ['gumbel (3-d)', D.gumbelCopula(2, 3)],
    ['frank (3-d)', D.frankCopula(4, 3)],
  ]
  for (const [name, c] of families) {
    it(`${name}: uniform marginals and the right Kendall's tau`, () => {
      const u = c.sample(key(1), [N]) as Tensor
      expect(u.shape).toEqual([N, c.dim])
      for (const m of toFlat(mean(u, 0))) expect(m).toBeCloseTo(0.5, 1)
      const want = toFlat(c.kendallTau())
      const tau = want.length === 1 ? (want[0] ?? 0) : (want[1] ?? 0)
      expect(Math.abs(empiricalTau(u) - tau)).toBeLessThan(0.04)
    })
  }
})

describe('copula densities and CDFs', () => {
  it('the bivariate t-copula CDF has uniform margins and the density as its mixed partial', () => {
    const c = D.studentTCopula(
      [
        [1, 0.5],
        [0.5, 1],
      ],
      5,
    )
    expect(item(c.cdf([0.3, 1 - 1e-15]))).toBeCloseTo(0.3, 8)
    const u = tensor([0.4, 0.7])
    const mixed = jvp((v: Value) => jvp((w: Value) => c.cdf(w), v, tensor([1, 0])).tangent, u, tensor([0, 1])).tangent
    expect(item(mixed)).toBeCloseTo(item(c.density(u)), 7)
    // Without correlation and with many degrees of freedom it approaches independence.
    expect(
      item(
        D.studentTCopula(
          [
            [1, 0],
            [0, 1],
          ],
          1e6,
        ).cdf([0.3, 0.6]),
      ),
    ).toBeCloseTo(0.18, 5)
  })

  it('the Gaussian copula CDF is the independence copula at zero correlation', () => {
    expect(
      item(
        D.gaussianCopula([
          [1, 0],
          [0, 1],
        ]).cdf([0.3, 0.6]),
      ),
    ).toBeCloseTo(0.18, 12)
    expect(
      item(
        D.gaussianCopula([
          [1, 0.3],
          [0.3, 1],
        ]).cdf([0.5, 0.5]),
      ),
    ).toBeCloseTo(0.25 + Math.asin(0.3) / (2 * Math.PI), 12)
  })
})

describe('joining marginals', () => {
  it('copula distributions keep their marginals and differentiate through reparameterised draws', () => {
    const make = (rho: Value): D.Distribution =>
      D.copulaDistribution(
        D.gaussianCopula([
          [1, 0.5],
          [0.5, 1],
        ]),
        [D.normal(rho, 2), D.exponential(2)],
      )
    const d = make(tensor(1))
    const x = d.rsample(key(2), [N]) as Tensor
    for (const [j, m] of toFlat(mean(x, 0)).entries()) expect(m).toBeCloseTo([1, 0.5][j] ?? 0, 1)
    const g = jvp((m: Value) => mean(make(m).rsample(key(3), [100]), 0), tensor(1), tensor(1)).tangent
    expect(toFlat(g)[0]).toBeCloseTo(1, 10)
    // The joint density factorises at zero correlation.
    const ind = D.copulaDistribution(
      D.gaussianCopula([
        [1, 0],
        [0, 1],
      ]),
      [D.normal(0, 1), D.exponential(2)],
    )
    expect(item(ind.logProb([0.3, 0.4]))).toBeCloseTo(
      item(D.normal(0, 1).logProb(0.3)) + item(D.exponential(2).logProb(0.4)),
      12,
    )
  })

  it('pseudo-observations and fitting recover the parameter', () => {
    const u = D.claytonCopula(3).sample(key(4), [2000]) as Tensor
    const fitted = D.fitCopula('clayton', D.pseudoObservations(u))
    expect(item(fitted.params['theta'] as Value)).toBeCloseTo(3, 0)
    expect(() =>
      D.fitCopula('clayton', [
        [1, 3],
        [2, 2],
        [3, 1],
      ]),
    ).toThrow(/tau/)
  })
})
