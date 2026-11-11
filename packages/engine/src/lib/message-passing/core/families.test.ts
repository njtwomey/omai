import { describe, expect, it } from 'vitest'
import { integrate, integratePositive } from '../../testing/message-passing.ts'
import {
  bernoulli as bernoulliDist,
  beta as betaDist,
  dirichlet as dirichletDist,
  exponentialFamily,
  gamma as gammaDist,
  multivariateNormal,
  normal,
} from '../dists/index.ts'
import { toFlat } from '../tensor/index.ts'
import * as M from './index.ts'

const close = (a: readonly number[], b: readonly number[], tol = 1e-9): void => {
  expect(a.length).toBe(b.length)
  for (const [i, v] of a.entries())
    expect(Math.abs(v - (b[i] as number)), `entry ${i}: ${v} vs ${b[i]}`).toBeLessThan(
      tol * (1 + Math.abs(b[i] as number)),
    )
}

describe('families agree with dists/exponential-family', () => {
  it('Gaussian: moments, and log-partition up to the base measure', () => {
    const eta = M.gaussianEta(1.5, 0.64)
    const ef = exponentialFamily(normal(1.5, 0.8))
    close(
      M.gaussian.moments(eta),
      ef.meanParameters().map((v) => toFlat(v)[0] as number),
    )
    const lp = toFlat(ef.logPartition())[0] as number
    expect(M.gaussian.logPartition(eta)).toBeCloseTo(lp + 0.5 * Math.log(2 * Math.PI), 12)
  })
  it('gamma, beta, Bernoulli and Dirichlet: moments and log-partition', () => {
    const g = exponentialFamily(gammaDist(3.2, 1.7))
    close(
      M.gamma.moments(M.gammaEta(3.2, 1.7)),
      g.meanParameters().map((v) => toFlat(v)[0] as number),
    )
    expect(M.gamma.logPartition(M.gammaEta(3.2, 1.7))).toBeCloseTo(toFlat(g.logPartition())[0] as number, 12)
    const b = exponentialFamily(betaDist(2.5, 4))
    close(
      M.beta.moments(M.betaEta(2.5, 4)),
      b.meanParameters().map((v) => toFlat(v)[0] as number),
    )
    expect(M.beta.logPartition(M.betaEta(2.5, 4))).toBeCloseTo(toFlat(b.logPartition())[0] as number, 12)
    const be = exponentialFamily(bernoulliDist({ probs: 0.3 }))
    close(
      M.bernoulli.moments(M.bernoulliEta(0.3)),
      be.meanParameters().map((v) => toFlat(v)[0] as number),
    )
    expect(M.bernoulli.logPartition(M.bernoulliEta(0.3))).toBeCloseTo(toFlat(be.logPartition())[0] as number, 12)
    const d = exponentialFamily(dirichletDist([2, 3.5, 1.5]))
    close(M.dirichlet(3).moments([1, 2.5, 0.5]), toFlat(d.meanParameters()[0] as never))
    expect(M.dirichlet(3).logPartition([1, 2.5, 0.5])).toBeCloseTo(toFlat(d.logPartition())[0] as number, 12)
  })
  it('multivariate Gaussian: moments and log-partition up to the base measure', () => {
    const cov = [2, 0.5, 0.5, 1]
    const eta = M.gaussianMvEta([1, -1], cov)
    const ef = exponentialFamily(
      multivariateNormal([1, -1], {
        covariance: [
          [2, 0.5],
          [0.5, 1],
        ],
      }),
    )
    const mean = toFlat(ef.meanParameters()[0] as never)
    const second = toFlat(ef.meanParameters()[1] as never)
    close(M.gaussianMv(2).moments(eta), [...mean, ...second])
    expect(M.gaussianMv(2).logPartition(eta)).toBeCloseTo(
      (toFlat(ef.logPartition())[0] as number) + Math.log(2 * Math.PI),
      10,
    )
  })
})

describe('families against numerical integration', () => {
  it('normal-gamma: the mass and the moments', () => {
    const eta = M.normalGammaEta(0.7, 3, 2.5, 1.2)
    const density = (mu: number, lam: number): number =>
      Math.exp(
        (eta[0] as number) * lam * mu +
          (eta[1] as number) * lam * mu * mu +
          (eta[2] as number) * lam +
          (eta[3] as number) * Math.log(lam),
      )
    const { m, beta: b0 } = M.normalGammaParams(eta)
    const inner = (lam: number, g: (mu: number) => number): number => {
      const w = 14 / Math.sqrt(b0 * lam)
      return integrate(g, m - w, m + w, 8, 16)
    }
    const mass = integratePositive((lam) => inner(lam, (mu) => density(mu, lam)), 40)
    expect(Math.log(mass)).toBeCloseTo(M.normalGamma.logPartition(eta), 6)
    const first = integratePositive((lam) => lam * inner(lam, (mu) => mu * density(mu, lam)), 40) / mass
    expect(first).toBeCloseTo(M.normalGamma.moments(eta)[0] as number, 5)
  })
  it('gamma moment matching recovers the shape and rate', () => {
    for (const [a, b] of [
      [0.7, 2],
      [3, 0.4],
      [40, 9],
    ] as const) {
      const eta = M.gammaEta(a, b)
      close(M.gamma.fromMoments(M.gamma.moments(eta)), eta, 1e-9)
    }
  })
  it('beta, Dirichlet and normal-gamma moment matching round trips', () => {
    for (const [a, b] of [
      [0.6, 0.9],
      [2, 5],
      [30, 4],
    ] as const)
      close(M.beta.fromMoments(M.beta.moments(M.betaEta(a, b))), M.betaEta(a, b), 1e-8)
    const d = M.dirichlet(4)
    close(d.fromMoments(d.moments([0.3, 1.2, 4, 0.1])), [0.3, 1.2, 4, 0.1], 1e-8)
    const ng = M.normalGammaEta(-1, 2, 3, 0.5)
    close(M.normalGamma.fromMoments(M.normalGamma.moments(ng)), ng, 1e-8)
    const mv = M.gaussianMv(2)
    const eta = M.gaussianMvEta([1, 2], [2, 0.3, 0.3, 1])
    close(mv.fromMoments(mv.moments(eta)), eta, 1e-10)
    const dc = M.discrete(3)
    close(dc.fromMoments(dc.moments(M.discreteEta([0.2, 0.5, 0.3]))), M.discreteEta([0.2, 0.5, 0.3]), 1e-12)
  })
  it('KL and entropy of the scalar families equal the integrals', () => {
    const cases: [
      typeof M.gaussian,
      readonly number[],
      readonly number[],
      (eta: readonly number[], x: number) => number,
      number,
      number,
    ][] = [
      [
        M.gaussian,
        M.gaussianEta(0.3, 1.5),
        M.gaussianEta(-0.4, 0.8),
        (e, x) => Math.exp((e[0] as number) * x - 0.5 * (e[1] as number) * x * x - M.gaussian.logPartition(e)),
        -14,
        14,
      ],
    ]
    for (const [fam, p, q, dens, lo, hi] of cases) {
      const kl = integrate(
        (x) => {
          const px = dens(p, x)
          return px > 0 ? px * (Math.log(px) - Math.log(dens(q, x))) : 0
        },
        lo,
        hi,
      )
      expect(fam.kl(p, q)).toBeCloseTo(kl, 8)
      expect(fam.entropy(p)).toBeCloseTo(
        -integrate(
          (x) => {
            const px = dens(p, x)
            return px > 0 ? px * Math.log(px) : 0
          },
          lo,
          hi,
        ),
        8,
      )
    }
    const gp = M.gammaEta(3, 2)
    const gq = M.gammaEta(2, 0.7)
    const gd = (e: readonly number[], x: number): number =>
      Math.exp((e[0] as number) * Math.log(x) + (e[1] as number) * x - M.gamma.logPartition(e))
    expect(M.gamma.kl(gp, gq)).toBeCloseTo(
      integratePositive((x) => gd(gp, x) * (Math.log(gd(gp, x)) - Math.log(gd(gq, x))), 80),
      7,
    )
    expect(M.gamma.entropy(gp)).toBeCloseTo(-integratePositive((x) => gd(gp, x) * Math.log(gd(gp, x)), 80), 7)
    const bp = M.betaEta(2, 3)
    const bq = M.betaEta(4, 1.5)
    const bd = (e: readonly number[], x: number): number =>
      Math.exp((e[0] as number) * Math.log(x) + (e[1] as number) * Math.log(1 - x) - M.beta.logPartition(e))
    expect(M.beta.kl(bp, bq)).toBeCloseTo(
      integrate((x) => bd(bp, x) * (Math.log(bd(bp, x)) - Math.log(bd(bq, x))), 0, 1, 24),
      6,
    )
  })
})

describe('product, quotient and power', () => {
  it('multiply Gaussian densities and divide them back, flagging an improper quotient', () => {
    const a = M.gaussianEta(1, 2)
    const b = M.gaussianEta(3, 0.5)
    const ab = M.gaussian.product(a, b)
    close(M.gaussian.quotient(ab, b).eta, a)
    expect(M.gaussian.quotient(ab, b).improper).toBe(false)
    expect(M.gaussian.quotient(a, ab).improper).toBe(true)
    expect(M.gaussian.summary(ab)['precision']).toBeCloseTo(1 / 2 + 1 / 0.5, 12)
    close(M.gaussian.power(a, 0), [0, 0])
    expect(M.gaussian.isProper(M.gaussian.flat)).toBe(false)
  })
  it('keep discrete messages normalised and finite', () => {
    const d = M.discrete(3)
    const p = d.product(M.discreteEta([0.2, 0.3, 0.5]), M.discreteEta([0.5, 0.5, 0]))
    close(M.discreteProbs(p), [(0.2 * 0.5) / (0.1 + 0.15), 0.15 / 0.25, 0])
    expect(d.isProper(d.flat)).toBe(true)
    expect(d.kl(M.discreteEta([0.2, 0.3, 0.5]), M.discreteEta([0.3, 0.3, 0.4]))).toBeCloseTo(
      0.2 * Math.log(0.2 / 0.3) + 0.3 * 0 + 0.5 * Math.log(0.5 / 0.4),
      12,
    )
  })
  it('build families from their JSON descriptions', () => {
    expect(M.familyOf({ kind: 'gaussian-mv', size: 3 }).size).toBe(12)
    expect(M.familyOf({ kind: 'discrete', size: 4 }).dim).toBe(4)
    expect(() => M.familyOf({ kind: 'discrete' })).toThrow()
  })
})
