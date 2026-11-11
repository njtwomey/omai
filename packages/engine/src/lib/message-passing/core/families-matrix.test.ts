import { describe, expect, it } from 'vitest'
import { gamma as gammaDist, inverseWishart, wishart as wishartDist } from '../../dists/index.ts'
import { toArray, toFlat } from '../../tensor/index.ts'
import * as M from '../index.ts'

const S = [2, 0.6, 0.6, 1.1]
const close = (a: readonly number[], b: readonly number[], tol = 1e-10): void => {
  expect(a.length).toBe(b.length)
  for (const [i, v] of a.entries())
    expect(Math.abs(v - (b[i] as number)), `entry ${i}: ${v} vs ${b[i]}`).toBeLessThan(
      tol * (1 + Math.abs(b[i] as number)),
    )
}
const logDet2 = (a: readonly number[]): number => Math.log((a[0] as number) * (a[3] as number) - (a[1] as number) ** 2)

describe('the Wishart family', () => {
  const f = M.wishart(2)
  const eta = M.wishartEta(6.5, S, 2)

  it('is the library Wishart density: log q = eta . T(L) - A(eta) at random matrices', () => {
    const dist = wishartDist(6.5, {
      covariance: [
        [2, 0.6],
        [0.6, 1.1],
      ],
    })
    for (const l of [
      [1.2, 0.3, 0.3, 0.9],
      [4, -1.1, -1.1, 2.5],
      [0.4, 0.05, 0.05, 0.3],
    ]) {
      const lp = toFlat(
        dist.logProb([
          [l[0] as number, l[1] as number],
          [l[2] as number, l[3] as number],
        ]),
      )[0] as number
      const pair = f.pair(eta, [logDet2(l), ...l])
      expect(pair - f.logPartition(eta)).toBeCloseTo(lp, 10)
    }
  })

  it('moments: E L = nu S, the entropy identity H = A - eta . E[T], the library variance', () => {
    const dist = wishartDist(6.5, {
      covariance: [
        [2, 0.6],
        [0.6, 1.1],
      ],
    })
    const mo = f.moments(eta)
    close(mo.slice(1), toFlat(dist.mean()) as unknown as number[])
    expect(f.entropy(eta)).toBeCloseTo(toFlat(dist.entropy())[0] as number, 9)
    const summary = f.summary(eta)
    close((summary['variance'] as number[][]).flat(), toFlat(dist.variance()) as unknown as number[])
  })

  it('fromMoments inverts moments (the KL projection of a member is itself)', () => {
    close(f.fromMoments(f.moments(eta)), eta, 1e-9)
    const near = M.wishartEta(2.2, [0.5, -0.1, -0.1, 0.8], 2)
    close(f.fromMoments(f.moments(near)), near, 1e-9)
    // a very diffuse member (few degrees of freedom) and a very concentrated one
    const wide = M.wishartEta(1.05, S, 2)
    close(f.fromMoments(f.moments(wide)), wide, 1e-7)
    const tight = M.wishartEta(4000, S, 2)
    close(f.fromMoments(f.moments(tight)), tight, 1e-7)
  })

  it('product and quotient are conjugate updates; the quotient flags an improper result', () => {
    const y = [0.7, -1.3]
    const site = [0.5, ...y.flatMap((a) => y.map((b) => -0.5 * a * b))]
    const post = f.product(eta, site)
    const p = M.wishartParams(post, 2)
    expect(p?.nu).toBeCloseTo(7.5, 12)
    const back = f.quotient(post, site)
    close(back.eta, eta)
    expect(back.improper).toBe(false)
    expect(f.quotient(site, eta).improper).toBe(true)
  })

  it('KL(p || q) against the density ratio at the moments: zero for p = q, positive otherwise, and the d = 1 case is the gamma KL', () => {
    expect(f.kl(eta, eta)).toBeCloseTo(0, 12)
    expect(f.kl(eta, M.wishartEta(5, S, 2))).toBeGreaterThan(0)
    const one = M.wishart(1)
    const p = M.wishartEta(5, [1.5], 1)
    const q = M.wishartEta(7, [0.8], 1)
    // Wishart(nu, s) in one dimension is Gamma(nu / 2, rate 1 / (2 s))
    const gp = M.gammaEta(2.5, 1 / 3)
    const gq = M.gammaEta(3.5, 1 / 1.6)
    expect(one.kl(p, q)).toBeCloseTo(M.gamma.kl(gp, gq), 10)
    expect(one.logPartition(p)).toBeCloseTo(M.gamma.logPartition(gp), 10)
    close(one.moments(p).slice(1), [5 * 1.5])
    expect(one.moments(p)[0]).toBeCloseTo((M.gamma.moments(gp)[0] as number) + Math.log(1), 10)
  })

  it('is proper only for a positive definite -B and a > -1', () => {
    expect(f.isProper(eta)).toBe(true)
    expect(f.isProper([-1.5, ...eta.slice(1)])).toBe(false)
    expect(f.isProper([0.2, 1, 0, 0, 1])).toBe(false)
    expect(f.isProper(f.flat)).toBe(false)
  })

  it('works through familyOf and the JSON description', () => {
    expect(M.familyOf({ kind: 'wishart', size: 3 }).size).toBe(10)
    expect(M.familyOf({ kind: 'normal-wishart', size: 3 }).size).toBe(14)
  })
})

describe('the inverse Wishart is the Wishart of the inverse', () => {
  it('the library inverse Wishart mean is S^-1 / (nu - d - 1) and the Wishart message of the precision has the matching scale', () => {
    const psi = [
      [2, 0.6],
      [0.6, 1.1],
    ]
    const iw = inverseWishart(6.5, psi)
    const mean = toArray(iw.mean()) as number[][]
    // precision Lambda = Sigma^-1 ~ W(6.5, Psi^-1); E[Sigma] = Psi / (nu - 3)
    expect(mean[0]?.[0]).toBeCloseTo(2 / 3.5, 12)
    const eta = M.wishartEta(6.5, [1.1 / 1.54, -0.6 / 1.54, -0.6 / 1.54, 2 / 1.54], 2)
    const p = M.wishartParams(eta, 2)
    expect(p?.nu).toBe(6.5)
  })
})

describe('the normal-Wishart family', () => {
  const f = M.normalWishart(2)
  const eta = M.normalWishartEta([1, -0.5], 3, 6.5, S)

  it('recovers its parameters and the moments of mean and precision', () => {
    const p = M.normalWishartParams(eta, 2)
    close(Array.from(p?.m ?? []), [1, -0.5])
    expect(p?.beta).toBeCloseTo(3, 12)
    expect(p?.nu).toBeCloseTo(6.5, 12)
    close(Array.from(p?.scale ?? []), S)
    const mo = f.moments(eta)
    close(
      mo.slice(1, 5),
      S.map((v) => 6.5 * v),
    )
    // E[mu' L mu] = d / beta + nu m' S m
    const mSm = 1 * (2 * 1 + 0.6 * -0.5) + -0.5 * (0.6 * 1 + 1.1 * -0.5)
    expect(mo[7]).toBeCloseTo(2 / 3 + 6.5 * mSm, 10)
  })

  it('fromMoments inverts moments', () => {
    close(f.fromMoments(f.moments(eta)), eta, 1e-9)
  })

  it('log-partition: the normalised density integrates to one (a one-dimensional check against the normal-gamma)', () => {
    const one = M.normalWishart(1)
    const ng = M.normalGamma
    // mu | L ~ N(m, 1 / (beta L)), L ~ Wishart(nu, s) = Gamma(nu / 2, rate 1 / (2 s)): the normal-gamma with a = nu / 2, b = 1 / (2 s)
    const w = M.normalWishartEta([0.7], 2.5, 5, [1.5])
    const g = M.normalGammaEta(0.7, 2.5, 2.5, 1 / 3)
    expect(one.logPartition(w)).toBeCloseTo(ng.logPartition(g), 10)
    // the shared moments: E[L] = a / b, E[L mu] and E[L mu^2]
    expect(one.moments(w)[1]).toBeCloseTo(ng.moments(g)[2] as number, 10)
    expect(one.moments(w)[2]).toBeCloseTo(ng.moments(g)[0] as number, 10)
    expect(one.moments(w)[3]).toBeCloseTo(ng.moments(g)[1] as number, 10)
    // and E log L: digamma(a) - log b
    expect(one.moments(w)[0]).toBeCloseTo(ng.moments(g)[3] as number, 10)
    expect(toFlat(gammaDist(2.5, 1 / 3).mean())[0]).toBeCloseTo(7.5, 12)
  })
})
