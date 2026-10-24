import { describe, expect, it } from 'vitest'
import { jvp } from '../../autodiff/index.ts'
import { key } from '../../random/index.ts'
import {
  cos,
  item,
  linspace,
  matmul,
  mean,
  mul,
  sin,
  stack,
  std,
  sum,
  swapaxes,
  type Tensor,
  tensor,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import * as D from '../index.ts'

const N = 10000
const close = (a: readonly number[], b: readonly number[], digits: number): void => {
  for (const [i, v] of a.entries()) expect(v).toBeCloseTo(b[i] ?? Number.NaN, digits)
}

describe('matrix distributions', () => {
  const sigma = tensor([
    [1, 0.3],
    [0.3, 0.5],
  ])

  it('Wishart draws average to df Sigma and differentiate in df', () => {
    const d = D.wishart(6, { covariance: sigma })
    const n = 4000
    const x = d.rsample(key(1), [n]) as Tensor
    const sd = toFlat(d.variance()).map((v) => Math.sqrt(v / n))
    for (const [i, m] of toFlat(mean(x, 0)).entries())
      expect(Math.abs(m - (toFlat(d.mean())[i] ?? 0))).toBeLessThan(5 * (sd[i] ?? 0))
    // d/d df E[X] = Sigma.
    const g = jvp(
      (nu: Value) => mean(D.wishart(nu, { covariance: sigma }).rsample(key(2), [n]), 0),
      tensor(6),
      tensor(1),
    )
    close(toFlat(g.tangent), toFlat(sigma), 1)
  })

  it('inverse Wishart draws average to Psi / (df - p - 1); entropy matches Monte Carlo', () => {
    const d = D.inverseWishart(8, sigma)
    const x = d.rsample(key(3), [8000]) as Tensor
    close(toFlat(mean(x, 0)), toFlat(d.mean()), 1)
    const mc = -item(mean(d.logProb(x)))
    expect(Math.abs(mc - item(d.entropy()))).toBeLessThan(0.06)
  })

  it('LKJ Cholesky draws are correlation factors and the density is normalised', () => {
    const d = D.lkjCholesky(3, 2)
    const l = d.rsample(key(4), [200]) as Tensor
    const c = matmul(l, swapaxes(l, -1, -2))
    const diag = toFlat(c).filter((_, i) => i % 9 === 0 || i % 9 === 4 || i % 9 === 8)
    expect(diag.every((v) => Math.abs(v - 1) < 1e-12)).toBe(true)
    // For d = 2, L = [[1, 0], [r, sqrt(1 - r^2)]]: the density in r integrates to one.
    const r = linspace(-0.9999, 0.9999, 4001)
    const rows = toFlat(r).map((v) => [
      [1, 0],
      [v, Math.sqrt(1 - v * v)],
    ])
    const dens = toFlat(D.lkjCholesky(2, 2).prob(tensor(rows)))
    const mass = (dens.reduce((s, v) => s + v, 0) * (2 * 0.9999)) / 4000
    expect(mass).toBeCloseTo(1, 3)
    // LKJ correlations: off-diagonal variance 1 / (2 eta + d - 1).
    const k = D.lkj(3, 2).rsample(key(5), [5000]) as Tensor
    const r01 = toFlat(k).filter((_, i) => i % 9 === 1)
    const v = r01.reduce((s, x) => s + x * x, 0) / r01.length
    expect(v).toBeCloseTo(1 / 6, 2)
  })
})

describe('directional distributions', () => {
  it('von Mises draws have the circular mean and variance, and differentiate in the location', () => {
    const d = D.vonMises(1, 4)
    const x = d.rsample(key(6), [N]) as Tensor
    const c = item(mean(cos(x)))
    const s = item(mean(sin(x)))
    expect(Math.atan2(s, c)).toBeCloseTo(1, 1)
    expect(1 - Math.hypot(c, s)).toBeCloseTo(item(d.variance()), 2)
    const g = jvp((m: Value) => mean(D.vonMises(m, 4).rsample(key(7), [100])), tensor(1), tensor(1)).tangent
    expect(item(g)).toBeCloseTo(1, 10)
  })

  it('von Mises-Fisher draws are unit vectors with mean A_d(kappa) mu', () => {
    const d = D.vonMisesFisher([0, 0.6, 0.8], 5)
    const x = d.sample(key(8), [N]) as Tensor
    expect(toFlat(sum(mul(x, x), -1)).every((v) => Math.abs(v - 1) < 1e-12)).toBe(true)
    // Within four standard errors of the sample mean, estimated from the draws.
    const se = toFlat(std(x, 0)).map((v) => v / Math.sqrt(N))
    const m = toFlat(d.mean())
    for (const [i, v] of toFlat(mean(x, 0)).entries()) expect(Math.abs(v - (m[i] ?? 0))).toBeLessThan(4 * (se[i] ?? 0))
    expect(d.hasRsample).toBe(false)
  })

  it('the power spherical density integrates to one on the circle and its draws differentiate in kappa', () => {
    const d = D.powerSpherical([0.6, 0.8], 3)
    const theta = linspace(0, 2 * Math.PI, 4001)
    const pts = stack([cos(theta), sin(theta)], -1)
    const mass = (toFlat(d.prob(pts)).reduce((s, v) => s + v, 0) * (2 * Math.PI)) / 4000
    expect(mass).toBeCloseTo(1, 3)
    const x = d.rsample(key(9), [N]) as Tensor
    close(toFlat(mean(x, 0)), toFlat(d.mean()), 2)
    // E[mu^T x] = kappa / (kappa + d - 1), with derivative (d - 1) / (kappa + d - 1)^2.
    const g = jvp(
      (k: Value) => mean(sum(mul(D.powerSpherical([0.6, 0.8], k).rsample(key(10), [N]), tensor([0.6, 0.8])), -1)),
      tensor(3),
      tensor(1),
    ).tangent
    expect(item(g)).toBeCloseTo(1 / 16, 2)
    expect(item(d.entropy())).toBeCloseTo(-item(mean(d.logProb(x))), 1)
  })
})
