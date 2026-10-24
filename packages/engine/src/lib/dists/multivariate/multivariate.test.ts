import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key } from '../../random/index.ts'
import { expandDims, item, mean, mul, sub, sum, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as D from '../index.ts'

const N = 10000
const empiricalCov = (x: Tensor): number[] => {
  const c = sub(x, mean(x, 0))
  return toFlat(mean(mul(expandDims(c, -1), expandDims(c, -2)), 0))
}

describe('multivariate normal families', () => {
  it('a batch of covariances gives batched densities and samples', () => {
    const covs = tensor([
      [
        [1, 0.3],
        [0.3, 2],
      ],
      [
        [0.5, -0.1],
        [-0.1, 0.4],
      ],
    ])
    const d = D.multivariateNormal([0.5, -1], { covariance: covs })
    expect(d.batchShape).toEqual([2])
    expect(d.sample(key(1), [3]).shape).toEqual([3, 2, 2])
    const one = D.multivariateNormal([0.5, -1], {
      covariance: [
        [0.5, -0.1],
        [-0.1, 0.4],
      ],
    })
    expect(toFlat(d.logProb([0.2, 0.1]))[1]).toBeCloseTo(item(one.logProb([0.2, 0.1])), 12)
    expect(toFlat(d.entropy())[1]).toBeCloseTo(item(one.entropy()), 12)
  })

  it('the low-rank normal matches the full one and samples its covariance', () => {
    const w = tensor([
      [1, 0.5],
      [0.2, -1],
      [0.3, 0.1],
    ])
    const diag = tensor([0.5, 0.2, 1])
    const lr = D.lowRankMultivariateNormal([1, 0, -1], w, diag)
    const full = D.multivariateNormal([1, 0, -1], { covariance: lr.covariance() })
    const x = tensor([
      [0.3, -0.2, 0.5],
      [1, 1, 1],
    ])
    for (const [i, v] of toFlat(lr.logProb(x)).entries()) expect(v).toBeCloseTo(toFlat(full.logProb(x))[i] ?? 0, 12)
    expect(item(lr.entropy())).toBeCloseTo(item(full.entropy()), 12)
    const s = lr.rsample(key(2), [N]) as Tensor
    for (const [i, v] of empiricalCov(s).entries()) expect(v).toBeCloseTo(toFlat(lr.covariance())[i] ?? 0, 1)
  })

  it('the matrix normal is the normal of the vectorised matrix with covariance V kron U', () => {
    const u = tensor([
      [1, 0.4],
      [0.4, 2],
    ])
    const v = tensor([[0.5]])
    const m = D.matrixNormal([[1], [-1]], u, v)
    const vec = D.multivariateNormal([1, -1], { covariance: mul(u, 0.5) })
    expect(item(m.logProb([[0.3], [0.2]]))).toBeCloseTo(item(vec.logProb([0.3, 0.2])), 12)
    const s = m.rsample(key(3), [N]) as Tensor
    expect(s.shape).toEqual([N, 2, 1])
    expect(toFlat(mean(s, 0))[0]).toBeCloseTo(1, 1)
  })

  it('conditioning and marginalising agree with the joint', () => {
    const d = D.multivariateNormal([1, -1, 0.5], {
      covariance: [
        [2, 0.6, 0.2],
        [0.6, 1, -0.3],
        [0.2, -0.3, 0.5],
      ],
    })
    // p(a, b) = p(b) p(a | b).
    const x = tensor([0.4, -0.3, 0.9])
    const joint = item(d.logProb(x))
    const split = item(d.marginal([2]).logProb([0.9])) + item(d.condition([2], [0.9]).logProb([0.4, -0.3]))
    expect(split).toBeCloseTo(joint, 12)
    const p = d.precision()
    expect(toFlat(p)[0]).toBeCloseTo(toFlat(D.multivariateNormal([0, 0, 0], { precision: p }).precision())[0] ?? 0, 10)
  })

  it('low-rank log densities differentiate in the factor and the diagonal', () => {
    const x = tensor([0.2, -0.4, 1])
    const f = (w: Value, dg: Value): Value => D.lowRankMultivariateNormal([0, 0, 0], w, dg).logProb(x)
    const report = checkPrimitive(f, [tensor([[1], [0.5], [-0.3]]), tensor([0.7, 1.2, 0.4])], {
      rtol: 1e-5,
      atol: 1e-8,
    })
    expect(report.failures).toEqual([])
  })
})

describe('simplex families', () => {
  it('Dirichlet-multinomial samples count n draws and match the mean', () => {
    const d = D.dirichletMultinomial(10, [1.5, 0.7, 3])
    const x = d.sample(key(4), [5000]) as Tensor
    expect(toFlat(sum(x, -1)).every((v) => v === 10)).toBe(true)
    for (const [j, v] of toFlat(mean(x, 0)).entries()) expect(v).toBeCloseTo(toFlat(d.mean())[j] ?? 0, 0)
  })

  it('the logistic normal with two categories is the logit-normal', () => {
    const d = D.logisticNormal([0.3], { scale: [0.8] })
    const ln = D.logitNormal(0.3, 0.8)
    // Density of (x, 1 - x): the logit-normal density of x (the simplex coordinate measure is dx).
    expect(item(d.logProb([0.6, 0.4]))).toBeCloseTo(item(ln.logProb(0.6)), 12)
    expect(toFlat(d.mean())[0]).toBeCloseTo(item(ln.mean()), 8)
    expect(item(d.entropy())).toBeCloseTo(item(ln.entropy()), 8)
  })

  it('logistic-normal samples lie on the simplex and match the quadrature mean', () => {
    const d = D.logisticNormal([0.3, -0.5], {
      covariance: [
        [0.5, 0.1],
        [0.1, 0.3],
      ],
    })
    const x = d.rsample(key(5), [N]) as Tensor
    expect(toFlat(sum(x, -1)).every((v) => Math.abs(v - 1) < 1e-12)).toBe(true)
    for (const [j, v] of toFlat(mean(x, 0)).entries()) expect(v).toBeCloseTo(toFlat(d.mean())[j] ?? 0, 2)
  })
})
