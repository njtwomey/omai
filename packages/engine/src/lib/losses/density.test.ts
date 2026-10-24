import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { grad } from '../autodiff/index.ts'
import { normal, weibull } from '../dists/index.ts'
import { exp, item, mul, tensor, toArray, type Value } from '../tensor/index.ts'
import * as L from './index.ts'

describe('likelihood losses', () => {
  it('differentiates the NLL of a distribution in its parameters', () => {
    expectConformant(
      (loc, scale) => L.negativeLogLikelihood(normal(loc, scale), [0.3, -1, 2]),
      [tensor([0.1, 0.2, -0.3]), tensor([1, 0.5, 2])],
    )
  })

  it('evaluates conditional distributions', () => {
    const w = tensor([0.5, -1])
    const model = (x: Value): ReturnType<typeof normal> => normal(x, exp(w))
    const direct = L.negativeLogLikelihood(normal([1, 2], exp(w)), [1.5, 1])
    expect(item(L.conditionalNll(model, [1, 2], [1.5, 1]))).toBe(item(direct))
  })

  it('reduces a one-component mixture to the Gaussian NLL', () => {
    const mdn = L.mixtureDensityNll([[0], [0]], [[0.5], [1]], [[2], [0.5]], [1, -1])
    const gaussian = L.negativeLogLikelihood(normal([0.5, 1], [2, 0.5]), [1, -1])
    expect(item(mdn)).toBeCloseTo(item(gaussian), 14)
    expectConformant(
      (z, m, s) => L.mixtureDensityNll(z, m, s, [0.3, -0.5]),
      [
        tensor([
          [0.2, -0.4],
          [1, 0.3],
        ]),
        tensor([
          [0, 1],
          [-1, 0.5],
        ]),
        tensor([
          [1, 0.7],
          [0.4, 1.3],
        ]),
      ],
    )
  })

  it('stops the beta-NLL weight from carrying gradient', () => {
    const m = tensor([0, 0.5])
    const s = tensor([0.2, -0.3])
    expectConformant((mm) => L.heteroscedasticGaussianNll(mm, s, [0.3, 1], { beta: 0.5 }), [m])
    // The gradient in s is the plain NLL's gradient scaled by the (constant) weight variance^beta.
    const weighted = grad((ss: Value) => L.heteroscedasticGaussianNll(m, ss, [0.3, 1], { beta: 0.5 }))(s)
    const plainGrad = grad((ss: Value) => L.heteroscedasticGaussianNll(m, ss, [0.3, 1]))(s)
    expect(toArray(weighted)).toEqual(
      (toArray(mul(plainGrad, exp(mul(s, 0.5)))) as number[]).map((v) => expect.closeTo(v, 14)),
    )
    // At beta = 0 the loss equals the Gaussian NLL with variance exp(s).
    const plain = L.heteroscedasticGaussianNll([0, 1], [0.2, -0.3], [0.3, 1], { full: true })
    expect(item(plain)).toBeCloseTo(item(L.gaussianNll([0, 1], [0.3, 1], exp([0.2, -0.3]), { full: true, eps: 0 })), 14)
  })

  it('handles every kind of censoring, with finite gradients', () => {
    const lo = [0.3, 0.5, -Infinity, 0.4, 3]
    const hi = [0.3, Infinity, 0.2, 0.9, 4]
    expectConformant(
      (k, s) => L.censoredNll(weibull(k, s), [0.3, 0.5, 0.01, 0.4, 3], [0.3, Infinity, 0.2, 0.9, 4]),
      [tensor([1.5, 0.8, 2, 1.2, 3]), tensor([1, 2, 0.5, 1, 3])],
    )
    expectConformant(
      (loc, s) => L.censoredNll(normal(loc, s), lo, hi),
      [tensor([0, 1, -1, 0.5, 3.5]), tensor([1, 1, 2, 0.5, 1])],
    )
    // An interval far in the upper tail keeps its accuracy through the survival function.
    const tail = item(L.censoredNll(normal(0, 1), [8], [9]))
    expect(tail).toBeCloseTo(35.01361859343716, 10)
    expect(() => L.censoredNll(normal(0, 1), [-Infinity], [Infinity])).toThrow()
    expect(() => L.censoredNll(normal(0, 1), [2], [1])).toThrow()
  })
})
