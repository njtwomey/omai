import { describe, expect, it } from 'vitest'
import { checkPrimitive, jvp } from '../autodiff/index.ts'
import { inv } from '../numerics/linalg/index.ts'
import { key } from '../random/index.ts'
import {
  expandDims,
  item,
  mean,
  mul,
  sliceAxis,
  square,
  sub,
  sum,
  type Tensor,
  tensor,
  toFlat,
  tril,
  type Value,
} from '../tensor/index.ts'
import * as D from './index.ts'

const N = 20000

const continuous: readonly (readonly [string, D.Distribution])[] = [
  ['normal', D.normal(1.5, 2)],
  ['halfNormal', D.halfNormal(0.7)],
  ['logNormal', D.logNormal(0.2, 0.4)],
  ['truncatedNormal', D.truncatedNormal(0.5, 1, -0.5, 2)],
  ['gamma (alpha < 1)', D.gamma(0.4, 2)],
  ['gamma', D.gamma(3, 0.5)],
  ['exponential', D.exponential(3)],
  ['uniform', D.uniform(-2, 5)],
  ['beta', D.beta(2, 5)],
  ['beta (small shapes)', D.beta(0.3, 0.6)],
  ['studentT', D.studentT(6, 1, 2)],
  ['laplace', D.laplace(-1, 0.5)],
  ['logistic', D.logistic(2, 0.7)],
  ['pareto', D.pareto(5, 2)],
  ['gumbel', D.gumbel(1, 2)],
  ['inverseGamma', D.inverseGamma(6, 2)],
  ['chi2', D.chi2(3)],
  ['weibull', D.weibull(1.5, 2)],
]

describe('sampling', () => {
  for (const [name, d] of continuous) {
    it(`${name}: sample moments match the analytic ones`, () => {
      const x = d.rsample(key(7), [N])
      const m = item(mean(x))
      const v = item(mean(square(sub(x, m))))
      const sd = Math.sqrt(item(d.variance()))
      // Five standard errors for the mean; a loose band for the variance.
      expect(Math.abs(m - item(d.mean()))).toBeLessThan((5 * sd) / Math.sqrt(N))
      expect(Math.abs(v / item(d.variance()) - 1)).toBeLessThan(0.1)
      expect(d.hasRsample).toBe(true)
    })
  }

  it('is reproducible from the key and differs across keys', () => {
    const d = D.gamma(2, 1)
    expect(toFlat(d.sample(key(1), [5]))).toEqual(toFlat(d.sample(key(1), [5])))
    expect(toFlat(d.sample(key(1), [5]))).not.toEqual(toFlat(d.sample(key(2), [5])))
  })

  it('gives samples of shape [...sampleShape, ...batchShape]', () => {
    expect(D.normal(tensor([0, 1, 2]), 1).sample(key(0), [4, 2]).shape).toEqual([4, 2, 3])
    expect(
      D.categorical({
        logits: tensor([
          [0, 1],
          [2, 0],
          [1, 1],
        ]),
      }).sample(key(0), [5]).shape,
    ).toEqual([5, 3])
  })

  it('discrete distributions sample their probabilities and refuse rsample', () => {
    const b = D.bernoulli({ probs: 0.3 })
    expect(item(mean(b.sample(key(3), [N])))).toBeCloseTo(0.3, 1)
    const c = D.categorical({ probs: [0.1, 0.6, 0.3] })
    expect(item(mean(c.sample(key(4), [N])))).toBeCloseTo(item(c.mean()), 1)
    expect(b.hasRsample).toBe(false)
    expect(() => b.rsample(key(0))).toThrow(/relaxed/)
  })
})

describe('pathwise gradients', () => {
  it('of a location-scale sample are the noise', () => {
    const eps = sub(D.normal(0, 1).rsample(key(5), [3]), 0)
    const d = jvp((s: Value) => D.normal(0, s).rsample(key(5), [3]), tensor(2), tensor(1)).tangent
    expect(toFlat(d)).toEqual(toFlat(eps))
  })

  // Implicit reparameterisation: a sample moves with its parameter at a fixed CDF value u, so dx/dalpha equals the
  // derivative of the quantile function ppf(u; alpha) (here by central differences of the scipy-checked ppf).
  const quantileDerivative = (make: (p: number) => D.Distribution, p: number, x: Tensor): number[] => {
    const u = make(p).cdf(x)
    const h = 1e-5 * p
    return toFlat(sub(make(p + h).ppf(u), make(p - h).ppf(u))).map((v) => v / (2 * h))
  }

  for (const [name, make, p] of [
    ['gamma concentration', (a: number) => D.gamma(a, 1.5), 2.5],
    ['gamma concentration below 1', (a: number) => D.gamma(a, 1), 0.3],
    ['gamma rate', (r: number) => D.gamma(2, r), 0.7],
  ] as const) {
    it(`of ${name} samples follow the quantile function`, () => {
      const k = key(11)
      const x = make(p).sample(k, [64]) as Tensor
      const got = toFlat(jvp((v: Value) => make(v as never).rsample(k, [64]), tensor(p), tensor(1)).tangent)
      const want = quantileDerivative(make, p, x)
      got.forEach((g, i) => {
        expect(g).toBeCloseTo(want[i] ?? Number.NaN, 5)
      })
    })
  }

  // Beta samples are G1 / (G1 + G2): a different (equally valid) reparameterisation from the inverse-CDF map, so their
  // per-sample derivatives differ from the quantile function's; the gradients of expectations agree.
  it('of beta samples give unbiased gradients of expectations: d/da E[X] = b / (a + b)^2', () => {
    const [a, b] = [1.8, 3]
    const g = jvp((v: Value) => mean(D.beta(v, b).rsample(key(13), [N])), tensor(a), tensor(1)).tangent
    expect(item(g)).toBeCloseTo(b / (a + b) ** 2, 2)
    const gb = jvp((v: Value) => mean(D.beta(a, v).rsample(key(14), [N])), tensor(b), tensor(1)).tangent
    expect(item(gb)).toBeCloseTo(-a / (a + b) ** 2, 2)
  })

  it('give unbiased gradients of expectations: d/dalpha E[X] = 1 / rate for a gamma', () => {
    const rate = 2
    const g = jvp((a: Value) => mean(D.gamma(a, rate).rsample(key(12), [N])), tensor(3), tensor(1)).tangent
    expect(item(g)).toBeCloseTo(1 / rate, 2)
  })

  it('of log densities pass the conformance check', () => {
    const x = tensor([0.3, 1.2, 2.5])
    for (const f of [
      (a: Value, b: Value) => D.gamma(a, b).logProb(x),
      (a: Value, b: Value) => D.beta(a, b).logProb(tensor([0.2, 0.5, 0.9])),
      (a: Value, b: Value) => D.normal(a, b).logProb(x),
      (a: Value, b: Value) => D.logNormal(a, b).cdf(x),
      (a: Value, b: Value) => D.truncatedNormal(a, b, -1, 3).logProb(x),
    ]) {
      expect(checkPrimitive(f, [tensor(1.7), tensor(0.8)], { rtol: 1e-5, atol: 1e-8 }).failures).toEqual([])
    }
  })

  it('of categorical log-probabilities are softmax residuals', () => {
    const logits = tensor([0.5, -1, 2])
    const g = jvp((l: Value) => sum(D.categorical({ logits: l }).logProb(2)), logits, tensor([0, 0, 1])).tangent
    expect(item(g)).toBeCloseTo(1 - item(D.categorical({ logits }).prob(2)), 12)
  })
})

describe('more families', () => {
  it('discrete count families sample their means', () => {
    for (const d of [D.poisson(3.5), D.binomial(12, 0.3), D.geometric(0.4), D.negativeBinomial(2.5, 0.35)]) {
      const x = d.sample(key(21), [N])
      const sd = Math.sqrt(item(d.variance()))
      expect(Math.abs(item(mean(x)) - item(d.mean()))).toBeLessThan((5 * sd) / Math.sqrt(N))
      expect(d.hasRsample).toBe(false)
    }
  })

  it('Student t samples differentiate in the degrees of freedom: d/dnu Var = -2 / (nu - 2)^2', () => {
    // E[X^2] for t with scale 1 is nu / (nu - 2), so d/dnu E[X^2] = -2 / (nu - 2)^2.
    const nu = 6
    const g = jvp((v: Value) => mean(square(D.studentT(v).rsample(key(22), [N * 5]))), tensor(nu), tensor(1)).tangent
    expect(item(g)).toBeCloseTo(-2 / (nu - 2) ** 2, 1)
  })
})

describe('multivariate', () => {
  const cov = tensor([
    [2, 0.6, 0.2],
    [0.6, 1, -0.3],
    [0.2, -0.3, 0.5],
  ])
  const mvn = D.multivariateNormal([1, -1, 0.5], { covariance: cov })

  it('the multivariate normal samples its mean and covariance', () => {
    const x = mvn.rsample(key(30), [N]) as Tensor
    expect(x.shape).toEqual([N, 3])
    const m = toFlat(mean(x, 0))
    m.forEach((v, j) => {
      expect(v).toBeCloseTo([1, -1, 0.5][j] ?? 0, 1)
    })
    const centred = sub(x, mean(x, 0))
    const empirical = toFlat(mean(mul(expandDims(centred, -1), expandDims(centred, -2)), 0))
    empirical.forEach((v, j) => {
      expect(v).toBeCloseTo(toFlat(cov)[j] ?? 0, 1)
    })
  })

  it('the multivariate normal agrees across parameterisations and marginalises', () => {
    const viaPrecision = D.multivariateNormal([1, -1, 0.5], { precision: inv(cov) })
    expect(item(viaPrecision.logProb([0, 0, 0]))).toBeCloseTo(item(mvn.logProb([0, 0, 0])), 12)
    const m = mvn.marginal([0])
    expect(item(m.logProb([0.3]))).toBeCloseTo(item(D.normal(1, Math.sqrt(2)).logProb(0.3)), 12)
  })

  it('log densities of the multivariate normal differentiate in the mean and covariance factor', () => {
    const x = tensor([
      [0.2, -0.4, 1],
      [1, 0, 0],
    ])
    const f = (mu: Value, l: Value): Value => sum(D.multivariateNormal(mu, { scaleTril: tril(l) }).logProb(x))
    expect(
      checkPrimitive(
        f,
        [
          tensor([0.5, -1, 0.2]),
          tensor([
            [1.2, 0, 0],
            [0.3, 0.9, 0],
            [0.1, -0.2, 0.7],
          ]),
        ],
        { rtol: 1e-5, atol: 1e-8 },
      ).failures,
    ).toEqual([])
  })

  it('Dirichlet samples lie on the simplex, match their mean, and differentiate in the concentrations', () => {
    const d = D.dirichlet([0.8, 2, 5])
    const x = d.rsample(key(31), [N]) as Tensor
    expect(toFlat(sum(x, -1)).every((v) => Math.abs(v - 1) < 1e-12)).toBe(true)
    toFlat(mean(x, 0)).forEach((v, j) => {
      expect(v).toBeCloseTo(toFlat(d.mean())[j] ?? 0, 2)
    })
    // d/dalpha_0 E[X_0] = (alpha_total - alpha_0) / alpha_total^2.
    const g = jvp(
      (a: Value) => sliceAxis(mean(D.dirichlet(a).rsample(key(32), [N]), 0), 0, 0, 1),
      tensor([0.8, 2, 5]),
      tensor([1, 0, 0]),
    ).tangent
    expect(item(sum(g))).toBeCloseTo(7 / 7.8 ** 2, 2)
  })

  it('multinomial samples count n draws and match the mean', () => {
    const d = D.multinomial(20, [0.1, 0.6, 0.3])
    const x = d.sample(key(33), [4000]) as Tensor
    expect(toFlat(sum(x, -1)).every((v) => v === 20)).toBe(true)
    toFlat(mean(x, 0)).forEach((v, j) => {
      expect(v).toBeCloseTo([2, 12, 6][j] ?? 0, 0)
    })
  })
})
