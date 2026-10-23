import { describe, expect, it } from 'vitest'
import { checkPrimitive, jvp } from '../autodiff/index.ts'
import { key } from '../random/index.ts'
import { item, mean, square, sub, sum, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
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
