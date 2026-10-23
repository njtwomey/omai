import { describe, expect, it } from 'vitest'
import { jvp } from '../../autodiff/index.ts'
import { key } from '../../random/index.ts'
import { item, mean, sum, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as D from '../index.ts'

const N = 20000

describe('discrete sampling', () => {
  const families: readonly (readonly [string, D.Distribution])[] = [
    ['poisson', D.poisson(3.5)],
    ['binomial', D.binomial(12, 0.3)],
    ['betaBinomial', D.betaBinomial(10, 2, 3)],
    ['geometric', D.geometric(0.4)],
    ['negativeBinomial', D.negativeBinomial(2.5, 0.35)],
    ['hypergeometric', D.hypergeometric(20, 7, 12)],
    ['discreteUniform', D.discreteUniform(-3, 4)],
    ['zeroInflatedPoisson', D.zeroInflatedPoisson(4, 0.3)],
    ['zeroInflatedNegativeBinomial', D.zeroInflatedNegativeBinomial(3, 0.4, 0.25)],
    ['orderedLogistic', D.orderedLogistic(0.4, [-1, 0.3, 1.5])],
  ]
  for (const [name, d] of families) {
    it(`${name} samples its mean and has no reparameterisation`, () => {
      const x = d.sample(key(21), [N])
      const sd = Math.sqrt(item(d.variance()))
      expect(Math.abs(item(mean(x)) - item(d.mean()))).toBeLessThan((5 * sd) / Math.sqrt(N))
      expect(d.hasRsample).toBe(false)
      expect(() => d.rsample(key(0))).toThrow(/relaxed/)
    })
  }

  it('one-hot categorical samples are one-hot with the category frequencies', () => {
    const d = D.oneHotCategorical({ probs: [0.1, 0.6, 0.3] })
    const x = d.sample(key(22), [N]) as Tensor
    expect(x.shape).toEqual([N, 3])
    expect(toFlat(sum(x, -1)).every((v) => v === 1)).toBe(true)
    for (const [j, v] of toFlat(mean(x, 0)).entries()) expect(v).toBeCloseTo([0.1, 0.6, 0.3][j] ?? 0, 1)
    expect(toFlat(d.mode())).toEqual([0, 1, 0])
  })
})

describe('discrete quantiles and moments', () => {
  it('quantiles invert the CDF on the integers', () => {
    const q = tensor([0.01, 0.3, 0.5, 0.9, 0.999])
    for (const d of [
      D.poisson(4),
      D.binomial(10, 0.3),
      D.betaBinomial(10, 2, 3),
      D.negativeBinomial(3, 0.4),
      D.hypergeometric(20, 7, 12),
      D.zeroInflatedPoisson(2, 0.3),
    ]) {
      const k = d.ppf(q)
      const at = toFlat(d.cdf(k))
      const below = toFlat(d.cdf(toFlat(k).map((v) => v - 1)))
      for (const [i, qi] of toFlat(q).entries()) {
        expect(at[i]).toBeGreaterThanOrEqual(qi)
        expect(below[i]).toBeLessThan(qi)
      }
    }
  })

  it('zero inflation moves mass to zero only', () => {
    const base = D.poisson(3)
    const d = D.zeroInflated(base, { logits: 0 })
    expect(item(d.prob(0))).toBeCloseTo(0.5 + 0.5 * Math.exp(-3), 14)
    expect(item(d.prob(4))).toBeCloseTo(0.5 * item(base.prob(4)), 14)
    const total = toFlat(d.prob(tensor(Array.from({ length: 40 }, (_, i) => i)))).reduce((a, b) => a + b, 0)
    expect(total).toBeCloseTo(1, 12)
    expect(item(d.mode())).toBe(0)
  })

  it('ordinal probabilities sum to one and move with the predictor', () => {
    const cut = tensor([-1, 0.3, 1.5])
    const d = D.orderedProbit(tensor([0.2, -0.5]), cut)
    expect(toFlat(sum(d.prob(tensor([[0], [1], [2], [3]])), 0))).toEqual([expect.closeTo(1, 14), expect.closeTo(1, 14)])
    // A larger predictor makes the top category likelier: d/deta P(Y = 3) = phi(1.5 - eta) > 0.
    const g = jvp((e: Value) => D.orderedProbit(e, cut).prob(3), tensor(0.2), tensor(1)).tangent
    expect(item(g)).toBeCloseTo(Math.exp(-0.5 * 1.3 ** 2) / Math.sqrt(2 * Math.PI), 10)
  })

  it('entropies sum the pmf', () => {
    const d = D.hypergeometric(20, 7, 12)
    const p = toFlat(d.prob(tensor(Array.from({ length: 13 }, (_, i) => i))))
    const h = -p.reduce((s, v) => s + (v > 0 ? v * Math.log(v) : 0), 0)
    expect(item(d.entropy())).toBeCloseTo(h, 12)
  })

  it('validates parameters', () => {
    expect(() => D.binomial(2.5, 0.3)).toThrow(/n must be nonNegativeInteger/)
    expect(() => D.poisson(-1)).toThrow(/rate/)
    expect(() => D.bernoulli({ probs: 1.5 })).toThrow(/probs/)
    expect(() => D.categorical({ probs: 0.5 })).toThrow(/category axis/)
  })
})
