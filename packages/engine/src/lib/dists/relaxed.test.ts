import { describe, expect, it } from 'vitest'
import { jvp } from '../autodiff/index.ts'
import { key } from '../random/index.ts'
import { item, mean, sum, type Tensor, tensor, toFlat, type Value } from '../tensor/index.ts'
import { integrate } from './_numeric.ts'
import * as D from './index.ts'

const N = 10000

describe('relaxed distributions', () => {
  it('the relaxed Bernoulli density integrates to one and its draws follow its CDF', () => {
    const d = D.relaxedBernoulli(0.7, { logits: -0.5 })
    expect(item(integrate((x) => d.prob(x), { kind: 'finite', low: 0, high: 1 }, 0))).toBeCloseTo(1, 8)
    const x = d.rsample(key(1), [N]) as Tensor
    const below = toFlat(x).filter((v) => v < 0.3).length / N
    expect(below).toBeCloseTo(item(d.cdf(0.3)), 1)
    expect(item(mean(x))).toBeCloseTo(item(d.mean()), 2)
  })

  it('relaxed Bernoulli draws differentiate in the logits', () => {
    // P(X > 1/2) = sigmoid(logits), and E[X] increases with the logits.
    const g = jvp(
      (l: Value) => mean(D.relaxedBernoulli(0.5, { logits: l }).rsample(key(2), [N])),
      tensor(0.2),
      tensor(1),
    )
    expect(item(g.tangent)).toBeGreaterThan(0)
  })

  it('Gumbel-softmax draws lie on the simplex and concentrate at low temperature', () => {
    const d = D.relaxedOneHotCategorical(0.05, { probs: [0.2, 0.5, 0.3] })
    const x = d.rsample(key(3), [N]) as Tensor
    expect(toFlat(sum(x, -1)).every((v) => Math.abs(v - 1) < 1e-12)).toBe(true)
    // Near-one-hot draws whose category frequencies are the probabilities.
    for (const [j, v] of toFlat(mean(x, 0)).entries()) expect(v).toBeCloseTo([0.2, 0.5, 0.3][j] ?? 0, 1)
    expect(() => d.mean()).toThrow(/no closed form/)
  })

  it('straight-through samples are hard with relaxed gradients', () => {
    const st = (l: Value): D.Distribution => D.straightThrough(D.relaxedOneHotCategorical(0.5, { logits: l }))
    const logits = tensor([0.1, 0.5, -0.3])
    const x = st(logits).rsample(key(4), [8]) as Tensor
    expect(toFlat(x).every((v) => v === 0 || v === 1)).toBe(true)
    expect(toFlat(sum(x, -1)).every((v) => v === 1)).toBe(true)
    const g = jvp((l: Value) => sum(st(l).rsample(key(4), [8])), logits, tensor([1, 0, 0])).tangent
    // The relaxed sums are constant (1 per draw), so the gradient of the total vanishes.
    expect(Math.abs(item(g))).toBeLessThan(1e-12)
    const b = D.straightThrough(D.bernoulli({ probs: 0.3 }))
    const gb = jvp(
      (p: Value) => mean(D.straightThrough(D.bernoulli({ probs: p })).rsample(key(5), [10])),
      tensor(0.3),
      tensor(1),
    ).tangent
    expect(item(gb)).toBeCloseTo(1, 12)
    expect(b.hasRsample).toBe(true)
    expect(() => D.straightThrough(D.normal(0, 1))).toThrow(/straight-through/)
  })
})
