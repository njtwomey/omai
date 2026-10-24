import { describe, expect, it } from 'vitest'
import { jvp } from '../../autodiff/index.ts'
import { key } from '../../random/index.ts'
import { item, mean, stack, sum, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as D from '../index.ts'
import * as C from './index.ts'

const N = 20000

describe('mixtures', () => {
  const m = C.mixture({ probs: [0.3, 0.7] }, [D.normal(0, 1), D.normal(3, 0.5)])

  it('samples match the mixture moments', () => {
    const x = m.sample(key(1), [N])
    const sd = Math.sqrt(item(m.variance()))
    expect(Math.abs(item(mean(x)) - item(m.mean()))).toBeLessThan((5 * sd) / Math.sqrt(N))
    expect(m.hasRsample).toBe(true)
  })

  it('reparameterised draws give unbiased gradients: d/dlogit E[X] for two components', () => {
    // With logits (0, l): w2 = sigmoid(l), dE/dl = w2 (1 - w2) (mu2 - mu1).
    const make = (l: Value): C.Mixture =>
      C.mixture({ logits: stack([tensor(0), l]) }, [D.normal(0, 1), D.normal(3, 0.5)])
    const g = jvp((l: Value) => mean(make(l).rsample(key(2), [N])), tensor(0.4), tensor(1)).tangent
    const w2 = 1 / (1 + Math.exp(-0.4))
    expect(item(g)).toBeCloseTo(w2 * (1 - w2) * 3, 1)
    // And in a component location: dE/dmu1 = w1.
    const h = jvp(
      (mu: Value) => mean(C.mixture({ probs: [0.3, 0.7] }, [D.normal(mu, 1), D.normal(3, 0.5)]).rsample(key(3), [N])),
      tensor(0),
      tensor(1),
    ).tangent
    expect(item(h)).toBeCloseTo(0.3, 1)
  })

  it('responsibilities are the posterior over components', () => {
    const r = toFlat(C.responsibilities(m, 0.5))
    const a = 0.3 * item(D.normal(0, 1).prob(0.5))
    const b = 0.7 * item(D.normal(3, 0.5).prob(0.5))
    expect(r[0]).toBeCloseTo(a / (a + b), 14)
    expect(item(sum(tensor(r)))).toBeCloseTo(1, 14)
  })

  it('the mode is the highest peak and quantiles invert the CDF', () => {
    // The higher peak sits just below 3 (the other component's tail pulls it); the score vanishes there.
    const mode = m.mode()
    expect(item(mode)).toBeCloseTo(3, 2)
    expect(Math.abs(item(jvp((x: Value) => m.logProb(x), mode, tensor(1)).tangent))).toBeLessThan(1e-8)
    expect(item(m.cdf(m.ppf(0.42)))).toBeCloseTo(0.42, 12)
  })

  it('same-family mixtures batch over the leading axes', () => {
    const comp = D.normal(
      tensor([
        [-1, 1],
        [0, 4],
      ]),
      1,
    )
    const mm = C.mixtureSameFamily({ probs: [0.5, 0.5] }, comp)
    expect(mm.batchShape).toEqual([2])
    expect(toFlat(mm.mean())).toEqual([0, 2])
    const x = mm.sample(key(4), [N]) as Tensor
    expect(x.shape).toEqual([N, 2])
    expect(toFlat(mean(x, 0))[1]).toBeCloseTo(2, 1)
    expect(mm.logProb([0, 0]).shape).toEqual([2])
  })
})
