import { describe, expect, it } from 'vitest'
import { checkPrimitive } from '../../autodiff/index.ts'
import { key } from '../../random/index.ts'
import { item, mean, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as D from '../index.ts'
import * as C from './index.ts'

describe('linear-Gaussian conditionals', () => {
  const prior = D.multivariateNormal([0.5, -1], {
    covariance: [
      [1, 0.3],
      [0.3, 0.8],
    ],
  })
  const obs = C.linearGaussian(
    [
      [1, 0.5],
      [0, 2],
      [1, -1],
    ],
    [0.1, 0, -0.2],
    {
      covariance: [
        [0.5, 0, 0],
        [0, 0.3, 0.1],
        [0, 0.1, 0.4],
      ],
    },
  )

  it('the posterior is the joint conditioned on y', () => {
    const y = [0.4, -1.5, 1.2]
    const viaJoint = obs.joint(prior).condition([2, 3, 4], y)
    const post = obs.posterior(prior, y)
    for (const [i, v] of toFlat(post.loc).entries()) expect(v).toBeCloseTo(toFlat(viaJoint.loc)[i] ?? 0, 12)
    for (const [i, v] of toFlat(post.covariance()).entries())
      expect(v).toBeCloseTo(toFlat(viaJoint.covariance())[i] ?? 0, 12)
    // Bayes: log p(x | y) = log p(y | x) + log p(x) - log p(y).
    const x = [0.2, -0.4]
    const lhs = item(post.logProb(x))
    const rhs = item(obs.logProb(y, x)) + item(prior.logProb(x)) - item(obs.marginal(prior).logProb(y))
    expect(lhs).toBeCloseTo(rhs, 10)
  })

  it('the marginal matches draws pushed through the conditional', () => {
    const xs = prior.sample(key(1), [20000]) as Tensor
    const ys = obs.sample(key(2), xs) as Tensor
    for (const [i, v] of toFlat(mean(ys, 0)).entries())
      expect(v).toBeCloseTo(toFlat(obs.marginal(prior).mean())[i] ?? 0, 1)
  })
})

describe('network heads', () => {
  it('turn unconstrained outputs into valid distributions, differentiably', () => {
    const raw = tensor([0.3, -0.2, 1.1, 0.5, -0.4, 0.2])
    const mdn = C.mixtureDensityFromOutput(raw, 2)
    const manual = C.mixture({ logits: [0.3, -0.2] }, [
      D.normal(1.1, Math.log1p(Math.exp(-0.4)) + 1e-6),
      D.normal(0.5, Math.log1p(Math.exp(0.2)) + 1e-6),
    ])
    expect(item(mdn.logProb(0.7))).toBeCloseTo(item(manual.logProb(0.7)), 12)
    const f = (r: Value): Value => C.mixtureDensityFromOutput(r, 2).logProb(tensor([0.7, -0.1]))
    expect(checkPrimitive(f, [raw], { rtol: 1e-5, atol: 1e-8 }).failures).toEqual([])
    const mvn = C.multivariateNormalFromOutput(tensor([[0, 0, 0.1, -0.5, 0.2]]), 2)
    expect(mvn.batchShape).toEqual([1])
    expect(item(C.normalFromOutput([0, 0]).stddev())).toBeCloseTo(Math.LN2, 5)
    expect(C.negativeBinomialFromOutput([1, 0]).support.kind).toBe('integerInterval')
    expect(() => C.normalFromOutput([1, 2, 3])).toThrow(/2 entries/)
  })

  it('distributional regressions give heteroscedastic conditionals', () => {
    const het = C.distributionalConditional(D.families.gamlssNormal(), [(x) => x, (x) => x])
    expect(item(het.condition(Math.log(3)).stddev())).toBeCloseTo(3, 12)
  })
})
