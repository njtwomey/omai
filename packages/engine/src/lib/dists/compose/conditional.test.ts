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
  it('builds each family from a raw network output by the formulas of its link', () => {
    const softplus = (x: number): number => Math.log1p(Math.exp(x))
    const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x))
    const [a, b] = [0.4, -0.7]
    const rows: readonly (readonly [string, () => number, number])[] = [
      ['bernoulli: mean is sigmoid(logit)', () => item(C.bernoulliFromOutput(2).mean()), sigmoid(2)],
      [
        'categorical: probability is softmax',
        () => item(C.categoricalFromOutput([0, 1, 2]).prob(2)),
        Math.exp(2) / (1 + Math.E + Math.exp(2)),
      ],
      ['poisson: rate is exp(output)', () => item(C.poissonFromOutput(Math.log(3)).mean()), 3],
      [
        'negative binomial: mean is exp of the first entry',
        () => item(C.negativeBinomialFromOutput([Math.log(4), 0.3]).mean()),
        4,
      ],
      [
        'beta: mean is the share of the two softplus parameters',
        () => item(C.betaFromOutput([a, b]).mean()),
        (softplus(a) + 1e-6) / (softplus(a) + softplus(b) + 2e-6),
      ],
      [
        'gamma: mean is concentration over rate',
        () => item(C.gammaFromOutput([a, b]).mean()),
        (softplus(a) + 1e-6) / (softplus(b) + 1e-6),
      ],
      [
        'beta: variance',
        () => item(C.betaFromOutput([a, b]).variance()),
        (() => {
          const p = softplus(a) + 1e-6
          const q = softplus(b) + 1e-6
          return (p * q) / ((p + q) ** 2 * (p + q + 1))
        })(),
      ],
      [
        'gamma: variance is concentration over rate squared',
        () => item(C.gammaFromOutput([a, b]).variance()),
        (softplus(a) + 1e-6) / (softplus(b) + 1e-6) ** 2,
      ],
    ]
    for (const [name, value, expected] of rows) expect(value(), name).toBeCloseTo(expected, 10)
    // A batch of outputs gives a batch of distributions, and a wrong last axis is refused.
    expect(
      C.betaFromOutput(
        tensor([
          [a, b],
          [b, a],
          [a, a],
        ]),
      ).batchShape,
    ).toEqual([3])
    expect(
      C.gammaFromOutput(
        tensor([
          [a, b],
          [b, a],
        ]),
      ).batchShape,
    ).toEqual([2])
    expect(
      C.categoricalFromOutput(
        tensor([
          [0, 1, 2],
          [2, 1, 0],
        ]),
      ).batchShape,
    ).toEqual([2])
    for (const make of [C.betaFromOutput, C.gammaFromOutput, C.negativeBinomialFromOutput])
      expect(() => make([1, 2, 3]), make.name).toThrow(/last axis/)
  })
})
