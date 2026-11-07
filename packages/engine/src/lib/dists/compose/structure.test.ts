import { describe, expect, it } from 'vitest'
import { expectRows } from '../../../testing/close.ts'
import { jvp } from '../../autodiff/index.ts'
import { key } from '../../random/index.ts'
import * as T from '../../tensor/index.ts'
import { item, mean, type NestedArray, sum, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as D from '../index.ts'
import * as C from './index.ts'

const N = 10000

describe('independent', () => {
  it('sums log densities and entropies over reinterpreted axes', () => {
    const base = D.normal(tensor([[0, 1, 2]]), tensor([[1, 2, 0.5]]))
    const d = C.independent(base)
    expect(d.batchShape).toEqual([1])
    expect(d.eventShape).toEqual([3])
    const x = tensor([0.2, 0.4, 1.9])
    expect(item(d.logProb(x))).toBeCloseTo(item(sum(base.logProb(x))), 14)
    expect(item(d.entropy())).toBeCloseTo(item(sum(base.entropy())), 14)
    expect(d.rsample(key(1), [4]).shape).toEqual([4, 1, 3])
    expect(() => C.independent(base, 3)).toThrow(/reinterpret/)
  })
})

describe('transformed', () => {
  it('a stick-breaking flow of independent normals is a distribution on the simplex', () => {
    const base = C.independent(D.normal(tensor([0, 0]), tensor([1, 1])))
    const d = C.transformed(base, D.bijectors.stickBreaking())
    expect(d.eventShape).toEqual([3])
    const x = d.rsample(key(2), [5]) as Tensor
    expect(toFlat(sum(x, -1)).every((v) => Math.abs(v - 1) < 1e-12)).toBe(true)
    // log p(y) = log p_X(x) - log |det J| at the preimage.
    const y = tensor([0.2, 0.5, 0.3])
    const pre = D.bijectors.stickBreaking().inverse(y)
    const want = item(base.logProb(pre)) - item(D.bijectors.stickBreaking().forwardLogDetJacobian(pre))
    expect(item(d.logProb(y))).toBeCloseTo(want, 12)
  })

  it('a decreasing map swaps the tails', () => {
    const d = C.transformed(D.exponential(1), D.bijectors.affine(0, -1))
    expect(item(d.cdf(-1))).toBeCloseTo(Math.exp(-1), 14)
    expect(item(d.ppf(Math.exp(-1)))).toBeCloseTo(-1, 12)
  })
})

describe('truncated', () => {
  it('samples stay in range and differentiate as the quantile function', () => {
    const make = (m: Value): D.Distribution => C.truncated(D.normal(m, 1), { low: -0.5, high: 1.5 })
    const x = make(tensor(0.3)).rsample(key(3), [N]) as Tensor
    expect(toFlat(x).every((v) => v >= -0.5 && v <= 1.5)).toBe(true)
    expect(item(mean(x))).toBeCloseTo(item(make(tensor(0.3)).mean()), 2)
    const ref = D.truncatedNormal(0.3, 1, -0.5, 1.5)
    expect(item(make(tensor(0.3)).logProb(0.7))).toBeCloseTo(item(ref.logProb(0.7)), 12)
    // d/dmu E[X] for the truncated normal matches the closed form's derivative.
    const g = jvp((m: Value) => make(m).mean(), tensor(0.3), tensor(1)).tangent
    const h = jvp((m: Value) => D.truncatedNormal(m, 1, -0.5, 1.5).mean(), tensor(0.3), tensor(1)).tangent
    expect(item(g)).toBeCloseTo(item(h), 8)
  })

  it('discrete truncation renormalises on the integers and samples there', () => {
    const d = C.truncated(D.binomial(10, 0.3), { low: 2, high: 5 })
    const p = toFlat(d.prob(tensor([1, 2, 3, 4, 5, 6])))
    expect(p[0]).toBe(0)
    expect(p[5]).toBe(0)
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 14)
    const x = d.sample(key(4), [N])
    expect(Math.abs(item(mean(x)) - item(d.mean()))).toBeLessThan((5 * Math.sqrt(item(d.variance()))) / Math.sqrt(N))
    expect(item(d.mode())).toBe(3)
  })
})

describe('censored', () => {
  it('puts the tail mass at the bound', () => {
    const d = C.censored(D.normal(0, 1), { low: 0 })
    // E[max(X, 0)] = phi(0).
    expect(item(d.mean())).toBeCloseTo(1 / Math.sqrt(2 * Math.PI), 10)
    expect(item(d.logProb(-3))).toBeCloseTo(Math.log(0.5), 14)
    const x = d.rsample(key(5), [N]) as Tensor
    expect(toFlat(x).filter((v) => v === 0).length / N).toBeCloseTo(0.5, 1)
    expect(() => d.entropy()).toThrow(/mixed/)
  })
})

describe('empirical and deterministic', () => {
  it('empirical draws resample atoms and differentiate in them', () => {
    const atoms = tensor([1, 2, 2, 5])
    const d = C.empirical(atoms)
    const x = d.sample(key(6), [N]) as Tensor
    expect(toFlat(x).every((v) => [1, 2, 5].includes(v))).toBe(true)
    expect(item(mean(x))).toBeCloseTo(2.5, 1)
    expect(item(d.mode())).toBe(2)
    expect(item(d.entropy())).toBeCloseTo(-(0.25 * Math.log(0.25) * 2 + 0.5 * Math.log(0.5)), 14)
    // Each draw moves with the atom it picked: the mean draw moves at rate 1 under a common shift.
    const g = jvp((a: Value) => mean(C.empirical(a).rsample(key(7), [100])), atoms, tensor([1, 1, 1, 1])).tangent
    expect(item(g)).toBeCloseTo(1, 12)
  })

  it('weighted vector atoms', () => {
    const d = C.empirical(
      tensor([
        [0, 1],
        [1, 0],
      ]),
      tensor([1, 3]),
    )
    expect(item(d.prob([1, 0]))).toBeCloseTo(0.75, 14)
    expect(toFlat(d.mean())).toEqual([0.75, 0.25])
  })

  it('deterministic samples are the value, with gradients', () => {
    const g = jvp((v: Value) => sum(C.deterministic(v).rsample(key(8), [3])), tensor(2), tensor(1)).tangent
    expect(item(g)).toBe(3)
    expect(item(C.deterministic([1, 2], 1).logProb([1, 2]))).toBe(0)
    expect(item(C.deterministic([1, 2], 1).logProb([1, 3]))).toBe(Number.NEGATIVE_INFINITY)
  })
})

describe('the structure combinators against scipy', () => {
  // [name, value under test, reference] (scipy.stats truncnorm, lognorm, binom, norm, cauchy; moments and entropies of
  // the derived distributions by quadrature in scipy.integrate)
  const rows: [string, () => Value, NestedArray][] = [
    [
      'truncated normal on [-1, 2]: mean',
      () => C.truncated(D.normal(0.3, 1), { low: -1, high: 2 }).mean(),
      0.3900494389102818,
    ],
    [
      'truncated normal on [-1, 2]: variance',
      () => C.truncated(D.normal(0.3, 1), { low: -1, high: 2 }).variance(),
      0.5462267319696479,
    ],
    [
      'truncated normal on [-1, 2]: entropy',
      () => C.truncated(D.normal(0.3, 1), { low: -1, high: 2 }).entropy(),
      1.0436938866107242,
    ],
    [
      'truncated normal on [-1, 2]: cdf(0.5)',
      () => C.truncated(D.normal(0.3, 1), { low: -1, high: 2 }).cdf(0.5),
      0.5618915571321623,
    ],
    [
      'truncated normal on [-1, 2]: sf(0.5)',
      () => C.truncated(D.normal(0.3, 1), { low: -1, high: 2 }).sf(0.5),
      0.4381084428678378,
    ],
    [
      'truncated normal on [-1, 2]: ppf(0.3)',
      () => C.truncated(D.normal(0.3, 1), { low: -1, high: 2 }).ppf(0.3),
      -0.07349320623546357,
    ],
    [
      'truncated normal on [-1, 2]: isf(0.3)',
      () => C.truncated(D.normal(0.3, 1), { low: -1, high: 2 }).isf(0.3),
      0.8182105752696778,
    ],
    [
      'truncated normal on [-1, 2]: logProb(0.2)',
      () => C.truncated(D.normal(0.3, 1), { low: -1, high: 2 }).logProb(0.2),
      -0.7715260699018717,
    ],
    [
      'truncated normal on [-1, 2]: logProb outside',
      () => C.truncated(D.normal(0.3, 1), { low: -1, high: 2 }).logProb(3),
      -Infinity,
    ],
    [
      'truncated normal(5, 1) below 2 has its mode at the bound',
      () => C.truncated(D.normal(5, 1), { high: 2 }).mode(),
      2,
    ],
    ['truncated normal(5, 1) below 2: mean', () => C.truncated(D.normal(5, 1), { high: 2 }).mean(), 1.71690134506956],
    [
      'truncated normal(5, 1) below 2: variance',
      () => C.truncated(D.normal(5, 1), { high: 2 }).variance(),
      0.07055918678525586,
    ],
    [
      'truncated binomial: cdf at 1, 2, 3, 5, 9',
      () => C.truncated(D.binomial(10, 0.3), { low: 2, high: 5 }).cdf(T.tensor([1, 2, 3, 5, 9])),
      [0, 0.29062870699881405, 0.6227758007117439, 1, 1],
    ],
    [
      'truncated binomial: sf at 1, 3, 5',
      () => C.truncated(D.binomial(10, 0.3), { low: 2, high: 5 }).sf(T.tensor([1, 3, 5])),
      [1, 0.3772241992882561, 0],
    ],
    [
      'truncated binomial: ppf at 0.05, 0.4, 0.99',
      () => C.truncated(D.binomial(10, 0.3), { low: 2, high: 5 }).ppf(T.tensor([0.05, 0.4, 0.99])),
      [2, 3, 5],
    ],
    [
      'truncated binomial: mean, variance, entropy',
      () =>
        T.stack([
          C.truncated(D.binomial(10, 0.3), { low: 2, high: 5 }).mean(),
          C.truncated(D.binomial(10, 0.3), { low: 2, high: 5 }).variance(),
          C.truncated(D.binomial(10, 0.3), { low: 2, high: 5 }).entropy(),
        ]),
      [3.2147093712930004, 1.0060944291767104, 1.33469874394381],
    ],
    [
      'exp of a normal is the lognormal: mean',
      () => C.transformed(D.normal(0.5, 0.8), D.bijectors.exp()).mean(),
      2.270499837532406,
    ],
    [
      'exp of a normal is the lognormal: variance',
      () => C.transformed(D.normal(0.5, 0.8), D.bijectors.exp()).variance(),
      4.621510897294227,
    ],
    [
      'exp of a normal is the lognormal: entropy',
      () => C.transformed(D.normal(0.5, 0.8), D.bijectors.exp()).entropy(),
      1.695794981890463,
    ],
    [
      'exp of a normal is the lognormal: cdf(2)',
      () => C.transformed(D.normal(0.5, 0.8), D.bijectors.exp()).cdf(2),
      0.595390608679215,
    ],
    [
      'exp of a normal is the lognormal: sf(2)',
      () => C.transformed(D.normal(0.5, 0.8), D.bijectors.exp()).sf(2),
      0.404609391320785,
    ],
    [
      'exp of a normal is the lognormal: ppf(0.3)',
      () => C.transformed(D.normal(0.5, 0.8), D.bijectors.exp()).ppf(0.3),
      1.0838067257408528,
    ],
    [
      'exp of a normal is the lognormal: isf(0.3)',
      () => C.transformed(D.normal(0.5, 0.8), D.bijectors.exp()).isf(0.3),
      2.5080872483061243,
    ],
    [
      'exp of a normal is the lognormal: logProb(1.7)',
      () => C.transformed(D.normal(0.5, 0.8), D.bijectors.exp()).logProb(1.7),
      -1.2271561155800765,
    ],
    [
      'minus a lognormal: cdf, sf at -2 (the tails swap)',
      () =>
        T.stack([
          C.transformed(D.normal(0.5, 0.8), [D.bijectors.exp(), D.bijectors.affine(0, -1)]).cdf(-2),
          C.transformed(D.normal(0.5, 0.8), [D.bijectors.exp(), D.bijectors.affine(0, -1)]).sf(-2),
        ]),
      [0.404609391320785, 0.595390608679215],
    ],
    [
      'minus a lognormal: ppf and isf at 0.3',
      () =>
        T.stack([
          C.transformed(D.normal(0.5, 0.8), [D.bijectors.exp(), D.bijectors.affine(0, -1)]).ppf(0.3),
          C.transformed(D.normal(0.5, 0.8), [D.bijectors.exp(), D.bijectors.affine(0, -1)]).isf(0.3),
        ]),
      [-2.5080872483061243, -1.0838067257408528],
    ],
    [
      'minus a lognormal: mean, entropy',
      () =>
        T.stack([
          C.transformed(D.normal(0.5, 0.8), [D.bijectors.exp(), D.bijectors.affine(0, -1)]).mean(),
          C.transformed(D.normal(0.5, 0.8), [D.bijectors.exp(), D.bijectors.affine(0, -1)]).entropy(),
        ]),
      [-2.270499837532406, 1.695794981890463],
    ],
    [
      'censored normal: mean and variance',
      () =>
        T.stack([
          C.censored(D.normal(0, 1), { low: -1, high: 2 }).mean(),
          C.censored(D.normal(0, 1), { low: -1, high: 2 }).variance(),
        ]),
      [0.07482476797085663, 0.7126989923971256],
    ],
    [
      'censored normal: cdf at -2, -1, 0, 2, 3',
      () => C.censored(D.normal(0, 1), { low: -1, high: 2 }).cdf(T.tensor([-2, -1, 0, 2, 3])),
      [0, 0.15865525393145707, 0.5, 1, 1],
    ],
    [
      'censored normal: sf at -2, -1, 0, 2, 3',
      () => C.censored(D.normal(0, 1), { low: -1, high: 2 }).sf(T.tensor([-2, -1, 0, 2, 3])),
      [1, 0.8413447460685429, 0.5, 0, 0],
    ],
    [
      'censored normal: the atoms have the tail masses; inside it is the density',
      () => C.censored(D.normal(0, 1), { low: -1, high: 2 }).logProb(T.tensor([-5, 0.5, 4])),
      [-1.8410216450092634, -1.0439385332046727, -3.783184333682032],
    ],
    [
      'censored normal: ppf clips to the bounds',
      () => C.censored(D.normal(0, 1), { low: -1, high: 2 }).ppf(T.tensor([0.01, 0.5, 0.999])),
      [-1, 0, 2],
    ],
    [
      'weighted empirical: cdf at 0, 1, 2, 4, 5',
      () => C.empirical(T.tensor([1, 2, 2, 5]), T.tensor([1, 1, 1, 2])).cdf(T.tensor([0, 1, 2, 4, 5])),
      [0, 0.2, 0.6000000000000001, 0.6000000000000001, 1],
    ],
    [
      'weighted empirical: mean, variance',
      () =>
        T.stack([
          C.empirical(T.tensor([1, 2, 2, 5]), T.tensor([1, 1, 1, 2])).mean(),
          C.empirical(T.tensor([1, 2, 2, 5]), T.tensor([1, 1, 1, 2])).variance(),
        ]),
      [3, 2.8],
    ],
    [
      'weighted empirical: the mass of a repeated atom adds',
      () => C.empirical(T.tensor([1, 2, 2, 5]), T.tensor([1, 1, 1, 2])).logProb(T.tensor([2, 5, 1])),
      [-0.916290731874155, -0.916290731874155, -1.6094379124341003],
    ],
    [
      'weighted empirical: ppf at 0.1, 0.2, 0.5, 0.61, 1',
      () => C.empirical(T.tensor([1, 2, 2, 5]), T.tensor([1, 1, 1, 2])).ppf(T.tensor([0.1, 0.2, 0.5, 0.61, 1])),
      [1, 1, 2, 5, 5],
    ],
    [
      'weighted empirical: entropy',
      () => C.empirical(T.tensor([1, 2, 2, 5]), T.tensor([1, 1, 1, 2])).entropy(),
      1.0549201679861442,
    ],
    ['deterministic at 2: cdf at 1, 2, 3', () => C.deterministic(2).cdf(T.tensor([1, 2, 3])), [0, 1, 1]],
    [
      'deterministic at 2: ppf, mean, variance, mode',
      () =>
        T.stack([
          C.deterministic(2).ppf(0.3),
          C.deterministic(2).mean(),
          C.deterministic(2).variance(),
          C.deterministic(2).mode(),
        ]),
      [2, 2, 0, 2],
    ],
    [
      'deterministic vector: entropy 0 and the mean is the value',
      () => T.concat([C.deterministic([1, 2], 1).mean(), T.reshape(C.deterministic([1, 2], 1).entropy(), [1])]),
      [1, 2, 0],
    ],
    [
      "independent: the mean, variance and mode are the base's, the log density the sum",
      () => C.independent(D.normal(T.tensor([[0, 1]]), T.tensor([[1, 2]]))).logProb(T.tensor([0.5, 0.5])),
      [-2.6872742469692907],
    ],
  ]
  it('truncated, transformed, censored, empirical, deterministic and independent have the reference moments, tails and quantiles', () => {
    expectRows(rows, 1e-6)
  })
})
