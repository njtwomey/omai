import { describe, expect, it } from 'vitest'
import { jvp } from '../../autodiff/index.ts'
import { key } from '../../random/index.ts'
import { item, mean, sum, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import * as D from '../index.ts'
import * as C from './index.ts'

const N = 20000

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
