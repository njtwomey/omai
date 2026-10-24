import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { quad } from '../numerics/quadrature/index.ts'
import { exp, item, log, mul, neg, onesLike, sqrt, tensor, type Value } from '../tensor/index.ts'
import * as F from './index.ts'

const weights = fc.array(fc.double({ min: 0.01, max: 5, noNaN: true }), { minLength: 2, maxLength: 6 })

describe('finite spaces and events', () => {
  it('validates weights and normalises them', () => {
    expect(() => F.probabilitySpace([1, 2], [1])).toThrow()
    expect(() => F.probabilitySpace([1, 2], [1, -1])).toThrow()
    expect(() => F.probabilitySpace([1, 2], [0, 0])).toThrow()
    const s = F.probabilitySpace(['a', 'b', 'c'], [1, 2, 1])
    expect(F.eventProbability(s, () => true)).toBeCloseTo(1, 15)
    expect(F.conditionSpace(s, (o) => o !== 'b').outcomes).toEqual(['a', 'c'])
    expect(() =>
      F.conditionalProbability(
        s,
        () => true,
        () => false,
      ),
    ).toThrow()
  })

  it('the law of total probability and Bayes agree with conditioning', () => {
    fc.assert(
      fc.property(weights, (w) => {
        const space = F.probabilitySpace(
          w.map((_, i) => i),
          w,
        )
        const a = (o: number): boolean => o % 2 === 0
        const parts = [(o: number) => o < 2, (o: number) => o >= 2]
        const total = parts.reduce((s, b) => {
          const pb = F.eventProbability(space, b)
          return pb > 0 ? s + F.conditionalProbability(space, a, b) * pb : s
        }, 0)
        expect(total).toBeCloseTo(F.eventProbability(space, a), 12)
        const prior = parts.map((b) => F.eventProbability(space, b))
        const likelihood = parts.map((b, i) => ((prior[i] ?? 0) > 0 ? F.conditionalProbability(space, a, b) : 0))
        if (F.eventProbability(space, a) === 0) return
        const post = F.bayesRule(prior, likelihood).posterior
        expect(post[0]).toBeCloseTo(F.conditionalProbability(space, parts[0] as (o: number) => boolean, a), 12)
      }),
    )
    expect(() => F.bayesRule([0.5, 0.5], [0, 0])).toThrow()
    expect(() => F.bayesRule([1], [0.5, 0.5])).toThrow()
    expect(() => F.diagnosticTest({ prevalence: 2, sensitivity: 0.9, specificity: 0.9 })).toThrow()
  })

  it('independence of product experiments, and pairwise without mutual', () => {
    const coins = F.powerSpace(F.probabilitySpace(['H', 'T']), 3)
    const events = [0, 1, 2].map((k) => (o: string[]) => o[k] === 'H')
    expect(F.mutuallyIndependent(coins, events)).toBe(true)
    const two = F.powerSpace(F.probabilitySpace(['H', 'T']), 2)
    const agree = (o: string[]): boolean => o[0] === o[1]
    expect(F.independent(two, (o) => o[0] === 'H', agree).independent).toBe(true)
    expect(F.independent(two, (o) => o[1] === 'H', agree).independent).toBe(true)
    expect(() =>
      F.mutuallyIndependent(
        coins,
        Array.from({ length: 17 }, () => () => true),
      ),
    ).toThrow()
  })
})

describe('random variables', () => {
  it('expectation is linear and variances of independent variables add', () => {
    const dice = F.powerSpace(F.probabilitySpace([1, 2, 3, 4, 5, 6]), 2)
    const x = (o: number[]): number => o[0] ?? 0
    const y = (o: number[]): number => o[1] ?? 0
    expect(F.expectation(dice, (o) => 2 * x(o) - 3 * y(o) + 1)).toBeCloseTo(2 * 3.5 - 3 * 3.5 + 1, 12)
    expect(F.varianceOf(dice, (o) => x(o) + y(o))).toBeCloseTo(2 * F.varianceOf(dice, x), 12)
    expect(F.covarianceOf(dice, x, y)).toBeCloseTo(0, 14)
    const pmf = F.distributionOf(dice, (o) => x(o) + y(o))
    expect(F.expectation(pmf)).toBeCloseTo(7, 12)
    expect(F.varianceOf(pmf)).toBeCloseTo(35 / 6, 12)
    expect(F.expectation(F.probabilitySpace([2, 4]))).toBe(3)
    expect(F.varianceOf(F.probabilitySpace([2, 4]))).toBe(1)
  })

  it('transformations keep total mass', () => {
    const p = F.diceSum(2)
    const t = F.transformPmf(p, (v) => v % 3)
    expect(t.values).toEqual([0, 1, 2])
    expect(t.probabilities.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 14)
    // Y = X^2 for X ~ U(0, 1) has density 1 / (2 sqrt y) on (0, 1].
    const f = F.transformDensity(
      (x: Value) => onesLike(x),
      (y: Value) => sqrt(y),
    )
    expect(item(f([0.25]))).toBeCloseTo(1, 14)
    const lognormal = F.transformDensity(
      (x: Value) => exp(neg(mul(mul(x, x), 0.5))),
      (y: Value) => log(y),
    )
    expect(item(quad((y) => lognormal(y), 0, Infinity).value) / Math.sqrt(2 * Math.PI)).toBeCloseTo(1, 8)
  })

  it('sums by convolution are associative and integer-only', () => {
    const a: F.Pmf = { values: [-1, 2], probabilities: [0.5, 0.5] }
    const b: F.Pmf = { values: [0, 1, 5], probabilities: [0.2, 0.3, 0.5] }
    const c: F.Pmf = { values: [3], probabilities: [1] }
    const left = F.sumPmf(F.sumPmf(a, b), c)
    const right = F.sumPmf(a, F.sumPmf(b, c))
    expect(left.values).toEqual(right.values)
    for (const [i, p] of left.probabilities.entries()) expect(p).toBeCloseTo(right.probabilities[i] ?? 0, 15)
    expect(left.values[0]).toBe(2)
    expect(() => F.sumPmf({ values: [0.5], probabilities: [1] }, a)).toThrow()
    expect(F.diceSum(2).values).toEqual([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
  })
})

describe('derivatives', () => {
  it('transformed densities are differentiable in y', () => {
    const lognormal = F.transformDensity(
      (x: Value) => exp(neg(mul(mul(x, x), 0.5))),
      (y: Value) => log(y),
    )
    expectConformant((y) => lognormal(y), [tensor([0.5, 1.5, 3])])
  })
})
