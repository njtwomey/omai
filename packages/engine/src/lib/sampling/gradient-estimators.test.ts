/**
 * Tests for the gradient estimators of an expectation with respect to distribution parameters, against closed forms:
 * `d/dmu, d/dsigma E[x^2] = 2 mu, 2 sigma` for a normal, the Bernoulli expectation, and a categorical payoff. They check
 * agreement within standard errors, that baselines keep the expectation (leave-one-out) or bias it by the known factor
 * (batch mean), that the pathwise estimator has much lower variance, the documented bias of straight-through, and the
 * Gumbel-softmax temperature trade-off.
 */

import { describe, expect, it } from 'vitest'
import { bernoulli, normal, poisson } from '../dists/index.ts'
import { foldIn, key } from '../random/index.ts'
import { add, mul, primalOf, square, sum, toFlat, type Value } from '../tensor/index.ts'
import {
  type GradientEstimate,
  gradientError,
  gumbelSoftmaxGradient,
  pathwiseGradient,
  scoreFunctionGradient,
  straightThroughGradient,
} from './index.ts'

const at = (e: GradientEstimate, name: string, i = 0): number => e.gradient[name]?.[i] ?? Number.NaN
const se = (e: GradientEstimate, name: string, i = 0): number => e.standardError[name]?.[i] ?? Number.NaN

const normalCase = {
  dist: (p: { loc: Value; scale: Value }) => normal(p.loc, p.scale),
  fn: (x: Value) => square(x),
  params: { loc: 1, scale: 1.5 },
}

describe('scoreFunctionGradient', () => {
  it('matches the exact gradient 2 mu, 2 sigma within its standard error, with every baseline', () => {
    for (const baseline of ['none', 'mean', 'leave-one-out', 3] as const) {
      const e = scoreFunctionGradient({ ...normalCase, key: key(1), n: 6000, baseline })
      // the batch-mean baseline carries a factor (1 - 1/n) bias, invisible at this n
      expect(Math.abs(at(e, 'loc') - 2)).toBeLessThan(5 * se(e, 'loc') + 2 / 6000)
      expect(Math.abs(at(e, 'scale') - 3)).toBeLessThan(5 * se(e, 'scale') + 3 / 6000)
    }
  })

  it('leave-one-out is unbiased and the batch-mean baseline is biased by (1 - 1/n), shown at n = 2', () => {
    const keys = Array.from({ length: 1200 }, (_, i) => key(i))
    const mean = (baseline: 'mean' | 'leave-one-out'): { value: number; err: number } => {
      const vals = keys.map((k) =>
        at(scoreFunctionGradient({ ...normalCase, params: { loc: 1, scale: 1 }, key: k, n: 2, baseline }), 'loc'),
      )
      const m = vals.reduce((a, b) => a + b, 0) / vals.length
      const sd = Math.sqrt(vals.reduce((a, b) => a + (b - m) ** 2, 0) / (vals.length - 1))
      return { value: m, err: sd / Math.sqrt(vals.length) }
    }
    const loo = mean('leave-one-out')
    const batch = mean('mean')
    expect(Math.abs(loo.value - 2)).toBeLessThan(5 * loo.err)
    expect(Math.abs(batch.value - 1)).toBeLessThan(5 * batch.err)
  })

  it('a baseline cuts the variance when f has a large offset', () => {
    const shifted = { ...normalCase, fn: (x: Value) => add(square(x), 100) }
    const plain = scoreFunctionGradient({ ...shifted, key: key(2), n: 2000 })
    const loo = scoreFunctionGradient({ ...shifted, key: key(2), n: 2000, baseline: 'leave-one-out' })
    expect(plain.variance['loc']?.[0] ?? 0).toBeGreaterThan(50 * (loo.variance['loc']?.[0] ?? 1))
    expect(Math.abs(at(loo, 'loc') - 2)).toBeLessThan(5 * se(loo, 'loc'))
  })

  it('works for a discrete distribution where nothing can be differentiated through', () => {
    // E f(x) for x ~ Bernoulli(p), f(x) = (x - 0.45)^2: derivative f(1) - f(0) = 0.1
    const e = scoreFunctionGradient({
      dist: (p: { probs: Value }) => bernoulli({ probs: p.probs }),
      fn: (x) => square(add(x, -0.45)),
      params: { probs: 0.3 },
      key: key(3),
      n: 4000,
      baseline: 'leave-one-out',
    })
    expect(Math.abs(at(e, 'probs') - 0.1)).toBeLessThan(5 * se(e, 'probs'))
  })

  it('reports what it did and validates', () => {
    const e = scoreFunctionGradient({ ...normalCase, key: key(0), n: 10, baseline: 2 })
    expect(e.baseline).toBe(2)
    expect(e.n).toBe(10)
    expect(e.gradient['loc']?.length).toBe(1)
    expect(() => scoreFunctionGradient({ ...normalCase, key: key(0), n: 1 })).toThrow(/at least 2/)
    expect(() => scoreFunctionGradient({ ...normalCase, key: key(0), n: 5, baseline: 'bogus' as never })).toThrow(
      /baseline/,
    )
  })
})

describe('pathwiseGradient', () => {
  it('matches the exact gradient and has far lower variance than the score function', () => {
    const path = pathwiseGradient({ ...normalCase, key: key(4), n: 4000 })
    const score = scoreFunctionGradient({ ...normalCase, key: key(4), n: 4000 })
    expect(Math.abs(at(path, 'loc') - 2)).toBeLessThan(5 * se(path, 'loc'))
    expect(Math.abs(at(path, 'scale') - 3)).toBeLessThan(5 * se(path, 'scale'))
    // Var[2x] = 4 sigma^2 for loc; the score estimator's is several times larger
    expect(path.variance['loc']?.[0] ?? 0).toBeCloseTo(4 * 1.5 ** 2, 0)
    expect(score.variance['loc']?.[0] ?? 0).toBeGreaterThan(3 * (path.variance['loc']?.[0] ?? 1))
    expect(score.variance['scale']?.[0] ?? 0).toBeGreaterThan(3 * (path.variance['scale']?.[0] ?? 1))
  })

  it('handles vector parameters and is reproducible from the key', () => {
    const fn = (x: Value): Value => sum(mul(x, [1, 2]))
    const a = pathwiseGradient({
      dist: (p: { loc: Value }) => normal(p.loc, 1),
      fn,
      params: { loc: [0.5, -0.5] },
      key: key(5),
      n: 50,
    })
    const b = pathwiseGradient({
      dist: (p: { loc: Value }) => normal(p.loc, 1),
      fn,
      params: { loc: [0.5, -0.5] },
      key: key(5),
      n: 50,
    })
    expect(Array.from(a.gradient['loc'] ?? [])).toEqual([1, 2])
    expect(Array.from(b.gradient['loc'] ?? [])).toEqual([1, 2])
    expect(a.method).toBe('pathwise')
  })

  it('refuses a distribution without a reparameterisation', () => {
    expect(() =>
      pathwiseGradient({
        dist: (p: { rate: Value }) => poisson(p.rate),
        fn: (x) => x,
        params: { rate: 3 },
        key: key(0),
        n: 3,
      }),
    ).toThrow()
  })
})

describe('straightThroughGradient', () => {
  const bern = {
    dist: (p: { probs: Value }) => bernoulli({ probs: p.probs }),
    fn: (x: Value) => square(add(x, -0.45)),
    params: { probs: 0.3 },
  }

  it("is biased: it returns f'(x) averaged over the hard sample, 2p - 0.9, not the true 0.1", () => {
    const st = straightThroughGradient({ ...bern, key: key(6), n: 4000 })
    const score = scoreFunctionGradient({ ...bern, key: key(6), n: 4000 })
    expect(Math.abs(at(st, 'probs') - (2 * 0.3 - 0.9))).toBeLessThan(5 * se(st, 'probs'))
    expect(Math.abs(at(st, 'probs') - 0.1)).toBeGreaterThan(0.3)
    // its single-sample values are only the two slopes f'(1) and f'(0)
    expect(st.variance['probs']?.[0] ?? 0).toBeCloseTo(4 * 0.3 * 0.7, 0)
    expect(score.n).toBe(st.n)
    expect(st.method).toBe('straight-through')
  })

  it('is exact for a linear f, where the surrogate is the truth', () => {
    const e = straightThroughGradient({ ...bern, fn: (x) => mul(x, 3), key: key(7), n: 20 })
    expect(at(e, 'probs')).toBeCloseTo(3, 6)
  })

  it('rejects a distribution it has no surrogate for', () => {
    expect(() => straightThroughGradient({ ...normalCase, key: key(0), n: 3 })).toThrow(/straight-through/)
  })
})

describe('gumbelSoftmaxGradient', () => {
  const payoff = (x: Value): Value => sum(mul(x, [1, 2, 3]))
  // E f = sum_i p_i v_i, so d/d logits_j = p_j (v_j - vbar) = (-1/3, 0, 1/3) at uniform logits
  const exact = { logits: [-1 / 3, 0, 1 / 3] }

  it('points along the exact gradient and approaches it as the temperature falls', () => {
    const warm = gumbelSoftmaxGradient({ fn: payoff, logits: [0, 0, 0], temperature: 1, key: key(8), n: 3000 })
    const cold = gumbelSoftmaxGradient({ fn: payoff, logits: [0, 0, 0], temperature: 0.2, key: key(8), n: 3000 })
    expect(at(warm, 'logits', 2)).toBeGreaterThan(at(warm, 'logits', 0))
    expect(at(cold, 'logits', 2)).toBeGreaterThan(at(cold, 'logits', 0))
    expect(gradientError(cold, exact)).toBeLessThan(gradientError(warm, exact))
    expect(gradientError(cold, exact)).toBeLessThan(0.1)
    // the price of the lower temperature is variance
    expect(cold.variance['logits']?.[2] ?? 0).toBeGreaterThan(warm.variance['logits']?.[2] ?? 1)
  })

  it('with hard samples the forward value is exactly one-hot while the gradient is kept', () => {
    const seen: number[][] = []
    const e = gumbelSoftmaxGradient({
      fn: (x) => {
        seen.push(toFlat(primalOf(x)))
        return payoff(x)
      },
      logits: [0.2, -0.1, 0.4],
      temperature: 0.5,
      hard: true,
      key: key(9),
      n: 30,
    })
    expect(seen.length).toBeGreaterThanOrEqual(30)
    for (const v of seen) expect([...v].sort()).toEqual([0, 0, 1])
    expect(e.method).toBe('gumbel-softmax (hard)')
    expect(at(e, 'logits', 2)).toBeGreaterThan(at(e, 'logits', 0))
  })

  it('validates the temperature', () => {
    expect(() => gumbelSoftmaxGradient({ fn: payoff, logits: [0, 0], temperature: 0, key: key(0), n: 3 })).toThrow(
      /temperature/,
    )
  })
})

describe('gradientError', () => {
  it('is the Euclidean distance over all parameters', () => {
    const e: GradientEstimate = {
      method: 'x',
      n: 2,
      gradient: { a: Float64Array.of(1, 2), b: Float64Array.of(0) },
      variance: {},
      standardError: {},
    }
    expect(gradientError(e, { a: [1, 4], b: [0] })).toBe(2)
    expect(() => gradientError(e, { a: [1, 4] })).toThrow(/exact gradient/)
  })

  it('shares its key derivation with the samples: sample i is foldIn(key, i)', () => {
    const a = pathwiseGradient({ ...normalCase, key: key(1), n: 3 })
    expect(foldIn(key(1), 0).words[0]).not.toBe(foldIn(key(1), 1).words[0])
    expect(a.n).toBe(3)
  })
})
