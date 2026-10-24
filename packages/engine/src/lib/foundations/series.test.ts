import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { div, exp, mul, neg, sign, tensor, toFlat, type Value } from '../tensor/index.ts'
import * as F from './index.ts'

describe('sums', () => {
  it('compensated partial sums keep their digits', () => {
    const terms = Array.from({ length: 10000 }, () => 0.1)
    expect(F.partialSums(terms)[9999]).toBe(1000)
  })

  it('accelerates geometric, alternating and logarithmic series', () => {
    expect(F.seriesSum((n) => 0.5 ** n, 10).accelerated).toBeCloseTo(2, 12)
    expect(F.seriesSum((n) => (-1) ** n / (2 * n + 1), 20).accelerated).toBeCloseTo(Math.PI / 4, 12)
    const basel = F.seriesSum((n) => 1 / (n + 1) ** 2)
    expect(basel.method).toBe('richardson')
    expect(basel.accelerated).toBeCloseTo(Math.PI ** 2 / 6, 9)
    expect(basel.partial).toBeLessThan(basel.accelerated)
  })
})

describe('convergence tests', () => {
  it('agree with the textbook verdicts', () => {
    expect(F.ratioTest((n) => 1 / 2 ** n).verdict).toBe('converges')
    expect(F.ratioTest((n) => 1 / n).verdict).toBe('inconclusive')
    expect(F.rootTest((n) => ((n + 1) / n) ** (n * n)).verdict).toBe('diverges')
    expect(F.rootTest((n) => -n * Math.log(n), { logAbs: true }).verdict).toBe('converges')
    expect(F.nthTermTest((n) => (-1) ** n).verdict).toBe('diverges')
    expect(F.pSeriesTest(0.5).verdict).toBe('diverges')
    expect(
      F.limitComparisonTest(
        (n) => 1 / (2 * n + 3),
        (n) => 1 / n,
        false,
      ).verdict,
    ).toBe('diverges')
    expect(
      F.limitComparisonTest(
        (n) => 1 / n ** 3,
        (n) => 1 / n,
        false,
      ).verdict,
    ).toBe('inconclusive')
    expect(F.alternatingSeriesTest((n) => (-1) ** n * (1 + 1 / (n + 1))).verdict).toBe('inconclusive')
    expect(F.alternatingSeriesTest((n) => 1 / (n + 1)).verdict).toBe('inconclusive')
  })

  it('the alternating-series bound holds for every partial sum', () => {
    const a = (n: number): number => (-1) ** n / (n + 1)
    const s = F.partialSums(F.sequenceTerms(a, 50))
    for (const [i, v] of s.entries()) expect(Math.abs(v - Math.LN2)).toBeLessThanOrEqual(1 / (i + 2) + 1e-15)
    expect(F.alternatingSeriesTest(a, 50).statistic).toBeCloseTo(1 / 51, 15)
  })

  it('the integral test compares with the tail integral', () => {
    const r = F.integralTest((x: Value) => exp(neg(x)), 0)
    expect(r.verdict).toBe('converges')
    expect(r.statistic).toBeCloseTo(1, 12)
  })

  it('radius of convergence of geometric-type coefficients', () => {
    fc.assert(
      fc.property(fc.double({ min: 0.2, max: 5, noNaN: true }), fc.integer({ min: 0, max: 3 }), (r, p) => {
        // c_n = n^p / r^n has radius r.
        const est = F.radiusOfConvergence((n) => p * Math.log(n) - n * Math.log(r), { logAbs: true })
        expect(est.root).toBeCloseTo(r, 6)
        expect(est.ratio).toBeCloseTo(r, 6)
      }),
      { numRuns: 30 },
    )
    expect(F.radiusOfConvergence((n) => -n * Math.log(n), { logAbs: true }).root).toBeGreaterThan(100)
  })
})

describe('Fourier series and the Gibbs phenomenon', () => {
  it('reproduces a smooth periodic function and converges in the mean at a jump', () => {
    const c = F.fourierCoefficients((x: Value) => exp(mul(neg(x), x)), 2 * Math.PI, 30)
    const xs = [-2, -0.5, 0, 1.3]
    const approx = toFlat(F.fourierPartialSum(c, xs))
    for (const [i, x] of xs.entries()) expect(approx[i]).toBeCloseTo(Math.exp(-x * x), 5)
    const sq = F.fourierCoefficients(sign, 2 * Math.PI, 9, { points: [0] })
    const closed = F.squareWaveCoefficients(9)
    for (const [k, b] of sq.b.entries()) expect(b).toBeCloseTo(closed.b[k] ?? 0, 10)
    // The partial sum is 0 at the jump, the average of the one-sided limits.
    expect(toFlat(F.fourierPartialSum(closed, [0]))[0]).toBeCloseTo(0, 15)
  })

  it('the overshoot does not vanish as N grows', () => {
    const fractions = [11, 41, 161].map((n) =>
      F.gibbsOvershoot(F.squareWaveCoefficients(n), n, { at: 0, left: -1, right: 1 }),
    )
    for (const g of fractions) expect(g.fraction).toBeCloseTo(0.0895, 2)
    expect(fractions[2]?.x ?? 1).toBeLessThan(fractions[0]?.x ?? 0)
    expect(() => F.gibbsOvershoot(F.squareWaveCoefficients(1), 0, { at: 0, left: -1, right: 1 })).toThrow()
  })
})

describe('generating functions', () => {
  it('a recurrence, its generating function and its series agree', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: -3, max: 3 }), { minLength: 1, maxLength: 3 }),
        fc.array(fc.integer({ min: -3, max: 3 }), { minLength: 3, maxLength: 3 }),
        (cs, init) => {
          const initial = init.slice(0, cs.length)
          const terms = F.linearRecurrence(cs, initial, 10)
          const { numerator, denominator } = F.recurrenceGeneratingFunction(cs, initial)
          expect(F.rationalSeries(numerator, denominator, 10)).toEqual(terms)
        },
      ),
    )
    expect(() => F.rationalSeries([1], [0, 1], 3)).toThrow()
    expect(() => F.linearRecurrence([1, 1], [1], 3)).toThrow()
    expect(F.seriesProduct([1, 1], [1, -1])).toEqual([1, 0, -1])
  })
})

describe('derivatives', () => {
  it('Fourier partial sums are differentiable in x', () => {
    const c = F.squareWaveCoefficients(7)
    expectConformant((x) => F.fourierPartialSum(c, x), [tensor([0.3, 1.7, -2])])
    expectConformant((x) => div(F.fourierPartialSum(c, x), 2), [tensor([0.1])])
  })
})
