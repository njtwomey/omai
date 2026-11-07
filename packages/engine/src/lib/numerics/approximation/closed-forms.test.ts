import { describe, expect, it } from 'vitest'
import { exp, item, log, mul, sin, sqrt, square, sum, toArray, toFlat, type Value } from '../../tensor/index.ts'
import { monicFromNodes, polysub } from '../polynomial/index.ts'
import { iv } from '../special/index.ts'
import * as A from './index.ts'

// Sequence acceleration, Pade approximants, continued fractions, minimax and Chebyshev approximation, and the
// floating-point helpers against closed forms (the exact value, an identity or a known constant), with the failure modes.
const arr = (v: unknown): number[] => toArray(v as never) as number[]
const PI = Math.PI
const partial = (term: (k: number) => number, n: number): number[] => {
  const out: number[] = []
  let s = 0
  for (let k = 0; k < n; k++) {
    s += term(k)
    out.push(s)
  }
  return out
}

describe('approximation against closed forms', () => {
  it('accelerate sequences, build Pade approximants and continued fractions, reach the minimax and Chebyshev results and satisfy the floating-point identities, as their closed forms say', () => {
    {
      const failures: string[] = []
      const check = (label: string, got: number, want: number, atol: number): void => {
        if (!(Math.abs(got - want) <= atol)) failures.push(`${label}: ${got}, expected ${want}`)
      }
      // Partial sums of a geometric series (limit 2) are exact after aitken and shanks of order 1; two geometric terms
      // (0.5^k + 0.2^k, limit 3.25) need order 2.
      const geometric = partial((k) => 0.5 ** k, 9)
      for (const [i, v] of arr(A.aitken(geometric)).entries()) check(`aitken geometric [${i}]`, v, 2, 1e-12)
      for (const [i, v] of arr(A.shanks(geometric, 1)).entries()) check(`shanks 1 [${i}]`, v, 2, 1e-12)
      const two = partial((k) => 0.5 ** k + 0.2 ** k, 9)
      for (const [i, v] of arr(A.shanks(two, 2)).entries()) check(`shanks 2 [${i}]`, v, 3.25, 1e-10)
      expect(arr(A.aitken([1, 1.5, 1.75]))).toEqual([2])
      // The Leibniz series for pi: 12 terms are 0.04 off; Aitken is 100 times closer, the epsilon algorithm 1e-8.
      const leibniz = partial((k) => (4 * (k % 2 ? -1 : 1)) / (2 * k + 1), 12)
      expect(Math.abs((arr(A.aitken(leibniz)).at(-1) ?? 0) - PI)).toBeLessThan(
        Math.abs((leibniz.at(-1) ?? 0) - PI) / 100,
      )
      check('wynnEpsilon pi', A.wynnEpsilon(leibniz).value, PI, 1e-7)
      check('wynnEpsilon constant', A.wynnEpsilon([1, 1, 1, 1]).value, 1, 1e-15)
      // Euler's transformation of the alternating harmonic series 1 - 1/2 + 1/3 - ... = ln 2.
      check('eulerTransform ln 2', A.eulerTransform(Array.from({ length: 20 }, (_, k) => 1 / (k + 1))), Math.LN2, 1e-7)
      // Richardson: Romberg on the trapezoid rule for int_0^1 exp (error h^2), and central differences of sin at 1.
      const trapezoid = [1, 2, 4, 8].map((n) => {
        let s = (Math.exp(0) + Math.exp(1)) / 2
        for (let i = 1; i < n; i++) s += Math.exp(i / n)
        return s / n
      })
      check('richardson trapezoid', A.richardson(trapezoid).value, Math.E - 1, 1e-7)
      const d = (h: number): number => (Math.sin(1 + h) - Math.sin(1 - h)) / (2 * h)
      check('richardson central difference', A.richardson([d(0.4), d(0.2), d(0.1)]).value, Math.cos(1), 1e-7)
      expect(() => A.richardson([])).toThrow(expect.objectContaining({ code: 'value' }))
      expect(failures).toEqual([])
    }
    {
      // exp [2/2]: (1 + x/2 + x^2/12) / (1 - x/2 + x^2/12); tan x [3/2] from its series: (x - x^3 / 15) / (1 - 2 x^2 / 5).
      const taylor = [1, 1, 1 / 2, 1 / 6, 1 / 24]
      const { numerator, denominator } = A.pade(taylor, 2)
      expect(arr(numerator)).toEqual([1, 0.5, 1 / 12].map((e) => expect.closeTo(e, 14)))
      expect(arr(denominator)).toEqual([1, -0.5, 1 / 12].map((e) => expect.closeTo(e, 14)))
      const tan = A.pade([0, 1, 0, 1 / 3, 0, 2 / 15], 2, 3)
      expect(arr(tan.numerator)).toEqual([0, 1, 0, -1 / 15].map((e) => expect.closeTo(e, 12)))
      expect(arr(tan.denominator)).toEqual([1, 0, -0.4].map((e) => expect.closeTo(e, 12)))
      expect(() => A.pade([1, 1], 2)).toThrow(expect.objectContaining({ code: 'value' }))
      // The golden ratio, sqrt 2 = 1 + 1 / (2 + 1 / (2 + ...)) and tan 1 = 1 / (1 - 1 / (3 - 1 / (5 - ...))).
      const golden = A.continuedFraction(
        () => 1,
        () => 1,
      )
      expect(golden.value).toBeCloseTo((1 + Math.sqrt(5)) / 2, 14)
      expect(
        A.continuedFraction(
          () => 1,
          (j) => (j === 0 ? 1 : 2),
        ).value,
      ).toBeCloseTo(Math.SQRT2, 14)
      expect(
        A.continuedFraction(
          (j) => (j === 1 ? 1 : -1),
          (j) => (j === 0 ? 0 : 2 * j - 1),
        ).value,
      ).toBeCloseTo(Math.tan(1), 14)
      expect(() =>
        A.continuedFraction(
          () => 1,
          () => 1,
          { maxTerms: 3 },
        ),
      ).toThrow(expect.objectContaining({ code: 'convergence' }))
    }
    {
      // The best degree-1 approximation of x^2 on [-1, 1] is the constant 1/2 with error 1/2; of x^3 by degree 2 it is 3x/4
      // with error 1/4 (x^3 - 3x/4 = T_3 / 4).
      const quad = A.remez((x) => x * x, 1)
      expect([Math.abs(quad.error), quad.converged, quad.status]).toEqual([expect.closeTo(0.5, 9), true, 'converged'])
      const cubic = A.remez((x) => x ** 3, 2)
      expect(Math.abs(cubic.error)).toBeCloseTo(0.25, 9)
      expect(cubic.evaluate(0.8)).toBeCloseTo(0.6, 9)
      // Chebyshev series by Clenshaw: sum c_k T_k(cos t) = sum c_k cos(k t).
      const c = [1, 2, 3, -1, 0.5]
      for (const t of [0.3, 1, 2.5]) {
        const want = c.reduce((s, ck, k) => s + ck * Math.cos(k * t), 0)
        expect(arr(A.chebyshevSeries(c, [Math.cos(t)]))[0]).toBeCloseTo(want, 13)
      }
      // exp on [-1, 1] has the coefficients 2 I_k(1) (halved for k = 0); sampled at the points cos(j pi / (n - 1)).
      const n = 17
      const values = Array.from({ length: n }, (_, j) => Math.exp(Math.cos((j * PI) / (n - 1))))
      arr(A.chebyshevCoefficients(values))
        .slice(0, 8)
        .forEach((ck, k) => {
          expect(ck, `T_${k}`).toBeCloseTo((k === 0 ? 1 : 2) * item(iv(k, 1)), 12)
        })
      // chebfun picks the degree itself: sin(5x) + exp(x) on [-2, 3] to 1e-13 everywhere, and the integral and a root agree.
      const f = A.chebfun((x: Value) => mul(sin(mul(x, 5)), 1), [-2, 3])
      for (const x of [-2, -1.3, 0, 0.2, 1.7, 3])
        expect(item(f.evaluate(x)), `chebfun at ${x}`).toBeCloseTo(Math.sin(5 * x), 12)
      expect(f.sum()).toBeCloseTo((Math.cos(-10) - Math.cos(15)) / 5, 11)
      expect(toFlat(f.roots()).filter((r) => Math.abs(r) < 1e-9)).toHaveLength(1)
      expect(() => A.chebfun((x) => exp(mul(sqrt(square(x)), 1)), [-1, 1], { maxPoints: 5 })).toThrow(
        expect.objectContaining({ code: 'convergence' }),
      )
    }
    // twoSum and twoProduct return the rounded result and its exact error, so that s + e is the true sum or product.
    expect(A.twoSum(0.1, 0.2)).toEqual([0.30000000000000004, -2.7755575615628914e-17])
    expect(A.twoSum(1e16, 1)).toEqual([1e16, 1])
    expect(A.twoProduct(1 + 2 ** -30, 1 + 2 ** -30)).toEqual([1 + 2 ** -29, 2 ** -60])
    expect(A.twoProduct(3, 0.1)).toEqual([0.30000000000000004, -2.7755575615628914e-17])
    // Neumaier's sum recovers the 1 in [1e16, 1, -1e16] that the naive and Kahan sums lose; fsum([0.1] * 10^6) = 100000.00000000001.
    expect(
      ['naive', 'kahan', 'neumaier', 'pairwise'].map((m) => A.compensatedSum([1e16, 1, -1e16], m as A.SumMethod)),
    ).toEqual([0, 0, 1, 0])
    expect(A.compensatedSum(new Array<number>(1e6).fill(0.1), 'neumaier')).toBeCloseTo(100000, 9)
    // The condition number |x f'(x) / f(x)|: sqrt 1/2, exp x, log at e is 1, x^2 is 2; the propagated error of x y.
    expect(A.conditionNumber((x) => sqrt(x), 4)).toBeCloseTo(0.5, 7)
    expect(A.conditionNumber((x) => exp(x), 10)).toBeCloseTo(10, 6)
    expect(A.conditionNumber((x) => log(x), Math.E)).toBeCloseTo(1, 6)
    expect(A.conditionNumber((x) => square(x), 7)).toBeCloseTo(2, 6)
    // f = x^2 + y^2 at (3, 4) with sigma (0.1, 0.2): sqrt((6 * 0.1)^2 + (8 * 0.2)^2).
    expect(A.propagateError((x) => sum(square(x)), [3, 4], [0.1, 0.2])).toBeCloseTo(Math.sqrt(0.36 + 2.56), 10)
    expect(A.significantDigits(1.0001, 1)).toBeCloseTo(4, 9)
    expect(A.significantDigits(Math.PI, PI)).toBeCloseTo(-Math.log10(Math.abs(Math.PI - PI) / PI), 12)
    expect(A.significantDigits(2, 2)).toBe(Number.POSITIVE_INFINITY)
    // polysub: [1, 2, 3] - [1, 2] = [0, 0, 3], a - a = [0], and the monic Gauss polynomials: Legendre n = 3, 4, Chebyshev n = 3.
    expect(arr(polysub([1, 2, 3], [1, 2]))).toEqual([0, 0, 3])
    expect(arr(polysub([1, 2], [1, 2]))).toEqual([0])
    expect(arr(monicFromNodes('legendre', 3))).toEqual([0, -0.6, 0, 1].map((e) => expect.closeTo(e, 14)))
    expect(arr(monicFromNodes('legendre', 4))).toEqual([3 / 35, 0, -6 / 7, 0, 1].map((e) => expect.closeTo(e, 14)))
    expect(arr(monicFromNodes('chebyshev', 3))).toEqual([0, -0.75, 0, 1].map((e) => expect.closeTo(e, 14)))
  })
})
