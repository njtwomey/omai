import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { grad } from '../../autodiff/index.ts'
import {
  abs,
  exp,
  mul,
  sin,
  sum,
  type Tensor,
  type TensorLike,
  tensor,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import * as A from './index.ts'

const smooth = (x: Value): Value => mul(exp(x), sin(mul(x, 5)))

describe('approximation against numpy and closed forms', () => {
  const f = A.chebfun(smooth, [0, 2])
  for (const c of loadFixture('numerics/approximation/approximation').cases) {
    it(c.name, () => {
      const t = c.inputs.map((v) => decode(v) as Tensor)
      const xs = (): number[] => toFlat(t[0] as Tensor)
      const actual: Value | number[] =
        c.fn === 'coefficients'
          ? A.chebyshevCoefficients(xs())
          : c.fn === 'derivative'
            ? [...A.chebyshev(xs()).derivative(2).coefficients]
            : c.fn === 'cumsum'
              ? [...A.chebyshev(xs()).cumsum().coefficients]
              : c.fn === 'roots'
                ? A.chebyshev(xs()).roots()
                : c.fn === 'chebfunSum'
                  ? [f.sum()]
                  : c.fn === 'chebfunValues'
                    ? f.evaluate(t[0] as Tensor)
                    : c.fn === 'chebfunRoots'
                      ? f.roots()
                      : c.fn === 'pade'
                        ? (({ numerator, denominator }) => [...toFlat(numerator), ...toFlat(denominator)])(
                            A.pade(xs(), c.params['m'] as number),
                          )
                        : c.fn === 'nextAfterUp'
                          ? xs().map((x) => A.nextAfter(x, Infinity))
                          : c.fn === 'nextAfterDown'
                            ? xs().map((x) => A.nextAfter(x, -Infinity))
                            : c.fn === 'ulp'
                              ? xs().map(A.ulp)
                              : c.fn === 'float32'
                                ? xs().map(A.toFloat32)
                                : c.fn === 'float16'
                                  ? xs().map(A.toFloat16)
                                  : [A.machineEpsilon(), A.machineEpsilon('float32')]
      expectMatches(
        Array.isArray(actual) ? tensor(c.fn === 'chebfunSum' ? (actual[0] as number) : actual) : actual,
        c.output,
        c.rtol,
        c.atol,
        c.name,
      )
    })
  }
})

describe('approximation properties', () => {
  it('Chebyshev coefficients decay geometrically for analytic functions and slowly for |x|', () => {
    const analytic = A.chebfun((x) => exp(x))
    expect(analytic.coefficients.length).toBeLessThan(20)
    const kink = A.chebfun((x) => abs(x), [-1, 1], { tol: 1e-6, maxPoints: 4097 })
    expect(kink.coefficients.length).toBeGreaterThan(200)
    const ex = A.chebfun(smooth, [0, 2]).extrema()
    expect(ex.points.length).toBeGreaterThan(3)
  })

  it('Remez solutions equioscillate and beat Chebyshev interpolation', () => {
    const r = A.remez(Math.exp, 3, { interval: [0, 1] })
    expect(r.converged).toBe(true)
    // The error alternates in sign with magnitude |E| at the reference points.
    const errs = r.reference.map((x) => Math.exp(x) - r.evaluate(x))
    for (let i = 0; i < errs.length; i++) {
      expect(Math.abs(Math.abs(errs[i] ?? 0) - Math.abs(r.error))).toBeLessThan(1e-9 * Math.abs(r.error) + 1e-14)
      if (i > 0) expect(Math.sign(errs[i] ?? 0)).toBe(-Math.sign(errs[i - 1] ?? 0))
    }
    const rational = A.remez(Math.exp, 2, { n: 1, interval: [-1, 1] })
    expect(rational.converged).toBe(true)
    expect(Math.abs(rational.error)).toBeLessThan(Math.abs(A.remez(Math.exp, 3).error))
  })

  it('series acceleration and continued fractions', () => {
    const s: number[] = []
    let acc = 0
    for (let n = 1; n <= 10; n++) {
      acc += (n % 2 ? 1 : -1) / n
      s.push(acc)
    }
    expect(Math.abs(A.wynnEpsilon(s).value - Math.LN2)).toBeLessThan(1e-7)
    expect(Math.abs((toFlat(A.aitken(s)).at(-1) ?? 0) - Math.LN2)).toBeLessThan(
      Math.abs((s.at(-1) ?? 0) - Math.LN2) / 50,
    )
    // tan(1) = 1 / (1 - 1 / (3 - 1 / (5 - ...))).
    const tan = A.continuedFraction(
      (j) => (j === 1 ? 1 : -1),
      (j) => (j === 0 ? 0 : 2 * j - 1),
    )
    expect(tan.value).toBeCloseTo(Math.tan(1), 14)
  })

  it('the sequence functions take tensor-likes (migration, D-009): the same for a tensor, a typed array and an array', () => {
    const calls: [string, (v: TensorLike) => number[]][] = [
      ['richardson', (v) => [A.richardson(v).value]],
      ['aitken', (v) => toFlat(A.aitken(v))],
      ['wynnEpsilon', (v) => [A.wynnEpsilon(v).value]],
      ['shanks', (v) => toFlat(A.shanks(v))],
      ['eulerTransform', (v) => [A.eulerTransform(v)]],
      ['chebyshev', (v) => [...A.chebyshev(v).coefficients]],
      ['chebyshevCoefficients', (v) => toFlat(A.chebyshevCoefficients(v))],
      ['chebyshevSeries', (v) => toFlat(A.chebyshevSeries(v, [0.5, -0.25]))],
      ['pade', (v) => toFlat(A.pade(v, 1).denominator)],
      ['propagateError', (v) => [A.propagateError((x) => sum(mul(x, x)), v, [0.1, 0.1, 0.1])]],
    ]
    const v = [1, 0.5, 0.25, 0.125]
    for (const [name, f] of calls) {
      const expected = f(v)
      expect(f(Float64Array.from(v)), name).toEqual(expected)
      expect(f(tensor(v)), name).toEqual(expected)
      expect(() => f(tensor([v, v])), name).toThrow(expect.objectContaining({ code: 'shape' }))
      expect(() => grad((x: Value) => sum(tensor(f(x as TensorLike))))(tensor(v)), name).toThrow(
        expect.objectContaining({ code: 'not-differentiable' }),
      )
    }
  })

  it('error-free transformations and compensated sums', () => {
    const [s, e] = A.twoSum(0.1, 0.2)
    expect(s + e === 0.1 + 0.2 || e !== 0).toBe(true)
    const terms = Array.from({ length: 10000 }, () => 0.1)
    expect(Math.abs(A.compensatedSum(terms, 'kahan') - 1000)).toBeLessThan(
      Math.abs(A.compensatedSum(terms, 'naive') - 1000),
    )
    expect(A.compensatedSum([1e16, 1, -1e16])).toBe(1)
    expect(A.compensatedSum(terms, 'pairwise')).toBeCloseTo(1000, 11)
  })
})
