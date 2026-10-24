import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { type Tensor, tensor, toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

// Coefficients on a grid of hundredths: realistic values, no subnormal or extreme magnitudes.
const coef = fc.integer({ min: -300, max: 300 }).map((k) => k / 100)
const lead = fc.integer({ min: 50, max: 300 }).map((k) => k / 100)
const tfArb = fc
  .record({
    den: fc.tuple(lead, fc.array(coef, { minLength: 1, maxLength: 4 })).map(([a, rest]) => [a, ...rest]),
    num: fc.array(coef, { minLength: 1, maxLength: 4 }),
  })
  .filter(({ num, den }) => num.length <= den.length && num.some((v) => v !== 0))

const ssArb = fc.integer({ min: 1, max: 4 }).chain((n) =>
  fc.record({
    a: fc.array(fc.array(coef, { minLength: n, maxLength: n }), { minLength: n, maxLength: n }),
    b: fc.array(fc.array(coef, { minLength: 2, maxLength: 2 }), { minLength: n, maxLength: n }),
    c: fc.array(fc.array(coef, { minLength: n, maxLength: n }), { minLength: 1, maxLength: 1 }),
    d: fc.array(fc.array(coef, { minLength: 2, maxLength: 2 }), { minLength: 1, maxLength: 1 }),
  }),
)

const w = [0.37, 1.3, 4.1]
const responseOf = (sys: S.Lti): number[] => {
  const h = S.freqresp(sys, w).h
  return [...toFlat(h.re as Tensor), ...toFlat(h.im as Tensor)]
}
const scaleOf = (xs: readonly number[]): number => xs.reduce((m, v) => Math.max(m, Math.abs(v)), 1)
const agree = (a: readonly number[], b: readonly number[], tol: number): void => {
  const scale = scaleOf([...a, ...b])
  for (const [i, v] of a.entries()) expect(Math.abs(v - (b[i] ?? Number.NaN))).toBeLessThan(tol * scale)
}

describe('conversions keep the transfer function', () => {
  it('tf -> ss -> tf returns the normalised transfer function', () => {
    fc.assert(
      fc.property(tfArb, ({ num, den }) => {
        const ref = S.normalize(num, den)
        const back = S.toTf(S.toSs(S.tf(num, den)))
        const want = toFlat(ref.num as Tensor)
        const got = toFlat(back.num as Tensor)
        agree(got.slice(got.length - want.length), want, 1e-8)
        agree(got.slice(0, got.length - want.length), new Array<number>(got.length - want.length).fill(0), 1e-8)
        agree(toFlat(back.den as Tensor), toFlat(ref.den as Tensor), 1e-8)
      }),
      { numRuns: 50 },
    )
  })

  it('tf -> zpk -> tf and tf -> sos keep the frequency response (away from poles)', () => {
    fc.assert(
      fc.property(tfArb, ({ num, den }) => {
        const g = S.tf(num, den)
        const poles = S.poles(g)
        fc.pre(poles.re.every((re, i) => w.every((x) => Math.hypot(re, (poles.im[i] ?? 0) - x) > 0.05)))
        const want = responseOf(g)
        agree(responseOf(S.toZpk(g)), want, 1e-6)
        agree(responseOf(S.toTf(S.toZpk(g))), want, 1e-6)
        const gd = S.tf(num, den, { dt: 0.1 })
        agree(responseOf(S.toSos(gd)), responseOf(gd), 1e-6)
      }),
      { numRuns: 50 },
    )
  })

  it('ss -> tf (each input) -> ss keeps every channel of the response', () => {
    fc.assert(
      fc.property(ssArb, ({ a, b, c, d }) => {
        const s = S.ss(tensor(a), tensor(b), tensor(c), tensor(d))
        const poles = S.poles(s)
        fc.pre(poles.re.every((re, i) => w.every((x) => Math.hypot(re, (poles.im[i] ?? 0) - x) > 0.05)))
        const h = S.freqresp(s, w).h
        const re = toFlat(h.re as Tensor)
        const im = toFlat(h.im as Tensor)
        for (const input of [0, 1]) {
          const g = S.toSs(S.toTf(s, { input }))
          const got = responseOf(g)
          const want = [
            ...[0, 1, 2].map((k) => re[2 * k + input] ?? 0),
            ...[0, 1, 2].map((k) => im[2 * k + input] ?? 0),
          ]
          agree(got, want, 1e-6)
        }
      }),
      { numRuns: 40 },
    )
  })
})
