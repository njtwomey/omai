import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import * as S from './index.ts'

const series = fc.array(fc.double({ min: -10, max: 10, noNaN: true, noDefaultInfinity: true }), {
  minLength: 12,
  maxLength: 60,
})

function bruteZDistance(q: Float64Array, w: Float64Array): number {
  const a = S.zNormalize(q)
  const b = S.zNormalize(w)
  let s = 0
  for (let i = 0; i < a.length; i++) s += ((a[i] ?? 0) - (b[i] ?? 0)) ** 2
  return Math.sqrt(s)
}

describe('sliding-window building blocks', () => {
  it('rolling statistics match direct computation on every window', () => {
    fc.assert(
      fc.property(series, fc.integer({ min: 2, max: 10 }), (xs, m) => {
        const t = Float64Array.from(xs)
        const { mean, std } = S.rollingMeanStd(t, m)
        for (let i = 0; i + m <= t.length; i++) {
          const w = t.subarray(i, i + m)
          let mu = 0
          for (const v of w) mu += v / m
          let v2 = 0
          for (const v of w) v2 += (v - mu) ** 2 / m
          expect(mean[i]).toBeCloseTo(mu, 9)
          expect(std[i]).toBeCloseTo(Math.sqrt(v2), 6)
        }
      }),
      { numRuns: 40 },
    )
  })

  it('FFT and direct sliding dot products agree, and the O(1) update reproduces the next row', () => {
    fc.assert(
      fc.property(series, fc.integer({ min: 3, max: 8 }), (xs, m) => {
        const t = Float64Array.from(xs)
        const q = t.slice(1, 1 + m)
        const a = S.slidingDotProduct(q, t)
        const b = S.slidingDotProductDirect(q, t)
        for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(b[i] ?? 0, 8)
        const row0 = S.slidingDotProductDirect(t.slice(0, m), t)
        const row1 = S.slidingDotProductDirect(t.slice(1, 1 + m), t)
        for (let j = 0; j + 1 < row1.length; j++)
          expect(S.nextDotProduct(row0[j] ?? 0, t, t, 0, j, m)).toBeCloseTo(row1[j + 1] ?? 0, 8)
      }),
      { numRuns: 40 },
    )
  })

  it('MASS equals brute-force z-normalised distances on non-constant windows', () => {
    fc.assert(
      fc.property(series, (xs) => {
        const t = Float64Array.from(xs, (v, i) => v + 0.01 * i * i)
        const m = 4
        const q = t.slice(2, 2 + m)
        const d = S.mass(q, t)
        const constant = S.rollingIsConstant(t, m)
        for (let i = 0; i + m <= t.length; i++) {
          if (constant[i] === 1) continue
          const w = t.subarray(i, i + m)
          if ((S.rollingMeanStd(w, m).std[0] ?? 0) < 1e-3) continue
          expect(d[i]).toBeCloseTo(bruteZDistance(q, w), 4)
        }
      }),
      { numRuns: 30 },
    )
  })
})
