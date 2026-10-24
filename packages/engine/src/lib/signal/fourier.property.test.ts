import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { expectConformant } from '../../testing/check.ts'
import { key } from '../random/index.ts'
import { item, tensor, toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

const signal = (n: number): fc.Arbitrary<number[]> =>
  fc.array(fc.double({ min: -2, max: 2, noNaN: true }), { minLength: n, maxLength: n })

describe('Fourier properties', () => {
  it("satisfies Parseval's identity", () => {
    fc.assert(
      fc.property(signal(13), signal(13), (re, im) => {
        const { time, frequency } = S.parseval({ re: tensor(re), im: tensor(im) })
        expect(item(frequency)).toBeCloseTo(item(time), 9)
      }),
      { numRuns: RUNS },
    )
  })

  it('takes the Hilbert transform of a cosine to a sine and is linear', () => {
    const n = 64
    const c = Array.from({ length: n }, (_, k) => Math.cos((2 * Math.PI * 5 * k) / n))
    const h = toFlat(S.hilbertTransform(c))
    for (let k = 0; k < n; k++) expect(h[k]).toBeCloseTo(Math.sin((2 * Math.PI * 5 * k) / n), 12)
    fc.assert(
      fc.property(signal(9), signal(9), fc.double({ min: -3, max: 3, noNaN: true }), (a, b, s) => {
        const lhs = toFlat(S.hilbertTransform(a.map((v, i) => v + s * (b[i] ?? 0))))
        const ha = toFlat(S.hilbertTransform(a))
        const hb = toFlat(S.hilbertTransform(b))
        lhs.forEach((v, i) => {
          expect(v).toBeCloseTo((ha[i] ?? 0) + s * (hb[i] ?? 0), 9)
        })
      }),
      { numRuns: RUNS },
    )
  })

  it('differentiates the DTFT, the analytic signal and sinc interpolation', () => {
    expectConformant((x) => S.dtft(x, [0.1, 0.3]).im, [tensor([1, -2, 0.5, 3])])
    expectConformant((x) => S.envelope(x), [tensor([1, 0.2, -1, 0.4, 0.7])])
    expectConformant((x) => S.sincInterpolate(x, 2, [0.3, 1.1]), [tensor([1, -2, 0.5])])
  })

  it('reconstructs a band-limited signal and finds peaks between bins', () => {
    const fs = 10
    const f = (t: number): number => Math.sin(2 * Math.PI * 1.3 * t) + 0.5 * Math.cos(2 * Math.PI * 2.1 * t)
    const x = Array.from({ length: 400 }, (_, k) => f(k / fs))
    const t = [19.03, 20.51, 21.77]
    const y = toFlat(S.sincInterpolate(x, fs, t))
    y.forEach((v, i) => {
      expect(v).toBeCloseTo(f(t[i] ?? 0), 2)
    })
    const tone = Array.from({ length: 128 }, (_, n) => Math.cos(2 * Math.PI * 0.2071 * n + 0.4))
    expect(S.peakFrequency(tone, { nfft: 512 })).toBeCloseTo(0.2071, 4)
    expect(S.frequencyResolution(10, 100)).toEqual({ resolution: 10, binSpacing: 10 })
    expect(S.leakageFraction(S.blackmanHarris(64, { sym: false }), 0.5, 3)).toBeLessThan(1e-5)
  })

  it('quantises, dithers and holds', () => {
    const x = Array.from({ length: 2000 }, (_, k) => 0.9 * Math.sin(0.0123 * k * 7))
    const q8 = toFlat(S.quantise(x, { bits: 8 }))
    const snr = S.sqnr(x, q8)
    expect(snr).toBeGreaterThan(S.idealSqnr(8) - 3)
    expect(snr).toBeLessThan(S.idealSqnr(8) + 3)
    const d = toFlat(S.dither(key(3), new Array(5000).fill(0), { step: 1 }))
    expect(d.reduce((a, v) => a + v, 0) / d.length).toBeCloseTo(0, 1)
    expect(d.reduce((a, v) => a + v * v, 0) / d.length).toBeCloseTo(1 / 6, 1)
    expect(toFlat(S.zeroOrderHold([1, 2], 2))).toEqual([1, 1, 2, 2])
    expect(S.alias([0, 50, 99], 100)).toEqual([0, 50, 1])
  })
})
