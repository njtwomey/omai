import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { expectConformant } from '../../testing/check.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

const arr = (min: number, max: number): fc.Arbitrary<number[]> =>
  fc.array(fc.double({ min: -2, max: 2, noNaN: true }), { minLength: min, maxLength: max })

const close = (a: readonly number[], b: readonly number[], digits = 10): void => {
  expect(a.length).toBe(b.length)
  for (const [i, v] of a.entries()) expect(v).toBeCloseTo(b[i] ?? 0, digits)
}

describe('multirate identities', () => {
  it('recombines polyphase components exactly', () => {
    fc.assert(
      fc.property(arr(1, 15), fc.integer({ min: 1, max: 4 }), (h, m) => {
        close(toFlat(S.polyphaseRecombine(S.polyphase(h, m), h.length)), h, 15)
      }),
      { numRuns: RUNS },
    )
  })

  it('satisfies the noble identities', () => {
    fc.assert(
      fc.property(arr(4, 6), arr(12, 20), fc.integer({ min: 2, max: 3 }), (h, x, m) => {
        // Filter with H(z^m) then downsample = downsample then filter with H(z).
        const a = toFlat(S.downsample(S.lfilter(S.expandFilter(h, m), [1], x), m))
        const b = toFlat(S.lfilter(h, [1], S.downsample(x, m)))
        close(a, b)
        // Upsample then filter with H(z^m) = filter with H(z) then upsample.
        close(
          toFlat(S.lfilter(S.expandFilter(h, m), [1], S.upsample(x, m))),
          toFlat(S.upsample(S.lfilter(h, [1], x), m)),
        )
      }),
      { numRuns: RUNS },
    )
  })

  it('upfirdn equals upsample, convolve and downsample', () => {
    fc.assert(
      fc.property(
        arr(1, 5),
        arr(3, 10),
        fc.integer({ min: 1, max: 3 }),
        fc.integer({ min: 1, max: 3 }),
        (h, x, up, down) => {
          const y = toFlat(S.upfirdn(h, x, { up, down }))
          const u = toFlat(S.upsample(x, up)).slice(0, (x.length - 1) * up + 1)
          const full = new Array<number>(u.length + h.length - 1).fill(0)
          for (const [i, ui] of u.entries())
            for (const [j, hj] of h.entries()) full[i + j] = (full[i + j] ?? 0) + ui * hj
          close(
            y,
            full.filter((_, k) => k % down === 0),
            9,
          )
        },
      ),
      { numRuns: RUNS },
    )
  })

  it('resamples a band-limited periodic signal exactly by Fourier resampling', () => {
    const n = 16
    const x = Array.from(
      { length: n },
      (_, k) => Math.cos((2 * Math.PI * 3 * k) / n) + 0.5 * Math.sin((2 * Math.PI * k) / n),
    )
    const y = toFlat(S.resample(x, 40))
    for (const [k, v] of y.entries()) {
      const t = (k * n) / 40
      expect(v).toBeCloseTo(Math.cos((2 * Math.PI * 3 * t) / n) + 0.5 * Math.sin((2 * Math.PI * t) / n), 10)
    }
    // Interpolation keeps a slowly varying signal and decimation keeps its low-frequency content.
    const slow = Array.from({ length: 200 }, (_, k) => Math.sin((2 * Math.PI * k) / 100))
    const up = toFlat(S.interpolate(slow, 2))
    for (let k = 40; k < 160; k++) expect(up[2 * k]).toBeCloseTo(slow[k] ?? 0, 2)
    const down = toFlat(S.decimate(slow, 2))
    for (let k = 20; k < 80; k++) expect(Math.abs((down[k] ?? 0) - (slow[2 * k] ?? 0))).toBeLessThan(0.02)
  })

  it('differentiates resampling', () => {
    const x = tensor([0.3, -1.2, 0.8, 2.1, -0.4, 0.9, 1.5, -0.7])
    expectConformant((h, xx) => S.upfirdn(h, xx, { up: 2, down: 3, mode: 'reflect' }), [tensor([0.5, 1, -0.3]), x])
    expectConformant((xx) => S.resamplePoly(xx, 3, 2, { padtype: 'mean' }), [x])
    expectConformant((xx) => S.resample(xx, 5), [x])
    expectConformant(
      (xx) => S.decimate(xx, 2, { n: 2 }),
      [tensor([0.3, -1.2, 0.8, 2.1, -0.4, 0.9, 1.5, -0.7, 0.2, 0.1, 0.6, -0.3, 0.4, 0.2, 0.5, 0.3, 1, 0.5, 0.2, 0.1])],
    )
  })
})
