import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../../testing/arbitrary.ts'
import { expectConformant } from '../../../testing/check.ts'
import { sliceAxis, tensor, toArray, toFlat } from '../../tensor/index.ts'
import * as S from '../index.ts'

const signal = (min: number, max: number): fc.Arbitrary<number[]> =>
  fc.array(fc.double({ min: -3, max: 3, noNaN: true }), { minLength: min, maxLength: max })
const names = fc.constantFrom('haar', 'db3', 'sym4', 'coif2', 'bior2.4', 'rbio3.3', 'db10')
const modes = fc.constantFrom<S.WaveletMode>(
  'symmetric',
  'periodization',
  'zero',
  'reflect',
  'periodic',
  'smooth',
  'antisymmetric',
)

describe('wavelet properties', () => {
  it('reconstructs perfectly from dwt and wavedec', () => {
    fc.assert(
      fc.property(signal(4, 40), names, modes, (x, w, mode) => {
        const { cA, cD } = S.dwt(x, w, { mode })
        const back = toFlat(S.idwt(cA, cD, w, { mode }))
        for (const [i, v] of x.entries()) expect(back[i]).toBeCloseTo(v, 9)
        const levels = Math.max(1, S.dwtMaxLevel(x.length, w))
        const rec = toFlat(S.waverec(S.wavedec(x, w, { mode, level: levels }), w, { mode }))
        for (const [i, v] of x.entries()) expect(rec[i]).toBeCloseTo(v, 8)
      }),
      { numRuns: RUNS },
    )
  })

  it('preserves energy for orthogonal wavelets with periodization and inverts the SWT', () => {
    fc.assert(
      fc.property(signal(16, 16), fc.constantFrom('haar', 'db2', 'sym3', 'coif1'), (x, w) => {
        const { cA, cD } = S.dwt(x, w, { mode: 'periodization' })
        const e = (v: number[]): number => v.reduce((a, b) => a + b * b, 0)
        // Relative to the energy: PyWavelets' sym3 filter (which the engine reproduces exactly, for parity) is
        // orthonormal only to 4.8e-12 (|sum h^2 - 1|), so for 16 samples ||W^T W - I||_2 = 1.1e-11 and a signal of
        // energy 144 can be off by 1.6e-9 (pywt too). A fixed 5e-10 failed now and then.
        expect(Math.abs(e(toFlat(cA)) + e(toFlat(cD)) - e(x))).toBeLessThanOrEqual(2e-11 * e(x) + 1e-12)
        const back = toFlat(S.iswt(S.swt(x, w, { level: 3 }), w))
        for (const [i, v] of x.entries()) expect(back[i]).toBeCloseTo(v, 9)
      }),
      { numRuns: RUNS },
    )
  })

  it('reconstructs images and differentiates the transforms', () => {
    const img = [
      [1, 2, 0, 3, 1],
      [4, 1, 2, 2, 0],
      [0, 3, 1, 4, 2],
      [2, 2, 3, 1, 1],
    ]
    // An odd width comes back one column wider, as pywt.waverec2.
    const rec = toArray(sliceAxis(S.waverec2(S.wavedec2(img, 'db2', { level: 1 }), 'db2'), 1, 0, 5)) as number[][]
    expect(rec.flat().map((v) => Math.round(v * 1e9) / 1e9)).toEqual(img.flat())
    const x = tensor([0.3, -1.2, 0.8, 2.1, -0.4, 0.9, 1.5, -0.7])
    expectConformant((v) => S.dwt(v, 'db2').cD, [x])
    expectConformant((v) => S.idwt(v, v, 'sym3', { mode: 'periodization' }), [x])
    expectConformant((v) => S.swt(v, 'db1', { level: 2 })[0]?.cA ?? v, [x])
    expectConformant((v) => S.cwt(v, [1, 2], 'mexh').coefficients.re, [x])
    expectConformant((v) => S.threshold(v, 0.5), [x])
  })

  it('removes noise', () => {
    const n = 256
    const clean = Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * i) / 64) + (i > 128 ? 1 : 0))
    const noisy = clean.map((v, i) => v + 0.3 * Math.sin(i * 12.9898) * Math.cos(i * 78.233))
    const out = toFlat(S.waveletDenoise(noisy, { wavelet: 'db4' }))
    const err = (a: readonly number[]): number => a.reduce((s, v, i) => s + (v - (clean[i] ?? 0)) ** 2, 0)
    expect(err(out)).toBeLessThan(err(noisy))
  })
})
