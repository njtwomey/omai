import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../../testing/arbitrary.ts'
import { expectConformant } from '../../../testing/check.ts'
import { tensor, toFlat } from '../../tensor/index.ts'
import * as A from './index.ts'

const freqs = fc.array(fc.double({ min: 0, max: 20000, noNaN: true }), { minLength: 1, maxLength: 8 })

describe('audio scales and features', () => {
  it('scale conversions invert', () => {
    fc.assert(
      fc.property(freqs, (f) => {
        for (const htk of [false, true]) {
          const back = toFlat(A.melToHz(A.hzToMel(f, htk), htk))
          for (const [i, v] of f.entries()) expect(back[i]).toBeCloseTo(v, 6)
        }
        for (const formula of ['traunmuller', 'schroeder'] as const) {
          const back = toFlat(A.barkToHz(A.hzToBark(f, formula), formula))
          for (const [i, v] of f.entries()) expect(back[i]).toBeCloseTo(v, 5)
        }
        const back = toFlat(A.erbToHz(A.hzToErb(f)))
        for (const [i, v] of f.entries()) expect(back[i]).toBeCloseTo(v, 6)
      }),
      { numRuns: RUNS },
    )
  })

  it('mel filters are non-negative triangles summing to about one per band under area normalisation', () => {
    const w = A.melFilterBank(8000, 256, { nMels: 20 })
    const data = toFlat(w)
    expect(data.every((v) => v >= 0)).toBe(true)
    const mel = toFlat(A.melFrequencies(22, 0, 4000))
    for (let i = 0; i < 20; i++) {
      let area = 0
      for (let k = 0; k < 129; k++) area += (data[i * 129 + k] ?? 0) * (8000 / 256)
      // Slaney normalisation: unit area in Hz up to the bin sampling of each triangle.
      const width = (mel[i + 2] ?? 0) - (mel[i] ?? 0)
      if (width > 4 * (8000 / 256)) expect(area).toBeCloseTo(1, 0)
    }
  })

  it('differentiates mel spectrograms and MFCCs in the signal', () => {
    const y = tensor(Array.from({ length: 64 }, (_, i) => Math.sin(i / 3) + 0.2 * Math.cos(i * 1.7)))
    expectConformant((v) => A.melSpectrogram(v, 8000, { nFft: 32, hopLength: 16, nMels: 6 }), [y])
    expectConformant((v) => A.mfccFromDb(A.spectrogram(v, { nFft: 16, hopLength: 8 }), { nMfcc: 4 }), [y])
    expectConformant((v) => A.spectralCentroid(A.spectrogram(v, { nFft: 16, hopLength: 8, power: 1 }), 8000), [y])
    expectConformant((v) => A.rms(v, { frameLength: 16, hopLength: 8 }), [y])
  })

  it('RMS from the signal equals RMS from its spectrogram without centring', () => {
    fc.assert(
      fc.property(fc.array(fc.double({ min: -1, max: 1, noNaN: true }), { minLength: 32, maxLength: 64 }), (xs) => {
        const s = A.spectrogram(xs, { nFft: 16, hopLength: 16, center: false, window: 'boxcar', power: 1 })
        const a = toFlat(A.rmsFromSpectrogram(s, 16))
        const b = toFlat(A.rms(xs, { frameLength: 16, hopLength: 16, center: false }))
        for (const [i, v] of b.entries()) expect(a[i]).toBeCloseTo(v, 9)
      }),
      { numRuns: RUNS },
    )
  })
})
