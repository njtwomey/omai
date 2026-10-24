import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../../testing/arbitrary.ts'
import { key } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import * as A from './index.ts'

const snr = (clean: ArrayLike<number>, est: ArrayLike<number>, from = 0): number => {
  let s = 0
  let e = 0
  for (let i = from; i < clean.length; i++) {
    s += (clean[i] ?? 0) ** 2
    e += ((clean[i] ?? 0) - (est[i] ?? 0)) ** 2
  }
  return 10 * Math.log10(s / e)
}

describe('enhancement, separation, coding and augmentation', () => {
  const n = 6000
  const clean = Float64Array.from({ length: n }, (_, i) => (i > 1600 ? Math.sin(i / 5) + 0.5 * Math.sin(i / 11) : 0))
  const noise = Float64Array.from({ length: n }, (_, i) => 0.3 * Math.sin(i * 2.37) + 0.2 * Math.sin(i * 1.91 + 1))
  const noisy = Float64Array.from(clean, (v, i) => v + (noise[i] ?? 0))

  it('spectral subtraction and Wiener denoising improve the SNR', () => {
    const before = snr(clean, noisy, 2000)
    expect(snr(clean, A.spectralSubtraction(noisy, { nFft: 256, noise }), 2000)).toBeGreaterThan(before)
    expect(snr(clean, A.wienerDenoise(noisy, { nFft: 256, noise }), 2000)).toBeGreaterThan(before)
  })

  it('voice activity follows the energy', () => {
    const v = toFlat(A.voiceActivity(Float64Array.from(clean, (x, i) => x + 1e-3 * Math.sin(i))))
    expect(v[0]).toBe(0)
    expect(v.at(-1)).toBe(1)
  })

  it('oracle Wiener masks sum to one and separate well', () => {
    const r = A.oracleSeparation([clean, noise], { nFft: 256 })
    const sum = Float64Array.from(r[0] ?? [], (v, i) => v + (r[1]?.[i] ?? 0))
    expect(snr(noisy, sum, 200)).toBeGreaterThan(80)
    expect(snr(clean, r[0] ?? [], 2000)).toBeGreaterThan(10)
    const ratio = toFlat(A.idealRatioMask(clean, noise, { nFft: 256 }))
    const bin = toFlat(A.idealBinaryMask(clean, noise, { nFft: 256 }))
    expect(ratio.every((v) => v >= 0 && v <= 1)).toBe(true)
    expect(bin.every((v) => v === 0 || v === 1)).toBe(true)
  })

  it('perceptual MDCT coding reconstructs exactly at fine steps and degrades gracefully', () => {
    const x = Float64Array.from(
      { length: 2048 },
      (_, i) => 0.4 * Math.sin((2 * Math.PI * 440 * i) / 16000) + 0.1 * Math.sin(i),
    )
    const fine = A.mdctCode(x, 16000, { stepScale: 1e-6 })
    expect(snr(x, fine.decoded)).toBeGreaterThan(80)
    const a = A.mdctCode(x, 16000, { stepScale: 1 })
    const b = A.mdctCode(x, 16000, { stepScale: 8 })
    expect(snr(x, a.decoded)).toBeGreaterThan(snr(x, b.decoded))
    expect(a.bits).toBeGreaterThan(b.bits)
  })

  it('mixing hits the requested SNR and SpecAugment only writes the mask value', () => {
    fc.assert(
      fc.property(fc.double({ min: -10, max: 30, noNaN: true }), fc.integer({ min: 0, max: 1000 }), (db, seed) => {
        const mixed = A.mixAtSnr(clean, noise, db)
        const n2 = Float64Array.from(mixed, (v, i) => v - (clean[i] ?? 0))
        let ps = 0
        let pn = 0
        for (let i = 0; i < n; i++) {
          ps += (clean[i] ?? 0) ** 2
          pn += (n2[i] ?? 0) ** 2
        }
        expect(10 * Math.log10(ps / pn)).toBeCloseTo(db, 8)
        const s = Array.from({ length: 12 }, () => new Array<number>(20).fill(3))
        const m = toFlat(A.specAugment(s, key(seed), { maskValue: -1 }))
        expect(m.every((v) => v === 3 || v === -1)).toBe(true)
      }),
      { numRuns: RUNS },
    )
  })

  it('a time stretch by r changes the length by 1 / r and keeps the pitch', () => {
    const tone = Float64Array.from({ length: 8000 }, (_, i) => Math.sin((2 * Math.PI * 500 * i) / 8000))
    const st = A.timeStretch(tone, 0.5, { nFft: 512 })
    expect(st.length).toBe(16000)
    const f0 = A.yin(st.slice(4000, 12000), { fmin: 100, fmax: 2000, sr: 8000, frameLength: 512 })
    expect(f0[5] ?? 0).toBeCloseTo(500, -1)
  })
})
