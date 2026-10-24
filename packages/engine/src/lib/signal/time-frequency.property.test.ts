import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { expectConformant } from '../../testing/check.ts'
import { argmaxOf } from '../base/index.ts'
import { absComplex, tensor, toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

const signal = (n: number): fc.Arbitrary<number[]> =>
  fc.array(fc.double({ min: -2, max: 2, noNaN: true }), { minLength: n, maxLength: n })

describe('time-frequency properties', () => {
  it('inverts the STFT exactly (NOLA windows)', () => {
    fc.assert(
      fc.property(
        signal(40),
        fc.constantFrom<[number, number]>([8, 4], [8, 6], [10, 5], [16, 12]),
        (x, [nperseg, noverlap]) => {
          const z = S.stft(x, { nperseg, noverlap }).values
          const back = toFlat(S.istft(z, { nperseg, noverlap }).x)
          for (let i = 0; i < x.length; i++) expect(back[i]).toBeCloseTo(x[i] ?? 0, 10)
        },
      ),
      { numRuns: RUNS },
    )
  })

  it('has the time marginal and chirp concentration of the Wigner-Ville distribution', () => {
    const n = 64
    const x = Array.from({ length: n }, (_, k) => Math.cos(2 * Math.PI * (0.05 * k + (0.3 / (2 * n)) * k * k)))
    const w = S.wignerVille(x)
    const v = toFlat(w.values)
    const z = S.hilbert(x)
    const zr = toFlat(z.re)
    const zi = toFlat(z.im)
    for (const t of [5, 20, 40]) {
      let s = 0
      for (let k = 0; k < n; k++) s += v[k * n + t] ?? 0
      expect(s / n).toBeCloseTo((zr[t] ?? 0) ** 2 + (zi[t] ?? 0) ** 2, 9)
      const column = Array.from({ length: n }, (_, k) => v[k * n + t] ?? 0)
      const f = argmaxOf(column) / (2 * n)
      expect(f).toBeCloseTo(0.05 + (0.3 / n) * t, 1)
    }
    for (const d of [S.pseudoWignerVille(x), S.smoothedPseudoWignerVille(x)]) {
      expect(d.values.shape).toEqual([n, n])
      expect(toFlat(d.values).every(Number.isFinite)).toBe(true)
    }
  })

  it('reconstructs from the synchrosqueezed transform and concentrates a tone', () => {
    fc.assert(
      fc.property(signal(48), (x) => {
        const t = S.synchrosqueeze(x, { sr: 1, nFft: 16, hop: 1 })
        const back = toFlat(S.inverseSynchrosqueeze(t, { nFft: 16 }))
        for (let i = 0; i < x.length; i++) expect(back[i]).toBeCloseTo(x[i] ?? 0, 9)
      }),
      { numRuns: RUNS },
    )
    const tone = Array.from({ length: 256 }, (_, n) => Math.cos(2 * Math.PI * 0.2 * n))
    const sq = toFlat(absComplex(S.synchrosqueeze(tone, { sr: 1, nFft: 64, hop: 1 }).values))
    const frames = 257
    const col = Array.from({ length: 33 }, (_, k) => sq[k * frames + 128] ?? 0)
    expect(argmaxOf(col)).toBe(Math.round(0.2 * 64))
  })

  it('meets the uncertainty bound and finds tones in the constant-Q transform', () => {
    expect(S.timeBandwidthProduct(S.gaussian(401, 20))).toBeCloseTo(1 / (4 * Math.PI), 6)
    expect(S.timeBandwidthProduct(S.hann(64))).toBeGreaterThan(1 / (4 * Math.PI))
    const sr = 8000
    const y = Array.from({ length: 4096 }, (_, n) => Math.cos((2 * Math.PI * 440 * n) / sr))
    const c = S.cqt(y, { sr, fmin: 110, nBins: 36, hop: 512 })
    const mag = toFlat(absComplex(c.values))
    const frames = c.values.re.shape[1] ?? 1
    const col = Array.from({ length: 36 }, (_, k) => mag[k * frames + 4] ?? 0)
    expect(argmaxOf(col)).toBe(24)
  })

  it('differentiates the transforms', () => {
    const x = tensor([0.3, -1.2, 0.8, 2.1, -0.4, 0.9, 1.5, -0.7, 0.2, 1.1, -0.3, 0.6, 1, -1, 0.5, 0.2])
    expectConformant((v) => absComplex(S.stft(v, { nperseg: 8 }).values), [x])
    expectConformant((v) => S.istft(S.stft(v, { nperseg: 8 }).values, { nperseg: 8 }).x, [x])
    expectConformant((v) => S.spectrogram(v, { nperseg: 8 }).values, [x])
    expectConformant((v) => S.instantaneousFrequency(v), [x])
  })
})
