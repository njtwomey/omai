import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { argmaxOf } from '../base/index.ts'
import { key, normal } from '../random/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

const x = tensor([0.3, -1.2, 0.8, 2.1, -0.4, 0.9, 1.5, -0.7, 0.2, 1.1, -0.3, 0.6])

describe('spectral estimates', () => {
  it('reduces the Blackman-Tukey estimate with all lags and a flat lag window to the periodogram', () => {
    const bt = toFlat(S.blackmanTukey(x, { maxLag: 11, window: 'boxcar', nfft: 32 }).values)
    const p = toFlat(S.periodogram(x, { nfft: 32 }).values)
    for (const [i, v] of bt.entries()) expect(v).toBeCloseTo(p[i] ?? 0, 10)
  })

  it('estimates the level of white noise and finds tones with multitapers', () => {
    const noise = toFlat(normal(key(7), [4096]))
    const psd = toFlat(S.multitaper(noise, { nw: 4, fs: 1 }).values)
    const inner = psd.slice(10, -10)
    const avg = inner.reduce((a, v) => a + v, 0) / inner.length
    const mu = noise.reduce((a, v) => a + v, 0) / noise.length
    const variance = noise.reduce((a, v) => a + (v - mu) ** 2, 0) / noise.length
    expect(avg / (2 * variance)).toBeCloseTo(1, 1)
    const n = 256
    const tone = Array.from(
      { length: n },
      (_, k) => Math.cos(2 * Math.PI * 0.2 * k) + 0.01 * (toFlat(normal(key(1), [n]))[k] ?? 0),
    )
    for (const adaptive of [false, true]) {
      const est = toFlat(S.multitaper(tone, { nw: 3, adaptive }).values)
      const at = argmaxOf(est)
      expect(Math.abs(at - 0.2 * n)).toBeLessThanOrEqual(3)
    }
  })

  it('gives the degrees of freedom of averaged estimates', () => {
    expect(S.welchDof(1000, 100, 0, 'boxcar')).toBe(20)
    const dof = S.welchDof(1000, 100, 50, 'hann')
    expect(dof).toBeGreaterThan(20)
    expect(dof).toBeLessThan(40)
  })

  it('differentiates the estimates in the signal', () => {
    expectConformant((v) => S.periodogram(v, { window: 'hann', detrend: 'linear' }).values, [x])
    expectConformant((v) => S.welch(v, { nperseg: 6 }).values, [x])
    expectConformant((v) => S.welch(v, { nperseg: 4, average: 'median' }).values, [x])
    expectConformant((v) => S.csd(v, x, { nperseg: 6 }).values.im, [x])
    expectConformant(
      (v) =>
        S.coherence(
          v,
          x.data.map((q) => q * q),
          { nperseg: 6 },
        ).values,
      [x],
    )
    expectConformant((v) => S.blackmanTukey(v, { maxLag: 3 }).values, [x])
    expectConformant((v) => S.multitaper(v, { nw: 2, k: 2 }).values, [x])
    const t = [0, 0.7, 1.3, 2.2, 3.1, 3.8, 4.6, 5.9]
    expectConformant(
      (v) => S.lombScargle(t, v, [0.5, 1.5], { normalize: 'normalize', floatingMean: true }),
      [tensor([1, 0.2, -0.5, 0.3, 1.2, -0.8, 0.4, 0.1])],
    )
  })
})
