import { describe, expect, it } from 'vitest'
import { expectConformant } from '../../testing/check.ts'
import { key, normal } from '../random/index.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import { recorder } from '../trace/index.ts'
import * as S from './index.ts'

describe('detection', () => {
  it('holds the false-alarm rate of CFAR on exponential noise and finds targets', () => {
    const n = 20000
    const g = toFlat(normal(key(5), [2 * n]))
    const power = Array.from({ length: n }, (_, i) => ((g[2 * i] ?? 0) ** 2 + (g[2 * i + 1] ?? 0) ** 2) / 2)
    for (const method of ['ca', 'os'] as const) {
      const r = S.cfar(power, { train: 16, guard: 2, pfa: 1e-2, method })
      const rate = r.detected.filter(Boolean).length / (n - 36)
      expect(rate).toBeGreaterThan(0.006)
      expect(rate).toBeLessThan(0.014)
    }
    const withTarget = [...power.slice(0, 100)]
    withTarget[50] = 200
    for (const method of ['ca', 'go', 'so', 'os'] as const)
      expect(S.cfar(withTarget, { method }).detected[50]).toBe(true)
  })

  it('gives the theoretical ROC for matched-filter statistics', () => {
    const pd = S.rocKnownSignal([1e-3, 0.1], 9)
    expect(pd[1]).toBeGreaterThan(pd[0] ?? 1)
    expect(S.rocEnergyDetector([0.05], 4, 0)[0]).toBeCloseTo(0.05, 6)
  })
})

describe('estimation and adaptive filtering', () => {
  it('identifies an FIR system by Wiener-Hopf and by adaptive filters', () => {
    const x = toFlat(normal(key(6), [3000]))
    const h = [0.8, -0.4, 0.2]
    const d = x.map((_, i) => h.reduce((a, c, k) => a + c * (x[i - k] ?? 0), 0))
    const w = toFlat(S.firWiener(x, d, 3).taps)
    for (const [k, c] of h.entries()) expect(w[k]).toBeCloseTo(c, 1)
    for (const method of ['lms', 'nlms', 'rls'] as const) {
      const rec = recorder()
      const r = toFlat(
        S.adaptiveFilter(method, x, d, { taps: 3, ...(method === 'lms' ? { mu: 0.05 } : {}), tracer: rec }).w,
      )
      for (const [k, c] of h.entries()) expect(r[k]).toBeCloseTo(c, 2)
      expect(rec.run().events.length).toBeGreaterThan(0)
    }
    const noise = toFlat(normal(key(7), [2000]))
    const s = noise.map((_, i) => Math.sin(0.01 * i))
    const primary = s.map((v, i) => v + 0.5 * (noise[i] ?? 0) + 0.25 * (noise[i - 1] ?? 0))
    const out = toFlat(S.adaptiveNoiseCancel(primary, noise, { taps: 4, mu: 0.02 }).cleaned)
    let err = 0
    for (let i = 1500; i < 2000; i++) err += ((out[i] ?? 0) - (s[i] ?? 0)) ** 2
    expect(err / 500).toBeLessThan(0.01)
  })

  it('recovers signals from complex cepstra and predicts with LPC', () => {
    const x = [2, -1, 0.5, 0.25, -0.3, 0.1]
    const back = toFlat(S.inverseComplexCepstrum(S.complexCepstrum(x, 16)))
    for (const [i, v] of x.entries()) expect(back[i]).toBeCloseTo(v, 10)
    const ar = Array.from({ length: 400 }, (_, i) => Math.cos(0.3 * i) * 0.99 ** i)
    const a = S.lpc(ar, 2)
    const res = toFlat(S.predictionResidual(ar, a)).slice(10)
    for (const v of res) expect(Math.abs(v)).toBeLessThan(1e-2)
    expect(toFlat(S.lifter([1, 2, 3, 4], 4, 'sine'))[0]).toBeCloseTo(1 + 2 * Math.sin(Math.PI / 4), 12)
  })

  it('differentiates the estimators', () => {
    const x = tensor([0.3, -1.2, 0.8, 2.1, -0.4, 0.9, 1.5, -0.7])
    expectConformant((v) => S.nonCausalWiener(v, [1, 2, 1, 0.5, 0.2], 0.3), [x])
    expectConformant((v) => S.firWiener(v, [1, 0, 2, 1, -1, 0, 1, 2], 2).taps, [x])
    expectConformant((v) => S.wiener(v, 3), [x])
    expectConformant((v) => S.matchedFilter(v, [1, -1, 0.5]), [x])
    expectConformant((v) => S.realCepstrum(v), [x])
  })
})
