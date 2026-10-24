import { describe, expect, it } from 'vitest'
import { foldIn, key, normal } from '../../random/index.ts'
import { toFlat } from '../../tensor/index.ts'
import { recorder } from '../../trace/index.ts'
import * as T from './index.ts'

describe('sequential tests', () => {
  it("Lan-DeMets boundaries match published O'Brien-Fleming and Pocock designs", () => {
    // gsDesign / Jennison and Turnbull: one-sided alpha 0.025, five equally spaced looks.
    const t = [0.2, 0.4, 0.6, 0.8, 1]
    const obf = toFlat(T.alphaSpending(t, { alpha: 0.025, sides: 1 }).boundaries)
    ;[4.8769, 3.3569, 2.6803, 2.2898, 2.031].forEach((v, i) => {
      expect(obf[i]).toBeCloseTo(v, 3)
    })
    const pocock = toFlat(T.alphaSpending(t, { alpha: 0.025, sides: 1, kind: 'pocock' }).boundaries)
    expect(pocock[0]).toBeCloseTo(2.438, 2)
    expect(toFlat(T.alphaSpending(t, { sides: 2 }).boundaries)[0]).toBeCloseTo(4.8769, 3)
    const spent = T.alphaSpending(t, { kind: 'power', rho: 2 }).spent
    expect(spent.at(-1)).toBeCloseTo(0.05, 12)
    expect(() => T.alphaSpending([0.5, 0.4])).toThrow()
  })

  it('the SPRT controls its error rates and stops early', () => {
    let rejections = 0
    let samples = 0
    for (let i = 0; i < 300; i++) {
      const x = toFlat(normal(foldIn(key(1), i), [400]))
      const r = T.sprt(x, { logLikelihoodRatio: T.sprtNormal(0, 0.5, 1), alpha: 0.05, beta: 0.1 })
      if (r.decision === 'reject H0') rejections++
      samples += r.n
    }
    expect(rejections / 300).toBeLessThan(0.08)
    expect(samples / 300).toBeLessThan(60)
    const rec = recorder()
    T.sprt([1, 1, 1, 1, 1], { logLikelihoodRatio: T.sprtBernoulli(0.3, 0.7), tracer: rec })
    expect(rec.run().events.length).toBeGreaterThan(0)
  })

  it('always-valid p-values and confidence sequences survive continuous monitoring', () => {
    let early = 0
    let misses = 0
    for (let i = 0; i < 200; i++) {
      const x = toFlat(normal(foldIn(key(2), i), [300]))
      const p = toFlat(T.alwaysValidPValues(x, { sigma: 1 }))
      if ((p.at(-1) ?? 1) < 0.05) early++
      for (let j = 1; j < p.length; j++) expect(p[j] ?? 0).toBeLessThanOrEqual(p[j - 1] ?? 0)
      const cs = T.confidenceSequence(x, { sigma: 1 })
      const lo = toFlat(cs.lower)
      const hi = toFlat(cs.upper)
      if (lo.some((v, j) => v > 0 || (hi[j] ?? 0) < 0)) misses++
    }
    expect(early / 200).toBeLessThan(0.08)
    expect(misses / 200).toBeLessThan(0.08)
    expect(T.msprt([1, 1.2, 0.9, 1.1, 1.3, 0.8, 1.0, 1.2, 1.1, 0.9], { sigma: 1 }).pValue).toBeLessThan(0.05)
  })
})
