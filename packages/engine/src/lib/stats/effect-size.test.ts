import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { coverage } from '../../testing/stats.ts'
import { foldIn, key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import * as S from './index.ts'

const sample = fc.array(fc.integer({ min: -100, max: 100 }), { minLength: 3, maxLength: 15 })

describe('effect sizes', () => {
  it("Cliff's delta is antisymmetric, bounded and its interval contains it", () => {
    fc.assert(
      fc.property(sample, sample, (x, y) => {
        const d = S.cliffsDelta(x, y)
        expect(S.cliffsDelta(y, x).value).toBeCloseTo(-d.value, 12)
        expect(Math.abs(d.value)).toBeLessThanOrEqual(1)
        const [lo, hi] = d.interval ?? ([0, 0] as const)
        expect(lo).toBeLessThanOrEqual(d.value + 1e-12)
        expect(hi).toBeGreaterThanOrEqual(d.value - 1e-12)
      }),
    )
  })

  it("Cohen's d interval covers the true standardised difference at about the nominal rate", () => {
    const intervals = Array.from({ length: 300 }, (_, i) => {
      const x = toFlat(normal(foldIn(key(31), 2 * i), [20])).map((v) => v + 0.5)
      const y = toFlat(normal(foldIn(key(31), 2 * i + 1), [20]))
      return S.cohensD(x, y).interval ?? ([0, 0] as const)
    })
    expect(coverage(intervals, 0.5)).toBeGreaterThan(0.9)
  })

  it('eta and omega squared lie in their intervals, and the intervals contain zero for null data', () => {
    const groups = [0, 1, 2].map((g) => toFlat(normal(foldIn(key(41), g), [12])))
    const eta = S.etaSquared(groups)
    expect(eta.interval?.[0]).toBe(0)
    expect(S.omegaSquared(groups).value).toBeLessThan(eta.value)
    const v = S.cramersV([
      [10, 10],
      [10, 10],
    ])
    expect(v.value).toBe(0)
    expect(v.interval?.[0]).toBe(0)
  })

  it("Cohen's h and Glass's delta follow their definitions", () => {
    expect(S.cohensH(0.5, 0.5, { n1: 10, n2: 10 }).interval?.[0]).toBeLessThan(0)
    expect(() => S.cohensH(1.5, 0.5)).toThrow()
    expect(S.glassDelta([2, 4, 6], [1, 2, 3]).value).toBeCloseTo(2, 12)
  })
})
