import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import * as V from './index.ts'

const bimodal = Array.from({ length: 8 }, (_, i) => Array.from({ length: 8 }, (_, j) => (j < 4 ? 0.2 : 0.8) + i / 200))

describe('thresholds', () => {
  it('chooses thresholds automatically between the modes', () => {
    for (const method of ['otsu', 'li', 'triangle'] as const) {
      const r = V.threshold(bimodal, method)
      expect(r.threshold).toBeGreaterThan(0.2)
      expect(r.threshold).toBeLessThan(0.84)
      expect(new Set(toFlat(r.image))).toEqual(new Set([0, 1]))
    }
    expect(() => V.threshold(bimodal, 0.5, { type: 'nope' as V.ThresholdType })).toThrow()
  })

  it('applies the five rules', () => {
    const x = [0.1, 0.5, 0.9]
    const rule = (type: V.ThresholdType): number[] => toFlat(V.threshold(x, 0.5, { type, maxValue: 2 }).image)
    expect(rule('binary')).toEqual([0, 0, 2])
    expect(rule('binary-inv')).toEqual([2, 2, 0])
    expect(rule('trunc')).toEqual([0.1, 0.5, 0.5])
    expect(rule('tozero')).toEqual([0, 0, 0.9])
    expect(rule('tozero-inv')).toEqual([0.1, 0.5, 0])
  })

  it('thresholds locally', () => {
    const ramp = Array.from({ length: 6 }, (_, i) =>
      Array.from({ length: 12 }, (_, j) => j / 12 + (i === j % 6 ? 0.3 : 0)),
    )
    const a = toFlat(V.adaptiveThreshold(ramp, { blockSize: 3, C: 3 / 255 }))
    const b = toFlat(V.adaptiveThreshold(ramp, { blockSize: 3, C: 3 / 255, type: 'binary-inv' }))
    for (const [i, v] of a.entries()) expect(v + (b[i] ?? 0)).toBe(1)
    expect(() => V.adaptiveThreshold(ramp, { blockSize: 4 })).toThrow()
    const n = toFlat(V.thresholdNiblack(bimodal, { windowSize: 3 }))
    const s = toFlat(V.thresholdSauvola(bimodal, { windowSize: 3 }))
    // Niblack lowers the local mean by k s; Sauvola scales it by 1 + k (s - 1) <= 1 for s <= 1.
    for (const [i, v] of s.entries()) expect(v).toBeLessThanOrEqual(n[i] ?? 0)
    expect(toFlat(V.thresholdSauvola([[0.5, 0.5, 0.5]], { windowSize: 3 }))).toEqual([0.4, 0.4, 0.4])
    expect(V.thresholdMultiOtsu(bimodal, { classes: 2, nbins: 16 }).length).toBe(1)
    expect(() => V.thresholdMultiOtsu([0.5, 0.5], { classes: 3 })).toThrow()
  })
})
