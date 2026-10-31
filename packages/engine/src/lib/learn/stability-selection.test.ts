import { describe, expect, it } from 'vitest'
import { foldIn, key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  lassoStabilityAlgorithm,
  lassoStabilitySelection,
  stabilitySelectionBound,
  stabilityThreshold,
} from './stability-selection.ts'

function planted(
  seed: number,
  n: number,
  p: number,
  support: readonly number[],
  amplitude: number,
): { x: number[][]; y: number[] } {
  const z = toFlat(normal(key(seed), [n * p]))
  const e = toFlat(normal(key(seed + 1), [n]))
  const x = Array.from({ length: n }, (_, i) => Array.from({ length: p }, (_, j) => z[i * p + j] ?? 0))
  return { x, y: x.map((r, i) => support.reduce((a, j) => a + amplitude * (r[j] ?? 0), 0) + (e[i] ?? 0)) }
}

describe('stability selection bound', () => {
  it('is q^2 / ((2 pi - 1) p) and inverts to the threshold', () => {
    expect(stabilitySelectionBound(8, 40, 0.9)).toBeCloseTo(64 / (0.8 * 40), 12)
    expect(stabilitySelectionBound(8, 40, 1)).toBeCloseTo(1.6, 12)
    expect(stabilityThreshold(4, 40, 1)).toBeCloseTo(0.7, 12)
    expect(stabilitySelectionBound(4, 40, stabilityThreshold(4, 40, 0.5))).toBeCloseTo(0.5, 12)
    expect(() => stabilityThreshold(8, 40, 1)).toThrow()
    expect(() => stabilitySelectionBound(8, 40, 0.5)).toThrow()
  })
})

describe('lasso stability selection', () => {
  const support = [1, 5, 9]
  it('selects the true variables with frequency near one and leaves the nulls below the threshold', () => {
    const { x, y } = planted(1, 200, 30, support, 1.5)
    const r = lassoStabilitySelection(x, y, { q: 6, subsamples: 40, threshold: 0.8, key: key(2) })
    expect(r.selected).toEqual(support)
    for (const j of support) expect(r.frequencies[j]).toBeGreaterThan(0.95)
    const nullMax = Math.max(...r.frequencies.filter((_, j) => !support.includes(j)))
    expect(nullMax).toBeLessThan(0.5)
    expect(r.meanSize).toBeGreaterThanOrEqual(6)
  })

  it('the expected number of false selections stays under the bound across replicates, at several thresholds', () => {
    const p = 30
    const reps = 20
    const q = 6
    const falseAt = { 0.6: 0, 0.7: 0, 0.9: 0 }
    let bound = 0
    for (let rep = 0; rep < reps; rep++) {
      // A harder problem: weak signals, so that nulls are often selected in some subsamples.
      const { x, y } = planted(100 + 2 * rep, 120, p, support, 0.4)
      const r = lassoStabilitySelection(x, y, { q, subsamples: 30, key: foldIn(key(3), rep), threshold: 0.6 })
      for (const pi of [0.6, 0.7, 0.9] as const) {
        const v = r.frequencies.filter((f, j) => f >= pi && !support.includes(j)).length
        falseAt[pi] += v / reps
      }
      bound = r.meanSize
    }
    for (const pi of [0.6, 0.7, 0.9] as const)
      expect(falseAt[pi]).toBeLessThanOrEqual(stabilitySelectionBound(bound, p, pi))
    // Raising the threshold removes false selections.
    expect(falseAt[0.9]).toBeLessThanOrEqual(falseAt[0.6])
  }, 60000)

  it('is reproducible by key and its state serialises', () => {
    const { x, y } = planted(7, 60, 6, [0], 2)
    const alg = lassoStabilityAlgorithm(x, y, { q: 2, subsamples: 5 })
    const a = runAlgorithm(alg, { maxSteps: 5, key: key(1) })
    const b = runAlgorithm(alg, { maxSteps: 5, key: key(1) })
    expect(a.state.counts).toEqual(b.state.counts)
    expect(alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(a.state))))?.counts).toEqual(a.state.counts)
    expect(a.state.counts[0]).toBe(5)
  })
})
