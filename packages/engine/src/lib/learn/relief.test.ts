import { describe, expect, it } from 'vitest'
import { key, normal } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { fClassif } from './feature-selection.ts'
import { relief, reliefAlgorithm } from './relief.ts'

describe('Relief', () => {
  it('matches the hand computation on four points (k = 1, every instance once)', () => {
    // Range 11. Hits are each point's partner at distance 1/11; the nearest miss of 0, 1, 10, 11 is at 10/11, 9/11, 9/11, 10/11.
    const r = relief([[0], [1], [10], [11]], [0, 0, 1, 1], { k: 1 })
    expect(r.weights[0]).toBeCloseTo((10 + 9 + 9 + 10) / 11 / 4 - 1 / 11, 12)
  })

  it('finds the two halves of an XOR that no univariate score sees', () => {
    const n = 300
    const z = toFlat(normal(key(1), [n * 8]))
    const x = Array.from({ length: n }, (_, i) => Array.from({ length: 8 }, (_, j) => z[i * 8 + j] ?? 0))
    const y = x.map((r) => ((r[0] ?? 0) * (r[1] ?? 0) > 0 ? 1 : 0))
    const r = relief(x, y, { k: 10 })
    expect(r.ranking.slice(0, 2).sort()).toEqual([0, 1])
    const noiseMax = Math.max(...r.weights.slice(2))
    expect(Math.min(r.weights[0] ?? 0, r.weights[1] ?? 0)).toBeGreaterThan(2 * noiseMax)
    // The ANOVA F score is blind to them: both p-values are large.
    const f = fClassif(x, y)
    expect(f.pValues?.[0]).toBeGreaterThan(0.01)
    expect(f.pValues?.[1]).toBeGreaterThan(0.01)
  })

  it('weights informative features of a three-class problem above noise, with class-prior weighted misses', () => {
    const n = 240
    const z = toFlat(normal(key(2), [n * 5]))
    const cls = Array.from({ length: n }, (_, i) => (i < 120 ? 0 : i < 180 ? 1 : 2))
    const x = Array.from({ length: n }, (_, i) =>
      Array.from(
        { length: 5 },
        (_, j) => (z[i * 5 + j] ?? 0) + (j === 0 ? 2.5 * (cls[i] ?? 0) : j === 1 ? 1.5 * ((cls[i] ?? 0) % 2) : 0),
      ),
    )
    const r = relief(x, cls, { k: 5 })
    expect(r.ranking.slice(0, 2)).toEqual([0, 1])
    expect(r.weights[0]).toBeGreaterThan(0)
    expect(Math.max(...r.weights.slice(2))).toBeLessThan(0.2 * (r.weights[0] ?? 0))
  })

  it('samples m < n instances reproducibly by key and honours the discrete mask', () => {
    const noise = toFlat(normal(key(3), [40]))
    const x = Array.from({ length: 40 }, (_, i) => [i % 2, noise[i] ?? 0])
    const y = x.map((r) => r[0] ?? 0)
    const a = relief(x, y, { k: 3, iterations: 15, key: key(4), discrete: [true, false] })
    const b = relief(x, y, { k: 3, iterations: 15, key: key(4), discrete: [true, false] })
    const c = relief(x, y, { k: 3, iterations: 15, key: key(5), discrete: [true, false] })
    expect(a.weights).toEqual(b.weights)
    expect(a.weights).not.toEqual(c.weights)
    expect(a.ranking[0]).toBe(0)
  })

  it('serialises the state and resumes: the run from a checkpoint equals the uninterrupted run', () => {
    const x = Array.from({ length: 30 }, (_, i) => [(i * 3) % 7, i % 3, Math.sin(i)])
    const y = x.map((r) => ((r[0] ?? 0) > 3 ? 1 : 0))
    const alg = reliefAlgorithm(x, y, { k: 2 })
    const full = runAlgorithm(alg, { maxSteps: 30, key: key(9) })
    const half = runAlgorithm(alg, { maxSteps: 12, key: key(9) })
    const restored = alg.deserialise?.(JSON.parse(JSON.stringify(alg.serialise?.(half.state))))
    const rest = runAlgorithm(alg, { maxSteps: 30, key: key(9), start: { state: restored, step: 12 } })
    expect(rest.state.weights).toEqual(full.state.weights)
  })

  it('rejects one class and a bad k', () => {
    expect(() => relief([[1], [2]], [0, 0])).toThrow()
    expect(() => relief([[1], [2]], [0, 1], { k: 0 })).toThrow()
  })
})
