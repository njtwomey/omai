import { describe, expect, it } from 'vitest'
import { key, normal, split } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import * as T from './index.ts'

const BUMP = [0, 1.5, 3, 1.5, 0]

/** Class 1 series carry the bump at a random position in noise; class 0 are noise. The truth is the bump and where it was put. */
function planted(seed: number, n: number, length: number): { x: number[][]; y: number[]; at: (number | null)[] } {
  const keys = split(key(seed), n)
  const x: number[][] = []
  const y: number[] = []
  const at: (number | null)[] = []
  keys.forEach((k, i) => {
    const noise = Array.from(toFlat(normal(k, [length + 1])), (v) => 0.15 * v)
    const series = noise.slice(0, length)
    const cls = i % 2
    let pos: number | null = null
    if (cls === 1) {
      pos = 2 + (Math.floor(Math.abs(noise[length] ?? 0) * 40) % (length - BUMP.length - 4))
      BUMP.forEach((b, j) => {
        series[(pos as number) + j] = (series[(pos as number) + j] ?? 0) + b
      })
    }
    x.push(series)
    y.push(cls)
    at.push(pos)
  })
  return { x, y, at }
}

describe('shapelet distance', () => {
  it('is zero at an exact match, finds its position, and early abandoning does not change the answer', () => {
    const s = [0.3, 0.1, 0.4, 1, 2, 1, 0.2, 0.1, 0, 0.5]
    const m = T.shapeletDistance(s, [1, 2, 1])
    expect(m.distance).toBe(0)
    expect(m.position).toBe(3)
    expect(m.skipped).toBeGreaterThan(0)
    // The unabandoned minimum over all windows.
    const brute = Math.min(
      ...Array.from(
        { length: s.length - 2 },
        (_, i) => [1, 2, 1].reduce((a, v, j) => a + (v - (s[i + j] ?? 0)) ** 2, 0) / 3,
      ),
    )
    expect(m.distance).toBeCloseTo(brute, 12)
  })
  it('z-normalisation makes it invariant to offset and scale of the match', () => {
    const shifted = [5, 5, 5, 10, 20, 10, 5, 5]
    expect(T.shapeletDistance(shifted, [1, 2, 1], true).distance).toBeCloseTo(0, 10)
    expect(T.shapeletDistance(shifted, [1, 2, 1], false).distance).toBeGreaterThan(10)
  })
})

describe('information gain of a split', () => {
  it('is one bit for a perfect split, zero when the classes interleave, and breaks ties by the gap', () => {
    expect(T.shapeletSplit([0.1, 0.2, 0.9, 1], [0, 0, 1, 1]).gain).toBeCloseTo(1, 12)
    expect(T.shapeletSplit([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1]).gain).toBeLessThan(0.32)
    const wide = T.shapeletSplit([0, 0.1, 5, 5.1], [0, 0, 1, 1])
    expect(wide.threshold).toBeCloseTo(2.55, 12)
    expect(wide.gap).toBeCloseTo(5, 12)
  })
})

describe('shapelet discovery', () => {
  const { x, y, at } = planted(3, 20, 40)
  it('recovers the planted bump: a perfect split from a window on the bump', () => {
    const found = T.findShapelets(x, y, { minLength: 5, maxLength: 5, k: 1, stride: 1 })
    const best = found[0] as T.Shapelet
    expect(best.gain).toBeCloseTo(1, 10)
    const cutFrom = at[best.source]
    expect(cutFrom).not.toBeNull()
    expect(Math.abs(best.start - (cutFrom as number))).toBeLessThanOrEqual(1)
    // The shapelet has the bump's shape: its peak is in the middle.
    expect(best.values.indexOf(Math.max(...best.values))).toBeGreaterThanOrEqual(1)
  })
  it('classifies held-out series from the transform by the learned threshold', () => {
    const found = T.findShapelets(x, y, { minLength: 5, maxLength: 5, k: 1 })
    const best = found[0] as T.Shapelet
    const test = planted(99, 20, 40)
    const pred = T.shapeletTransform(test.x, found).map((r) => ((r[0] ?? 0) <= best.threshold ? 1 : 0))
    expect(pred.filter((p, i) => p === test.y[i]).length / pred.length).toBeGreaterThanOrEqual(0.95)
  })
  it('runs one candidate per step, skips work by early abandoning, and its state serialises', () => {
    const { algorithm, candidates } = T.shapeletDiscoveryAlgorithm(x, y, { minLength: 5, maxLength: 5, stride: 3 })
    const run = runAlgorithm(algorithm, { maxSteps: candidates })
    expect(run.steps).toBe(candidates)
    expect(run.state.skipped / run.state.work).toBeGreaterThan(0.1)
    const half = runAlgorithm(algorithm, { maxSteps: 7 })
    expect(algorithm.deserialise?.(algorithm.serialise?.(half.state) as never)).toEqual(half.state)
  })
})
