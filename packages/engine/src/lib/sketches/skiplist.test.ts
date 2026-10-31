/**
 * Tests for the skip list: it behaves as a sorted map under random operations (model-based), its tower heights are
 * geometric, searches take logarithmically many comparisons even for sorted input, the search trace is a staircase,
 * and the stepped algorithm's comparison count agrees with the real structure's.
 */

import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { skipList, skipListAlgorithm, skipListSearchPath } from './index.ts'

describe('skipList', () => {
  it('behaves as a sorted map under random inserts and deletes (property)', () => {
    const op = fc.record({ insert: fc.boolean(), k: fc.integer({ min: 0, max: 40 }), v: fc.nat(1000) })
    fc.assert(
      fc.property(fc.array(op, { maxLength: 150 }), fc.nat(1000), (ops, seed) => {
        const sl = skipList<number, number>({ key: key(seed), maxLevel: 6 })
        const model = new Map<number, number>()
        for (const { insert, k, v } of ops) {
          if (insert) {
            sl.insert(k, v)
            model.set(k, v)
          } else {
            const had = model.delete(k)
            if (sl.delete(k) !== had) return false
          }
        }
        const expected = [...model].sort((a, b) => a[0] - b[0])
        const got = sl.entries().map((e) => [e.key, e.value])
        return (
          sl.size === model.size &&
          JSON.stringify(got) === JSON.stringify(expected) &&
          expected.every(([k, v]) => sl.get(k as number) === v)
        )
      }),
      { numRuns: RUNS * 5 },
    )
  })

  it('draws geometric tower heights', () => {
    const p = 0.5
    const sl = skipList<number>({ key: key(1), p, maxLevel: 20 })
    const n = 20_000
    for (let i = 0; i < n; i++) sl.insert(i)
    const heights = sl.entries().map((e) => e.height)
    for (const h of [2, 3, 4]) {
      const frac = heights.filter((x) => x >= h).length / n
      const expected = p ** (h - 1)
      expect(Math.abs(frac - expected)).toBeLessThan(5 * Math.sqrt((expected * (1 - expected)) / n))
    }
    expect(sl.level).toBeLessThanOrEqual(20)
  })

  it('searches in logarithmically many comparisons even for sorted input', () => {
    const n = 8192
    const sl = skipList<number>({ key: key(2) })
    for (let i = 0; i < n; i++) sl.insert(i)
    let total = 0
    const probes = 400
    for (let t = 0; t < probes; t++) total += skipListSearchPath(sl, Math.floor((t * n) / probes)).comparisons
    const mean = total / probes
    // expected comparisons are about log_2(n) / p + 1 = 27 for p = 1/2 and far below n / 2
    expect(mean).toBeLessThan(40)
    expect(mean).toBeGreaterThan(8)
  })

  it('traces a staircase: levels from the top down, each visited key at least as small as the next level starts', () => {
    const sl = skipList<number>({ key: key(7) })
    for (let i = 0; i < 200; i++) sl.insert(i * 2)
    const trace = sl.searchPath(101)
    expect(trace.found).toBe(false)
    expect(trace.levels.at(-1)?.level).toBe(0)
    expect(trace.levels.map((l) => l.level)).toEqual(Array.from({ length: sl.level }, (_, i) => sl.level - 1 - i))
    expect(trace.comparisons).toBe(trace.levels.reduce((s, l) => s + l.visitedKeys.length, 0))
    const hit = sl.searchPath(100)
    expect(hit.found).toBe(true)
    expect(hit.value).toBe(100)
  })

  it('accepts a comparator and validates its options', () => {
    const sl = skipList<string>({ key: key(0), compare: (a, b) => a.length - b.length || (a < b ? -1 : a > b ? 1 : 0) })
    for (const s of ['ccc', 'a', 'bb', 'aa']) sl.insert(s)
    expect(sl.entries().map((e) => e.key)).toEqual(['a', 'aa', 'bb', 'ccc'])
    expect(() => skipList({ key: key(0), p: 1 })).toThrow(/p must/)
    expect(() => skipList({ key: key(0), maxLevel: 0 })).toThrow(/maxLevel/)
  })
})

describe('skipListAlgorithm', () => {
  it('inserts in order of arrival, ends sorted, and keeps comparisons logarithmic on sorted input', () => {
    const keys = Array.from({ length: 1024 }, (_, i) => i)
    const alg = skipListAlgorithm({ keys })
    const run = runAlgorithm(alg, { seed: 3, maxSteps: 2000 })
    expect(run.state.keys).toEqual(keys)
    expect(run.state.heights.every((h) => h >= 1)).toBe(true)
    const late = runAlgorithm(alg, { seed: 3, maxSteps: 1024 })
    expect(late.state.comparisons).toBeLessThan(60)
    const again = alg.deserialise?.(alg.serialise?.(run.state) ?? null)
    expect(again?.heights).toEqual(run.state.heights)
  })

  it('ignores a repeated key', () => {
    const run = runAlgorithm(skipListAlgorithm({ keys: [5, 3, 5, 9, 3] }), { seed: 1, maxSteps: 10 })
    expect(run.state.keys).toEqual([3, 5, 9])
    expect(run.state.inserted).toBe(5)
  })

  it('counts the comparisons the real structure makes for the same towers', () => {
    const keys = Array.from({ length: 120 }, (_, i) => (i * 53) % 211)
    const alg = skipListAlgorithm({ keys })
    for (const m of [10, 40, 120]) {
      const before = runAlgorithm(alg, { seed: 9, maxSteps: m - 1 }).state
      const after = runAlgorithm(alg, { seed: 9, maxSteps: m }).state
      // rebuild the structure before the last insertion from its keys and heights, in arrival order
      const arrival = keys.slice(0, m - 1)
      const height = new Map(before.keys.map((k, i) => [k, before.heights[i] ?? 1]))
      const real = skipList<number>({ key: key(0), heights: arrival.map((k) => height.get(k) ?? 1) })
      for (const k of arrival) real.insert(k)
      expect(skipListSearchPath(real, keys[m - 1] ?? 0).comparisons).toBe(after.comparisons)
    }
  })
})
