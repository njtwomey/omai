import { describe, expect, it } from 'vitest'
import { foldIn, key } from '../../../lib/random/index.ts'
import { discrete } from '../../../lib/space/index.ts'
import {
  encodedSize,
  encodeObservation,
  reservoir,
  reservoirAdd,
  reservoirSample,
  sampleIndex,
  solveZeroSum,
  tabularQ,
} from './_multi-agent.ts'

describe('zero-sum matrix games', () => {
  it('solve pure saddle points, mixed equilibria and non-square games exactly', () => {
    const saddle = solveZeroSum([
      [3, 1],
      [4, 2],
    ])
    expect(saddle).toMatchObject({ row: [0, 1], column: [0, 1], value: 2, gap: 0 })
    const pennies = solveZeroSum([
      [1, -1],
      [-1, 1],
    ])
    expect(pennies.value).toBeCloseTo(0, 12)
    for (const p of [...pennies.row, ...pennies.column]) expect(p).toBeCloseTo(0.5, 12)
    const rps = solveZeroSum([
      [0, -1, 1],
      [1, 0, -1],
      [-1, 1, 0],
    ])
    for (const p of rps.row) expect(p).toBeCloseTo(1 / 3, 12)
    // A 2 x 3 game: the third column is never played; on the first two both players mix evenly, value 1/2.
    const wide = solveZeroSum([
      [2, -1, 0],
      [-1, 2, 3],
    ])
    expect(wide.gap).toBeLessThan(1e-9)
    expect(wide.value).toBeCloseTo(0.5, 12)
    expect(() => solveZeroSum([])).toThrow(/empty/)
  })

  it('fall back to regret matching+ for larger games, to the tolerance', () => {
    // 7 x 7 generalised rock-paper-scissors: uniform equilibrium, value 0.
    const n = 7
    const m = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => (i === j ? 0 : (j - i + n) % n <= 3 ? -1 : 1)),
    )
    const r = solveZeroSum(m, { tol: 1e-4 })
    expect(r.gap).toBeLessThan(1e-4)
    expect(Math.abs(r.value)).toBeLessThan(1e-4)
    expect(r.iterations).toBeGreaterThan(0)
  })
})

describe('reservoir sampling', () => {
  it('keeps a uniform sample of everything offered, persistently', () => {
    let r = reservoir<number>(100)
    for (let i = 0; i < 1000; i++) r = reservoirAdd(r, i, foldIn(key(0), i))
    expect(r.size).toBe(100)
    expect(r.seen).toBe(1000)
    const sample = reservoirSample(r, key(1), 2000)
    let mean = 0
    for (const x of sample) mean += x / sample.length
    expect(Math.abs(mean - 500)).toBeLessThan(80)
    const small = reservoirAdd(reservoir<number>(2), 7, key(0))
    expect(reservoirSample(small, key(0), 3)).toEqual([7, 7, 7])
    expect(() => reservoirSample(reservoir(2), key(0), 1)).toThrow(/empty/)
  })
})

describe('helpers', () => {
  it('sample indices, encode observations and learn tabular Q-values', () => {
    expect(sampleIndex(key(0), [0, 0, 1])).toBe(2)
    expect(encodeObservation(discrete(3), 1)).toEqual([0, 1, 0])
    expect(encodeObservation(undefined, 2.5)).toEqual([2.5])
    expect(encodeObservation(undefined, [1, 2])).toEqual([1, 2])
    expect(() => encodeObservation(undefined, 'x')).toThrow(/numbers/)
    expect(encodedSize(discrete(4))).toBe(4)
    const q = tabularQ(2, { alpha: 1, gamma: 0.5 })
    const s = q.learn(
      q.learn(
        q.init(key(0)),
        { observation: 1, action: 1, reward: 2, nextObservation: 2, terminated: true, truncated: false },
        key(0),
      ),
      { observation: 0, action: 0, reward: 1, nextObservation: 1, terminated: false, truncated: false },
      key(0),
    )
    expect(s.q['0']).toEqual([2, 0])
    expect(q.greedy(s, 1)).toBe(1)
  })
})
