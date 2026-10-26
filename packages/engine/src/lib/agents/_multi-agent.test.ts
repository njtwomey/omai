import { describe, expect, it } from 'vitest'
import { foldIn, key } from '../random/index.ts'
import { discrete } from '../space/index.ts'
import {
  encodedSize,
  encodeObservation,
  reservoir,
  reservoirAdd,
  reservoirSample,
  sampleIndex,
  tabularQ,
} from './_multi-agent.ts'

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
