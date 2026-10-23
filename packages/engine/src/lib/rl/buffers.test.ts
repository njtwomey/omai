import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { minOf } from '../base/index.ts'
import { key } from '../random/index.ts'
import {
  bufferAdd,
  bufferGet,
  bufferItems,
  bufferSample,
  episodeAdd,
  episodeStore,
  minibatchIndices,
  nStepAdd,
  nStepBuffer,
  prioritizedAdd,
  prioritizedReplay,
  prioritizedSample,
  prioritizedUpdate,
  replayBuffer,
  rolloutAdd,
  rolloutAdvantages,
  rolloutBuffer,
  sampleSequences,
  sumTree,
  sumTreeFind,
  sumTreeGet,
  sumTreeMin,
  sumTreeSet,
  sumTreeTotal,
} from './buffers.ts'

describe('replay buffers', () => {
  it('keep the last capacity items in order and leave old versions intact', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 20 }), fc.array(fc.integer(), { maxLength: 60 }), (cap, xs) => {
        let b = replayBuffer<number>(cap, { chunkSize: 3 })
        const versions = [b]
        for (const x of xs) {
          b = bufferAdd(b, x)
          versions.push(b)
        }
        expect(bufferItems(b)).toEqual(xs.slice(-cap))
        expect(b.added).toBe(xs.length)
        // Persistence: every earlier version still holds exactly its own items.
        for (const [i, v] of versions.entries()) expect(bufferItems(v)).toEqual(xs.slice(0, i).slice(-cap))
      }),
      { numRuns: 50 },
    )
  })

  it('sample uniformly with replacement and reject empty buffers', () => {
    let b = replayBuffer<number>(10)
    for (let i = 0; i < 10; i++) b = bufferAdd(b, i)
    const counts = new Array<number>(10).fill(0)
    const { slots, items } = bufferSample(b, key(0), 5000)
    for (const s of slots) counts[s] = (counts[s] ?? 0) + 1
    expect(items[0]).toBe(bufferGet(b, slots[0] ?? 0))
    for (const c of counts) expect(Math.abs(c / 5000 - 0.1)).toBeLessThan(0.02)
    expect(() => bufferSample(replayBuffer<number>(3), key(0), 1)).toThrow(/empty/)
    expect(() => bufferGet(b, 10)).toThrow(/empty/)
    expect(() => replayBuffer(0)).toThrow(/positive/)
  })
})

describe('sum trees and prioritised replay', () => {
  it('maintain sums, minima and prefix-sum search under updates', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 33 }),
        fc.array(fc.tuple(fc.nat(), fc.oneof(fc.constant(0), fc.double({ min: 1e-3, max: 10, noNaN: true }))), {
          maxLength: 40,
        }),
        (cap, updates) => {
          let t = sumTree(cap)
          const values = new Array<number>(cap).fill(0)
          const set = new Set<number>()
          for (const [i, v] of updates) {
            t = sumTreeSet(t, i % cap, v)
            values[i % cap] = v
            set.add(i % cap)
          }
          const total = values.reduce((a, b) => a + b, 0)
          expect(sumTreeTotal(t)).toBeCloseTo(total, 9)
          for (let i = 0; i < cap; i++) expect(sumTreeGet(t, i)).toBe(values[i])
          const min = set.size === 0 ? Infinity : minOf([...set].map((i) => values[i] ?? 0))
          expect(sumTreeMin(t)).toBe(min)
          if (total > 0) {
            let prefix = 0
            for (let i = 0; i < cap; i++) {
              const v = values[i] ?? 0
              if (v > 0) expect(sumTreeFind(t, prefix + v / 2)).toBe(i)
              prefix += v
            }
          }
        },
      ),
      { numRuns: 60 },
    )
    expect(() => sumTreeSet(sumTree(2), 2, 1)).toThrow(/out of range/)
    expect(() => sumTreeSet(sumTree(2), 0, -1)).toThrow(/non-negative/)
  })

  it('sample in proportion to priority^alpha with correcting weights', () => {
    let b = prioritizedReplay<string>(4, { alpha: 1 })
    for (const x of ['a', 'b', 'c', 'd']) b = prioritizedAdd(b, x)
    b = prioritizedUpdate(b, [0, 1, 2, 3], [1, 1, 1, 7])
    const counts: Record<string, number> = {}
    for (let i = 0; i < 200; i++) {
      for (const item of prioritizedSample(b, key(i), 10).items) counts[item] = (counts[item] ?? 0) + 1
    }
    expect(Math.abs((counts['d'] ?? 0) / 2000 - 0.7)).toBeLessThan(0.04)
    const s = prioritizedSample(b, key(0), 4, 1)
    for (const [i, w] of s.weights.entries()) {
      // With beta = 1 the weights are 1 / (N P) normalised by the largest, that of the smallest priority.
      expect(w).toBeCloseTo(0.1 / (s.probabilities[i] ?? 1), 5)
    }
    expect(b.maxPriority).toBeCloseTo(7, 5)
    expect(() => prioritizedSample(prioritizedReplay<number>(2), key(0), 1)).toThrow(/empty/)
  })
})

describe('n-step, rollout and episode buffers', () => {
  const step = (o: number, r: number, terminated = false, truncated = false) => ({
    observation: o,
    action: 0,
    reward: r,
    nextObservation: o + 1,
    terminated,
    truncated,
  })

  it('emit n-step transitions and flush at episode ends', () => {
    let b = nStepBuffer<number, number>(3, 0.5)
    const out: ReturnType<typeof nStepAdd<number, number>>['ready'] = []
    for (const t of [step(0, 1), step(1, 2), step(2, 4), step(3, 8, true)]) {
      const r = nStepAdd(b, t)
      b = r.buffer
      out.push(...r.ready)
    }
    expect(out.map((t) => [t.observation, t.reward, t.nextObservation, t.steps, t.discount, t.terminated])).toEqual([
      [0, 1 + 1 + 1, 3, 3, 0.125, false],
      [1, 2 + 2 + 2, 4, 3, 0.125, true],
      [2, 4 + 4, 4, 2, 0.25, true],
      [3, 8, 4, 1, 0.5, true],
    ])
    expect(b.pending).toHaveLength(0)
    expect(() => nStepBuffer(0, 1)).toThrow(/positive/)
  })

  it('compute rollout advantages with truncation bootstraps and shuffle minibatches', () => {
    let r = rolloutBuffer<number, number>()
    const base = { observation: 0, action: 0, value: 0, logProb: 0 }
    r = rolloutAdd(r, { ...base, reward: 1, terminated: false, truncated: true, bootstrapValue: 10 })
    r = rolloutAdd(r, { ...base, reward: 1, terminated: false, truncated: false })
    const { advantages, returns } = rolloutAdvantages(r, 2, 0.5, 1)
    expect(advantages).toEqual([1 + 0.5 * 10, 1 + 0.5 * 2])
    expect(returns).toEqual(advantages)
    const batches = minibatchIndices(10, 4, key(1))
    expect(batches.map((x) => x.length)).toEqual([4, 4, 2])
    expect(batches.flat().sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
  })

  it('store whole episodes and sample windows within them', () => {
    let s = episodeStore<number>(2)
    for (let i = 0; i < 12; i++) s = episodeAdd(s, i, i % 4 === 3)
    expect(s.episodes).toEqual([
      [4, 5, 6, 7],
      [8, 9, 10, 11],
    ])
    for (const seq of sampleSequences(s, key(0), 20, 2)) {
      expect(seq).toHaveLength(2)
      expect((seq[1] ?? 0) - (seq[0] ?? 0)).toBe(1)
    }
    expect(() => sampleSequences(episodeStore<number>(1), key(0), 1, 1)).toThrow(/no finished/)
  })
})
