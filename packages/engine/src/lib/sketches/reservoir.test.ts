/**
 * Tests for reservoir sampling: inclusion probabilities `k / n` and pair probabilities `k (k - 1) / (n (n - 1))` for
 * the unweighted rules, the successive-sampling inclusion probabilities for the weighted ones, determinism from the
 * key, and the stepped algorithm.
 */

import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { reservoirAlgorithm, reservoirSample, reservoirSampler } from './index.ts'

const range = (n: number): number[] => Array.from({ length: n }, (_, i) => i)

describe('unweighted reservoir rules', () => {
  for (const method of ['R', 'L'] as const) {
    it(`${method}: every item and every pair is included with the uniform probability`, () => {
      const n = 20
      const k = 5
      const trials = 6000
      const inclusion = new Array<number>(n).fill(0)
      let pairA = 0
      let pairB = 0
      for (let t = 0; t < trials; t++) {
        const { sample } = reservoirSample(range(n), { k, key: key(t), method })
        expect(new Set(sample).size).toBe(k)
        for (const x of sample) inclusion[x] = (inclusion[x] ?? 0) + 1
        if (sample.includes(0) && sample.includes(1)) pairA += 1
        if (sample.includes(18) && sample.includes(19)) pairB += 1
      }
      const p = k / n
      const sd = Math.sqrt((p * (1 - p)) / trials)
      for (const c of inclusion) expect(Math.abs(c / trials - p)).toBeLessThan(5 * sd)
      const pp = (k * (k - 1)) / (n * (n - 1))
      const sdPair = Math.sqrt((pp * (1 - pp)) / trials)
      expect(Math.abs(pairA / trials - pp)).toBeLessThan(5 * sdPair)
      expect(Math.abs(pairB / trials - pp)).toBeLessThan(5 * sdPair)
    })
  }

  it('L draws far fewer random numbers than items on a long stream yet stays spread over the stream', () => {
    const r = reservoirSampler<number>({ k: 10, key: key(1), method: 'L' })
    for (let i = 0; i < 200_000; i++) r.add(i)
    const mean = r.sample().reduce((a, b) => a + b, 0) / 10
    expect(r.seen).toBe(200_000)
    expect(mean).toBeGreaterThan(40_000)
    expect(mean).toBeLessThan(160_000)
  })

  it('keeps everything while the stream is shorter than k and is reproducible from the key', () => {
    expect(reservoirSample([1, 2, 3], { k: 5, key: key(0) }).sample.toSorted()).toEqual([1, 2, 3])
    const a = reservoirSample(range(100), { k: 7, key: key(3) }).sample
    expect(reservoirSample(range(100), { k: 7, key: key(3) }).sample).toEqual(a)
    expect(reservoirSample(range(100), { k: 7, key: key(4) }).sample).not.toEqual(a)
  })

  it('validates k and weights', () => {
    expect(() => reservoirSampler({ k: 0, key: key(0) })).toThrow(/k must/)
    expect(() => reservoirSample([1, 2], { k: 1, key: key(0), weights: [1] })).toThrow(/weights/)
    expect(() => reservoirSample([1, 2], { k: 1, key: key(0), weights: [1, -1] })).toThrow(/positive/)
  })
})

describe('weighted reservoir rules', () => {
  const weights = [1, 2, 3, 4]
  const W = 10
  // successive sampling without replacement: inclusion probability of item i in a sample of two
  const inclusion2 = weights.map(
    (wi, i) => wi / W + weights.reduce((s, wj, j) => (j === i ? s : s + (wj / W) * (wi / (W - wj))), 0),
  )

  for (const method of ['A-Res', 'A-ExpJ'] as const) {
    it(`${method}: k = 1 picks item i with probability w_i / sum w`, () => {
      const trials = 8000
      const counts = [0, 0, 0, 0]
      for (let t = 0; t < trials; t++) {
        const pick = reservoirSample([0, 1, 2, 3], { k: 1, key: key(t), weights, method }).sample[0] ?? 0
        counts[pick] = (counts[pick] ?? 0) + 1
      }
      weights.forEach((w, i) => {
        const p = w / W
        expect(Math.abs((counts[i] ?? 0) / trials - p)).toBeLessThan(5 * Math.sqrt((p * (1 - p)) / trials))
      })
    })

    it(`${method}: k = 2 inclusion matches successive sampling`, () => {
      const trials = 8000
      const counts = [0, 0, 0, 0]
      for (let t = 0; t < trials; t++) {
        for (const x of reservoirSample([0, 1, 2, 3], { k: 2, key: key(50_000 + t), weights, method }).sample)
          counts[x] = (counts[x] ?? 0) + 1
      }
      inclusion2.forEach((p, i) => {
        expect(Math.abs((counts[i] ?? 0) / trials - p)).toBeLessThan(5 * Math.sqrt((p * (1 - p)) / trials))
      })
    })
  }

  it('the jump rule gives the same inclusion on a long weighted stream as A-Res', () => {
    const n = 400
    const w = range(n).map((i) => 1 + (i % 10))
    const share = (method: 'A-Res' | 'A-ExpJ'): number => {
      let heavy = 0
      const trials = 500
      for (let t = 0; t < trials; t++) {
        const { sample } = reservoirSample(range(n), { k: 20, key: key(t), weights: w, method })
        heavy += sample.filter((i) => (w[i] ?? 0) >= 6).length
      }
      return heavy / (trials * 20)
    }
    expect(Math.abs(share('A-Res') - share('A-ExpJ'))).toBeLessThan(0.02)
    expect(share('A-Res')).toBeGreaterThan(0.6)
  })
})

describe('reservoirAlgorithm', () => {
  it('samples indices uniformly and serialises, step by step', () => {
    const n = 30
    const k = 6
    const counts = new Array<number>(n).fill(0)
    const trials = 800
    for (let s = 0; s < trials; s++) {
      for (const i of runAlgorithm(reservoirAlgorithm({ length: n, k }), { seed: s, maxSteps: 100 }).state.indices)
        counts[i] = (counts[i] ?? 0) + 1
    }
    const p = k / n
    for (const c of counts) expect(Math.abs(c / trials - p)).toBeLessThan(5 * Math.sqrt((p * (1 - p)) / trials))
    const alg = reservoirAlgorithm({ length: n, k, weights: range(n).map((i) => 1 + i) })
    const run = runAlgorithm(alg, { seed: 1, maxSteps: 100 })
    expect(run.state.seen).toBe(n)
    const again = alg.deserialise?.(alg.serialise?.(run.state) ?? null)
    expect(again?.indices).toEqual(run.state.indices)
    // late items are heavier, so they are over-represented
    expect(run.state.indices.reduce((a, b) => a + b, 0) / k).toBeGreaterThan(14)
  })

  it('resumes to the same sample as an uninterrupted run (steps depend only on the key and t)', () => {
    const alg = reservoirAlgorithm({ length: 40, k: 4 })
    const whole = runAlgorithm(alg, { seed: 5, maxSteps: 100 })
    const half = runAlgorithm(alg, { seed: 5, maxSteps: 20 })
    const rest = runAlgorithm(alg, { seed: 5, maxSteps: 100, start: { state: half.state, step: half.steps } })
    expect(rest.state.indices).toEqual(whole.state.indices)
  })
})
