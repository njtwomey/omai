/**
 * Tests for MinHash and banded LSH: the signature estimate is unbiased for the Jaccard similarity with the binomial
 * spread, merge is the union's signature, the empirical candidate rate follows the S-curve, and the banding optimiser
 * agrees with an independent integration.
 */

import { describe, expect, it } from 'vitest'
import { quad } from '../numerics/quadrature/index.ts'
import { key } from '../random/index.ts'
import { item as itemOf, pow, sub, type Value } from '../tensor/index.ts'
import { exactJaccard, lshCollisionProbability, lshIndex, lshOptimalParams, minHash, minHashJaccard } from './index.ts'

const range = (from: number, to: number, prefix = 'e'): string[] =>
  Array.from({ length: to - from }, (_, i) => `${prefix}${from + i}`)

describe('minHash', () => {
  const A = range(0, 300)
  const B = range(100, 400)
  const J = exactJaccard(A, B)

  it('estimates the Jaccard similarity without bias and with binomial spread', () => {
    const k = 128
    const estimates = Array.from({ length: 150 }, (_, s) =>
      minHashJaccard(minHash(A, { numHashes: k, seed: s }), minHash(B, { numHashes: k, seed: s })),
    )
    const mean = estimates.reduce((a, b) => a + b, 0) / estimates.length
    const sd = Math.sqrt(estimates.reduce((a, b) => a + (b - mean) ** 2, 0) / (estimates.length - 1))
    const theory = Math.sqrt((J * (1 - J)) / k)
    expect(J).toBe(0.5)
    expect(Math.abs(mean - J)).toBeLessThan(4 * (theory / Math.sqrt(estimates.length)))
    expect(sd).toBeGreaterThan(0.75 * theory)
    expect(sd).toBeLessThan(1.25 * theory)
  })

  it('gives 1 for equal sets and about 0 for disjoint ones', () => {
    expect(minHashJaccard(minHash(A), minHash([...A].reverse()))).toBe(1)
    expect(
      minHashJaccard(minHash(range(0, 200, 'p'), { numHashes: 256 }), minHash(range(0, 200, 'q'), { numHashes: 256 })),
    ).toBeLessThan(0.03)
  })

  it('merging signatures is the signature of the union', () => {
    const a = minHash(A, { numHashes: 64, seed: 3 })
    const b = minHash(B, { numHashes: 64, seed: 3 })
    const union = minHash([...A, ...B], { numHashes: 64, seed: 3 })
    expect(Array.from(a.merge(b).values)).toEqual(Array.from(union.values))
    expect(() => a.merge(minHash(B, { numHashes: 64, seed: 4 }))).toThrow(/seed/)
  })

  it('is the same with a key as with the number it derives', () => {
    const a = minHash(A, { numHashes: 32, seed: key(2) })
    const b = minHash(A, { numHashes: 32, seed: key(2) })
    expect(Array.from(a.values)).toEqual(Array.from(b.values))
  })
})

describe('lshIndex', () => {
  it('follows the S-curve 1 - (1 - s^r)^b', () => {
    const bands = 20
    const rows = 5
    for (const shared of [30, 55, 80]) {
      let hits = 0
      let s = 0
      const pairs = 300
      for (let p = 0; p < pairs; p++) {
        const left = range(1000 * p, 1000 * p + 100, `x${p}-`)
        const right = [...range(1000 * p, 1000 * p + shared, `x${p}-`), ...range(0, 100 - shared, `y${p}-`)]
        s = exactJaccard(left, right)
        const idx = lshIndex({ bands, rows })
        idx.add('left', minHash(left, { numHashes: bands * rows, seed: p }))
        if (idx.query(minHash(right, { numHashes: bands * rows, seed: p })).includes('left')) hits += 1
      }
      const theory = lshCollisionProbability(s, bands, rows)
      expect(Math.abs(hits / pairs - theory)).toBeLessThan(4 * Math.sqrt((theory * (1 - theory)) / pairs) + 0.01)
    }
  })

  it('lists candidate pairs once and exposes its threshold', () => {
    const idx = lshIndex({ bands: 16, rows: 4 })
    const sig = minHash(['a', 'b', 'c'], { numHashes: 64 })
    idx.add('one', sig)
    idx.add('two', sig)
    idx.add('three', minHash(range(0, 50, 'z'), { numHashes: 64 }))
    expect(idx.candidatePairs()).toEqual([['one', 'two']])
    expect(idx.threshold).toBeCloseTo((1 / 16) ** (1 / 4), 12)
    expect(() => idx.add('short', minHash(['a'], { numHashes: 8 }))).toThrow(/need at least/)
  })
})

describe('lshOptimalParams', () => {
  it('minimises the weighted error against an independent adaptive integration', () => {
    const threshold = 0.5
    const n = 64
    const best = lshOptimalParams(threshold, n)
    expect(best.bands * best.rows).toBeLessThanOrEqual(n)
    const independent = (b: number, r: number): number => {
      const miss = (s: Value): Value => pow(sub(1, pow(s, r)), b)
      const fp = quad((s) => sub(1, miss(s)), 0, threshold)
      const fn = quad(miss, threshold, 1)
      return 0.5 * itemOf(fp.value) + 0.5 * itemOf(fn.value)
    }
    expect(independent(best.bands, best.rows)).toBeCloseTo(best.error, 6)
    for (const [b, r] of [
      [4, 16],
      [8, 8],
      [16, 4],
      [32, 2],
      [64, 1],
      [1, 64],
    ] as const)
      expect(best.error).toBeLessThanOrEqual(independent(b, r) + 1e-12)
  })

  it('weighting false negatives more buys more bands', () => {
    const balanced = lshOptimalParams(0.6, 64)
    const recall = lshOptimalParams(0.6, 64, [0.1, 0.9])
    expect(recall.bands).toBeGreaterThanOrEqual(balanced.bands)
    expect(() => lshOptimalParams(1, 64)).toThrow(/threshold/)
  })
})
