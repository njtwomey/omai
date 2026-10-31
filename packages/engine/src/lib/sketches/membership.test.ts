/**
 * Tests for the membership filters: no false negatives, the Bloom false-positive rate against its theory, the optimal
 * sizing, merge, counting-filter deletion, the cuckoo filter's load factor, deletion and rollback on refusal, the
 * false-positive bound, and the stepped algorithms.
 */

import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { RUNS } from '../../testing/arbitrary.ts'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  bloomFalsePositiveRate,
  bloomFilter,
  bloomFilterAlgorithm,
  countingBloomFilter,
  cuckooFilter,
  cuckooFilterAlgorithm,
  optimalBloomParams,
} from './index.ts'

const names = (prefix: string, n: number, from = 0): string[] =>
  Array.from({ length: n }, (_, i) => `${prefix}-${from + i}`)

describe('optimalBloomParams', () => {
  it('sizes by m = -n ln p / (ln 2)^2 and k = (m / n) ln 2', () => {
    const p = optimalBloomParams(1000, 0.01)
    expect(p.bits).toBe(Math.ceil((-1000 * Math.log(0.01)) / Math.LN2 ** 2))
    expect(p.hashes).toBe(7)
    expect(p.expectedFpr).toBeCloseTo(0.01, 3)
  })

  it('rejects impossible requests', () => {
    expect(() => optimalBloomParams(0, 0.01)).toThrow(/expectedItems/)
    expect(() => optimalBloomParams(10, 1)).toThrow(/targetFpr/)
  })
})

describe('bloomFilter', () => {
  it('never forgets an inserted item (property)', () => {
    fc.assert(
      fc.property(
        fc.array(fc.oneof(fc.string(), fc.integer()), { minLength: 1, maxLength: 60 }),
        fc.nat(1000),
        (items, seed) => {
          const bf = bloomFilter({ bits: 512, hashes: 4, seed })
          for (const x of items) bf.add(x)
          return items.every((x) => bf.has(x))
        },
      ),
      { numRuns: RUNS },
    )
  })

  it('has a false-positive rate matching the fill-based and the classical theory', () => {
    const shape = optimalBloomParams(1000, 0.01)
    const bf = bloomFilter({ expectedItems: 1000, falsePositiveRate: 0.01 })
    for (const x of names('in', 1000)) bf.add(x)
    const probes = names('out', 40000)
    const rate = probes.filter((x) => bf.has(x)).length / probes.length
    const sigma = Math.sqrt((0.01 * 0.99) / probes.length)
    // the rate given the fill is bf.observedFpr(); the classical formula is its expectation
    expect(Math.abs(rate - bf.observedFpr())).toBeLessThan(5 * sigma)
    expect(Math.abs(rate - shape.expectedFpr)).toBeLessThan(6 * sigma)
    expect(Math.abs(bf.theoreticalFpr() - shape.expectedFpr)).toBeLessThan(1e-12)
    expect(bf.estimateCount()).toBeGreaterThan(950)
    expect(bf.estimateCount()).toBeLessThan(1050)
  })

  it('merges to the union filter', () => {
    const a = bloomFilter({ bits: 2048, hashes: 3, seed: 5 })
    const b = bloomFilter({ bits: 2048, hashes: 3, seed: 5 })
    const both = bloomFilter({ bits: 2048, hashes: 3, seed: 5 })
    for (const x of names('a', 50)) {
      a.add(x)
      both.add(x)
    }
    for (const x of names('b', 50)) {
      b.add(x)
      both.add(x)
    }
    const merged = a.merge(b)
    expect(Array.from(merged.words)).toEqual(Array.from(both.words))
    expect(() => a.merge(bloomFilter({ bits: 1024, hashes: 3, seed: 5 }))).toThrow(/share bits/)
  })

  it('is reproducible from a key and differs between keys', () => {
    const fill = (k: number): Uint32Array => {
      const bf = bloomFilter({ bits: 256, hashes: 3, seed: key(k) })
      for (const x of names('x', 20)) bf.add(x)
      return bf.words
    }
    expect(Array.from(fill(1))).toEqual(Array.from(fill(1)))
    expect(Array.from(fill(1))).not.toEqual(Array.from(fill(2)))
  })

  it('the stepped algorithm tracks the theoretical rate and serialises', () => {
    const items = names('s', 300)
    const alg = bloomFilterAlgorithm({ items, bits: 4096, hashes: 5, queries: names('neg', 4000) })
    const run = runAlgorithm(alg, { maxSteps: 1000 })
    expect(run.steps).toBe(300)
    expect(run.state.inserted).toBe(300)
    expect(run.state.theoreticalFpr).toBeCloseTo(bloomFalsePositiveRate(4096, 5, 300), 12)
    expect(Math.abs(run.state.empiricalFpr - run.state.theoreticalFpr)).toBeLessThan(0.01)
    const again = alg.deserialise?.(alg.serialise?.(run.state) ?? null)
    expect(Array.from(again?.words ?? [])).toEqual(Array.from(run.state.words))
  })
})

describe('countingBloomFilter', () => {
  it('removes items without disturbing the others (property)', () => {
    fc.assert(
      fc.property(fc.uniqueArray(fc.integer({ min: 0, max: 10_000 }), { minLength: 2, maxLength: 40 }), (xs) => {
        const cbf = countingBloomFilter({ bits: 1024, hashes: 3 })
        for (const x of xs) cbf.add(x)
        const half = xs.slice(0, Math.floor(xs.length / 2))
        const rest = xs.slice(half.length)
        for (const x of half) if (!cbf.remove(x)) return false
        return rest.every((x) => cbf.has(x)) && cbf.size === rest.length
      }),
      { numRuns: RUNS },
    )
  })

  it('refuses to remove what is not there', () => {
    const cbf = countingBloomFilter({ bits: 4096, hashes: 3 })
    cbf.add('a')
    expect(cbf.remove('zzz')).toBe(false)
    expect(cbf.size).toBe(1)
    expect(cbf.count('a')).toBe(1)
  })
})

describe('cuckooFilter', () => {
  it('holds every inserted item and finds none of the absent ones beyond the bound', () => {
    const cf = cuckooFilter({ capacity: 4096, bucketSize: 4, fingerprintBits: 12 })
    const items = names('c', Math.floor(0.9 * cf.numBuckets * cf.bucketSize))
    for (const x of items) expect(cf.add(x)).toBe(true)
    expect(items.every((x) => cf.has(x))).toBe(true)
    expect(cf.loadFactor()).toBeGreaterThan(0.89)
    const probes = names('absent', 40000)
    const rate = probes.filter((x) => cf.has(x)).length / probes.length
    // at most 2 b / 2^f per probe (both buckets full of distinct fingerprints), about 2 b alpha / 2^f
    expect(rate).toBeLessThan((2 * 4) / 2 ** 12)
    expect(rate).toBeGreaterThan(0.2 * ((2 * 4 * 0.9) / 2 ** 12))
  })

  it('fills to a high load factor before refusing, and a refusal loses nothing', () => {
    const cf = cuckooFilter({ capacity: 2048, bucketSize: 4, maxKicks: 500 })
    const slots = cf.numBuckets * cf.bucketSize
    const stored: string[] = []
    let refused = 0
    for (const x of names('f', slots + 200)) {
      if (cf.add(x)) stored.push(x)
      else refused += 1
    }
    expect(refused).toBeGreaterThan(0)
    expect(cf.loadFactor()).toBeGreaterThan(0.93)
    expect(cf.size).toBe(stored.length)
    expect(stored.every((x) => cf.has(x))).toBe(true)
  })

  it('removes an inserted item and only that item (property)', () => {
    fc.assert(
      fc.property(fc.uniqueArray(fc.integer({ min: 0, max: 1_000_000 }), { minLength: 2, maxLength: 80 }), (xs) => {
        const cf = cuckooFilter({ capacity: 512, seed: 3 })
        for (const x of xs) if (!cf.add(x)) return false
        const gone = xs[0] as number
        if (!cf.remove(gone)) return false
        return cf.size === xs.length - 1 && xs.slice(1).every((x) => cf.has(x))
      }),
      { numRuns: RUNS },
    )
  })

  it('validates its sizes', () => {
    expect(() => cuckooFilter({ fingerprintBits: 3 })).toThrow(/fingerprintBits/)
    expect(() => cuckooFilter({ capacity: 0 })).toThrow(/capacity/)
  })

  it('the stepped algorithm shows kicks rising with the load', () => {
    const items = names('k', 1100)
    const alg = cuckooFilterAlgorithm({ items, capacity: 1024 })
    const early = runAlgorithm(alg, { maxSteps: 300 })
    const full = runAlgorithm(alg, { maxSteps: 2000 })
    expect(early.state.kicks).toBeLessThanOrEqual(1)
    expect(full.state.size + full.state.failures).toBe(1100)
    const again = alg.deserialise?.(alg.serialise?.(full.state) ?? null)
    expect(Array.from(again?.slots ?? [])).toEqual(Array.from(full.state.slots))
  })
})
