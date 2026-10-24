import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toFlat } from '../tensor/index.ts'
import { choice, key, permutation, reservoir, shuffle, subsample } from './index.ts'

const seed = fc.integer({ min: 0, max: 2 ** 31 })
const range = (n: number): number[] => Array.from({ length: n }, (_, i) => i)

/** Largest standardised deviation of inclusion counts from their binomial expectation `trials * p`. */
function worstDeviation(counts: readonly number[], trials: number, p: number): number {
  const sd = Math.sqrt(trials * p * (1 - p))
  let worst = 0
  for (const c of counts) worst = Math.max(worst, Math.abs(c - trials * p) / sd)
  return worst
}

describe('permutation and shuffle', () => {
  it('permutation is a permutation of 0..n-1', () => {
    fc.assert(
      fc.property(seed, fc.integer({ min: 0, max: 200 }), (s, n) => {
        expect(Array.from(permutation(key(s), n).data).sort((a, b) => a - b)).toEqual(range(n))
      }),
    )
  })

  it('shuffle permutes the rows of nested and typed arrays alike', () => {
    fc.assert(
      fc.property(seed, fc.integer({ min: 1, max: 30 }), (s, n) => {
        const typed = Float64Array.from(range(n))
        const out = toFlat(shuffle(key(s), typed))
        expect([...out].sort((a, b) => a - b)).toEqual(range(n))
        expect(toFlat(shuffle(key(s), range(n)))).toEqual(out)
      }),
    )
    expect(() => shuffle(key(0), 3)).toThrow(/dimension/)
  })
})

describe('choice', () => {
  it('without replacement gives distinct indices and never a zero weight', () => {
    fc.assert(
      fc.property(
        seed,
        fc.array(fc.double({ min: 0, max: 5, noNaN: true }), { minLength: 1, maxLength: 20 }),
        (s, w) => {
          const positive = w.filter((x) => x > 0).length
          const got = Array.from(choice(key(s), w.length, positive, { replace: false, weights: w }).data)
          expect(new Set(got).size).toBe(positive)
          for (const i of got) expect(w[i]).toBeGreaterThan(0)
        },
      ),
    )
  })

  it('rejects impossible requests', () => {
    expect(() => choice(key(0), 3, 3, { replace: false, weights: [1, 0, 1] })).toThrow(/positive weights/)
    expect(() => choice(key(0), 3, 1, { weights: [1, 1] })).toThrow(/2 weights/)
    expect(() => choice(key(0), 3, 4, { replace: false })).toThrow(/without replacement/)
  })
})

describe('subsample', () => {
  it('gives size distinct sorted indices below n', () => {
    fc.assert(
      fc.property(seed, fc.integer({ min: 0, max: 10 ** 9 }), fc.integer({ min: 0, max: 50 }), (s, n, k) => {
        const size = Math.min(n, k)
        const got = Array.from(subsample(key(s), n, size).data)
        expect(got.length).toBe(size)
        got.forEach((v, i) => {
          expect(v).toBeGreaterThanOrEqual(0)
          expect(v).toBeLessThan(n)
          if (i > 0) expect(v).toBeGreaterThan(got[i - 1] ?? -1)
        })
      }),
    )
  })

  it('includes every index with probability size / n', () => {
    const n = 12
    const size = 4
    const trials = 6000
    const counts = new Array<number>(n).fill(0)
    for (let t = 0; t < trials; t++) for (const i of subsample(key(t), n, size).data) counts[i] = (counts[i] ?? 0) + 1
    expect(worstDeviation(counts, trials, size / n)).toBeLessThan(4.5)
  })

  it('rejects a size above n', () => {
    expect(() => subsample(key(0), 3, 4)).toThrow(/size/)
  })
})

describe('reservoir', () => {
  for (const method of ['L', 'R'] as const) {
    it(`Algorithm ${method} includes every item with probability size / N`, () => {
      const items = range(30)
      const size = 5
      const trials = 4000
      const counts = new Array<number>(items.length).fill(0)
      for (let t = 0; t < trials; t++) {
        for (const i of reservoir(key(t), items, size, { method })) counts[i] = (counts[i] ?? 0) + 1
      }
      expect(worstDeviation(counts, trials, size / items.length)).toBeLessThan(4.5)
    })

    it(`Algorithm ${method} gives the same sample from an array and from an iterator`, () => {
      fc.assert(
        fc.property(seed, fc.integer({ min: 0, max: 300 }), fc.integer({ min: 0, max: 12 }), (s, n, size) => {
          const fromArray = reservoir(key(s), range(n), size, { method })
          function* stream(): Generator<number> {
            for (let i = 0; i < n; i++) yield i
          }
          expect(reservoir(key(s), stream(), size, { method })).toEqual(fromArray)
          expect(fromArray.length).toBe(Math.min(n, size))
          expect(new Set(fromArray).size).toBe(fromArray.length)
        }),
      )
    })
  }

  it('is deterministic per key, keeps short streams whole and samples long ones lazily', () => {
    expect(reservoir(key(1), 'abc', 5)).toEqual(['a', 'b', 'c'])
    expect(reservoir(key(2), range(1000), 3)).toEqual(reservoir(key(2), range(1000), 3))
    let pulled = 0
    function* endless(): Generator<number> {
      for (let i = 0; i < 100000; i++) {
        pulled++
        yield i
      }
    }
    expect(reservoir(key(3), endless(), 10, { method: 'L' }).length).toBe(10)
    expect(pulled).toBe(100000)
    expect(() => reservoir(key(0), [1], -1)).toThrow(/size/)
  })
})
