import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import * as F from './index.ts'

const small = fc.integer({ min: 0, max: 12 })

describe('counting', () => {
  it("Pascal's rule, Vandermonde's identity and row sums", () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 60 }), fc.integer({ min: 0, max: 60 }), (n, k) => {
        expect(F.binomialExact(n, k)).toBe(F.binomialExact(n - 1, k) + (k > 0 ? F.binomialExact(n - 1, k - 1) : 0n))
      }),
    )
    let vandermonde = 0n
    for (let j = 0; j <= 7; j++) vandermonde += F.binomialExact(10, j) * F.binomialExact(12, 7 - j)
    expect(vandermonde).toBe(F.binomialExact(22, 7))
    for (const [n, row] of F.pascalTriangle(30).entries()) expect(row.reduce((s, v) => s + v, 0)).toBe(2 ** n)
    expect(() => F.pascalTriangle(60)).toThrow()
    expect(F.pascalTriangle(200, 2)).toHaveLength(200)
    expect(F.pascalTriangle(3, 1)).toEqual([[0], [0, 0], [0, 0, 0]])
    expect(() => F.binomialExact(-1, 2)).toThrow()
  })

  it('multinomials count arrangements of a multiset', () => {
    const word = ['a', 'a', 'b', 'c', 'c', 'c']
    const distinct = new Set(F.permutations(word).map((p) => p.join('')))
    expect(BigInt(distinct.size)).toBe(F.multinomialExact([2, 1, 3]))
    expect(F.multinomialExact([])).toBe(1n)
  })
})

describe('enumeration and ranking', () => {
  it('enumerations have the counted sizes and are in rank order', () => {
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 6 }), fc.integer({ min: 0, max: 6 }), (n, r) => {
        const items = Array.from({ length: n }, (_, i) => i)
        const perms = F.permutations(items, r)
        expect(BigInt(perms.length)).toBe(r > n ? 0n : F.factorialExact(n) / F.factorialExact(n - r))
        const combs = F.combinations(items, r)
        expect(BigInt(combs.length)).toBe(F.binomialExact(n, r))
        for (const [i, c] of combs.entries()) {
          expect(F.combinationRank(c, n)).toBe(BigInt(i))
          expect(F.combinationUnrank(n, r, i)).toEqual(c)
        }
      }),
      { numRuns: 30 },
    )
  })

  it('permutation rank and unrank are inverse', () => {
    const all = F.permutations([0, 1, 2, 3, 4])
    for (const [i, p] of all.entries()) {
      expect(F.permutationRank(p).rank).toBe(BigInt(i))
      expect(F.permutationUnrank(5, i)).toEqual(p)
    }
    expect(() => F.permutationRank([0, 0, 1])).toThrow()
    expect(() => F.permutationUnrank(3, 6)).toThrow()
    expect(() => F.combinationUnrank(4, 2, 6)).toThrow()
    expect(() => F.combinationRank([2, 1], 4)).toThrow()
    expect(F.permutationUnrank(20, F.factorialExact(20) - 1n)[0]).toBe(19)
  })
})

describe('special numbers', () => {
  it('derangements match brute force and approach n!/e', () => {
    for (let n = 0; n <= 7; n++) {
      const items = Array.from({ length: n }, (_, i) => i)
      const fixedFree = F.permutations(items).filter((p) => p.every((v, i) => v !== i)).length
      expect(F.derangements(n)).toBe(BigInt(fixedFree))
    }
    expect(Number(F.derangements(15))).toBe(Math.round(Number(F.factorialExact(15)) / Math.E))
  })

  it('Stirling numbers sum to Bell numbers and factorials', () => {
    fc.assert(
      fc.property(small, (n) => {
        let s2 = 0n
        let s1 = 0n
        let signed = 0n
        for (let k = 0; k <= n; k++) {
          s2 += F.stirling2(n, k)
          s1 += F.stirling1(n, k)
          signed += F.stirling1(n, k, true)
        }
        expect(s2).toBe(F.bell(n))
        expect(s1).toBe(F.factorialExact(n))
        // The signed numbers are the coefficients of the falling factorial x(x-1)...(x-n+1); at x = 1 it vanishes.
        expect(signed).toBe(n === 1 || n === 0 ? 1n : 0n)
      }),
    )
  })

  it('Catalan numbers count balanced bracketings', () => {
    for (let n = 0; n <= 8; n++) {
      // Choose the positions of the n opening brackets among 2n.
      const balanced = F.combinations(
        Array.from({ length: 2 * n }, (_, i) => i),
        n,
      ).filter((open) => {
        const opens = new Set(open)
        let depth = 0
        for (let i = 0; i < 2 * n; i++) {
          depth += opens.has(i) ? 1 : -1
          if (depth < 0) return false
        }
        return true
      })
      expect(F.catalan(n)).toBe(BigInt(balanced.length))
    }
  })

  it('partition counts match the enumeration and the conjugate symmetry', () => {
    for (let n = 0; n <= 20; n++) {
      const ps = F.integerPartitions(n)
      expect(BigInt(ps.length)).toBe(F.partitionCount(n))
      for (const p of ps) expect(p.reduce((s, v) => s + v, 0)).toBe(n)
    }
    // Partitions into k parts equal partitions with largest part k.
    for (let k = 1; k <= 6; k++) {
      expect(F.integerPartitions(12, { parts: k }).length).toBe(
        F.integerPartitions(12).filter((p) => p[0] === k).length,
      )
    }
    expect(F.integerPartitions(5, { maxPart: 2 })).toEqual([
      [2, 2, 1],
      [2, 1, 1, 1],
      [1, 1, 1, 1, 1],
    ])
  })
})

describe('inclusion-exclusion', () => {
  it('equals the union size and the Bonferroni bounds alternate', () => {
    fc.assert(
      fc.property(
        fc.array(fc.array(fc.integer({ min: 0, max: 15 }), { maxLength: 8 }), { minLength: 1, maxLength: 6 }),
        (families) => {
          const r = F.inclusionExclusion(families)
          expect(r.union).toBe(new Set(families.flat()).size)
          for (const [k, s] of r.partialSums.entries()) {
            if (k % 2 === 0) expect(s).toBeGreaterThanOrEqual(r.union)
            else expect(s).toBeLessThanOrEqual(r.union)
          }
        },
      ),
      { numRuns: 50 },
    )
    expect(() => F.inclusionExclusion({ count: 21, intersection: () => 0 })).toThrow()
  })
})
