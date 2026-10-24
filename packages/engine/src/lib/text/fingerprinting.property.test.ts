import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import {
  characterShingles,
  hammingDistance64,
  jaccard,
  minhash,
  minhashJaccard,
  minhashPermutations,
  rabinKarp,
  randomIndexing,
  rollingHashes,
  simhash,
  winnow,
} from './index.ts'

const text = fc.string({ unit: fc.constantFrom('a', 'b', 'c', 'é'), maxLength: 30 })

describe('fingerprinting invariants', () => {
  it('Rabin-Karp finds exactly the occurrences, and equal windows hash equally', () => {
    fc.assert(
      fc.property(text, fc.string({ unit: fc.constantFrom('a', 'b'), minLength: 1, maxLength: 3 }), (t, p) => {
        const chars = Array.from(t)
        const pc = Array.from(p)
        const brute: number[] = []
        for (let i = 0; i + pc.length <= chars.length; i++) if (pc.every((c, j) => chars[i + j] === c)) brute.push(i)
        expect(rabinKarp(p, t)).toEqual(brute)
        const hs = rollingHashes(t, 3)
        for (let i = 0; i < hs.length; i++)
          for (let j = 0; j < hs.length; j++)
            if (chars.slice(i, i + 3).join('') === chars.slice(j, j + 3).join('')) expect(hs[i]).toBe(hs[j])
      }),
    )
  })

  it('winnowing picks a fingerprint in every window, and shared long substrings share fingerprints', () => {
    fc.assert(
      fc.property(
        text,
        text,
        text,
        fc.integer({ min: 1, max: 4 }),
        fc.integer({ min: 1, max: 4 }),
        (a, common, b, k, w) => {
          const fps = winnow(a, k, w)
          const n = rollingHashes(a, k).length
          const positions = fps.map((f) => f.position)
          for (let s = 0; s + w <= n; s++) expect(positions.some((p) => p >= s && p < s + w)).toBe(true)
          if (Array.from(common).length >= w + k - 1) {
            const x = new Set(winnow(a + common, k, w).map((f) => f.hash))
            expect(winnow(common + b, k, w).some((f) => x.has(f.hash))).toBe(true)
          }
        },
      ),
    )
  })

  it('SimHash is a 64-bit signature with distance 0 to itself; MinHash estimates Jaccard', () => {
    const p = minhashPermutations(key(5), 512)
    fc.assert(
      fc.property(text, text, (a, b) => {
        const s = simhash(characterShingles(a, 2))
        expect(hammingDistance64(s, s)).toBe(0)
        expect(s < 1n << 64n).toBe(true)
        const sa = characterShingles(a, 2)
        const sb = characterShingles(b, 2)
        const est = minhashJaccard(minhash(sa, p), minhash(sb, p))
        if (sa.size + sb.size > 0) expect(Math.abs(est - jaccard(sa, sb))).toBeLessThan(0.15)
      }),
      { numRuns: 60 },
    )
  })

  it('random indexing gives each word a context vector built from ternary index vectors', () => {
    const v = randomIndexing(
      [
        ['a', 'b', 'c'],
        ['a', 'c'],
      ],
      key(9),
      { dimension: 32, nonzeros: 4, window: 1 },
    )
    expect(v.words).toEqual(['a', 'b', 'c'])
    for (const r of v.rows) expect(r.every((x) => Number.isInteger(x))).toBe(true)
    // 'b' co-occurs once with 'a' and once with 'c': the sum of two index vectors with 4 entries each.
    const b = v.rows[1] ?? new Float64Array()
    expect(b.reduce((s, x) => s + Math.abs(x), 0)).toBeLessThanOrEqual(8)
    expect(
      randomIndexing(
        [
          ['a', 'b', 'c'],
          ['a', 'c'],
        ],
        key(9),
        { dimension: 32, nonzeros: 4, window: 1 },
      ),
    ).toEqual(v)
  })
})
