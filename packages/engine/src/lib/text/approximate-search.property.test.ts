import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { metricSeries, recorder } from '../trace/index.ts'
import {
  bkSearch,
  bkTree,
  levenshtein,
  qgramIndex,
  qgramSearch,
  sellers,
  topKFuzzy,
  trie,
  trieSearch,
} from './index.ts'

const word = fc.string({ unit: fc.constantFrom('a', 'b', 'c', 'é'), maxLength: 7 })
const words = fc.uniqueArray(word, { maxLength: 25 })

/** Every word within k, sorted by distance then by the given tie order. */
function brute(ws: readonly string[], q: string, k: number, byWord: boolean): { value: string; distance: number }[] {
  const out = ws.map((w, i) => ({ value: w, distance: levenshtein(q, w), i })).filter((m) => m.distance <= k)
  out.sort(
    (a, b) => a.distance - b.distance || (byWord ? (a.value < b.value ? -1 : a.value > b.value ? 1 : 0) : a.i - b.i),
  )
  return out.map(({ value, distance }) => ({ value, distance }))
}

describe('approximate search agrees with brute force', () => {
  it('q-gram indexes, BK-trees and tries find exactly the words within k', () => {
    fc.assert(
      fc.property(words, word, fc.integer({ min: 0, max: 3 }), fc.integer({ min: 1, max: 3 }), (ws, q, k, g) => {
        expect(qgramSearch(qgramIndex(ws, g), q, k).matches).toEqual(brute(ws, q, k, false))
        expect(bkSearch(bkTree(ws), q, k).matches).toEqual(brute(ws, q, k, true))
        expect(trieSearch(trie(ws), q, k).matches).toEqual(brute(ws, q, k, true))
      }),
    )
  })

  it('top-k matching returns the k nearest and accounts for every candidate in its trace', () => {
    fc.assert(
      fc.property(words, word, fc.integer({ min: 1, max: 4 }), (ws, q, k) => {
        const rec = recorder()
        const got = topKFuzzy(q, ws, k, { tracer: rec })
        expect(got).toEqual(brute(ws, q, Number.POSITIVE_INFINITY, false).slice(0, k))
        if (ws.length === 0) return
        const run = rec.run()
        const last = (name: string): number => metricSeries(run, name).values.at(-1) ?? 0
        // Every candidate is either pruned by length or verified (possibly cut off early).
        expect(last('prunedByLength') + last('verified')).toBe(ws.length)
        expect(last('prunedEarly')).toBeLessThanOrEqual(last('verified'))
      }),
    )
  })

  it("Sellers' occurrences have the least distance over substrings ending at each position", () => {
    fc.assert(
      fc.property(
        word,
        fc.string({ unit: fc.constantFrom('a', 'b', 'c'), maxLength: 9 }),
        fc.integer({ min: 0, max: 2 }),
        (p, t, k) => {
          const chars = Array.from(t)
          const best = (j: number): number => {
            let m = Number.POSITIVE_INFINITY
            for (let i = 0; i <= j; i++) m = Math.min(m, levenshtein(p, chars.slice(i, j).join('')))
            return m
          }
          const occ = sellers(p, t, k)
          const ends = new Set(occ.map((o) => o.end))
          for (let j = 0; j <= chars.length; j++) expect(ends.has(j)).toBe(best(j) <= k)
          for (const o of occ) {
            expect(o.distance).toBe(best(o.end))
            expect(levenshtein(p, chars.slice(o.start, o.end).join(''))).toBe(o.distance)
          }
        },
      ),
    )
  })
})
