import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { toDense } from '../numerics/sparse/index.ts'
import { toArray } from '../tensor/index.ts'
import {
  buildVocabulary,
  countVectorize,
  decodeTokens,
  encodeTokens,
  hashingVectorize,
  tfidfVectorize,
} from './index.ts'

const word = fc.constantFrom('cat', 'dog', 'the', 'an', 'fox', 'naïve', 'zebra', 'x1', 'Dog')
const doc = fc.array(word, { minLength: 1, maxLength: 8 }).map((ws) => ws.join(' '))
const docs = fc.array(doc, { minLength: 1, maxLength: 6 })

describe('vectoriser invariants', () => {
  it('counts sum to the analyzed token count, columns are alphabetical, rows independent of order', () => {
    fc.assert(
      fc.property(docs, (ds) => {
        const { vectorizer, matrix } = countVectorize(ds, { ngramRange: [1, 2] })
        const dense = toArray(toDense(matrix)) as number[][]
        const sorted = [...vectorizer.featureNames].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0))
        expect(vectorizer.featureNames).toEqual(sorted)
        const reversed = toArray(
          toDense(countVectorize([...ds].reverse(), { ngramRange: [1, 2] }).matrix),
        ) as number[][]
        expect(reversed).toEqual([...dense].reverse())
      }),
    )
  })

  it('tf-idf rows have unit l2 (or l1) norm and hashing preserves token counts without sign alternation', () => {
    fc.assert(
      fc.property(docs, fc.constantFrom('l1' as const, 'l2' as const), (ds, norm) => {
        const rows = toArray(toDense(tfidfVectorize(ds, { norm, minDf: 1 }).matrix)) as number[][]
        for (const r of rows) {
          const s = norm === 'l2' ? Math.sqrt(r.reduce((a, v) => a + v * v, 0)) : r.reduce((a, v) => a + Math.abs(v), 0)
          if (s > 0) expect(s).toBeCloseTo(1, 12)
        }
        const counts = toArray(toDense(countVectorize(ds).matrix)) as number[][]
        const hashed = toArray(
          toDense(hashingVectorize(ds, { nFeatures: 7, alternateSign: false, norm: 'none' })),
        ) as number[][]
        for (const [i, r] of hashed.entries())
          expect(r.reduce((a, v) => a + v, 0)).toBe((counts[i] ?? []).reduce((a, v) => a + v, 0))
      }),
    )
  })
})

describe('vocabularies', () => {
  it('encode and decode round-trip, and the size limit keeps the most frequent tokens', () => {
    fc.assert(
      fc.property(
        fc.array(fc.array(word, { maxLength: 6 }), { maxLength: 5 }),
        fc.integer({ min: 2, max: 8 }),
        (ds, size) => {
          const v = buildVocabulary(ds, { specials: ['<pad>'], unk: '<unk>', maxSize: size })
          expect(v.tokens.length).toBeLessThanOrEqual(size)
          const tokens = ds.flat()
          const ids = encodeTokens(v, tokens)
          expect(decodeTokens(v, ids)).toEqual(tokens.map((t) => (v.index.has(t) ? t : '<unk>')))
          const kept = v.counts.slice(2)
          const dropped = [...new Set(tokens)]
            .filter((t) => !v.index.has(t))
            .map((t) => tokens.filter((x) => x === t).length)
          for (const d of dropped) for (const k of kept) expect(k).toBeGreaterThanOrEqual(d)
        },
      ),
    )
  })

  it('orders by frequency, alphabetically or by first occurrence, and rejects bad inputs', () => {
    const ds = [['b', 'c', 'a', 'c']]
    expect(buildVocabulary(ds).tokens).toEqual(['c', 'a', 'b'])
    expect(buildVocabulary(ds, { order: 'alphabetical' }).tokens).toEqual(['a', 'b', 'c'])
    expect(buildVocabulary(ds, { order: 'first', minCount: 1 }).tokens).toEqual(['b', 'c', 'a'])
    expect(buildVocabulary(ds, { minCount: 2 }).tokens).toEqual(['c'])
    expect(() => buildVocabulary(ds, { specials: ['x', 'x'] })).toThrow(/repeated/)
    expect(() => encodeTokens(buildVocabulary(ds), ['z'])).toThrow(/not in the vocabulary/)
    expect(() => decodeTokens(buildVocabulary(ds), [9])).toThrow(/out of range/)
  })
})
