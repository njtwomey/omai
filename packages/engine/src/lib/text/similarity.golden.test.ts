import { describe, expect, it } from 'vitest'
import { expectMatches, loadFixture } from '../../testing/golden.ts'
import { tensor } from '../tensor/index.ts'
import {
  alignmentScore,
  containment,
  damerauLevenshtein,
  everygrams,
  fastTextBuckets,
  fastTextNgrams,
  fnv1a,
  hamming,
  indel,
  jaccard,
  jaro,
  jaroWinkler,
  lcsTable,
  levenshtein,
  levenshteinSimilarity,
  metaphone,
  murmurhash3,
  needlemanWunsch,
  ngrams,
  ratio,
  skipgrams,
  smithWaterman,
  soundex,
  tokenSetRatio,
  tokenSortRatio,
} from './index.ts'

describe('text/similarity against rapidfuzz', () => {
  for (const c of loadFixture('text/similarity').cases) {
    it(c.name, () => {
      if (c.fn === 'sets') {
        const [a, b] = c.inputs as [string[], string[]]
        expectMatches(tensor([jaccard(a, b), containment(a, b)]), c.output, c.rtol, c.atol, c.name)
        return
      }
      const [a, b] = c.inputs as [string, string]
      const lcs = lcsTable(a, b)
      const row = [
        levenshtein(a, b),
        levenshtein(a, b, { insertion: 1, deletion: 1, substitution: 2 }),
        levenshtein(a, b, { insertion: 2, deletion: 3, substitution: 4 }),
        levenshteinSimilarity(a, b),
        damerauLevenshtein(a, b, { restricted: true }),
        damerauLevenshtein(a, b),
        indel(a, b),
        lcs[lcs.length - 1]?.[Array.from(b).length] ?? 0,
        jaro(a, b),
        jaroWinkler(a, b),
        jaroWinkler(a, b, 0.2),
        ratio(a, b),
        tokenSortRatio(a, b),
        tokenSetRatio(a, b),
        Array.from(a).length === Array.from(b).length ? hamming(a, b) : -1,
      ]
      expectMatches(tensor(row), c.output, c.rtol, c.atol, c.name)
    })
  }
})

describe('text/alignment against Biopython', () => {
  for (const c of loadFixture('text/alignment').cases) {
    it(c.name, () => {
      const [a, b] = c.inputs as [string, string]
      const scoring = {
        match: c.params['match'] as number,
        mismatch: c.params['mismatch'] as number,
        gapOpen: c.params['gapOpen'] as number,
        gapExtend: c.params['gapExtend'] as number,
      }
      const g = needlemanWunsch(a, b, scoring)
      const l = smithWaterman(a, b, scoring)
      expectMatches(tensor([g.score, l.score]), c.output, c.rtol, c.atol, c.name)
      // The traced alignments achieve their scores.
      expect(alignmentScore(a, b, g.pairs, scoring)).toBeCloseTo(g.score, 10)
      expect(alignmentScore(a, b, l.pairs, scoring)).toBeCloseTo(l.score, 10)
      expect(g.pairs.filter(([i]) => i >= 0).length).toBe(Array.from(a).length)
    })
  }
})

describe('text/phonetic against jellyfish', () => {
  for (const c of loadFixture('text/phonetic').cases) {
    it(c.name, () => {
      const words = c.inputs[0] as string[]
      expect(words.map(c.fn === 'soundex' ? soundex : metaphone)).toEqual(c.output)
    })
  }
})

describe('text/shingles against nltk and gensim', () => {
  for (const c of loadFixture('text/shingles').cases) {
    it(c.name, () => {
      const run = (): unknown => {
        switch (c.fn) {
          case 'ngrams':
            return ngrams(c.inputs[0] as string[], c.inputs[1] as number)
          case 'ngramsPadded':
            return ngrams(c.inputs[0] as string[], c.inputs[1] as number, { padLeft: true, padRight: true })
          case 'everygrams':
            return everygrams(c.inputs[0] as string[], c.inputs[1] as number, c.inputs[2] as number)
          case 'skipgrams':
            return skipgrams(c.inputs[0] as string[], c.inputs[1] as number, c.inputs[2] as number)
          case 'fastTextNgrams':
            return fastTextNgrams(c.inputs[0] as string, c.inputs[1] as number, c.inputs[2] as number)
          default:
            return fastTextBuckets(
              c.inputs[0] as string,
              c.inputs[1] as number,
              c.inputs[2] as number,
              c.inputs[3] as number,
            )
        }
      }
      expect(run()).toEqual(c.output)
    })
  }
})

describe('text/hashing against scikit-learn and gensim', () => {
  for (const c of loadFixture('text/hashing').cases) {
    it(c.name, () => {
      const k = c.inputs[0] as string
      const got =
        c.fn === 'murmurhash3'
          ? [0, 1, 42].map((s) => murmurhash3(k, s))
          : c.fn === 'murmurhash3Positive'
            ? murmurhash3(k, 0, true)
            : c.fn === 'fnv1a'
              ? fnv1a(k)
              : fnv1a(k, { signedBytes: true })
      expect(got).toEqual(c.output)
    })
  }
})
