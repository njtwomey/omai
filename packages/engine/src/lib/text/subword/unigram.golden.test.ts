import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { wordCounts } from '../../../testing/text.ts'
import { compareCodePoints } from '../_classes.ts'
import { splitSpans, utf8Bytes } from '../index.ts'
import {
  trainUnigram,
  trainWordPiece,
  type UnigramTrainingOptions,
  unigramModel,
  unigramTokenize,
  type WordPieceTrainingOptions,
  wordPieceModel,
  wordPieceTokenize,
} from './index.ts'

/** HF's Metaspace pre-tokeniser (prepend_scheme 'always', split on the replacement, merged with the next piece). */
function metaspaceCounts(corpus: readonly string[]): Map<string, number> {
  const counts = new Map<string, number>()
  for (const line of corpus) {
    const s = `▁${line.replace(/ /g, '▁')}`
    for (const [a, b] of splitSpans(s, '▁', 'mergedWithNext')) {
      const w = s.slice(a, b)
      counts.set(w, (counts.get(w) ?? 0) + 1)
    }
  }
  return counts
}

describe('subword/unigram against HF tokenizers', () => {
  for (const c of loadFixture('text/subword/unigram').cases) {
    it(c.name, () => {
      if (c.fn === 'unigramTokenize') {
        const [pieces, word] = c.inputs as [[string, number][], string]
        const m = unigramModel(pieces, { unkId: 0 })
        const bytes = (i: number): number => utf8Bytes(word.slice(0, i)).length
        expect(unigramTokenize(m, word).map((t) => [t.token, t.id, [bytes(t.start), bytes(t.end)]])).toEqual(c.output)
        return
      }
      const [corpus, options] = c.inputs as [string[], UnigramTrainingOptions]
      const model = trainUnigram(metaspaceCounts(corpus), options)
      const expected = c.output as [string, number][]
      const byName = (xs: readonly (readonly [string, number])[]): (readonly [string, number])[] =>
        [...xs].sort((a, b) => compareCodePoints(a[0], b[0]))
      const ours = byName(model.pieces)
      expect(ours.map(([p]) => p)).toEqual(byName(expected).map(([p]) => p))
      for (const [k, [, s]] of byName(expected).entries())
        expect(Math.abs((ours[k]?.[1] ?? 0) - s)).toBeLessThan(c.atol)
      // Scores never increase along the vocabulary after the specials.
      const specials =
        (options.specialTokens ?? []).length +
        (options.unkToken !== undefined && !(options.specialTokens ?? []).includes(options.unkToken) ? 1 : 0)
      const rest = model.pieces.slice(specials).map(([, s]) => s)
      for (let i = 1; i < rest.length; i++) expect(rest[i] ?? 0).toBeLessThanOrEqual(rest[i - 1] ?? 0)
      const held = c.params['heldOut'] as string[]
      const enc = c.params['encodings'] as string[][]
      for (const [k, w] of held.entries()) expect(unigramTokenize(model, w).map((t) => t.token)).toEqual(enc[k])
      const words = c.params['words'] as string[]
      const wenc = c.params['wordEncodings'] as string[][]
      for (const [k, w] of words.entries()) expect(unigramTokenize(model, w).map((t) => t.token)).toEqual(wenc[k])
    })
  }
})

describe('subword/wordpiece against HF tokenizers', () => {
  for (const c of loadFixture('text/subword/wordpiece').cases) {
    it(c.name, () => {
      if (c.fn === 'wordPieceTokenize') {
        const [vocab, word] = c.inputs as [string[], string]
        const m = wordPieceModel(vocab, { maxInputCharsPerWord: 12 })
        const bytes = (i: number): number => utf8Bytes(word.slice(0, i)).length
        expect(wordPieceTokenize(m, word).map((t) => [t.token, t.id, [bytes(t.start), bytes(t.end)]])).toEqual(c.output)
        return
      }
      const [corpus, options] = c.inputs as [string[], WordPieceTrainingOptions]
      const model = trainWordPiece(wordCounts(corpus), options)
      expect([...model.vocab.keys()].sort()).toEqual(c.output)
    })
  }
})
