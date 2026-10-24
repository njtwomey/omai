import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../../testing/golden.ts'
import { PRE_TOKENIZER_PATTERNS, preTokenize, splitSpans, utf8Bytes } from '../index.ts'
import { type BpeTrainingOptions, bpeModel, bpeTokenize, toByteLevel, trainBpe } from './index.ts'

/** Word counts as HF's Whitespace (or GPT-2 byte-level) pre-tokeniser produces them. */
export function wordCounts(corpus: readonly string[], byteLevel = false): Map<string, number> {
  const counts = new Map<string, number>()
  for (const text of corpus) {
    const words = byteLevel
      ? preTokenize(text, 'gpt2')
      : splitSpans(text, PRE_TOKENIZER_PATTERNS.whitespace, 'removed', true).map(([a, b]) => text.slice(a, b))
    for (const w of words) counts.set(w, (counts.get(w) ?? 0) + 1)
  }
  return counts
}

describe('subword/bpe against HF tokenizers', () => {
  for (const c of loadFixture('text/subword/bpe').cases) {
    it(c.name, () => {
      const [corpus, options] = c.inputs as [string[], BpeTrainingOptions]
      const byteLevel = options.byteLevel === true
      const model = trainBpe(wordCounts(corpus, byteLevel), { ...options, ...(byteLevel ? {} : { unkToken: '[UNK]' }) })
      const expected = c.output as string[]
      const ours = [...model.vocab].sort((a, b) => a[1] - b[1]).map(([t]) => t)
      expect(model.merges).toEqual(c.params['merges'])
      const affixed = c.params['affixed'] === true
      if (affixed) expect([...ours].sort()).toEqual(expected)
      else expect(ours).toEqual(expected)
      // Encoding held-out words with HF's own vocabulary and merges.
      const hf = bpeModel(
        affixed ? model.vocab : new Map(expected.map((t, i) => [t, i])),
        c.params['merges'] as [string, string][],
        {
          ...(byteLevel ? {} : { unkToken: '[UNK]' }),
          ...(options.continuingSubwordPrefix === undefined
            ? {}
            : { continuingSubwordPrefix: options.continuingSubwordPrefix }),
          ...(options.endOfWordSuffix === undefined ? {} : { endOfWordSuffix: options.endOfWordSuffix }),
        },
      )
      const held = c.params['heldOut'] as string[]
      const encodings = c.params['encodings'] as [string, number, [number, number]][][]
      for (const [k, w] of held.entries()) {
        const word = byteLevel ? toByteLevel(w) : w
        const got = bpeTokenize(hf, word)
        // HF's model offsets count UTF-8 bytes of the word; ours count UTF-16 units.
        const bytes = (i: number): number => utf8Bytes(word.slice(0, i)).length
        expect(got.map((t) => [t.token, affixed ? -1 : t.id, [bytes(t.start), bytes(t.end)]])).toEqual(encodings[k])
        expect(bpeTokenize(model, word).map((t) => t.token)).toEqual(encodings[k]?.map((e) => e[0]))
      }
    })
  }
})
