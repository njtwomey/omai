import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { key, split } from '../../random/index.ts'
import { recorder } from '../../trace/index.ts'
import {
  bpeTokenize,
  fromByteLevel,
  toByteLevel,
  trainBpe,
  trainUnigram,
  trainWordPiece,
  unigramModel,
  unigramNbest,
  unigramSample,
  unigramTokenize,
  wordPieceTokenize,
} from './index.ts'

const word = fc.string({ unit: fc.constantFrom('a', 'b', 'c', 'd', 'é', '😀'), minLength: 1, maxLength: 7 })
const counts = fc
  .array(fc.tuple(word, fc.integer({ min: 1, max: 9 })), { minLength: 1, maxLength: 8 })
  .map((xs) => new Map(xs))

describe('BPE', () => {
  it('training is deterministic in the order of the word counts, and encodings spell the word', () => {
    fc.assert(
      fc.property(counts, fc.integer({ min: 4, max: 30 }), (wc, size) => {
        const a = trainBpe(wc, { vocabSize: size, unkToken: '[UNK]', specialTokens: ['[UNK]'] })
        const b = trainBpe(new Map([...wc].reverse()), { vocabSize: size, unkToken: '[UNK]', specialTokens: ['[UNK]'] })
        expect(a.merges).toEqual(b.merges)
        for (const w of wc.keys()) {
          const toks = bpeTokenize(a, w)
          expect(toks.map((t) => t.token).join('')).toBe(w)
          expect(toks.map((t) => w.slice(t.start, t.end)).join('')).toBe(w)
        }
      }),
    )
  })

  it('dropout keeps the spelling, is reproducible by key, and 0 or 1 give the extremes', () => {
    fc.assert(
      fc.property(counts, fc.nat(1000), (wc, seed) => {
        const m = trainBpe(wc, { vocabSize: 25 })
        for (const w of wc.keys()) {
          const k = key(seed)
          const d = bpeTokenize(m, w, { dropout: 0.4, key: k })
          expect(d.map((t) => t.token).join('')).toBe(w)
          expect(bpeTokenize(m, w, { dropout: 0.4, key: k })).toEqual(d)
          expect(bpeTokenize(m, w, { dropout: 1, key: k }).length).toBe(Array.from(w).length)
          expect(bpeTokenize(m, w, { dropout: 0, key: k })).toEqual(bpeTokenize(m, w))
        }
      }),
    )
  })

  it('byte-level mapping round-trips any text and byte-level BPE needs no unknown token', () => {
    fc.assert(
      fc.property(fc.string({ unit: 'grapheme', maxLength: 12 }), (s) => {
        expect(fromByteLevel(toByteLevel(s))).toBe(s)
        const m = trainBpe(new Map([[s.length > 0 ? s : 'x', 3]]), { vocabSize: 270, byteLevel: true })
        const bl = toByteLevel(s)
        expect(
          bpeTokenize(m, bl)
            .map((t) => t.token)
            .join(''),
        ).toBe(bl)
      }),
    )
  })

  it('records one merge event per step and stops at the vocabulary size or when pairs run out', () => {
    const rec = recorder()
    const m = trainBpe(
      new Map([
        ['abab', 3],
        ['ab', 1],
      ]),
      { vocabSize: 100, tracer: rec },
    )
    expect(m.merges).toEqual([
      ['a', 'b'],
      ['ab', 'ab'],
    ])
    expect(m.vocab.size).toBe(4)
    expect(trainBpe(new Map([['a', 5]]), { vocabSize: 10 }).merges).toEqual([])
    expect(trainBpe(new Map([['aaaa', 5]]), { vocabSize: 10, minFrequency: 100 }).merges).toEqual([])
  })
})

describe('WordPiece', () => {
  it('pieces spell the word (prefixes removed) or the word becomes the unknown token', () => {
    fc.assert(
      fc.property(counts, fc.constantFrom('frequency' as const, 'likelihood' as const), (wc, scoring) => {
        const m = trainWordPiece(wc, { vocabSize: 30, scoring })
        for (const w of [...wc.keys(), 'zzz']) {
          const toks = wordPieceTokenize(m, w)
          if (toks.length === 1 && toks[0]?.token === '[UNK]') continue
          expect(toks.map((t, i) => (i > 0 ? t.token.slice(2) : t.token)).join('')).toBe(w)
        }
        expect(wordPieceTokenize(m, 'zzz').map((t) => t.token)).toEqual(['[UNK]'])
      }),
    )
  })
})

describe('Unigram', () => {
  const model = unigramModel(
    [
      ['<unk>', 0],
      ['a', -1.2],
      ['b', -1.5],
      ['ab', -2.1],
      ['ba', -2.4],
      ['aba', -3.3],
      ['bab', -3.9],
    ],
    { unkId: 0 },
  )
  /** Every segmentation of a word over the model's pieces, with its score. */
  function segmentations(w: string): [string[], number][] {
    if (w.length === 0) return [[[], 0]]
    const out: [string[], number][] = []
    for (const [p, s] of model.pieces.slice(1))
      if (w.startsWith(p)) for (const [rest, r] of segmentations(w.slice(p.length))) out.push([[p, ...rest], s + r])
    return out
  }

  it('Viterbi and n-best agree with brute-force enumeration', () => {
    fc.assert(
      fc.property(fc.string({ unit: fc.constantFrom('a', 'b'), minLength: 1, maxLength: 8 }), (w) => {
        const all = segmentations(w).sort((x, y) => y[1] - x[1])
        const nb = unigramNbest(model, w, 4)
        expect(nb.length).toBe(Math.min(4, all.length))
        for (const [i, s] of nb.entries()) expect(s.score).toBeCloseTo(all[i]?.[1] ?? 0, 12)
        expect(unigramTokenize(model, w).map((t) => t.token)).toEqual(nb[0]?.tokens.map((t) => t.token))
      }),
    )
  })

  it('sampling follows exp(alpha * score) over segmentations', () => {
    const w = 'abab'
    const all = segmentations(w)
    const alpha = 0.7
    const z = all.reduce((s, [, sc]) => s + Math.exp(alpha * sc), 0)
    const freq = new Map<string, number>()
    let k = key(11)
    const n = 4000
    for (let i = 0; i < n; i++) {
      const [a, b] = split(k)
      k = b ?? k
      const s = unigramSample(model, w, a ?? k, alpha)
        .map((t) => t.token)
        .join(' ')
      freq.set(s, (freq.get(s) ?? 0) + 1)
    }
    for (const [seg, sc] of all) {
      const p = Math.exp(alpha * sc) / z
      const got = (freq.get(seg.join(' ')) ?? 0) / n
      expect(Math.abs(got - p)).toBeLessThan(4 * Math.sqrt((p * (1 - p)) / n) + 1e-3)
    }
  })

  it('training terminates on degenerate inputs and covers every character', () => {
    for (const wc of [
      new Map([['a', 1]]),
      new Map([
        ['▁aaaa', 3],
        ['▁b', 1],
      ]),
      new Map([['▁x😀y', 2]]),
    ]) {
      const m = trainUnigram(wc, { vocabSize: 8 })
      for (const w of wc.keys())
        expect(
          unigramTokenize(m, w)
            .map((t) => t.token)
            .join(''),
        ).toBe(w)
    }
    expect(() => trainUnigram(new Map([['abcdef', 1]]), { vocabSize: 3 })).toThrow(/too small/)
  })
})
