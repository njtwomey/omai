import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { decode, encode, encodeBatch, tokenizer } from './index.ts'
import { toByteLevel, trainBpe } from './subword/index.ts'

const text = fc.string({ unit: fc.constantFrom('a', 'b', ' ', 'é', '😀', '!', 'z', '\n'), maxLength: 16 })

const corpus = ['ab ab abba', 'a b é😀!', 'zz top a']
const words = new Map<string, number>()
for (const line of corpus) for (const w of line.match(/ ?[^ ]+| +/gu) ?? []) words.set(w, (words.get(w) ?? 0) + 1)
const bpe = trainBpe(words, { vocabSize: 290, byteLevel: true })
const gpt2 = tokenizer({
  preTokenizer: { type: 'ByteLevel', addPrefixSpace: false },
  model: { type: 'BPE', model: bpe },
  decoder: { type: 'ByteLevel' },
})

describe('tokenizer pipelines', () => {
  it('byte-level BPE round-trips any text and offsets cover the original', () => {
    fc.assert(
      fc.property(text, (s) => {
        const e = encode(gpt2, s)
        expect(decode(gpt2, e.ids)).toBe(s)
        expect(e.tokens.join('')).toBe(toByteLevel(s))
        for (const [a, b] of e.offsets) {
          expect(a).toBeGreaterThanOrEqual(0)
          expect(b).toBeLessThanOrEqual(s.length)
          expect(a).toBeLessThanOrEqual(b)
        }
      }),
    )
  })

  it('truncation keeps the maximum length and overflowing windows overlap by the stride', () => {
    fc.assert(
      fc.property(text, fc.integer({ min: 2, max: 6 }), fc.integer({ min: 0, max: 1 }), (s, max, stride) => {
        const plain = encode(gpt2, s)
        const t = encode(tokenizer({ ...gpt2, truncation: { maxLength: max, stride } }), s)
        expect(t.ids.length).toBe(Math.min(max, plain.ids.length))
        const windows = [t, ...t.overflowing]
        for (const [i, w] of windows.entries()) {
          if (i === 0) continue
          const prev = windows[i - 1]
          expect(w.ids.slice(0, stride)).toEqual(prev?.ids.slice(prev.ids.length - stride))
        }
        if (plain.ids.length > 0) expect(windows[windows.length - 1]?.ids.at(-1)).toBe(plain.ids.at(-1))
      }),
    )
  })

  it('batch padding gives equal lengths with masks marking the padding', () => {
    fc.assert(
      fc.property(fc.array(text, { minLength: 1, maxLength: 4 }), fc.integer({ min: 1, max: 4 }), (xs, m) => {
        const tok = tokenizer({ ...gpt2, padding: { padId: 0, padToken: 'p', padToMultipleOf: m, direction: 'left' } })
        const encs = encodeBatch(tok, xs)
        const n = encs[0]?.ids.length ?? 0
        expect(n % m).toBe(0)
        for (const [i, e] of encs.entries()) {
          expect(e.ids.length).toBe(n)
          const real = encode(gpt2, xs[i] ?? '').ids
          expect(e.ids.slice(n - real.length)).toEqual(real)
          expect(e.attentionMask.reduce((a, v) => a + v, 0)).toBe(real.length)
        }
      }),
    )
  })
})
