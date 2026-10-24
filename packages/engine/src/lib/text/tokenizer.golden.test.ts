import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { decode, type Encoding, encode, encodeBatch, loadTokenizer } from './index.ts'

/** UTF-16 offsets to code point offsets (HF reports character offsets). */
function charOffsets(text: string, offsets: readonly (readonly [number, number])[]): number[][] {
  const index: number[] = []
  let k = 0
  for (const ch of text) {
    for (let u = 0; u < ch.length; u++) index.push(k)
    k++
  }
  index.push(k)
  return offsets.map(([a, b]) => [index[a] ?? 0, index[b] ?? 0])
}

function summary(e: Encoding, text: string, pair?: string): Record<string, unknown> {
  // Offsets of the second sequence refer to the pair's text.
  const offsets = e.offsets.map(
    (o, i) => charOffsets(e.sequenceIds[i] === 1 && pair !== undefined ? pair : text, [o])[0] ?? [0, 0],
  )
  return {
    ids: e.ids,
    tokens: e.tokens,
    offsets,
    typeIds: e.typeIds,
    attentionMask: e.attentionMask,
    specialTokensMask: e.specialTokensMask,
    wordIds: e.wordIds,
    sequenceIds: e.sequenceIds,
    overflowing: e.overflowing.map((o) => summary(o, text, pair)),
  }
}

describe('text/tokenizer against HF tokenizers', () => {
  for (const c of loadFixture('text/tokenizer').cases) {
    it(c.name, () => {
      const tok = loadTokenizer(c.inputs[0] as string)
      if (c.fn === 'encodeTexts') {
        const texts = c.inputs[1] as string[]
        expect(texts.map((t) => summary(encode(tok, t), t))).toEqual(c.output)
        return
      }
      if (c.fn === 'encodeBatch') {
        const pairs = c.inputs[1] as [string, string][]
        expect(encodeBatch(tok, pairs).map((e, i) => summary(e, pairs[i]?.[0] ?? '', pairs[i]?.[1]))).toEqual(c.output)
        return
      }
      for (const row of c.output as {
        text: string
        pair?: string
        encoding: unknown
        decoded?: string
        decodedAll?: string
      }[]) {
        const e = encode(tok, row.text, row.pair)
        expect(summary(e, row.text, row.pair)).toEqual(row.encoding)
        if (row.decoded !== undefined) {
          expect(decode(tok, e.ids)).toBe(row.decoded)
          expect(decode(tok, e.ids, { skipSpecialTokens: false })).toBe(row.decodedAll)
        }
      }
    })
  }
})
