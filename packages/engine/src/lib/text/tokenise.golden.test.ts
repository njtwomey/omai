import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import {
  bertPreTokenize,
  graphemeTokenize,
  PRE_TOKENIZER_PATTERNS,
  preTokenize,
  regexTokenize,
  type Span,
  type SplitBehavior,
  splitSpans,
  whitespaceTokenize,
  wordPunctTokenize,
  wordTokenize,
} from './index.ts'

/** UTF-16 spans to code point spans (the offsets HF tokenizers reports). */
export function codePointSpans(text: string, spans: readonly Span[]): number[][] {
  const index: number[] = []
  let k = 0
  for (const ch of text) {
    for (let u = 0; u < ch.length; u++) index.push(k)
    k++
  }
  index.push(k)
  return spans.map(([a, b]) => [index[a] ?? 0, index[b] ?? 0])
}

describe('text/tokenise against nltk, scikit-learn, the regex module and HF pre-tokenizers', () => {
  for (const c of loadFixture('text/tokenise').cases) {
    it(c.name, () => {
      const t = c.inputs[0] as string
      const run = (): unknown => {
        switch (c.fn) {
          case 'regexTokenize':
            return c.params['gaps'] === true
              ? regexTokenize(t, /[\p{White_Space}]+/u, { gaps: true })
              : regexTokenize(t, /[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]+/u)
          case 'wordPunctTokenize':
            return wordPunctTokenize(t)
          case 'whitespaceTokenize':
            return whitespaceTokenize(t)
          case 'wordTokenize':
            return wordTokenize(t)
          case 'graphemeTokenize':
            return graphemeTokenize(t)
          case 'preTokenize':
            return preTokenize(t, c.params['pattern'] as 'gpt2')
          case 'hfWhitespace':
            return codePointSpans(t, splitSpans(t, PRE_TOKENIZER_PATTERNS.whitespace, 'removed', true))
          case 'bertPreTokenize':
            return codePointSpans(t, bertPreTokenize(t))
          default:
            return codePointSpans(
              t,
              splitSpans(
                t,
                c.params['delimiter'] as string,
                c.params['behavior'] as SplitBehavior,
                c.params['invert'] as boolean,
              ),
            )
        }
      }
      expect(run()).toEqual(c.output)
    })
  }
})
