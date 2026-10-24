// Shared helpers of the text golden tests. Node only; not published.

import { PRE_TOKENIZER_PATTERNS, preTokenize, type Span, splitSpans } from '../lib/text/index.ts'

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

/** Word counts of a corpus as HF trainers see it: Whitespace pre-tokenisation, or GPT-2's pattern when byte level. */
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
