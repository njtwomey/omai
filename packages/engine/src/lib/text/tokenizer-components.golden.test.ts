import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { codePointSpans } from '../../testing/text.ts'
import {
  alignedText,
  applyNormalizer,
  applyPreTokenizer,
  decodeChain,
  loadTokenizer,
  originalSpan,
  type Span,
} from './index.ts'

describe('text/tokenizer-components against HF tokenizers', () => {
  for (const c of loadFixture('text/tokenizer_components').cases) {
    it(c.name, () => {
      const [json, inputs] = c.inputs as [string, string[]]
      const tok = loadTokenizer(json)
      switch (c.fn) {
        case 'normalizer': {
          const spec = tok.normalizer
          expect(spec).toBeDefined()
          if (spec === undefined) return
          expect(inputs.map((t) => applyNormalizer(spec, alignedText(t)).text)).toEqual(c.output)
          return
        }
        case 'preTokenizer': {
          const spec = tok.preTokenizer
          expect(spec).toBeDefined()
          if (spec === undefined) return
          const got = inputs.map((t) => {
            const pieces = applyPreTokenizer(spec, [alignedText(t)])
            const spans = codePointSpans(
              t,
              pieces.map((p): Span => originalSpan(p, 0, p.text.length)),
            )
            return pieces.map((p, i) => [p.text, spans[i]])
          })
          expect(got).toEqual(c.output)
          return
        }
        default: {
          const spec = tok.decoder
          expect(spec).toBeDefined()
          if (spec === undefined) return
          expect(decodeChain(spec, inputs).join('')).toBe(c.output)
        }
      }
    })
  }
})
