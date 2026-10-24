import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { type AccentStripping, caseFold, type NormalForm, normalizeUnicode, stripAccents } from './index.ts'

describe('text/normalise against unicodedata, str.casefold, scikit-learn and HF tokenizers', () => {
  for (const c of loadFixture('text/normalise').cases) {
    it(c.name, () => {
      const text = c.inputs[0] as string
      const got =
        c.fn === 'normalizeUnicode'
          ? normalizeUnicode(text, c.params['form'] as NormalForm)
          : c.fn === 'caseFold'
            ? caseFold(text)
            : stripAccents(text, c.params['mode'] as AccentStripping)
      expect(got).toBe(c.output)
    })
  }
})
