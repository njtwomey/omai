import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { hyphenate, hyphenationPatterns, insertHyphens } from './index.ts'

describe('text/hyphenation against pyphen', () => {
  for (const c of loadFixture('text/hyphenation').cases) {
    it(c.name, () => {
      const [patterns, words] = c.inputs as [string[], string[]]
      const p = hyphenationPatterns(patterns)
      const options = { leftMin: c.params['left'] as number, rightMin: c.params['right'] as number }
      expect(words.map((w) => [hyphenate(w, p, options).points, insertHyphens(w, p, options)])).toEqual(c.output)
    })
  }
})
