import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { nltkWordTokenize, treebankDetokenize, treebankSpans, treebankTokenize } from './index.ts'

describe('text/treebank against nltk', () => {
  for (const c of loadFixture('text/treebank').cases) {
    it(c.name, () => {
      const options = { convertParentheses: c.params['convertParentheses'] === true }
      const got =
        c.fn === 'treebankTokenize'
          ? treebankTokenize(c.inputs[0] as string, options)
          : c.fn === 'treebankSpans'
            ? treebankSpans(c.inputs[0] as string)
            : c.fn === 'nltkWordTokenize'
              ? nltkWordTokenize(c.inputs[0] as string)
              : treebankDetokenize(c.inputs[0] as string[], options)
      expect(got).toEqual(c.output)
    })
  }
})
