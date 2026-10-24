import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { cyk, earley, grammar, sentenceProbability, toChomskyNormalForm, treeToString, viterbiParse } from './index.ts'

describe('text/parsing against nltk', () => {
  for (const c of loadFixture('text/parsing').cases) {
    it(c.name, () => {
      const [source, tokens] = c.inputs as [string, string[]]
      const g = grammar(source)
      if (c.fn === 'parses') {
        const trees = earley(g, tokens).trees.map(treeToString).sort()
        expect(trees).toEqual(c.output)
        expect(trees).toEqual(c.params['earley'])
        // The CNF grammar recognises the same sentences, with as many parses when no unit chain merges them.
        expect(cyk(toChomskyNormalForm(g), tokens).recognised).toBe(trees.length > 0)
        return
      }
      const [best, prob, total] = c.output as [string, number, number]
      const v = viterbiParse(g, tokens)
      expect(v === undefined ? '' : treeToString(v.tree)).toBe(best)
      expect(v?.probability ?? 0).toBeCloseTo(prob, 15)
      const p = sentenceProbability(g, tokens)
      expect(Math.abs(p - total)).toBeLessThanOrEqual(1e-12 * Math.abs(total) + 1e-18)
      expect(Math.abs(sentenceProbability(toChomskyNormalForm(g), tokens) - total)).toBeLessThanOrEqual(
        1e-12 * total + 1e-18,
      )
    })
  }
})
