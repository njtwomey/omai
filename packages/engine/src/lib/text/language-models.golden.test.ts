import { describe, expect, it } from 'vitest'
import { loadFixture } from '../../testing/golden.ts'
import { type NgramSmoothing, ngramModel } from './index.ts'

describe('text/language-models against the Chen and Goodman definitions', () => {
  for (const c of loadFixture('text/language_models').cases) {
    it(c.name, () => {
      const model = ngramModel(c.inputs[0] as string[][], {
        order: c.params['order'] as number,
        smoothing: c.params['smoothing'] as NgramSmoothing,
        unigram: c.params['unigram'] as 'uniform' | 'mle',
      })
      const queries = c.inputs[1] as [string, string[]][]
      const expected = c.output as number[]
      for (const [i, [w, h]] of queries.entries()) {
        const got = model.prob(w, h)
        expect(Math.abs(got - (expected[i] ?? 0)), `${w} | ${h.join(' ')}`).toBeLessThanOrEqual(
          c.atol + c.rtol * Math.abs(expected[i] ?? 0),
        )
      }
    })
  }
})
