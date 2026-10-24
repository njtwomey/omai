import { describe, expect, it } from 'vitest'
import { expectMatches, loadFixture } from '../../testing/golden.ts'
import { tensor } from '../tensor/index.ts'
import { heapsFit, lexicalDiversity, topicCoherence, zipfFit } from './index.ts'

describe('text/statistics against numpy, scipy and gensim', () => {
  for (const c of loadFixture('text/statistics').cases) {
    it(c.name, () => {
      switch (c.fn) {
        case 'zipf': {
          const z = zipfFit(c.inputs[0] as string[])
          expectMatches(tensor([z.exponent, z.constant]), c.output, c.rtol, c.atol, c.name)
          return
        }
        case 'zipfMle':
          expectMatches(
            tensor(zipfFit(c.inputs[0] as string[], { method: 'mle' }).exponent),
            c.output,
            c.rtol,
            c.atol,
            c.name,
          )
          return
        case 'heaps': {
          const h = heapsFit(c.inputs[0] as string[])
          expectMatches(tensor([h.k, h.beta]), c.output, c.rtol, c.atol, c.name)
          return
        }
        default: {
          const [topics, texts] = c.inputs as [string[][], string[][]]
          const window = c.params['window'] as number | undefined
          const r = topicCoherence(topics, texts, {
            measure: c.params['measure'] as 'u_mass' | 'c_npmi',
            ...(window === undefined ? {} : { windowSize: window }),
          })
          expectMatches(tensor([...r.perTopic, r.mean]), c.output, c.rtol, c.atol, c.name)
        }
      }
    })
  }

  it('lexical diversity measures agree with their definitions', () => {
    const tokens = 'a b a c b a d e'.split(' ')
    const d = lexicalDiversity(tokens, { window: 3 })
    expect(d.ttr).toBe(5 / 8)
    expect(d.rootTtr).toBeCloseTo(5 / Math.sqrt(8), 14)
    // Windows: aba bac acb cba bad ade (2, 3, 3, 3, 3, 3 types).
    expect(d.mattr).toBeCloseTo(17 / 18, 14)
    expect(d.hapaxRatio).toBe(3 / 5)
    expect(Number.isNaN(lexicalDiversity([]).ttr)).toBe(true)
  })
})
