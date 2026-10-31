import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { analogyCorpus } from '../datasets/index.ts'
import { key } from '../random/index.ts'
import { cooccurrence, pmi, svdWordVectors } from '../text/index.ts'
import { ppmiEmbedding } from './count-based.ts'
import { analogyAccuracy } from './evaluation.ts'

describe('ppmiEmbedding', () => {
  it('is the composition of co-occurrence, shifted PPMI and the SVD', () => {
    const s = [
      ['a', 'b', 'c', 'd'],
      ['b', 'c', 'd', 'a'],
      ['c', 'a', 'b', 'd'],
    ]
    const direct = svdWordVectors(pmi(cooccurrence(s, { window: 2 }), { positive: true, shift: 2, alpha: 0.75 }), 3, {
      power: 0.5,
    })
    const ours = ppmiEmbedding(s, { window: 2, dimension: 3, shift: 2 })
    expect(ours.words).toEqual(direct.words)
    for (const [i, r] of ours.rows.entries()) expect(Array.from(r)).toEqual(Array.from(direct.rows[i] as Float64Array))
  })

  it('recovers every planted analogy of the planted corpus, with no training', () => {
    const { data, truth } = generate(analogyCorpus, {
      key: key(5),
      knobs: { concepts: 4, attributes: 3, contexts: 3, sentences: 2500, noise: 0.15 },
    })
    const sentences = (data.documents ?? []).map((d) => d.split(' '))
    const quads = truth.extra?.['analogies'] as [string, string, string, string][]
    const v = ppmiEmbedding(sentences, { window: 4, dimension: 16 })
    const acc = analogyAccuracy(v, quads)
    expect(acc.total).toBe(quads.length)
    expect(acc.accuracy).toBeGreaterThan(0.9)
  })

  it('rejects a dimension above the vocabulary', () => {
    expect(() => ppmiEmbedding([['a', 'b']], { dimension: 5 })).toThrow()
  })
})
