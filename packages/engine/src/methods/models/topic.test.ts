import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { topicCorpus } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { documentTermCounts, latentDirichletAllocation } from './topic.ts'

const g = generate(topicCorpus, { key: key(1), knobs: { topics: 'bars', nDocs: 150, docLength: 40, alpha: 0.5 } })
const counts = documentTermCounts(g.data)
const truth = (g.truth?.latents?.['topics'] ?? []) as number[][]

describe('latent Dirichlet allocation', () => {
  it('collapsed Gibbs recovers the bars topics and records its sweeps', () => {
    const f = latentDirichletAllocation({ nTopics: 10, nIter: 60, alpha: 0.5, eta: 0.1 }).fit(counts, { key: key(2) })
    // Every true bar is matched by a learned topic (cosine similarity).
    const cos = (a: readonly number[], b: readonly number[]) => {
      let ab = 0
      let aa = 0
      let bb = 0
      for (let i = 0; i < a.length; i++) {
        ab += (a[i] ?? 0) * (b[i] ?? 0)
        aa += (a[i] ?? 0) ** 2
        bb += (b[i] ?? 0) ** 2
      }
      return ab / Math.sqrt(aa * bb)
    }
    const matched = truth.filter((t) => Math.max(...f.params.topicWord.map((k) => cos(t, k))) > 0.9).length
    expect(matched).toBeGreaterThanOrEqual(8)
    const [steps] = recordsOf(f.exposition, 'mixtureSteps')
    expect(steps?.objective.at(-1)).toBeGreaterThan((steps?.objective[0] ?? 0) + 0.2)
    expect(steps?.means[0]?.length).toBe(10)
    const [comp] = recordsOf(f.exposition, 'components')
    expect(comp?.features.length).toBe(25)
    expect(comp?.scores?.length).toBe(150)
    expect(recordsOf(f.exposition, 'coefficientTable')[0]?.names.length).toBe(80)
    expect(f.transform(counts.x).shape).toEqual([150, 10])
    expect(f.params.perplexity).toBeLessThan(25)
  })
})
