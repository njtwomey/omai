import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { swissRoll } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key } from '../../lib/random/index.ts'
import { FactorAnalysis, ProbabilisticPCA, rotateLoadings } from './probabilistic.ts'

const { data } = generate(swissRoll, { key: key(0), knobs: { n: 60 } })

describe('latent-Gaussian decompositions', () => {
  it('probabilistic PCA and factor analysis give scores and finite likelihoods', () => {
    for (const est of [new ProbabilisticPCA({ nComponents: 2 }), new FactorAnalysis({ nComponents: 2 })]) {
      const f = est.fit(data.x).model
      expect(f.transform(data.x).shape).toEqual([60, 2])
      expect(Number.isFinite(f.score({ x: data.x }))).toBe(true)
      expect(recordsOf(f.exposition, 'components')[0]?.scores?.length).toBe(60)
    }
  })

  it('varimax keeps the communalities (row norms) of the loadings', () => {
    const a = [
      [0.8, 0.3],
      [0.7, 0.4],
      [0.2, 0.9],
      [0.1, 0.8],
    ]
    const r = rotateLoadings(a)
    for (const [i, row] of a.entries()) expect(Math.hypot(...(r[i] ?? []))).toBeCloseTo(Math.hypot(...row), 10)
  })
})
