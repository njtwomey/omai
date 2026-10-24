import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { blobs, swissRoll } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { key, uniform } from '../../lib/random/index.ts'
import { toArray } from '../../lib/tensor/index.ts'
import { isomap } from './isomap.ts'
import { locallyLinearEmbedding } from './lle.ts'
import { MDS, mds } from './mds.ts'
import { continuity, coRankingMatrix, embeddingStress, neighbourhoodPreservation, trustworthiness } from './quality.ts'
import { SelfOrganisingMap, uMatrix } from './som.ts'
import { diffusionMap, spectralEmbedding } from './spectral.ts'
import { TSNE } from './tsne.ts'
import { umap } from './umap.ts'

const roll = generate(swissRoll, { key: key(0), knobs: { n: 70, noise: 0 } }).data
const blob = generate(blobs, { key: key(1), knobs: { n: 60, centers: 3, nFeatures: 5, clusterStd: 0.5 } }).data
const random = toArray(uniform(key(9), [60, 2])) as number[][]

describe('manifold learning', () => {
  it('quality measures: perfect for the identity, lower for a random layout', () => {
    expect(trustworthiness(blob.x, blob.x)).toBeCloseTo(1, 12)
    expect(continuity(blob.x, blob.x)).toBeCloseTo(1, 12)
    expect(neighbourhoodPreservation(blob.x, blob.x)).toBeCloseTo(1, 12)
    expect(trustworthiness(blob.x, random)).toBeLessThan(0.8)
    const q = coRankingMatrix(blob.x, blob.x)
    expect(q[0]?.[0]).toBe(60)
    expect(embeddingStress(blob.x, blob.x).raw).toBeCloseTo(0, 12)
  })

  it('MDS: classical recovers distances of planar data; SMACOF lowers its stress', () => {
    const flat = (toArray(roll.x as never) as number[][]).map((r) => [r[0] ?? 0, r[2] ?? 0])
    const c = mds({ classical: true }).fit({ x: flat })
    expect(embeddingStress(flat, c.params.embedding).normalised).toBeLessThan(1e-8)
    const s = new MDS({ maxIter: 100 }).fit(blob.x, undefined, { key: key(2) }).model
    const [steps] = recordsOf(s.exposition, 'embeddingSteps')
    const obj = steps?.objective ?? []
    expect(obj.at(-1) ?? 0).toBeLessThan(obj[0] ?? 0)
    expect(steps?.positions[0]?.length).toBe(60)
  })

  it('Isomap, LLE and the spectral embeddings keep the swiss roll neighbourhoods', () => {
    for (const est of [
      isomap({ nNeighbors: 8 }),
      locallyLinearEmbedding({ nNeighbors: 10 }),
      spectralEmbedding({ nNeighbors: 10 }),
      diffusionMap({ epsilon: 4 }),
    ]) {
      const f = est.fit(roll)
      expect(trustworthiness(roll.x, f.params.embedding)).toBeGreaterThan(0.8)
      expect(recordsOf(f.exposition, 'embeddingSteps')[0]?.positions[0]?.length).toBe(70)
    }
  })

  it('t-SNE and UMAP keep neighbours far better than random, with traced steps and bandwidths', () => {
    const t = new TSNE({ perplexity: 10, maxIter: 400 }).fit(blob.x, undefined, { key: key(3) }).model
    expect(trustworthiness(blob.x, t.params.embedding)).toBeGreaterThan(0.9)
    const [ts] = recordsOf(t.exposition, 'embeddingSteps')
    expect(ts?.bandwidths?.length).toBe(60)
    expect(ts?.iterations.at(-1)).toBe(400)
    const u = umap({ nNeighbors: 10, nEpochs: 100 }).fit(blob, { key: key(4) })
    expect(trustworthiness(blob.x, u.params.embedding)).toBeGreaterThan(0.9)
    const [us] = recordsOf(u.exposition, 'embeddingSteps')
    expect(us?.graph?.length).toBeGreaterThan(0)
    const uo = us?.objective ?? []
    expect(uo.at(-1) ?? 0).toBeLessThan(uo[0] ?? 0)
  })

  it('a self-organising map lowers its quantisation error as the neighbourhood shrinks', () => {
    const m = new SelfOrganisingMap({ rows: 4, cols: 4, epochs: 10 }).fit(blob.x, undefined, { key: key(5) }).model
    const [steps] = recordsOf(m.exposition, 'embeddingSteps')
    const obj = steps?.objective ?? []
    expect(obj.length).toBe(11)
    expect(obj.at(-1) ?? 0).toBeLessThan(obj[1] ?? 0)
    expect(uMatrix(m.params.weights, m.params.grid).length).toBe(4)
    expect(m.transform(blob.x).shape).toEqual([60, 2])
  })
})
