import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { circles } from '../datasets/index.ts'
import { adjustedRandScore } from '../metrics/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { affinityMatrix, spectralClustering, spectralClusteringEmbedding } from './spectral.ts'

const ring = generate(circles, { key: key(4), knobs: { n: 80, noise: 0.03, factor: 0.3 } }).data

describe('spectral clustering', () => {
  it('separates concentric circles with a neighbour graph; the embedding has one column per cluster', () => {
    const f = spectralClustering
      .with({ nClusters: 2, affinity: 'nearest_neighbors', nNeighbors: 8 })
      .fit(ring, { key: key(5) })
    expect(adjustedRandScore(ring.y, f.params.labels).estimate.value).toBeGreaterThan(0.95)
    const e = spectralClusteringEmbedding(ring.x, { nClusters: 2, affinity: 'nearest_neighbors', nNeighbors: 8 })
    expect(e.data.length).toBe(160)
    expect(toFlat(f.predict(ring.x))).toEqual(f.params.labels)
  })

  it('a session steps the k-means on the embedding and can be probed early', () => {
    const s = spectralClustering.with({ nClusters: 2, gamma: 5 }).session(ring, { key: key(1), logs: 'all' })
    s.advance({ steps: 1 })
    expect((s.probe(1, 'model/assignments') as number[]).length).toBe(80)
  })

  it('affinity matrices are symmetric', () => {
    const a = affinityMatrix([[0], [1], [5], [6]], 'nearest_neighbors', { nNeighbors: 2 })
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) expect(a[i * 4 + j]).toBe(a[j * 4 + i])
    expect(() => affinityMatrix([[0], [1]], 'nearest_neighbors', { nNeighbors: 9 })).toThrow()
  })
})
