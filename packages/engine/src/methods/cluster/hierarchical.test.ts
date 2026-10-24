import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { blobs } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { adjustedRandScore } from '../../lib/metrics/index.ts'
import { key } from '../../lib/random/index.ts'
import { AgglomerativeClustering, agglomerativeClustering, cutTree, dendrogram, linkageMatrix } from './hierarchical.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 45, centers: 3, clusterStd: 0.5 } })

describe('agglomerative clustering', () => {
  for (const linkage of ['ward', 'single', 'complete', 'average'] as const) {
    it(`${linkage} linkage recovers well-separated blobs with n - 1 recorded merges`, () => {
      const f = agglomerativeClustering({ nClusters: 3, linkage }).fit(data)
      expect(adjustedRandScore(data.y, f.params.labels).estimate.value).toBeGreaterThan(0.99)
      const [steps] = recordsOf(f.exposition, 'clusteringSteps')
      expect(steps?.merges?.length).toBe(44)
      // The rows ride along, labelled by the data's classes, so the dendrogram view needs nothing else.
      expect(steps?.rows?.points).toHaveLength(45)
      expect(steps?.rows?.labels).toHaveLength(45)
      const heights = (steps?.merges ?? []).map((m) => m[2])
      for (let t = 1; t < heights.length; t++)
        expect(heights[t] ?? 0).toBeGreaterThanOrEqual((heights[t - 1] ?? 0) - 1e-12)
      expect(f.params.copheneticCorrelation).toBeGreaterThan(0.5)
    })
  }

  it('cuts by height, lays out a dendrogram and has a class form', () => {
    const merges = linkageMatrix(data.x, 'single')
    expect(cutTree(merges, 45, { height: Number.POSITIVE_INFINITY }).every((l) => l === 0)).toBe(true)
    const tree = dendrogram(merges, 45)
    expect([...tree.leaves].sort((a, b) => a - b)).toEqual(Array.from({ length: 45 }, (_, i) => i))
    expect(tree.links.length).toBe(44)
    const c = new AgglomerativeClustering({ distanceThreshold: 1e9 }).fit(data.x)
    expect(c.model.params.nClusters).toBe(1)
  })
})
