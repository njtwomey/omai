import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs } from '../datasets/index.ts'
import { adjustedRandScore } from '../metrics/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson, tensorSeries } from '../trace/index.ts'
import {
  agglomerativeAlgorithm,
  agglomerativeClustering,
  birch,
  birchAlgorithm,
  birchLeaves,
  copheneticCorrelation,
  cutTree,
  dendrogram,
  linkageMatrix,
  singleLinkageTree,
} from './hierarchical.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 45, centers: 3, clusterStd: 0.5 } })

describe('agglomerative clustering', () => {
  for (const linkage of ['ward', 'single', 'complete', 'average'] as const) {
    it(`${linkage} linkage recovers well-separated blobs with n - 1 merges of rising height`, () => {
      const f = agglomerativeClustering.with({ nClusters: 3, linkage }).fit(data, { logs: 'all' })
      expect(adjustedRandScore(data.y, f.params.labels).estimate.value).toBeGreaterThan(0.99)
      expect(f.params.merges.length).toBe(44)
      const heights = f.params.merges.map((m) => m[2])
      for (let t = 1; t < heights.length; t++)
        expect(heights[t] ?? 0).toBeGreaterThanOrEqual((heights[t - 1] ?? 0) - 1e-12)
      expect(metricSeries(f.run, 'solver/clusters').values.at(-1)).toBe(1)
      expect(tensorSeries(f.run, 'solver/labels').length).toBeGreaterThan(1)
      expect(copheneticCorrelation(data.x, f.params.merges)).toBeGreaterThan(0.5)
    })
  }

  it('one merge per step; the partition at an early step has n - t clusters; the state round-trips', () => {
    const s = agglomerativeClustering.with({ nClusters: 3 }).session(data, { key: key(0), logs: 'all' })
    s.advance({ steps: 10 })
    expect(new Set(s.probe(10, 'model/assignments') as number[]).size).toBe(35)
    expect(s.fittedAt(10).params.merges.length).toBe(10)
    const alg = agglomerativeAlgorithm(data.x)
    const mid = runAlgorithm(alg, { maxSteps: 5 }).state
    expect(stateFromJson(stateToJson(mid))).toEqual(mid)
  })

  it('cuts by height, lays out a dendrogram, takes a precomputed matrix', () => {
    const merges = linkageMatrix(data.x, 'single')
    expect(cutTree(merges, 45, { height: Number.POSITIVE_INFINITY }).every((l) => l === 0)).toBe(true)
    const tree = dendrogram(merges, 45)
    expect([...tree.leaves].sort((a, b) => a - b)).toEqual(Array.from({ length: 45 }, (_, i) => i))
    expect(tree.links.length).toBe(44)
    expect(agglomerativeClustering.with({ distanceThreshold: 1e9 }).fit(data).params.nClusters).toBe(1)
    expect(
      linkageMatrix(
        [
          [0, 1, 5],
          [1, 0, 4],
          [5, 4, 0],
        ],
        'single',
        'precomputed',
      ),
    ).toEqual([
      [0, 1, 1, 2],
      [2, 3, 4, 3],
    ])
    expect(toFlat(agglomerativeClustering.fit(data).predict(data.x)).length).toBe(45)
  })

  it('single linkage of an edge list is the minimum spanning tree in order of weight', () => {
    expect(
      singleLinkageTree(4, [
        [0, 1, 3],
        [1, 2, 1],
        [2, 3, 2],
        [0, 3, 0.5],
      ]),
    ).toEqual([
      [0, 3, 0.5, 2],
      [1, 2, 1, 2],
      [4, 5, 2, 4],
    ])
  })
})

describe('BIRCH', () => {
  it('inserts one row per step; subclusters appear, and the clusters recover the blobs', () => {
    const f = birch.with({ nClusters: 3, threshold: 0.8 }).fit(data)
    expect(adjustedRandScore(data.y, f.params.labels).estimate.value).toBeGreaterThan(0.95)
    expect(toFlat(f.predict(data.x)).length).toBe(45)
    const s = birch.with({ threshold: 0.8 }).session(data, { key: key(0), logs: 'all' })
    s.advance({ steps: 5 })
    expect(s.fittedAt(5).params.subclusterCentres.length).toBeGreaterThan(0)
  })

  it('splits nodes past the branching factor and keeps the chain of leaves', () => {
    const rows = Array.from({ length: 30 }, (_, i) => [i * 3])
    const state = runAlgorithm(birchAlgorithm(rows, { threshold: 0.1, branchingFactor: 3 }), { maxSteps: 30 }).state
    const leaves = birchLeaves(state)
    expect(leaves.length).toBe(30)
    expect(leaves.reduce((t, l) => t + l.n, 0)).toBe(30)
    expect(stateFromJson(stateToJson(state))).toEqual(state)
  })
})
