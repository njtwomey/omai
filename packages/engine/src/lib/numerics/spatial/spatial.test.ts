import { describe, expect, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import { key, normal } from '../../random/index.ts'
import { type Tensor, tensor, toFlat } from '../../tensor/index.ts'
import type { DistanceMetric } from '../geometry/index.ts'
import { toDense } from '../sparse/index.ts'
import * as S from './index.ts'

describe('neighbour search against sklearn', () => {
  for (const c of loadFixture('numerics/spatial/spatial').cases) {
    it(c.name, () => {
      const t = c.inputs.map((v) => decode(v) as Tensor)
      const [a, b] = t as [Tensor, Tensor]
      const metric = c.params['metric'] as DistanceMetric
      const actual =
        c.fn === 'bruteIndices'
          ? S.kNearest(a, b, 5).indices
          : c.fn === 'bruteDistances'
            ? S.kNearest(a, b, 5).distances
            : c.fn === 'knnGraph'
              ? toDense(S.kNeighboursGraph(a, 3, { mode: 'distance' }))
              : c.fn === 'radiusGraph'
                ? toDense(S.radiusNeighboursGraph(a, 0.8, { mode: 'distance' }))
                : c.fn === 'kdIndices'
                  ? S.kdTree(a, { metric, leafSize: 8 }).query(b, 4).indices
                  : c.fn === 'kdDistances'
                    ? S.kdTree(a, { metric, leafSize: 8 }).query(b, 4).distances
                    : c.fn === 'kdCounts'
                      ? tensor(S.kdTree(a, { metric, leafSize: 8 }).countNeighbours(b, 1))
                      : c.fn === 'ballIndices'
                        ? S.ballTree(a, { metric, leafSize: 8 }).query(b, metric === 'haversine' ? 3 : 4).indices
                        : c.fn === 'ballDistances'
                          ? S.ballTree(a, { metric, leafSize: 8 }).query(b, metric === 'haversine' ? 3 : 4).distances
                          : c.fn === 'kmeansCentroids'
                            ? S.kmeansCodebook(a, 3, { init: b, maxIterations: 300, tol: 0 }).centroids
                            : tensor(S.kmeansCodebook(a, 3, { init: b, maxIterations: 300, tol: 0 }).inertia)
      expectMatches(actual, c.output, c.rtol, c.atol, c.name)
    })
  }
})

const data = normal(key(11), [400, 8])
const queries = normal(key(12), [30, 8])
const truth = S.kNearest(data, queries, 10)
const recall = (found: readonly (readonly number[])[]): number => {
  let hit = 0
  found.forEach((row, q) => {
    const set = new Set(Array.from(truth.indices.data.slice(q * 10, q * 10 + 10)))
    for (const i of row.slice(0, 10)) if (set.has(i)) hit++
  })
  return hit / (found.length * 10)
}
const rows = (q: number): number[] => toFlat(queries).slice(q * 8, q * 8 + 8)

describe('exact trees', () => {
  it('KD, ball and VP trees agree with brute force for every metric, and log their pruning', () => {
    for (const metric of ['euclidean', 'cityblock', 'chebyshev'] as const) {
      const brute = S.kNearest(data, queries, 6, { metric })
      for (const tree of [
        S.kdTree(data, { metric, leafSize: 10 }),
        S.ballTree(data, { metric, leafSize: 10 }),
        S.vpTree(data, { metric, leafSize: 10 }),
      ]) {
        const r = tree.query(queries, 6, { log: true })
        expect(Array.from(r.indices.data)).toEqual(Array.from(brute.indices.data))
        const evaluations = (r.logs ?? []).reduce((s, l) => s + l.distanceEvaluations, 0)
        expect(evaluations).toBeLessThan(400 * 30)
        expect((r.logs ?? []).some((l) => l.pruned.length > 0)).toBe(true)
        const radius = tree.queryRadius(queries, 2.5)
        const bruteRadius = S.radiusNeighbours(data, queries, 2.5, { metric })
        expect(radius.indices).toEqual(bruteRadius.map((h) => h.indices))
      }
    }
  })

  it('tree nodes describe a partition for figures', () => {
    const tree = S.kdTree(data, { leafSize: 20 })
    const leaves = tree.nodes.filter((n) => n.children.length === 0)
    expect(leaves.reduce((s, n) => s + n.points.length, 0)).toBe(400)
    expect(tree.nodes[0]?.bounds.lo?.length).toBe(8)
  })
})

describe('approximate indexes', () => {
  it('HNSW reaches high recall and its search log descends the layers', () => {
    const index = S.hnswIndex(data, { m: 8, efConstruction: 100, key: key(3) })
    const found = Array.from({ length: 30 }, (_, q) => index.search(rows(q), 10, { ef: 64 }).indices)
    expect(recall(found)).toBeGreaterThan(0.9)
    const logged = index.search(rows(0), 10, { log: true })
    expect(logged.log?.path.length).toBe((index.levels[index.entryPoint] ?? 0) + 1)
    expect(logged.log?.distanceEvaluations ?? 0).toBeLessThan(400)
  })

  it('NN-descent builds an accurate k-NN graph and records fewer updates as it converges', () => {
    const truthGraph = S.kNearest(data, undefined, 10, { excludeSelf: true })
    const r = S.nnDescent(data, 10, { key: key(4), truth: truthGraph.indices.data })
    let hit = 0
    for (let v = 0; v < 400; v++) {
      const set = new Set(Array.from(truthGraph.indices.data.slice(v * 10, v * 10 + 10)))
      for (let j = 0; j < 10; j++) if (set.has(r.indices.data[v * 10 + j] ?? -1)) hit++
    }
    expect(hit / 4000).toBeGreaterThan(0.9)
    expect(r.updates.at(-1) ?? 0).toBeLessThan(r.updates[0] ?? 0)
  })

  it('LSH recall grows with tables and multi-probe, at the cost of more candidates', () => {
    const few = S.lshIndex(data, { family: 'pstable', tables: 4, k: 6, w: 3, key: key(5) })
    const many = S.lshIndex(data, { family: 'pstable', tables: 16, k: 6, w: 3, key: key(5) })
    const r1 = recall(Array.from({ length: 30 }, (_, q) => few.query(rows(q), 10).indices))
    const r2 = recall(Array.from({ length: 30 }, (_, q) => many.query(rows(q), 10).indices))
    const r3 = recall(Array.from({ length: 30 }, (_, q) => few.query(rows(q), 10, { probes: 20 }).indices))
    expect(r2).toBeGreaterThan(r1)
    expect(r3).toBeGreaterThan(r1)
    const cosine = S.lshIndex(data, { family: 'hyperplane', tables: 10, k: 4, key: key(6) })
    expect(cosine.query(rows(0), 5).indices.length).toBe(5)
  })

  it('collision curves and MinHash banding behave as their formulas say', () => {
    expect(S.collisionProbability('pstable', 1, { w: 4 })).toBeGreaterThan(
      S.collisionProbability('pstable', 4, { w: 4 }),
    )
    expect(S.collisionProbability('pstable', 1e-9, { w: 4, p: 1 })).toBeCloseTo(1, 6)
    expect(S.bandingProbability(0.3, 5, 20)).toBeLessThan(0.1)
    expect(S.bandingProbability(0.9, 5, 20)).toBeGreaterThan(0.99)
    const sets = [
      Array.from({ length: 100 }, (_, i) => i),
      Array.from({ length: 100 }, (_, i) => i + 20),
      Array.from({ length: 100 }, (_, i) => i + 500),
    ]
    const sig = S.minHash(sets, key(7), 200)
    // The estimate's standard error is sqrt(J (1 - J) / 200), about 0.033: allow three.
    expect(Math.abs(S.estimateJaccard(sig[0] as number[], sig[1] as number[]) - 80 / 120)).toBeLessThan(0.1)
    // 40 bands of 5 rows: similarity 2/3 becomes a candidate with probability 1 - (1 - (2/3)^5)^40, about 0.996.
    expect(S.bandingProbability(2 / 3, 5, 40)).toBeGreaterThan(0.99)
    expect(S.minHashBanding(sig, 40, 5)).toEqual([[0, 1]])
  })

  it('product quantisation reconstructs approximately; IVF trades probes for recall', () => {
    const pq = S.productQuantiser(data, { m: 4, k: 32, key: key(8) })
    const codes = pq.encode(data)
    const recon = toFlat(pq.decode(codes))
    const x = toFlat(data)
    let err = 0
    let total = 0
    for (let i = 0; i < x.length; i++) {
      err += ((recon[i] ?? 0) - (x[i] ?? 0)) ** 2
      total += (x[i] ?? 0) ** 2
    }
    expect(err / total).toBeLessThan(0.5)
    // Asymmetric distances equal the distances to the reconstructions.
    const adc = pq.asymmetricDistances(rows(0), codes.subarray(0, 4 * 5))
    for (let i = 0; i < 5; i++) {
      const direct = rows(0).reduce((s, v, j) => s + (v - (recon[i * 8 + j] ?? 0)) ** 2, 0)
      expect(adc[i]).toBeCloseTo(direct, 10)
    }
    const ivf = S.ivfIndex(data, { lists: 16, key: key(9) })
    const r1 = recall(Array.from({ length: 30 }, (_, q) => ivf.search(rows(q), 10, { nprobe: 1 }).indices))
    const r4 = recall(Array.from({ length: 30 }, (_, q) => ivf.search(rows(q), 10, { nprobe: 6 }).indices))
    expect(r4).toBeGreaterThan(r1)
    const ivfpq = S.ivfIndex(data, { lists: 8, key: key(10), pq: { m: 4, k: 16 } })
    expect(ivfpq.search(rows(1), 3, { nprobe: 8 }).indices.length).toBe(3)
  })
})
