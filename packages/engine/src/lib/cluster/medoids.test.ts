import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs } from '../datasets/index.ts'
import { adjustedRandScore } from '../metrics/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { claraAlgorithm, kMedoids, pamAlgorithm, pamBuild, pamSolve } from './medoids.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 60, centers: 3, clusterStd: 0.5 } })
const ari = (labels: readonly number[]): number => adjustedRandScore(data.y, labels).estimate.value

describe('k-medoids', () => {
  it('PAM recovers the blobs, building then swapping with the cost falling every step', () => {
    const f = kMedoids.with({ nClusters: 3 }).fit(data, { logs: 'all' })
    expect(ari(f.params.labels)).toBeGreaterThan(0.99)
    const cost = metricSeries(f.run, 'solver/cost').values
    for (let t = 1; t < cost.length; t++) expect(cost[t] ?? 0).toBeLessThanOrEqual((cost[t - 1] ?? 0) + 1e-12)
    expect(f.params.medoidIndices.length).toBe(3)
    expect(toFlat(f.predict(data.x)).length).toBe(60)
  })

  it('a session shows the medoids being built one per step', () => {
    const s = kMedoids.with({ nClusters: 3 }).session(data, { key: key(0), logs: 'all' })
    s.advance({ steps: 1 })
    expect(s.fittedAt(0).params.medoidIndices.length).toBe(1)
    expect(s.fittedAt(1).params.medoidIndices.length).toBe(2)
  })

  it('CLARA draws a sample per step and keeps the best medoids', () => {
    const f = kMedoids.with({ nClusters: 3, method: 'clara', sampleSize: 25, samples: 5 }).fit(data, { key: key(1) })
    expect(ari(f.params.labels)).toBeGreaterThan(0.95)
    const alg = claraAlgorithm(data.x, { nClusters: 3, sampleSize: 25 })
    const state = runAlgorithm(alg, { maxSteps: 3, key: key(2) }).state
    expect(stateFromJson(stateToJson(state))).toEqual(state)
  })

  it('works on a precomputed dissimilarity matrix; the helpers agree with the algorithm', () => {
    const d = [0, 1, 9, 9, 1, 0, 9, 9, 9, 9, 0, 1, 9, 9, 1, 0]
    const square = [
      [0, 1, 9, 9],
      [1, 0, 9, 9],
      [9, 9, 0, 1],
      [9, 9, 1, 0],
    ]
    expect(pamBuild(d, 4, 2).length).toBe(2)
    expect(pamSolve(d, 4, 2).medoids).toEqual([0, 2])
    const out = runAlgorithm(pamAlgorithm(square, { nClusters: 2, metric: 'precomputed' }), { maxSteps: 10 })
    expect(out.state.medoids).toEqual([0, 2])
    expect(() => pamAlgorithm([[0], [1]], { nClusters: 5 })).toThrow()
  })
})
