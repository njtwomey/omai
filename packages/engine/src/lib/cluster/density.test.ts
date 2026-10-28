import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs, moons } from '../datasets/index.ts'
import { adjustedRandScore } from '../metrics/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import {
  dbscan,
  dbscanAlgorithm,
  estimateBandwidth,
  hdbscan,
  hdbscanClustering,
  meanShift,
  optics,
  opticsAlgorithm,
  opticsClustersAt,
  opticsOrdering,
} from './density.ts'

const moonData = generate(moons, { key: key(1), knobs: { n: 120, noise: 0.05 } }).data
const blobData = generate(blobs, { key: key(3), knobs: { n: 90, centers: 3, clusterStd: 0.5 } }).data
const ari = (y: unknown, labels: readonly number[]): number => adjustedRandScore(y as number[], labels).estimate.value

describe('density clustering', () => {
  it('DBSCAN separates the moons, one cluster per step, and predicts like its labels', () => {
    const f = dbscan.with({ eps: 0.3, minSamples: 5 }).fit(moonData, { logs: 'all' })
    expect(ari(moonData.y, f.params.labels)).toBeGreaterThan(0.95)
    const clusters = metricSeries(f.run, 'solver/clusters').values
    expect(clusters.at(-1)).toBe(Math.max(...f.params.labels) + 1)
    expect(f.params.roles.length).toBe(120)
    expect(toFlat(f.predict(moonData.x))).toEqual(
      f.params.labels.map((l, i) => (f.params.roles[i] === 'noise' ? -1 : l)),
    )
    expect(
      dbscan
        .with({ eps: 0.01, minSamples: 5 })
        .fit({ x: moonData.x })
        .params.labels.every((l) => l === -1),
    ).toBe(true)
  })

  it('DBSCAN session: an early probe shows the first cluster only; the state serialises', () => {
    const s = dbscan.with({ eps: 0.3 }).session(moonData, { key: key(0), logs: 'all' })
    s.advance({ steps: 1 })
    const early = s.probe(1, 'model/assignments') as number[]
    expect(Math.max(...early)).toBe(0)
    expect(early.includes(-1)).toBe(true)
    const alg = dbscanAlgorithm(moonData.x, { eps: 0.3 })
    const state = alg.init(key(0))
    expect(stateFromJson(stateToJson(state))).toEqual(state)
  })

  it('DBSCAN on a precomputed distance matrix', () => {
    const d = [
      [0, 1, 9],
      [1, 0, 9],
      [9, 9, 0],
    ]
    const out = runAlgorithm(dbscanAlgorithm(d, { eps: 2, minSamples: 2, metric: 'precomputed' }), { maxSteps: 4 })
    expect(out.state.labels).toEqual([0, 0, -1])
  })

  it('OPTICS emits one point per step and extracts DBSCAN-like clusters', () => {
    const f = optics.with({ minSamples: 5, eps: 0.3 }).fit(moonData)
    expect(f.params.ordering.length).toBe(120)
    expect(ari(moonData.y, f.params.labels)).toBeGreaterThan(0.9)
    const o = opticsOrdering([[0], [0.1], [5], [5.1]], { minSamples: 2 })
    expect(opticsClustersAt(o, 0.5)).toEqual([0, 0, 1, 1])
    const alg = opticsAlgorithm([[0], [1], [5]], { minSamples: 2 })
    const first = alg.step(alg.init(key(0)), undefined as never)
    expect(first.ordering).toEqual([0])
    expect(() => opticsAlgorithm([[0], [1]], { minSamples: 5 })).toThrow()
  })

  it('HDBSCAN finds the blobs and returns the condensed tree', () => {
    const r = hdbscan(blobData.x, { minClusterSize: 10 })
    expect(ari(blobData.y, r.labels)).toBeGreaterThan(0.9)
    expect(r.linkage.length).toBe(89)
    expect(r.condensedTree.length).toBeGreaterThan(90)
    expect(r.selected.length).toBe(3)
    for (const p of r.probabilities) expect(p >= 0 && p <= 1).toBe(true)
    const f = hdbscanClustering({ minClusterSize: 10 }).fit(blobData)
    expect(f.params.labels).toEqual(r.labels)
    expect(toFlat(f.predict(blobData.x)).length).toBe(90)
  })

  it('mean shift finds the blob modes and runs one move per step', () => {
    const f = meanShift.with({ bandwidth: 2 }).fit(blobData)
    expect(f.params.centres.length).toBe(3)
    expect(ari(blobData.y, f.params.labels)).toBeGreaterThan(0.99)
    expect(toFlat(f.predict(blobData.x))).toEqual(f.params.labels)
    expect(estimateBandwidth(blobData.x)).toBeGreaterThan(0)
    const s = meanShift.with({ bandwidth: 2 }).session(blobData, { key: key(0), logs: 'all' })
    s.advance({ steps: 1 })
    expect((s.probe(1, 'model/assignments') as number[]).length).toBe(90)
  })
})
