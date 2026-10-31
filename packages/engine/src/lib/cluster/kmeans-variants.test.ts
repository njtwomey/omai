import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs, circles, moons } from '../datasets/index.ts'
import { adjustedRandScore } from '../metrics/index.ts'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { initialCentres, lloydAlgorithm } from './kmeans.ts'
import {
  bisectingKMeans,
  bisectingKMeansAlgorithm,
  elkanAlgorithm,
  elkanKMeans,
  featureSpaceDistances,
  kernelKMeans,
} from './kmeans-variants.ts'

const blob = generate(blobs, { key: key(3), knobs: { n: 90, centers: 3, clusterStd: 0.5 } }).data
const ring = generate(circles, { key: key(4), knobs: { n: 80, noise: 0.03, factor: 0.3 } }).data

describe('Elkan k-means', () => {
  it('reaches the same fixed point as Lloyd from the same centres, skipping distances', () => {
    const init = initialCentres(blob.x, 3, 'k-means++', key(0))
    const e = runAlgorithm(elkanAlgorithm(blob.x, init, { tol: 1e-12 }), { maxSteps: 100, key: key(0) })
    const l = runAlgorithm(lloydAlgorithm(blob.x, init, { tol: 1e-12 }), { maxSteps: 100, key: key(0) })
    expect(e.state.labels).toEqual(l.state.labels)
    for (const [c, row] of e.state.centres.entries())
      for (const [q, v] of row.entries()) expect(v).toBeCloseTo(l.state.centres[c]?.[q] ?? 0, 9)
    expect(e.state.skipped).toBeGreaterThan(e.state.computed)
    expect(stateFromJson(stateToJson(e.state))).toEqual(e.state)
  })

  it('the estimator fits and a session probes early', () => {
    const f = elkanKMeans.with({ nClusters: 3 }).fit(blob, { key: key(1), logs: 'all' })
    expect(adjustedRandScore(blob.y, f.params.labels).estimate.value).toBeGreaterThan(0.99)
    expect(metricSeries(f.run, 'solver/computed').values.length).toBeGreaterThan(1)
    const s = elkanKMeans.session(blob, { key: key(1), logs: 'all' })
    s.advance({ steps: 1 })
    expect((s.probe(1, 'model/assignments') as number[]).length).toBe(90)
  })
})

describe('bisecting k-means', () => {
  it('splits one cluster per step and recovers the blobs', () => {
    const f = bisectingKMeans.with({ nClusters: 3 }).fit(blob, { key: key(2), logs: 'all' })
    expect(adjustedRandScore(blob.y, f.params.labels).estimate.value).toBeGreaterThan(0.95)
    expect(f.params.centres.length).toBe(3)
    const inertia = metricSeries(f.run, 'solver/inertia').values
    for (let t = 1; t < inertia.length; t++) expect(inertia[t] ?? 0).toBeLessThan(inertia[t - 1] ?? 0)
    const alg = bisectingKMeansAlgorithm(blob.x, 3, { strategy: 'largest_cluster' })
    const state = runAlgorithm(alg, { maxSteps: 2, key: key(0) }).state
    expect(stateFromJson(stateToJson(state))).toEqual(state)
  })
})

describe('kernel k-means', () => {
  it('separates concentric circles that k-means cannot', () => {
    const f = kernelKMeans.with({ nClusters: 2, gamma: 5 }).fit(ring, { key: key(6) })
    expect(adjustedRandScore(ring.y, f.params.labels).estimate.value).toBeGreaterThan(0.5)
    expect(toFlat(f.predict(ring.x))).toEqual(f.params.labels)
  })

  it('separates two moons at gamma 10 and more restarts never raise the inertia of the first start', () => {
    const moon = generate(moons, { key: key(8), knobs: { n: 200, noise: 0.06 } }).data
    const best = kernelKMeans.with({ nClusters: 2, gamma: 10, nInit: 5, init: 'spectral' }).fit(moon, { key: key(9) })
    expect(adjustedRandScore(moon.y, best.params.labels).estimate.value).toBeGreaterThan(0.9)
    const one = kernelKMeans.with({ nClusters: 2, gamma: 10, nInit: 1, init: 'spectral' }).fit(moon, { key: key(9) })
    expect(best.params.objective).toBeLessThanOrEqual(one.params.objective + 1e-9)
  })

  it('feature-space distances are zero at the cluster mean of a point alone', () => {
    expect(featureSpaceDistances([1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1], 2)).toEqual([0, 2, 2, 0])
  })
})
