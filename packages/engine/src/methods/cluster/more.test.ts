import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { blobs, circles } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { adjustedRandScore } from '../../lib/metrics/index.ts'
import { key } from '../../lib/random/index.ts'
import { sum, toFlat } from '../../lib/tensor/index.ts'
import { COPKMeans } from './constrained.ts'
import { AffinityPropagation, birch } from './exemplar.ts'
import { FuzzyCMeans } from './fuzzy.ts'
import { kernelKMeans } from './kmeans.ts'
import { KMedoids, kMedoids } from './medoids.ts'
import { SpectralClustering } from './spectral.ts'

const blob = generate(blobs, { key: key(3), knobs: { n: 60, centers: 3, clusterStd: 0.5 } }).data
const ring = generate(circles, { key: key(4), knobs: { n: 80, noise: 0.03, factor: 0.3 } }).data
const ari = (y: unknown, labels: readonly number[]): number => adjustedRandScore(y as number[], labels).estimate.value

describe('more clustering', () => {
  it('k-medoids (PAM and CLARA), fuzzy c-means, affinity propagation and BIRCH recover the blobs', () => {
    const pam = new KMedoids({ nClusters: 3 }).fit(blob.x).model
    expect(ari(blob.y, pam.params.labels)).toBeGreaterThan(0.99)
    const obj = recordsOf(pam.exposition, 'clusteringSteps')[0]?.objective ?? []
    for (let t = 1; t < obj.length; t++) expect(obj[t] ?? 0).toBeLessThan(obj[t - 1] ?? 0)
    expect(
      ari(blob.y, kMedoids({ nClusters: 3, method: 'clara' }).fit(blob, { key: key(1) }).params.labels),
    ).toBeGreaterThan(0.95)
    const fcm = new FuzzyCMeans({ nClusters: 3 }).fit(blob.x, undefined, { key: key(2) })
    expect(ari(blob.y, toFlat(fcm.predict(blob.x)))).toBeGreaterThan(0.99)
    for (const s of toFlat(sum(fcm.predictProba(blob.x), 1))) expect(s).toBeCloseTo(1, 12)
    const ap = new AffinityPropagation({ preference: -50 }).fit(blob.x).model
    expect(ap.params.exemplarIndices.length).toBe(3)
    expect(ari(blob.y, ap.params.labels)).toBeGreaterThan(0.99)
    expect(recordsOf(ap.exposition, 'clusteringSteps')[0]?.assignments.length).toBe(ap.params.nIter)
    const b = birch({ nClusters: 3, threshold: 0.8 }).fit(blob)
    expect(ari(blob.y, b.params.labels)).toBeGreaterThan(0.95)
  })

  it('spectral clustering and kernel k-means separate concentric circles; k-means cannot', () => {
    const sc = new SpectralClustering({ nClusters: 2, affinity: 'nearest_neighbors', nNeighbors: 8 }).fit(
      ring.x,
      undefined,
      {
        key: key(5),
      },
    ).model
    expect(ari(ring.y, sc.params.labels)).toBeGreaterThan(0.95)
    const kk = kernelKMeans({ nClusters: 2, gamma: 5, maxIter: 50 }).fit(ring, { key: key(6) })
    expect(ari(ring.y, kk.params.labels)).toBeGreaterThan(0.5)
    expect(toFlat(kk.predict(ring.x))).toEqual(kk.params.labels)
  })

  it('COP-k-means honours its constraints', () => {
    const m = new COPKMeans({ nClusters: 3, mustLink: [[0, 1]], cannotLink: [[2, 3]] }).fit(blob.x, undefined, {
      key: key(7),
    })
    const labels = m.model.params.labels
    expect(labels[0]).toBe(labels[1])
    expect(labels[2]).not.toBe(labels[3])
  })
})
