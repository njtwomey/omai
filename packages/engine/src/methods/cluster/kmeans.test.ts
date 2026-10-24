import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { blobs } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { adjustedRandScore } from '../../lib/metrics/index.ts'
import { key } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import { BisectingKMeans, KMeans, kmeans, kmeansElbow, miniBatchKMeans } from './kmeans.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 90, centers: 3, clusterStd: 0.5 } })
const ari = (labels: readonly number[]): number => adjustedRandScore(data.y, labels).estimate.value

describe('k-means family', () => {
  it('k-means recovers well-separated blobs and records its Lloyd steps', () => {
    const f = kmeans({ nClusters: 3, nInit: 3 }).fit(data, { key: key(0) })
    expect(ari(f.params.labels)).toBeGreaterThan(0.99)
    const [steps] = recordsOf(f.exposition, 'clusteringSteps')
    expect(steps?.assignments[0]?.length).toBe(90)
    expect(steps?.centres?.length).toBe(steps?.assignments.length)
    const obj = steps?.objective ?? []
    for (let t = 1; t < obj.length; t++) expect(obj[t] ?? 0).toBeLessThanOrEqual((obj[t - 1] ?? 0) + 1e-9)
    expect(toFlat(f.predict(data.x))).toEqual(f.params.labels)
    expect(f.transform(data.x).shape).toEqual([90, 3])
    expect(kmeans().revive(f.params).exposition).toEqual([])
  })

  it('the class form, mini-batch and bisecting variants cluster the blobs', () => {
    expect(
      ari(toFlat(new KMeans({ nClusters: 3 }).fit(data.x, undefined, { key: key(1) }).predict(data.x))),
    ).toBeGreaterThan(0.9)
    const mb = miniBatchKMeans({ nClusters: 3, batchSize: 30, maxIter: 30 }).fit(data, { key: key(2) })
    expect(ari(mb.params.labels)).toBeGreaterThan(0.9)
    expect(recordsOf(mb.exposition, 'clusteringSteps')[0]?.assignments.length).toBe(31)
    const bi = new BisectingKMeans({ nClusters: 3 }).fit(data.x, undefined, { key: key(4) }).model
    expect(ari(bi.params.labels)).toBeGreaterThan(0.9)
    expect(recordsOf(bi.exposition, 'clusteringSteps')[0]?.assignments.length).toBe(3)
  })

  it('the elbow curve falls with k', () => {
    const { inertia } = kmeansElbow(data.x, [1, 2, 3, 4], { key: key(5) })
    for (let i = 1; i < inertia.length; i++) expect(inertia[i] ?? 0).toBeLessThan(inertia[i - 1] ?? 0)
  })
})
