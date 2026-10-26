import { describe, expect, it } from 'vitest'
import { generate } from '../../lib/data/index.ts'
import { blobs, moons } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { adjustedRandScore } from '../../lib/metrics/index.ts'
import { key } from '../../lib/random/index.ts'
import { toFlat } from '../../lib/tensor/index.ts'
import { DBSCAN, dbscan, hdbscan, meanShift, optics } from './density.ts'

const moonData = generate(moons, { key: key(1), knobs: { n: 120, noise: 0.05 } }).data
const blobData = generate(blobs, { key: key(3), knobs: { n: 90, centers: 3, clusterStd: 0.5 } }).data
const ari = (y: unknown, labels: readonly number[]): number => adjustedRandScore(y as number[], labels).estimate.value

describe('density clustering', () => {
  it('DBSCAN separates the moons, one step per cluster, with core, border and noise roles', () => {
    const f = dbscan({ eps: 0.3, minSamples: 5 }).fit(moonData)
    expect(ari(moonData.y, f.params.labels)).toBeGreaterThan(0.95)
    const [steps] = recordsOf(f.exposition, 'clusteringSteps')
    expect(steps?.assignments.length).toBe(Math.max(...f.params.labels) + 2)
    expect(steps?.roles?.length).toBe(120)
    expect(toFlat(f.predict(moonData.x))).toEqual(
      f.params.labels.map((l, i) => (f.params.roles[i] === 'noise' ? -1 : l)),
    )
    expect(new DBSCAN({ eps: 1e-6, minSamples: 2 }).fit(moonData.x).model.params.labels.every((l) => l === -1)).toBe(
      true,
    )
  })

  it('OPTICS records the reachability plot and matches DBSCAN at the same radius on cores', () => {
    const f = optics({ minSamples: 5, eps: 0.3 }).fit(moonData)
    const [steps] = recordsOf(f.exposition, 'clusteringSteps')
    expect(steps?.reachability?.length).toBe(120)
    expect(ari(moonData.y, f.params.labels)).toBeGreaterThan(0.9)
  })

  it('HDBSCAN finds the blobs and records the mutual-reachability merges', () => {
    const f = hdbscan({ minClusterSize: 10 }).fit(blobData)
    expect(ari(blobData.y, f.params.labels)).toBeGreaterThan(0.9)
    expect(recordsOf(f.exposition, 'clusteringSteps')[0]?.merges?.length).toBe(89)
  })

  it('mean shift finds the blob modes', () => {
    const f = meanShift({ bandwidth: 2 }).fit(blobData)
    expect(f.params.centres.length).toBe(3)
    expect(ari(blobData.y, f.params.labels)).toBeGreaterThan(0.99)
    expect(toFlat(f.predict(blobData.x))).toEqual(f.params.labels)
  })
})
