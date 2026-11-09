import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs } from '../datasets/index.ts'
import { adjustedRandScore } from '../metrics/index.ts'
import { foldIn, key } from '../random/index.ts'
import { toArray } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  initialCentres,
  kMeans,
  kmeansPlusPlus,
  lloydAlgorithm,
  miniBatchKMeans,
  miniBatchKMeansAlgorithm,
  nearestCentres,
} from './kmeans.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 90, centers: 3, clusterStd: 0.5 } })
const labelsTrue = toArray(data.y) as number[]
const ari = (labels: readonly number[]): number => adjustedRandScore(data.y, labels).estimate.value

describe('k-means core', () => {
  it('Lloyd from k-means++ recovers well-separated blobs and never raises the inertia', () => {
    const alg = lloydAlgorithm(data.x, initialCentres(data.x, 3, 'k-means++', key(0)))
    const inertias: number[] = []
    const result = runAlgorithm(
      {
        ...alg,
        step: (s, c) => {
          inertias.push(s.inertia)
          return alg.step(s, c)
        },
      },
      { maxSteps: 100, key: key(0) },
    )
    expect(ari(result.state.labels)).toBeGreaterThan(0.99)
    expect(result.status).toBe('converged')
    for (let t = 1; t < inertias.length; t++)
      expect(inertias[t] ?? 0).toBeLessThanOrEqual((inertias[t - 1] ?? 0) + 1e-9)
    expect(Object.keys(result.state)).not.toContain('history')
  })

  it('observes inertia, moved, centres and labels as its schema declares', () => {
    const alg = lloydAlgorithm(data.x, initialCentres(data.x, 3, 'random', key(1)))
    const seen = alg.observe?.(alg.init(key(0)), 0)
    expect(Object.keys(seen?.metrics ?? {})).toEqual(['inertia', 'moved'])
    expect(seen?.tensors?.['centres']?.shape).toEqual([3, 2])
    expect(seen?.tensors?.['labels']?.shape).toEqual([90])
    expect(Object.keys(alg.schema?.tensors ?? {})).toEqual(['centres', 'labels'])
    expect(Object.keys(alg.schema?.metrics ?? {})).toEqual(['inertia', 'moved'])
  })

  it('mini-batch k-means clusters the blobs', () => {
    const init = initialCentres(data.x, 3, 'k-means++', key(2))
    const result = runAlgorithm(miniBatchKMeansAlgorithm(data.x, init, 30), { maxSteps: 30, key: key(2) })
    expect(ari(result.state.labels)).toBeGreaterThan(0.9)
  })

  it('initial centres have the right shape, given centres pass through, and bad k throws', () => {
    for (const rule of ['k-means++', 'random'] as const) expect(initialCentres(data.x, 4, rule, key(5)).length).toBe(4)
    const given = [
      [0, 0],
      [1, 1],
    ]
    expect(initialCentres(data.x, 2, given, key(0))).toEqual(given)
    expect(() => initialCentres(data.x, 0, 'random', key(0))).toThrow()
  })

  it('nearestCentres labels each row by its closest centre', () => {
    const rows = [
      [0, 0],
      [10, 0],
      [9, 1],
    ]
    const { labels, inertia } = nearestCentres(rows, [
      [0, 0],
      [10, 0],
    ])
    expect(labels).toEqual([0, 1, 1])
    expect(inertia).toBeCloseTo(2, 12)
  })
})

describe('kmeansPlusPlus', () => {
  it('returns k rows of the data with their indices, deterministically per key', () => {
    const a = kmeansPlusPlus(data.x, 3, { key: key(7) })
    const b = kmeansPlusPlus(data.x, 3, { key: key(7) })
    expect(a).toEqual(b)
    expect(a.centres.length).toBe(3)
    const rows = toArray(data.x) as number[][]
    for (const [j, i] of a.indices.entries()) expect(a.centres[j]).toEqual(rows[i])
  })

  it('spreads the seeds: one per well-separated blob far more often than not', () => {
    let hits = 0
    for (let s = 0; s < 20; s++) {
      const { indices } = kmeansPlusPlus(data.x, 3, { key: key(s) })
      if (new Set(indices.map((i) => labelsTrue[i])).size === 3) hits++
    }
    expect(hits).toBeGreaterThanOrEqual(18)
  })

  it('k = n picks every distinct row and a bad k throws', () => {
    const x = [[0], [5], [9]]
    expect(kmeansPlusPlus(x, 3, { key: key(1) }).indices.toSorted()).toEqual([0, 1, 2])
    expect(() => kmeansPlusPlus(x, 4, { key: key(1) })).toThrow()
    expect(() => kmeansPlusPlus(x, 0, { key: key(1) })).toThrow()
  })
})

describe('kMeans and miniBatchKMeans estimators', () => {
  it('fit recovers the blobs; predict, transform and score agree with the centres', () => {
    const f = kMeans.with({ nClusters: 3, nInit: 3 }).fit(data, { key: key(1) })
    expect(ari(f.params.labels)).toBeGreaterThan(0.99)
    expect(toArray(f.predict(data.x))).toEqual(f.params.labels)
    const dist = f.transform(data.x)
    expect(dist.shape).toEqual([90, 3])
    const rows = toArray(dist) as number[][]
    expect(rows.map((r) => r.indexOf(Math.min(...r)))).toEqual(f.params.labels)
    expect(f.score(data)).toBeCloseTo(-f.params.inertia, 9)
    expect(f.featureNamesOut()).toEqual(['kmeans0', 'kmeans1', 'kmeans2'])
  })

  it('keeps the restart with the lowest inertia', () => {
    const many = kMeans.with({ nClusters: 5, nInit: 8 }).fit(data, { key: key(4) })
    const singles = Array.from(
      { length: 8 },
      (_, r) => kMeans.with({ nClusters: 5 }).fit(data, { key: foldIn(key(4), r) }).params.inertia,
    )
    expect(many.params.inertia).toBeCloseTo(Math.min(...singles), 9)
  })

  it('given centres run once and revive rebuilds the same verbs', () => {
    const init = [
      [-5, -5],
      [0, 0],
      [5, 5],
    ]
    const f = kMeans.with({ nClusters: 3, init, nInit: 10 }).fit(data)
    const again = kMeans.revive(f.params)
    expect(toArray(again.predict(data.x))).toEqual(toArray(f.predict(data.x)))
  })

  it('a session steps the fit and fittedAt gives the model so far', () => {
    const s = kMeans.with({ nClusters: 3 }).session(data, { key: key(2), logs: 'all' })
    s.advance({ steps: 1 })
    expect((s.probe(1, 'model/assignments') as number[]).length).toBe(90)
    expect(s.fittedAt(1).params.centres.length).toBe(3)
  })

  it('mini-batch k-means clusters the blobs', () => {
    const f = miniBatchKMeans.with({ nClusters: 3, batchSize: 30, maxIter: 40 }).fit(data, { key: key(2) })
    expect(ari(f.params.labels)).toBeGreaterThan(0.9)
    expect((toArray(f.transform(data.x)) as number[][])[0]?.length).toBe(3)
  })
})
