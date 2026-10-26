import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs } from '../datasets/index.ts'
import { adjustedRandScore } from '../metrics/index.ts'
import { key } from '../random/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { initialCentres, lloydAlgorithm, miniBatchKMeansAlgorithm, nearestCentres } from './kmeans.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 90, centers: 3, clusterStd: 0.5 } })
const ari = (labels: readonly number[]): number => adjustedRandScore(data.y, labels).estimate.value

describe('k-means core', () => {
  it('Lloyd from k-means++ recovers well-separated blobs and never raises the inertia', () => {
    const alg = lloydAlgorithm(data.x, initialCentres(data.x, 3, 'k-means++', key(0)))
    const inertias: number[] = []
    const result = runAlgorithm(
      { ...alg, step: (s, c) => (inertias.push(s.inertia), alg.step(s, c)) },
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
