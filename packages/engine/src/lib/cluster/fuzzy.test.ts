import { describe, expect, it } from 'vitest'
import { generate } from '../data/index.ts'
import { blobs } from '../datasets/index.ts'
import { adjustedRandScore } from '../metrics/index.ts'
import { key } from '../random/index.ts'
import { sum, toFlat } from '../tensor/index.ts'
import { metricSeries, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { fuzzyCMeans, fuzzyCMeansAlgorithm, fuzzyMemberships } from './fuzzy.ts'

const { data } = generate(blobs, { key: key(3), knobs: { n: 60, centers: 3, clusterStd: 0.5 } })

describe('fuzzy c-means', () => {
  it('recovers the blobs with memberships summing to one and a falling objective', () => {
    const f = fuzzyCMeans.with({ nClusters: 3 }).fit(data, { key: key(2), logs: 'all' })
    expect(adjustedRandScore(data.y, toFlat(f.predict(data.x))).estimate.value).toBeGreaterThan(0.99)
    for (const s of toFlat(sum(f.predictProba(data.x), 1))) expect(s).toBeCloseTo(1, 12)
    const objective = metricSeries(f.run, 'solver/objective').values
    for (let t = 1; t < objective.length; t++)
      expect(objective[t] ?? 0).toBeLessThanOrEqual((objective[t - 1] ?? 0) + 1e-9)
    expect(f.params.memberships.length).toBe(60)
  })

  it('a point on a centre is wholly in it; the fuzzifier must exceed one; the state round-trips', () => {
    expect(fuzzyMemberships([[0], [1]], [[0], [1]], 2).memberships).toEqual([1, 0, 0, 1])
    expect(() => fuzzyCMeansAlgorithm([[0], [1]], [[0], [1]], 1)).toThrow()
    const s = runAlgorithm(fuzzyCMeansAlgorithm([[0], [1], [9], [10]], [[0], [10]]), { maxSteps: 3 }).state
    expect(stateFromJson(stateToJson(s))).toEqual(s)
  })

  it('a session probe at an early step has the assignments', () => {
    const s = fuzzyCMeans.session(data, { key: key(0), logs: 'all' })
    s.advance({ steps: 2 })
    expect((s.probe(2, 'model/assignments') as number[]).length).toBe(60)
  })
})
