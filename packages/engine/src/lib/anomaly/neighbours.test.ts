import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { recordsOf } from '../exposition/index.ts'
import { key, normal } from '../random/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import {
  averageChainingDistance,
  ConnectivityOutlierFactor,
  cofAlgorithm,
  connectivityOutlierFactors,
  KnnDetector,
  knnAlgorithm,
  knnDetector,
  LocalOutlierFactor,
  localOutlierFactor,
  localOutlierFactors,
  lofAlgorithm,
} from './neighbours.ts'

const rank = (scores: readonly number[], i: number): number => scores.filter((s) => s > (scores[i] ?? 0)).length

/** A dense cluster, a sparse cluster, and one point just outside the dense one (a local outlier). */
function twoDensities(): { x: number[][]; planted: number } {
  const dense = (toArray(normal(key(1), [40, 2])) as number[][]).map(([a, b]) => [0.1 * (a ?? 0), 0.1 * (b ?? 0)])
  const sparse = (toArray(normal(key(2), [40, 2])) as number[][]).map(([a, b]) => [20 + 3 * (a ?? 0), 3 * (b ?? 0)])
  return { x: [...dense, ...sparse, [1.2, 0]], planted: 80 }
}

describe('the local outlier factor', () => {
  it('matches a hand computation on three points on a line', () => {
    // k = 1: neighbours 0 -> 1, 1 -> 0, 3 -> 1; k-distances 1, 1, 2; lrd 1, 1, 1/2; factors 1, 1, 2.
    const r = localOutlierFactors([[0], [1], [3]], { nNeighbors: 1 })
    expect(r.neighbours).toEqual([[1], [0], [1]])
    expect(r.kDistance).toEqual([1, 1, 2])
    r.lrd.forEach((v, i) => {
      expect(v).toBeCloseTo([1, 1, 0.5][i] ?? 0, 8)
    })
    r.lof.forEach((v, i) => {
      expect(v).toBeCloseTo([1, 1, 2][i] ?? 0, 8)
    })
  })

  it('goes through its three stages, serialises its state and gives equal factors on every index', () => {
    const { x } = twoDensities()
    const alg = lofAlgorithm(x, { nNeighbors: 8, algorithm: 'brute' })
    const stages = [1, 2, 3].map((n) => runAlgorithm(alg, { maxSteps: n }).state)
    expect(stages.map((s) => s.stage)).toEqual([1, 2, 3])
    expect(stages[0]?.lrd).toEqual([])
    expect(alg.deserialise?.(alg.serialise?.(stages[2] as never) ?? null)).toEqual(stages[2])
    for (const algorithm of ['kd-tree', 'ball-tree'] as const) {
      const other = localOutlierFactors(x, { nNeighbors: 8, algorithm })
      other.lof.forEach((v, i) => {
        expect(v).toBeCloseTo(stages[2]?.lof[i] ?? 0, 9)
      })
    }
  })

  it('catches the local outlier next to a dense cluster that the k-nearest-neighbour distance misses', () => {
    const { x, planted } = twoDensities()
    const lof = localOutlierFactor({ nNeighbors: 10 }).fit({ x }).params.scores
    const knn = knnDetector({ nNeighbors: 10 }).fit({ x }).params.scores
    expect(rank(lof, planted)).toBe(0)
    expect(rank(knn, planted)).toBeGreaterThan(10)
  })

  it('is near 1 inside a uniform cluster and does not divide by zero on duplicates', () => {
    const same = localOutlierFactors(new Array(6).fill([2, 2]), { nNeighbors: 3 })
    expect(same.lof.every((v) => v === 1)).toBe(true)
    const { x } = twoDensities()
    const inliers = localOutlierFactors(x, { nNeighbors: 10 }).lof.slice(0, 40)
    expect(Math.max(...inliers)).toBeLessThan(2)
  })

  it('scores a new point against the training set, and the fit leaves each row out of its own neighbourhood', () => {
    const { x } = twoDensities()
    const f = localOutlierFactor({ nNeighbors: 10, contamination: 0.05 }).fit({ x })
    // A training row scored as a new row has itself as nearest neighbour, so it differs from its training factor.
    expect(toFlat(f.scoreSamples([x[3] ?? []]))[0]).not.toBeCloseTo(f.params.scores[3] ?? 0, 3)
    expect(
      toFlat(
        f.predict([
          [0, 0],
          [1.2, 0],
        ]),
      ),
    ).toEqual([1, -1])
    expect(f.params.threshold).toBe(
      [...f.params.scores].sort((a, b) => a - b)[Math.floor(0.95 * (x.length - 1))] ?? Number.NaN,
    )
    const [record] = recordsOf(f.exposition, 'anomalyScores')
    expect(record?.scores).toEqual(f.params.scores)
    expect(f.kneighbors([[0, 0]])[0]?.indices.length).toBe(10)
  })

  it('uses the threshold 1.5 by default and a session shows the fit stage by stage', () => {
    const { x } = twoDensities()
    const est = localOutlierFactor({ nNeighbors: 10 })
    expect(est.fit({ x }).params.threshold).toBe(1.5)
    const sess = est.session({ x }, { mode: 'eager' })
    expect(sess.t).toBe(3)
    expect(sess.fittedAt(3).params.scores).toEqual(est.fit({ x }).params.scores)
    expect(new LocalOutlierFactor({ nNeighbors: 5 }).fit(x).model.params.k).toBe(5)
  })

  it('refuses a single row and a bad neighbour count', () => {
    expect(() => localOutlierFactors([[1, 2]])).toThrow()
    expect(() => localOutlierFactors([[1], [2]], { nNeighbors: 0 })).toThrow()
  })

  it('property: factors are positive, and scale and translation do not change them', () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.integer({ min: -20, max: 20 }), fc.integer({ min: -20, max: 20 })), {
          minLength: 6,
          maxLength: 20,
        }),
        fc.integer({ min: 1, max: 5 }),
        (pts, k) => {
          const x = pts.map(([a, b]) => [a, b])
          const base = localOutlierFactors(x, { nNeighbors: k, algorithm: 'brute' }).lof
          expect(base.every((v) => v > 0 && Number.isFinite(v))).toBe(true)
          const moved = x.map(([a, b]) => [4 * (a ?? 0) + 3, 4 * (b ?? 0) - 7])
          const other = localOutlierFactors(moved, { nNeighbors: k, algorithm: 'brute' }).lof
          base.forEach((v, i) => {
            expect(other[i]).toBeCloseTo(v, 6)
          })
        },
      ),
      { numRuns: 40 },
    )
  })
})

describe('the connectivity-based outlier factor', () => {
  it('chains the nearest links from the point with weights that fall along the chain', () => {
    expect(
      averageChainingDistance([
        [0, 1, 3],
        [1, 0, 2],
        [3, 2, 0],
      ]),
    ).toBeCloseTo(4 / 3, 12)
    expect(averageChainingDistance([[0]])).toBe(0)
    // The chain 0 -> 1 -> 2 -> 3 on a line costs 1 each: the weights sum to 1.
    expect(
      averageChainingDistance([
        [0, 1, 2, 3],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [3, 2, 1, 0],
      ]),
    ).toBeCloseTo(1, 12)
  })

  it('scores a point off the end of a line above the points on it', () => {
    const x = [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
      [4, 0],
      [5, 0],
      [2.5, 3],
    ]
    const r = connectivityOutlierFactors(x, { nNeighbors: 3 })
    expect(rank(r.cof, 6)).toBe(0)
    expect(r.cof[6]).toBeGreaterThan(2)
    expect(Math.max(...r.cof.slice(0, 6))).toBeLessThan(1.6)
    const alg = cofAlgorithm(x, { nNeighbors: 3 })
    const s = runAlgorithm(alg, { maxSteps: 3 }).state
    expect(alg.deserialise?.(alg.serialise?.(s) ?? null)).toEqual(s)
  })

  it('is a detector with the outlier verbs and an exposition', () => {
    const x = [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
      [4, 0],
      [5, 0],
      [2.5, 3],
    ]
    const f = new ConnectivityOutlierFactor({ nNeighbors: 3, contamination: 0.15 }).fit(x).model
    expect(
      toFlat(
        f.predict([
          [2.5, 0],
          [2.5, 6],
        ]),
      ),
    ).toEqual([1, -1])
    expect(recordsOf(f.exposition, 'anomalyScores')[0]?.scores.length).toBe(7)
    expect(toFlat(f.decisionFunction([[2.5, 0]]))[0]).toBeGreaterThan(0)
  })
})

describe('the k-nearest-neighbour distance', () => {
  it('is the distance to the k-th neighbour, or the mean or median of the k distances', () => {
    const x = [[0], [1], [3], [10]]
    expect(runAlgorithm(knnAlgorithm(x, { nNeighbors: 1 }), { maxSteps: 2 }).state.scores).toEqual([1, 1, 2, 7])
    const two = (aggregation: 'largest' | 'mean' | 'median'): number[] =>
      runAlgorithm(knnAlgorithm(x, { nNeighbors: 2, aggregation }), { maxSteps: 2 }).state.scores
    expect(two('largest')).toEqual([3, 2, 3, 9])
    expect(two('mean')).toEqual([2, 1.5, 2.5, 8])
    expect(two('median')).toEqual([2, 1.5, 2.5, 8])
  })

  it('thresholds at the contamination quantile and scores new points', () => {
    const x = (toArray(normal(key(5), [60, 2])) as number[][]).concat([[9, 9]])
    const f = new KnnDetector({ nNeighbors: 5, contamination: 0.05 }).fit(x).model
    expect(
      toFlat(
        f.predict([
          [0, 0],
          [9.5, 9],
        ]),
      ),
    ).toEqual([1, -1])
    const flagged = f.params.scores.filter((s) => s > f.params.threshold).length
    expect(flagged).toBeLessThanOrEqual(Math.ceil(0.05 * 61))
    expect(flagged).toBeGreaterThanOrEqual(1)
    expect(knnDetector({ nNeighbors: 5 }).with({ aggregation: 'mean' }).hyper.aggregation).toBe('mean')
  })
})
