import { describe, expect, it } from 'vitest'
import { key } from '../random/index.ts'
import { toFlat } from '../tensor/index.ts'
import { revive, serialiseFitted } from './estimator.ts'
import {
  kNeighboursClassifier,
  kNeighboursRegressor,
  nearestCentroid,
  neighbourIndex,
  neighbourWeights,
  radiusNeighboursClassifier,
  radiusNeighboursRegressor,
  shrunkenCentroidPath,
} from './neighbours.ts'

const data = {
  x: [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [8, 8],
    [8, 9],
    [9, 8],
    [9, 9],
  ],
  y: [0, 0, 0, 0, 1, 1, 1, 1],
}

describe('neighbour search', () => {
  it('agrees across the brute force and both trees', () => {
    const pts = Array.from({ length: 60 }, (_, i) => [Math.sin(i * 1.7) * 5, Math.cos(i * 0.9) * 5, (i % 7) - 3])
    const q = [
      [0.3, 0.2, 0.1],
      [4, -4, 2],
    ]
    const brute = neighbourIndex(pts, { algorithm: 'brute' }).nearest(q, 5)
    for (const algorithm of ['kd-tree', 'ball-tree'] as const) {
      const tree = neighbourIndex(pts, { algorithm, leafSize: 4 }).nearest(q, 5)
      expect(tree.map((h) => h.indices)).toEqual(brute.map((h) => h.indices))
    }
    const radius = neighbourIndex(pts, { algorithm: 'kd-tree', leafSize: 4 }).within(q, 3)
    expect(radius.map((h) => h.indices)).toEqual(
      neighbourIndex(pts)
        .within(q, 3)
        .map((h) => h.indices),
    )
  })

  it('refuses a tree for a metric it cannot serve, and defaults seuclidean to the data variances', () => {
    expect(() => neighbourIndex([[0], [1]], { algorithm: 'kd-tree', metric: 'cosine' })).toThrow(/supports/)
    const hits = neighbourIndex(
      [
        [0, 0],
        [1, 10],
        [2, 0],
      ],
      { metric: 'seuclidean' },
    ).nearest([[0, 0]], 3)[0]
    expect(hits?.indices).toEqual([0, 1, 2])
    expect(
      neighbourIndex([
        [0, 0],
        [1, 10],
        [3, 0],
      ]).nearest([[0, 0]], 3)[0]?.indices,
    ).toEqual([0, 2, 1])
  })

  it('weights by inverse distance and gives a zero distance all the weight', () => {
    expect(neighbourWeights([1, 2, 4], 'distance')).toEqual([1, 0.5, 0.25])
    expect(neighbourWeights([0, 0, 3], 'distance')).toEqual([1, 1, 0])
    expect(neighbourWeights([5, 7], 'uniform')).toEqual([1, 1])
  })
})

describe('neighbour learners', () => {
  it('classifies, gives probabilities and neighbours, and survives serialisation', () => {
    const fitted = kNeighboursClassifier({ nNeighbors: 3 }).fit(data)
    expect(
      toFlat(
        fitted.predict([
          [0.4, 0.4],
          [8.5, 8.5],
        ]),
      ),
    ).toEqual([0, 1])
    expect(toFlat(fitted.predictProba([[1, 1]]))).toEqual([1, 0])
    expect(fitted.neighbours([[0, 0.1]])[0]?.indices[0]).toBe(0)
    expect(fitted.score(data)).toBe(1)
    const est = kNeighboursClassifier({ nNeighbors: 3 })
    const back = revive(est, JSON.parse(JSON.stringify(serialiseFitted(est.fit(data)))))
    expect(toFlat(back.predict([[8, 8]]))).toEqual([1])
  })

  it('shows the memory filling: a session probes the model on the first rows', () => {
    const est = kNeighboursClassifier({ nNeighbors: 1 })
    const s = est.session(data, { key: key(0), logs: 'all' })
    s.advance({ steps: 5 })
    expect(s.probe(2, 'model/params')).toMatchObject({
      x: [
        [0, 0],
        [0, 1],
        [1, 0],
      ],
    })
    // Early on only the first class has been seen, so every query is class 0.
    expect(toFlat(s.fittedAt(3).predict([[9, 9]]))).toEqual([0])
    expect(toFlat(s.fittedAt(5).predict([[9, 9]]))).toEqual([1])
    expect(Object.keys(s.contract)).toContain('solver/rows')
    expect(s.probe(3, 'model/predictions')).toEqual([0, 0, 0, 0, 0, 0, 0, 0])
  })

  it('regresses with the weighted mean of the neighbours', () => {
    const f = kNeighboursRegressor({ nNeighbors: 2, weights: 'distance' }).fit({ x: [[0], [1], [3]], y: [0, 10, 30] })
    expect(toFlat(f.predict([[0.25]]))[0]).toBeCloseTo((0 / 0.25 + 10 / 0.75) / (1 / 0.25 + 1 / 0.75), 12)
    expect(toFlat(f.predict([[1]]))).toEqual([10])
  })

  it('falls back on the outlier label in the radius learners', () => {
    const rc = radiusNeighboursClassifier({ radius: 1.5, outlierLabel: 7 }).fit(data)
    expect(
      toFlat(
        rc.predict([
          [0.5, 0.5],
          [50, 50],
        ]),
      ),
    ).toEqual([0, 7])
    expect(Array.from(toFlat(rc.predictProba([[50, 50]])))).toEqual([0, 0])
    expect(() =>
      radiusNeighboursClassifier({ radius: 1.5 })
        .fit(data)
        .predict([[50, 50]]),
    ).toThrow(/outlierLabel/)
    expect(
      toFlat(
        radiusNeighboursClassifier({ radius: 1.5, outlierLabel: 'most-frequent' })
          .fit(data)
          .predict([[50, 50]]),
      ),
    ).toEqual([0])
    expect(
      toFlat(
        radiusNeighboursRegressor({ radius: 1 })
          .fit(data)
          .predict([[50, 50]]),
      )[0],
    ).toBeNaN()
  })
})

describe('nearest centroid', () => {
  it('predicts the nearest centroid and shrinks features away along the path', () => {
    const x = [
      [0, 5],
      [0.2, 4],
      [4, 5.1],
      [4.2, 4.9],
      [0.1, 5.2],
      [4.1, 5],
    ]
    const y = [0, 0, 1, 1, 0, 1]
    const f = nearestCentroid().fit({ x, y })
    expect(
      toFlat(
        f.predict([
          [0, 5],
          [4, 5],
        ]),
      ),
    ).toEqual([0, 1])
    const path = shrunkenCentroidPath({ x, y }, [0, 0.5, 1000])
    expect(path.map((p) => p.nActive)).toEqual([2, path[1]?.nActive ?? 0, 0])
    expect(path[0]?.active).toEqual([true, true])
    expect(path[2]?.active).toEqual([false, false])
    const shrunk = nearestCentroid({ shrinkThreshold: 1000 }).fit({ x, y })
    expect(shrunk.params.centroids[0]).toEqual(shrunk.params.datasetCentroid)
  })

  it('runs as a session over the rows', () => {
    const s = nearestCentroid().session(data, { key: key(0), logs: 'all' })
    s.advance({ steps: 8 })
    expect(s.fittedAt(7).params.classes).toEqual([0, 1])
    expect(s.fittedAt(3).params.classes).toEqual([0])
  })
})
