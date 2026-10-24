import { describe, expect, it } from 'vitest'
import { iris } from '../../lib/datasets/index.ts'
import { recordsOf } from '../../lib/exposition/index.ts'
import { classificationSet, regressionSet } from '../../testing/supervised.ts'
import { NearestCentroid, nearestCentroid } from './centroid.ts'
import { KNeighborsClassifier, kNeighborsRegressor, radiusNeighborsClassifier } from './knn.ts'

const moons = classificationSet('moons', { n: 100, noise: 0.2 }, 1)
const sine = regressionSet('sineCurve', { n: 60 }, 2)

describe('neighbours', () => {
  it('k-NN records neighbours and surfaces', () => {
    const f = new KNeighborsClassifier({ nNeighbors: 7 }).fit(moons.x, moons.y)
    expect(f.score(moons.x, moons.y)).toBeGreaterThan(0.9)
    const record = recordsOf(f.exposition, 'neighbours')[0]
    expect(record?.indices).toHaveLength(50)
    expect(record?.indices[0]).toHaveLength(7)
    expect(record?.indices[0]).not.toContain(0)
    expect(recordsOf(f.exposition, 'decisionSurface')).toHaveLength(1)
    expect(f.model.kneighbors([[0, 0]]).indices[0]).toHaveLength(7)
    const r = kNeighborsRegressor({ weights: 'distance' }).fit(sine)
    expect(r.score?.(sine)).toBeGreaterThan(0.9)
    expect(r.predictDistribution(sine.x).batchShape).toEqual([60])
    expect(radiusNeighborsClassifier({ radius: 0.5 }).fit(moons).score?.(moons)).toBeGreaterThan(0.8)
  })

  it('nearest centroids', () => {
    const flowers = iris()
    const f = nearestCentroid().fit(flowers)
    expect(f.score?.(flowers)).toBeGreaterThan(0.85)
    expect(recordsOf(f.exposition, 'coefficientTable')[0]?.names[0]).toBe('setosa:sepal length (cm)')
    expect(new NearestCentroid({ shrinkThreshold: 0.5 }).fit(moons.x, moons.y).score(moons.x, moons.y)).toBeGreaterThan(
      0.7,
    )
  })
})
