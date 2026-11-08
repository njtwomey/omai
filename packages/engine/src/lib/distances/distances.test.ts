import { describe, expect, it } from 'vitest'
import { isOmaiError } from '../base/index.ts'
import { TRUE_METRICS } from '../numerics/geometry/index.ts'
import { contains, defaults } from '../space/index.ts'
import { tensor, toArray, toFlat } from '../tensor/index.ts'
import {
  type DistanceAlias,
  type DistanceName,
  type DistanceOptions,
  distance,
  distanceAliases,
  distanceEntry,
  distanceNames,
  distanceSpace,
  distances,
  euclideanDistance,
  haversineDistance,
  mahalanobisDistance,
  pairwiseDistances,
} from './index.ts'

const near = (got: number, want: number, digits = 12): void => expect(got).toBeCloseTo(want, digits)

describe('the distance registry', () => {
  it('lists every entry under its own name, in table order', () => {
    expect(distanceNames).toEqual(Object.keys(distances))
    for (const name of distanceNames) expect(distances[name].name).toBe(name)
    expect(distances.euclidean).toBe(euclideanDistance)
    expect(distances.haversine).toBe(haversineDistance)
    expect(distanceNames).toHaveLength(14)
  })

  it('resolves names and aliases, and rejects unknown names with a value error', () => {
    expect(distanceEntry('euclidean')).toBe(euclideanDistance)
    for (const [alias, name] of Object.entries(distanceAliases)) expect(distanceEntry(alias).name).toBe(name)
    expect(() => distanceEntry('euclid')).toThrow(/unknown distance/)
    try {
      distance([0], [1], 'euclid' as DistanceName)
    } catch (e) {
      expect(isOmaiError(e, 'value')).toBe(true)
    }
  })

  it('describes every entry and marks the true metrics as numerics does', () => {
    for (const name of distanceNames) {
      const e = distances[name]
      expect(e.description.length).toBeGreaterThan(20)
      expect(e.parameters.kind).toBe('dict')
    }
    expect(distances.euclidean.metric).toBe(true)
    expect(distances.squaredEuclidean.metric).toBe(false)
    expect(distances.cosine.metric).toBe(false)
    expect(distances.braycurtis.metric).toBe(false)
    expect(TRUE_METRICS.has('cityblock')).toBe(distances.manhattan.metric)
    expect(TRUE_METRICS.has('canberra')).toBe(distances.canberra.metric)
  })

  it('offers the knobs as spaces and the whole registry as one union a control can pick from', () => {
    expect(distances.minkowski.parameters.spaces['p']).toMatchObject({ kind: 'real', default: 2 })
    expect(distances.haversine.parameters.spaces['degrees']).toMatchObject({ kind: 'bool', default: true })
    expect(Object.keys(distances.euclidean.parameters.spaces)).toEqual([])
    expect(distanceSpace.kind).toBe('union')
    expect(distanceSpace.tag).toBe('name')
    expect(Object.keys(distanceSpace.options)).toEqual(distanceNames)
    expect(defaults(distanceSpace)).toEqual({ name: 'euclidean' })
    const chosen = { name: 'minkowski', p: 3 }
    expect(contains(distanceSpace, chosen)).toBe(true)
    // A value of the union is the name plus the options, ready for the entry points.
    near(toArray(distance([0, 0], [1, 1], chosen.name as DistanceName, chosen)) as number, 2 ** (1 / 3))
  })
})

describe('the named distances on hand-computed values', () => {
  const a = [0, 0]
  const b = [3, 4]
  const value = (name: DistanceName | DistanceAlias, options: DistanceOptions = {}, x = a, y = b): number =>
    toArray(distance(x, y, name, options)) as number

  it('euclidean family', () => {
    near(value('euclidean'), 5)
    near(value('squaredEuclidean'), 25)
    near(value('manhattan'), 7)
    near(value('chebyshev'), 4)
    near(value('minkowski', { p: 1 }), 7)
    near(value('minkowski', { p: 2 }), 5)
    near(value('minkowski', { p: 3 }), (27 + 64) ** (1 / 3))
    near(value('euclidean', { weights: [4, 1] }), Math.sqrt(36 + 16))
    near(value('l2'), 5)
    near(value('l1'), 7)
    near(value('linf'), 4)
  })

  it('standardised and whitened', () => {
    near(value('seuclidean', { variance: [9, 16] }), Math.SQRT2)
    near(
      value('mahalanobis', {
        covariance: [
          [9, 0],
          [0, 16],
        ],
      }),
      Math.SQRT2,
    )
    near(
      value('mahalanobis', {
        inverseCovariance: [
          [1 / 9, 0],
          [0, 1 / 16],
        ],
      }),
      Math.SQRT2,
    )
    // The inverse covariance wins when both are given.
    near(
      value('mahalanobis', {
        covariance: [
          [1, 0],
          [0, 1],
        ],
        inverseCovariance: [
          [1 / 9, 0],
          [0, 1 / 16],
        ],
      }),
      Math.SQRT2,
    )
    expect(() => mahalanobisDistance.compute(tensor(a), tensor(b), {})).toThrow(/needs a covariance/)
    expect(() => distances.seuclidean.scalar(2, {})).toThrow(/needs the variances/)
  })

  it('angles and correlation', () => {
    near(value('cosine', {}, [1, 0], [0, 1]), 1)
    near(value('cosine', {}, [1, 1], [2, 2]), 0)
    near(value('cosine', {}, [1, 0], [-1, 0]), 2)
    near(value('cosine', {}, [0, 0], [1, 0]), 1)
    near(value('correlation', {}, [1, 2, 3], [2, 4, 6]), 0)
    near(value('correlation', {}, [1, 2, 3], [3, 2, 1]), 2)
  })

  it('discrete and compositional', () => {
    near(value('hamming', {}, [1, 0, 1, 1], [1, 1, 1, 0]), 0.5)
    near(value('jaccard', {}, [1, 1, 0, 0], [1, 0, 1, 0]), 2 / 3)
    near(value('jaccard', {}, [0, 0], [0, 0]), 0)
    near(value('canberra', {}, [1, 2], [3, 2]), 0.5)
    near(value('canberra', {}, [0, 1], [0, 3]), 0.5)
    near(value('braycurtis', {}, [1, 3], [3, 1]), 0.5)
  })

  it('haversine on degrees, radians and a radius', () => {
    near(value('haversine', {}, [0, 0], [0, 180]), Math.PI)
    near(value('haversine', {}, [0, 0], [90, 0]), Math.PI / 2)
    near(value('haversine', { degrees: false }, [0, 0], [0, Math.PI]), Math.PI)
    near(value('haversine', { radius: 6371 }, [0, 0], [0, 90]), (6371 * Math.PI) / 2, 9)
    expect(() => distance([0, 0, 0], [1, 1, 1], 'haversine')).toThrow(/latitude, longitude/)
    expect(() => haversineDistance.scalar(3, {})).toThrow(/latitude, longitude/)
  })
})

describe('the plain-number form agrees with the tensor form', () => {
  const points = [
    [0.3, -1.2, 2.5],
    [1.1, 0.4, -0.7],
    [-2.0, 0.9, 0.1],
  ]
  const binary = [
    [1, 0, 1, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 1, 1],
  ]
  const geo = [
    [51.5, -0.1],
    [48.9, 2.4],
    [-33.9, 151.2],
  ]
  const cases: readonly (readonly [DistanceName, DistanceOptions, number[][]])[] = [
    ['euclidean', { weights: [1, 2, 0.5] }, points],
    ['squaredEuclidean', {}, points],
    ['manhattan', {}, points],
    ['chebyshev', {}, points],
    ['minkowski', { p: 3 }, points],
    ['seuclidean', { variance: [0.5, 1, 2] }, points],
    ['cosine', {}, points],
    ['correlation', {}, points],
    [
      'mahalanobis',
      {
        covariance: [
          [2, 0.3, 0],
          [0.3, 1, 0.2],
          [0, 0.2, 1.5],
        ],
      },
      points,
    ],
    ['hamming', {}, binary],
    ['jaccard', {}, binary],
    ['canberra', {}, points],
    ['braycurtis', {}, points],
    ['haversine', { radius: 6371 }, geo],
    ['haversine', { degrees: false }, geo.map((p) => p.map((v) => (v * Math.PI) / 180))],
  ]
  for (const [name, options, data] of cases) {
    it(name, () => {
      const dim = data[0]?.length ?? 0
      const flat = Float64Array.from(data.flat())
      const scalar = distances[name].scalar(dim, options)
      const matrix = toFlat(pairwiseDistances(data, undefined, name, options))
      const n = data.length
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) near(scalar(flat, i * dim, flat, j * dim), matrix[i * n + j] ?? Number.NaN, 11)
    })
  }
})
