import { describe, expect, it } from 'vitest'
import { isOmaiError } from '../base/index.ts'
import { activeDims, contains, defaults } from '../space/index.ts'
import { tensor, toArray, toFlat } from '../tensor/index.ts'
import {
  cosineSimilarity,
  distance,
  distanceNames,
  distanceToSimilarity,
  gaussianSimilarity,
  inverseDistanceSimilarity,
  pairwiseDistances,
  pairwiseSimilarities,
  type SimilarityAlias,
  type SimilarityName,
  type SimilarityOptions,
  similarities,
  similarity,
  similarityAliases,
  similarityEntry,
  similarityNames,
  similaritySpace,
  similarityToDistance,
} from './index.ts'

const near = (got: number, want: number, digits = 12): void => expect(got).toBeCloseTo(want, digits)
const value = (
  x: number[],
  y: number[],
  name: SimilarityName | SimilarityAlias,
  options: SimilarityOptions = {},
): number => toArray(similarity(x, y, name, options)) as number

describe('the similarity registry', () => {
  it('lists every entry under its own name with a range and a description', () => {
    expect(similarityNames).toEqual(Object.keys(similarities))
    for (const name of similarityNames) {
      const e = similarities[name]
      expect(e.name).toBe(name)
      expect(e.description.length).toBeGreaterThan(20)
      expect(e.range[0]).toBeLessThan(e.range[1])
      expect(e.parameters.kind).toBe('dict')
    }
    expect(similarities.cosine).toBe(cosineSimilarity)
    expect(similarities.gaussian).toBe(gaussianSimilarity)
  })

  it('resolves names and aliases, and rejects unknown names with a value error', () => {
    for (const [alias, name] of Object.entries(similarityAliases)) expect(similarityEntry(alias).name).toBe(name)
    expect(() => similarityEntry('cos')).toThrow(/unknown similarity/)
    try {
      similarity([0], [1], 'cos' as SimilarityName)
    } catch (e) {
      expect(isOmaiError(e, 'value')).toBe(true)
    }
  })

  it('brings every distance knob into the inverse distance, each active only for its distance', () => {
    const spaces = inverseDistanceSimilarity.parameters.spaces
    expect(spaces['distance']).toMatchObject({ kind: 'choice', options: distanceNames, default: 'euclidean' })
    expect(spaces['scale']).toMatchObject({ kind: 'real', default: 1 })
    expect(spaces['p']).toMatchObject({ kind: 'real', default: 2, when: { key: 'distance', in: ['minkowski'] } })
    expect(spaces['degrees']).toMatchObject({ kind: 'bool', when: { key: 'distance', in: ['haversine'] } })
    expect(defaults(inverseDistanceSimilarity.parameters)).toEqual({ distance: 'euclidean', scale: 1 })
    expect(contains(inverseDistanceSimilarity.parameters, { distance: 'minkowski', scale: 1, p: 3 })).toBe(true)
    expect(activeDims(inverseDistanceSimilarity.parameters, { distance: 'minkowski', scale: 1, p: 3 })).toEqual([
      'distance',
      'scale',
      'p',
    ])
    expect(activeDims(inverseDistanceSimilarity.parameters, { distance: 'euclidean', scale: 1 })).toEqual([
      'distance',
      'scale',
    ])
  })

  it('is one union a control can pick from', () => {
    expect(similaritySpace.tag).toBe('name')
    expect(Object.keys(similaritySpace.options)).toEqual(similarityNames)
    expect(defaults(similaritySpace)).toEqual({ name: 'cosine' })
    const chosen = { name: 'gaussian', gamma: 0.5 }
    expect(contains(similaritySpace, chosen)).toBe(true)
    near(toArray(similarity([0, 0], [1, 1], chosen.name as SimilarityName, chosen)) as number, Math.exp(-1))
  })
})

describe('the named similarities on hand-computed values', () => {
  it('cosine, dot and pearson', () => {
    near(value([1, 0], [1, 1], 'cosine'), Math.SQRT1_2)
    near(value([1, 1], [-2, -2], 'cosine'), -1)
    near(value([1, 2], [3, 4], 'dot'), 11)
    near(value([1, 2], [3, 4], 'linear'), 11)
    near(value([1, 2, 3], [2, 4, 6], 'pearson'), 1)
    near(value([1, 2, 3], [3, 2, 1], 'correlation'), -1)
  })

  it('gaussian and laplacian with gamma or a bandwidth', () => {
    near(value([0, 0], [1, 0], 'gaussian'), Math.exp(-1))
    near(value([0, 0], [1, 0], 'gaussian', { gamma: 2 }), Math.exp(-2))
    near(value([0, 0], [1, 0], 'rbf', { bandwidth: 2 }), Math.exp(-1 / 8))
    near(value([0, 0], [1, 0], 'gaussian', { gamma: 3, bandwidth: 2 }), Math.exp(-3))
    near(value([0, 0], [1, 1], 'laplacian', { gamma: 0.5 }), Math.exp(-1))
    near(value([0, 0], [1, 1], 'laplacian', { bandwidth: 4 }), Math.exp(-0.5))
    near(value([0, 0], [0, 0], 'gaussian'), 1)
  })

  it('inverse distance over any registered distance, with its options beside it', () => {
    near(value([0, 0], [3, 4], 'inverseDistance'), 1 / 6)
    near(value([0, 0], [3, 4], 'inverseDistance', { distance: 'manhattan' }), 1 / 8)
    near(value([0, 0], [3, 4], 'inverseDistance', { distance: 'l1', scale: 7 }), 0.5)
    near(value([0, 0], [3, 4], 'inverseDistance', { distance: 'minkowski', p: 1 }), 1 / 8)
    near(value([0, 0], [3, 4], 'inverseDistance', { distance: 'seuclidean', variance: [9, 16] }), 1 / (1 + Math.SQRT2))
    expect(() => value([0, 0], [3, 4], 'inverseDistance', { distance: 'nope' as 'euclidean' })).toThrow(
      /unknown distance/,
    )
  })

  it('resolves the data-dependent defaults of an inner distance from the points', () => {
    const x = [
      [0.1, 1.2],
      [2.3, -0.4],
      [1.1, 0.8],
      [-0.9, 2.2],
    ]
    const s = toFlat(pairwiseSimilarities(x, undefined, 'inverseDistance', { distance: 'mahalanobis' }))
    const d = toFlat(pairwiseDistances(x, undefined, 'mahalanobis'))
    s.forEach((v, i) => {
      near(v, 1 / (1 + (d[i] ?? Number.NaN)))
    })
    const rows = toFlat(similarity(x, x.slice().reverse(), 'inverseDistance', { distance: 'seuclidean' }))
    const expected = toFlat(distance(x, x.slice().reverse(), 'seuclidean'))
    rows.forEach((v, i) => {
      near(v, 1 / (1 + (expected[i] ?? Number.NaN)))
    })
  })
})

describe('conversions between distances and similarities', () => {
  const d = [0, 1, 3]

  it('map a distance to a similarity by each form', () => {
    expect(toArray(distanceToSimilarity(d))).toEqual([1, 0.5, 0.25])
    expect(toArray(distanceToSimilarity(d, { form: 'inverse', scale: 2 }))).toEqual([1, 2 / 3, 0.4])
    const exponential = toFlat(distanceToSimilarity(d, { form: 'exponential' }))
    d.forEach((v, i) => {
      near(exponential[i] ?? Number.NaN, Math.exp(-v))
    })
    const gaussian = toFlat(distanceToSimilarity(d, { form: 'gaussian', scale: 2 }))
    d.forEach((v, i) => {
      near(gaussian[i] ?? Number.NaN, Math.exp(-(v * v) / 8))
    })
    expect(toArray(distanceToSimilarity(d, { form: 'linear', scale: 2 }))).toEqual([1, 0.5, 0])
    expect(() => distanceToSimilarity(d, { form: 'other' as 'linear' })).toThrow(/unknown form/)
  })

  it('map a similarity back to a distance, undoing the inverse and linear forms', () => {
    const back = toFlat(similarityToDistance(distanceToSimilarity(d), { form: 'inverse' }))
    d.forEach((v, i) => {
      near(back[i] ?? Number.NaN, v)
    })
    expect(toArray(similarityToDistance([1, 0.5, 0]))).toEqual([0, 0.5, 1])
    const linear = toFlat(similarityToDistance(distanceToSimilarity([0.2, 0.7], { form: 'linear' })))
    near(linear[0] ?? Number.NaN, 0.2)
    near(linear[1] ?? Number.NaN, 0.7)
    expect(() => similarityToDistance(d, { form: 'other' as 'linear' })).toThrow(/unknown form/)
  })

  it('give the kernel-induced distance of a normalised kernel through the chord form', () => {
    // The chord of a cosine similarity is the Euclidean distance between the unit vectors.
    const x = [
      [3, 4],
      [-1, 0],
      [0, 2],
    ]
    const unit = x.map((p) => {
      const n = Math.hypot(p[0] ?? 0, p[1] ?? 0)
      return p.map((v) => v / n)
    })
    const chord = toFlat(similarityToDistance(pairwiseSimilarities(x, undefined, 'cosine'), { form: 'chord' }))
    const euclid = toFlat(pairwiseDistances(unit, undefined, 'euclidean'))
    chord.forEach((v, i) => {
      near(v, euclid[i] ?? Number.NaN)
    })
    // The Gaussian form of a distance with bandwidth sigma is the Gaussian similarity with that bandwidth.
    const gaussianOfDistance = toFlat(
      distanceToSimilarity(pairwiseDistances(x, undefined, 'euclidean'), { form: 'gaussian', scale: 1.5 }),
    )
    const gaussian = toFlat(pairwiseSimilarities(x, undefined, 'gaussian', { bandwidth: 1.5 }))
    gaussianOfDistance.forEach((v, i) => {
      near(v, gaussian[i] ?? Number.NaN)
    })
    expect(toArray(similarityToDistance(tensor([1, 1.0000001]), { form: 'chord' }))).toEqual([0, 0])
  })
})
