import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { awayFromZero, distinctTensor, RUNS, tensorOf } from '../../testing/arbitrary.ts'
import { expectConformant } from '../../testing/check.ts'
import { add, fromStorage, sum, type Tensor, type TensorLike, toFlat, type Value } from '../tensor/index.ts'
import {
  type DistanceName,
  type DistanceOptions,
  distance,
  distanceNames,
  distances,
  pairwiseDistances,
  pairwiseSimilarities,
  type SimilarityName,
  type SimilarityOptions,
  similarity,
  similarityNames,
} from './index.ts'

const N = 5
const D = 3

/** Binary rows, for hamming and jaccard. */
const binary = (n: number, d: number): fc.Arbitrary<Tensor<'float64'>> =>
  fc.array(fc.boolean(), { minLength: n * d, maxLength: n * d }).map((bits) =>
    fromStorage(
      Float64Array.from(bits, (b) => (b ? 1 : 0)),
      [n, d],
      'float64',
    ),
  )

/** Latitude-longitude rows in degrees, away from the poles. */
const geo = (n: number): fc.Arbitrary<Tensor<'float64'>> =>
  fc.array(fc.double({ min: -80, max: 80, noNaN: true }), { minLength: n, maxLength: n }).chain((lats) =>
    fc.array(fc.double({ min: -170, max: 170, noNaN: true }), { minLength: n, maxLength: n }).map((lons) => {
      const data = new Float64Array(2 * n)
      lats.forEach((lat, i) => {
        data[2 * i] = lat
        data[2 * i + 1] = lons[i] ?? 0
      })
      return fromStorage(data, [n, 2], 'float64')
    }),
  )

const COVARIANCE = [
  [2, 0.3, 0],
  [0.3, 1, 0.2],
  [0, 0.2, 1.5],
]

/** The data each distance is tested on, with the parameters it needs. */
const setups: Readonly<
  Record<DistanceName, { readonly points: fc.Arbitrary<Tensor>; readonly options: DistanceOptions }>
> = {
  euclidean: { points: tensorOf([N, D]), options: {} },
  squaredEuclidean: { points: tensorOf([N, D]), options: {} },
  manhattan: { points: tensorOf([N, D]), options: { weights: [1, 0.5, 2] } },
  chebyshev: { points: tensorOf([N, D]), options: {} },
  minkowski: { points: tensorOf([N, D]), options: { p: 3 } },
  seuclidean: { points: tensorOf([N, D]), options: { variance: [0.5, 1, 2] } },
  cosine: { points: awayFromZero([N, D]), options: {} },
  correlation: { points: distinctTensor([N, D]), options: {} },
  mahalanobis: { points: tensorOf([N, D]), options: { covariance: COVARIANCE } },
  hamming: { points: binary(N, D), options: {} },
  jaccard: { points: binary(N, D), options: {} },
  canberra: { points: awayFromZero([N, D]), options: {} },
  braycurtis: { points: awayFromZero([N, D], 0.1, 2), options: {} },
  haversine: { points: geo(N), options: {} },
}

const matrix = (x: TensorLike, y: TensorLike | undefined, name: DistanceName, options: DistanceOptions): number[] =>
  Array.from(toFlat(pairwiseDistances(x, y, name, options)))

describe('pairwise distances (properties)', () => {
  for (const name of distanceNames) {
    const { points, options } = setups[name]
    it(`${name}: symmetric, zero on the diagonal and non-negative`, () => {
      fc.assert(
        fc.property(points, (x) => {
          const m = matrix(x, undefined, name, options)
          const scale = Math.max(1, ...m.map(Math.abs))
          for (let i = 0; i < N; i++) {
            expect(Math.abs(m[i * N + i] ?? Number.NaN)).toBeLessThanOrEqual(1e-12 * scale)
            for (let j = 0; j < N; j++) {
              expect(m[i * N + j]).toBeGreaterThanOrEqual(-1e-12 * scale)
              expect(Math.abs((m[i * N + j] ?? 0) - (m[j * N + i] ?? 0))).toBeLessThanOrEqual(1e-12 * scale)
            }
          }
        }),
        { numRuns: RUNS },
      )
    })

    it(`${name}: the elementwise form is the diagonal of the pairwise form`, () => {
      fc.assert(
        fc.property(points, points, (x, y) => {
          const rows = toFlat(distance(x, y, name, options))
          const m = matrix(x, y, name, options)
          for (let i = 0; i < N; i++) {
            const want = m[i * N + i] ?? Number.NaN
            expect(Math.abs((rows[i] ?? Number.NaN) - want)).toBeLessThanOrEqual(1e-12 * Math.max(1, Math.abs(want)))
          }
          // A single point against the rows is its row of the matrix.
          const first = Array.from(toFlat(distance(toFlat(x).slice(0, x.shape[1] ?? 0), y, name, options)))
          first.forEach((v, j) => {
            const want = m[j] ?? Number.NaN
            expect(Math.abs(v - want)).toBeLessThanOrEqual(1e-12 * Math.max(1, Math.abs(want)))
          })
        }),
        { numRuns: RUNS },
      )
    })

    if (distances[name].metric) {
      it(`${name}: satisfies the triangle inequality`, () => {
        fc.assert(
          fc.property(points, (x) => {
            const m = matrix(x, undefined, name, options)
            const scale = Math.max(1, ...m.map(Math.abs))
            for (let i = 0; i < N; i++)
              for (let j = 0; j < N; j++)
                for (let k = 0; k < N; k++)
                  expect(m[i * N + k]).toBeLessThanOrEqual((m[i * N + j] ?? 0) + (m[j * N + k] ?? 0) + 1e-10 * scale)
          }),
          { numRuns: RUNS },
        )
      })
    }
  }
})

const CONTINUOUS: readonly DistanceName[] = [
  'euclidean',
  'squaredEuclidean',
  'manhattan',
  'chebyshev',
  'minkowski',
  'seuclidean',
  'cosine',
  'correlation',
  'mahalanobis',
  'canberra',
  'braycurtis',
  'haversine',
]

describe('distances are differentiable in both operands (conformance)', () => {
  for (const name of CONTINUOUS) {
    const { options } = setups[name]
    // Canberra and Bray-Curtis also take absolute values of the points and their sums, so positive points keep the
    // check away from those kinks.
    const points = name === 'canberra' || name === 'braycurtis' ? tensorOf([N, D], 0.1, 2) : setups[name].points
    it(name, () => {
      // Offsets with distinct magnitudes keep every coordinate difference away from zero and from ties, where the
      // absolute values and maxima of the L1, L-infinity and Canberra distances have no derivative.
      const offsets = name === 'haversine' ? distinctTensor([N, 2], 1, 3) : distinctTensor([N, D], 0.2, 1)
      fc.assert(
        fc.property(points, offsets, (x, s) => {
          const y = add(x, s) as Tensor
          expectConformant((a: Value, b: Value) => sum(distance(a, b, name, options)), [x, y], { vmap: false })
        }),
        { numRuns: Math.max(3, Math.floor(RUNS / 4)) },
      )
    })
  }
})

const SIMILARITY_SETUPS: Readonly<
  Record<SimilarityName, { readonly points: fc.Arbitrary<Tensor>; readonly options: SimilarityOptions }>
> = {
  cosine: { points: awayFromZero([N, D]), options: {} },
  dot: { points: tensorOf([N, D]), options: {} },
  pearson: { points: distinctTensor([N, D]), options: {} },
  gaussian: { points: tensorOf([N, D]), options: { gamma: 0.7 } },
  laplacian: { points: tensorOf([N, D]), options: { bandwidth: 1.5 } },
  inverseDistance: { points: tensorOf([N, D]), options: { distance: 'minkowski', p: 3, scale: 2 } },
}

describe('similarities (properties)', () => {
  for (const name of similarityNames) {
    const { points, options } = SIMILARITY_SETUPS[name]
    it(`${name}: symmetric, within its range, largest on the diagonal of a normalised similarity`, () => {
      fc.assert(
        fc.property(points, points, (x, y) => {
          const m = Array.from(toFlat(pairwiseSimilarities(x, undefined, name, options)))
          const [lo, hi] = [-1, 1]
          for (let i = 0; i < N; i++)
            for (let j = 0; j < N; j++) {
              const v = m[i * N + j] ?? Number.NaN
              expect(Math.abs(v - (m[j * N + i] ?? 0))).toBeLessThanOrEqual(1e-12 * Math.max(1, Math.abs(v)))
              if (name !== 'dot') {
                expect(v).toBeGreaterThanOrEqual(lo - 1e-12)
                expect(v).toBeLessThanOrEqual(hi + 1e-12)
              }
              if (name === 'gaussian' || name === 'laplacian' || name === 'inverseDistance') {
                expect(v).toBeGreaterThan(0)
                expect(v).toBeLessThanOrEqual(m[i * N + i] ?? Number.NaN)
              }
            }
          if (
            name === 'cosine' ||
            name === 'pearson' ||
            name === 'gaussian' ||
            name === 'laplacian' ||
            name === 'inverseDistance'
          )
            for (let i = 0; i < N; i++) expect(m[i * N + i]).toBeCloseTo(1, 12)
          const rows = toFlat(similarity(x, y, name, options))
          const cross = toFlat(pairwiseSimilarities(x, y, name, options))
          for (let i = 0; i < N; i++)
            expect(Math.abs((rows[i] ?? Number.NaN) - (cross[i * N + i] ?? Number.NaN))).toBeLessThanOrEqual(1e-12)
        }),
        { numRuns: RUNS },
      )
    })
  }
})
