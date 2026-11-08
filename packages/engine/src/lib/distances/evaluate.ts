/**
 * The entry points over the two registries: `distance` and `similarity` between corresponding rows (or two points),
 * `pairwiseDistances` and `pairwiseSimilarities` between every pair of rows, each by name with the entry's options.
 * Shapes are checked once here, scipy's data-dependent defaults (seuclidean's variances, mahalanobis's covariance) are
 * filled from the points, and the rest is the entry's `compute` on the broadcast layout it expects.
 */

import { OmaiError } from '../base/index.ts'
import { resolveMetricOptions } from '../numerics/geometry/index.ts'
import { concat, promoteFloat, reshape, shapesEqual, type TensorLike, toFlat, type Value } from '../tensor/index.ts'
import { type DistanceAlias, type DistanceName, type DistanceOptions, distanceEntry } from './distances.ts'
import { type SimilarityAlias, type SimilarityName, type SimilarityOptions, similarityEntry } from './similarities.ts'

/** Points `[d]` or rows `[n, d]` as rows, for the pairwise forms. */
function asRows(x: Value, name: string): Value {
  if (x.shape.length === 1) return reshape(x, [1, -1])
  if (x.shape.length === 2) return x
  throw new OmaiError('shape', `${name}: points must be [n, d] or [d], got [${x.shape.join(', ')}]`)
}

/** Both operands as floats, with the same coordinate axis and leading shapes that are equal or a single point. */
function elementwise(a: TensorLike, b: TensorLike, name: string): [Value, Value] {
  const [x, y] = promoteFloat(a, b) as [Value, Value]
  const describe = (): string => `[${x.shape.join(', ')}] and [${y.shape.join(', ')}]`
  if (x.shape.length === 0 || y.shape.length === 0 || x.shape.length > 2 || y.shape.length > 2)
    throw new OmaiError('shape', `${name}: points are [d] or rows [n, d], got ${describe()}`)
  if (x.shape[x.shape.length - 1] !== y.shape[y.shape.length - 1])
    throw new OmaiError('shape', `${name}: the points have different dimensions, ${describe()}`)
  if (x.shape.length === 2 && y.shape.length === 2 && !shapesEqual(x.shape, y.shape))
    throw new OmaiError('shape', `${name}: rows must match one to one, got ${describe()}`)
  return [x, y]
}

/** Both operands as rows `[m, d]` and `[n, d]` (b defaulting to a), for the pairwise forms. */
function pairwise(a: TensorLike, b: TensorLike | undefined, name: string): [Value, Value] {
  const x = asRows(promoteFloat(a)[0] as Value, name)
  if (b === undefined) return [x, x]
  const y = asRows(promoteFloat(b)[0] as Value, name)
  if (x.shape[1] !== y.shape[1])
    throw new OmaiError(
      'shape',
      `${name}: the points have different dimensions, [${x.shape.join(', ')}] and [${y.shape.join(', ')}]`,
    )
  return [x, y]
}

/**
 * scipy's defaults for a seuclidean or mahalanobis distance, from the points: the stacked rows of both operands when
 * two are given, one operand's rows when the other is the same.
 */
function withStatistics<O extends DistanceOptions>(
  name: DistanceName | DistanceAlias | undefined,
  points: () => Value,
  options: O,
): O {
  if (name === undefined) return options
  const canonical = distanceEntry(name).name
  if (canonical === 'seuclidean' && options.variance === undefined)
    return { ...options, variance: resolveMetricOptions('seuclidean', points()).v }
  if (canonical === 'mahalanobis' && options.covariance === undefined && options.inverseCovariance === undefined)
    return { ...options, inverseCovariance: resolveMetricOptions('mahalanobis', points()).vi }
  return options
}

/** The rows of both operands stacked `[m + n, d]` (an operand `[d]` is one row). */
function stacked(x: Value, y: Value): Value {
  const d = x.shape[x.shape.length - 1] ?? 0
  return concat([reshape(x, [-1, d]), reshape(y, [-1, d])], 0)
}

/**
 * The named distance between corresponding points: two points `[d]` give a scalar, rows `[n, d]` against rows of the
 * same shape give one distance per row `[n]`, and a point against rows gives its distance to each. Differentiable in
 * the points where the distance is (every continuous one). seuclidean's variances and mahalanobis's covariance default
 * to the sample statistics of all the points given, as scipy.
 *
 * @param a - A point `[d]` or rows `[n, d]`.
 * @param b - A point `[d]` or rows `[n, d]`.
 * @param name - A registered distance or alias (`distanceNames`).
 * @param options - The distance's parameters (`distances[name].parameters` lists the knobs).
 * @returns The distances, `[]` for two points or `[n]`.
 * @throws OmaiError `value` for an unknown name, `shape` for points of different dimensions or rows that do not pair up.
 * @concept distances
 * @index function Named distance between points, from a registry of scipy's metrics.
 * @example
 * ```ts
 * import { distance } from 'omai-engine/distances'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(distance([0, 0], [3, 4], 'euclidean')) // => 5
 * toArray(distance([[0, 0], [1, 1]], [[3, 4], [1, 2]], 'manhattan')) // => [7, 1]
 * toArray(distance([0, 0], [[1, 1], [2, 2]], 'minkowski', { p: 3 })) // => [1.259921, 2.519842]
 * ```
 */
export function distance(
  a: TensorLike,
  b: TensorLike,
  name: DistanceName | DistanceAlias,
  options: DistanceOptions = {},
): Value {
  const entry = distanceEntry(name)
  const [x, y] = elementwise(a, b, 'distance')
  return entry.compute(
    x,
    y,
    withStatistics(name, () => stacked(x, y), options),
  )
}

/**
 * The named distance between every row of `a` and every row of `b` (scipy's `cdist`, or `pdist` as a square matrix
 * when `b` is left out): the matrix `[m, n]`. seuclidean's variances and mahalanobis's covariance default to the
 * sample statistics of both inputs stacked (of `a` alone without `b`), as scipy.
 *
 * @param a - Rows `[m, d]` (a point `[d]` is one row).
 * @param b - Rows `[n, d]`, or undefined for the distances among the rows of `a`.
 * @param name - A registered distance or alias (`distanceNames`).
 * @param options - The distance's parameters.
 * @returns Distances `[m, n]`, differentiable in the points where the distance is.
 * @throws OmaiError `value` for an unknown name, `shape` for points of different dimensions.
 * @concept distances
 * @example
 * ```ts
 * import { pairwiseDistances } from 'omai-engine/distances'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(pairwiseDistances([[0, 0], [3, 4]], undefined, 'euclidean')) // => [[0, 5], [5, 0]]
 * toArray(pairwiseDistances([[0, 0]], [[1, 0], [0, 2]], 'chebyshev')) // => [[1, 2]]
 * ```
 */
export function pairwiseDistances(
  a: TensorLike,
  b: TensorLike | undefined,
  name: DistanceName | DistanceAlias,
  options: DistanceOptions = {},
): Value {
  const entry = distanceEntry(name)
  const [x, y] = pairwise(a, b, 'pairwiseDistances')
  const d = x.shape[1] ?? 0
  const points = (): Value => (b === undefined ? x : concat([x, y], 0))
  return entry.compute(reshape(x, [-1, 1, d]), reshape(y, [1, -1, d]), withStatistics(name, points, options))
}

/** The name of a metric slot: a registered distance or alias, or `'precomputed'` (the points are already distances). */
export type MetricSlot = DistanceName | DistanceAlias | 'precomputed'

/**
 * The square matrix of distances among the rows of `x`, flat row-major, for the algorithms that work on distances
 * rather than coordinates (density clustering, medoids, embeddings): the metric slot of a clusterer or a manifold
 * method. With `'precomputed'` the input is taken to be the distance matrix itself and checked to be square.
 *
 * @param x - Rows `[n, d]`, or with `'precomputed'` the distances `[n, n]`.
 * @param metric - A registered distance or alias (default `'euclidean'`), or `'precomputed'`.
 * @param options - The distance's parameters.
 * @returns The `n * n` distances, and `n`.
 * @throws OmaiError `shape` for a precomputed input that is not square.
 * @concept distances
 * @example
 * ```ts
 * import { distanceMatrix } from 'omai-engine/distances'
 * distanceMatrix([[0, 0], [3, 4]]).data // => [0, 5, 5, 0]
 * distanceMatrix([[0, 2], [2, 0]], 'precomputed').n // => 2
 * ```
 */
export function distanceMatrix(
  x: TensorLike,
  metric: MetricSlot = 'euclidean',
  options: DistanceOptions = {},
): { readonly n: number; readonly data: Float64Array } {
  const d = metric === 'precomputed' ? promoteFloat(x)[0] : pairwiseDistances(x, undefined, metric, options)
  const shape = (d as Value).shape
  if (shape.length !== 2 || shape[0] !== shape[1])
    throw new OmaiError('shape', `distanceMatrix: expected a square matrix, got [${shape.join(', ')}]`)
  return { n: shape[0] ?? 0, data: Float64Array.from(toFlat(d as Value)) }
}

/** The distance an entry is built on, whose data-dependent defaults the points must fill. */
function innerDistance(name: SimilarityName, options: SimilarityOptions): DistanceName | DistanceAlias | undefined {
  return name === 'inverseDistance' ? (options.distance ?? 'euclidean') : undefined
}

/**
 * The named similarity between corresponding points: two points `[d]` give a scalar, rows `[n, d]` against rows of
 * the same shape give one value per row `[n]`, and a point against rows gives its similarity to each. Differentiable
 * in the points.
 *
 * @param a - A point `[d]` or rows `[n, d]`.
 * @param b - A point `[d]` or rows `[n, d]`.
 * @param name - A registered similarity or alias (`similarityNames`).
 * @param options - The similarity's parameters (`similarities[name].parameters` lists the knobs).
 * @returns The similarities, `[]` for two points or `[n]`.
 * @throws OmaiError `value` for an unknown name, `shape` for points of different dimensions or rows that do not pair up.
 * @concept distances
 * @index function Named similarity between points.
 * @example
 * ```ts
 * import { similarity } from 'omai-engine/distances'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(similarity([1, 0], [1, 1], 'cosine')) // => 0.707107
 * toArray(similarity([[0, 0], [0, 0]], [[1, 0], [3, 4]], 'inverseDistance', { distance: 'euclidean' })) // => [0.5, 0.166667]
 * ```
 */
export function similarity(
  a: TensorLike,
  b: TensorLike,
  name: SimilarityName | SimilarityAlias,
  options: SimilarityOptions = {},
): Value {
  const entry = similarityEntry(name)
  const [x, y] = elementwise(a, b, 'similarity')
  return entry.compute(
    x,
    y,
    withStatistics(innerDistance(entry.name, options), () => stacked(x, y), options),
  )
}

/**
 * The named similarity between every row of `a` and every row of `b` (the Gram matrix of the rows of `a` when `b` is
 * left out): the matrix `[m, n]`, as sklearn's `cosine_similarity`, `rbf_kernel` and `laplacian_kernel`.
 *
 * @param a - Rows `[m, d]` (a point `[d]` is one row).
 * @param b - Rows `[n, d]`, or undefined for the similarities among the rows of `a`.
 * @param name - A registered similarity or alias (`similarityNames`).
 * @param options - The similarity's parameters.
 * @returns Similarities `[m, n]`, differentiable in the points.
 * @throws OmaiError `value` for an unknown name, `shape` for points of different dimensions.
 * @concept distances
 * @example
 * ```ts
 * import { pairwiseSimilarities } from 'omai-engine/distances'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(pairwiseSimilarities([[1, 0], [0, 1]], undefined, 'dot')) // => [[1, 0], [0, 1]]
 * toArray(pairwiseSimilarities([[0, 0]], [[0, 0], [1, 0]], 'gaussian', { gamma: 1 })) // => [[1, 0.367879]]
 * ```
 */
export function pairwiseSimilarities(
  a: TensorLike,
  b: TensorLike | undefined,
  name: SimilarityName | SimilarityAlias,
  options: SimilarityOptions = {},
): Value {
  const entry = similarityEntry(name)
  const [x, y] = pairwise(a, b, 'pairwiseSimilarities')
  const d = x.shape[1] ?? 0
  const points = (): Value => (b === undefined ? x : concat([x, y], 0))
  return entry.compute(
    reshape(x, [-1, 1, d]),
    reshape(y, [1, -1, d]),
    withStatistics(innerDistance(entry.name, options), points, options),
  )
}
