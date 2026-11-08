/**
 * The registry of named distances: each entry is a plain record with its name, a description, whether it is a true
 * metric, the knobs a page can draw as a `DictSpace`, a differentiable `compute` over the coordinate axis of broadcast
 * operands and the plain-number `scalar` form for loops over points. The computations are numerics/geometry's
 * (scipy.spatial.distance plus haversine); this file names them under one vocabulary, documents their parameters and
 * lays them out as one table, with a union space over the names so a control can pick a distance and its knobs.
 */

import { OmaiError } from '../base/index.ts'
import {
  type DistanceMetric,
  distanceBetween,
  type MetricOptions,
  metricFunction,
  type PointDistance,
} from '../numerics/geometry/index.ts'
import { inv } from '../numerics/linalg/index.ts'
import {
  type BoolSpace,
  bool,
  type DictSpace,
  dict,
  type RealSpace,
  real,
  type UnionSpace,
  union,
} from '../space/index.ts'
import { mul, type TensorLike, type Value } from '../tensor/index.ts'

/** The parameters of the named distances; each entry's `parameters` and description say which it reads. */
export interface DistanceOptions {
  /** Per-coordinate weights `[d]` (euclidean, squaredEuclidean, manhattan, chebyshev, minkowski). */
  readonly weights?: TensorLike
  /** The order of the Minkowski distance (default 2). */
  readonly p?: number
  /**
   * Per-coordinate variances `[d]` of the standardised Euclidean distance (default: the sample variances of the
   * points, as scipy).
   */
  readonly variance?: TensorLike
  /** The covariance `[d, d]` of the Mahalanobis distance (default: the sample covariance of the points, as scipy). */
  readonly covariance?: TensorLike
  /** The inverse covariance `[d, d]` of the Mahalanobis distance; takes precedence over `covariance`. */
  readonly inverseCovariance?: TensorLike
  /** Whether haversine coordinates are in degrees (default true) rather than radians. */
  readonly degrees?: boolean
  /** The sphere radius haversine distances are scaled by (default 1, the central angle; 6371 gives kilometres on Earth). */
  readonly radius?: number
}

/** The names of the registered distances. */
export type DistanceName =
  | 'euclidean'
  | 'squaredEuclidean'
  | 'manhattan'
  | 'chebyshev'
  | 'minkowski'
  | 'seuclidean'
  | 'cosine'
  | 'correlation'
  | 'mahalanobis'
  | 'hamming'
  | 'jaccard'
  | 'canberra'
  | 'braycurtis'
  | 'haversine'

/** Familiar aliases accepted wherever a distance is named. */
export type DistanceAlias = 'l1' | 'l2' | 'cityblock' | 'sqeuclidean' | 'linf'

/** The aliases of {@link DistanceAlias} resolved to registered names. */
export const distanceAliases: Readonly<Record<DistanceAlias, DistanceName>> = {
  l1: 'manhattan',
  l2: 'euclidean',
  cityblock: 'manhattan',
  sqeuclidean: 'squaredEuclidean',
  linf: 'chebyshev',
}

/**
 * A registered distance: a plain record a page can list, describe and draw controls for, and the computation behind
 * it. `compute` takes the parameters already resolved (`distance` and `pairwiseDistances` fill the data-dependent
 * defaults of seuclidean and mahalanobis from the points first).
 *
 * @concept distances
 */
export interface DistanceEntry {
  readonly name: DistanceName
  /** One sentence: the formula and what the distance is for. */
  readonly description: string
  /** Whether the distance satisfies the triangle inequality, so trees and bounds may rely on it. */
  readonly metric: boolean
  /** The knobs a page can draw; tensor parameters (weights, variances, covariances) are options only. */
  readonly parameters: DictSpace
  /**
   * The distance over the coordinate (last) axis of `a` and `b` `[..., d]` broadcast together, differentiable in both
   * where the formula is: rows against rows give `[n]`, `[m, 1, d]` against `[1, n, d]` the matrix `[m, n]`.
   */
  readonly compute: (a: Value, b: Value, options: DistanceOptions) => Value
  /** The plain-number form on flat arrays (the same values), for loops over points. */
  readonly scalar: (dim: number, options: DistanceOptions) => PointDistance
}

const P: RealSpace = real(1, 10, { default: 2, label: 'p', description: 'The order of the norm' })
const DEGREES: BoolSpace = bool({ default: true, label: 'degrees', description: 'Coordinates in degrees, not radians' })
const NONE: DictSpace = dict({})
const RADIANS_PER_DEGREE = Math.PI / 180

/** The options in scipy's names, the covariance inverted when only it is given. */
function metricOptions(options: DistanceOptions): MetricOptions {
  const out: { p?: number; w?: TensorLike; v?: TensorLike; vi?: TensorLike } = {}
  if (options.p !== undefined) out.p = options.p
  if (options.weights !== undefined) out.w = options.weights
  if (options.variance !== undefined) out.v = options.variance
  if (options.inverseCovariance !== undefined) out.vi = options.inverseCovariance
  else if (options.covariance !== undefined) out.vi = inv(options.covariance)
  return out
}

function entry(
  name: DistanceName,
  metric: DistanceMetric,
  description: string,
  isMetric: boolean,
  parameters: DictSpace,
): DistanceEntry {
  return {
    name,
    description,
    metric: isMetric,
    parameters,
    compute: (a, b, options) => distanceBetween(a, b, metric, metricOptions(options)),
    scalar: (dim, options) => metricFunction(metric, dim, metricOptions(options)),
  }
}

function requiring(e: DistanceEntry, has: (options: DistanceOptions) => boolean, message: string): DistanceEntry {
  const check = (options: DistanceOptions): void => {
    if (!has(options)) throw new OmaiError('value', `${e.name}: ${message}`)
  }
  return {
    ...e,
    compute: (a, b, options) => {
      check(options)
      return e.compute(a, b, options)
    },
    scalar: (dim, options) => {
      check(options)
      return e.scalar(dim, options)
    },
  }
}

/**
 * The Euclidean distance sqrt(sum w_i (a_i - b_i)^2), the straight-line distance (scipy `euclidean`).
 *
 * @example
 * ```ts
 * import { euclideanDistance } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(euclideanDistance.compute(tensor([0, 0]), tensor([3, 4]), {})) // => 5
 * ```
 */
export const euclideanDistance: DistanceEntry = entry(
  'euclidean',
  'euclidean',
  'The straight-line distance sqrt(sum (a - b)^2), optionally with per-coordinate weights.',
  true,
  NONE,
)

/**
 * The squared Euclidean distance sum w_i (a_i - b_i)^2, the Euclidean distance without its square root (scipy
 * `sqeuclidean`): cheaper, smooth at zero, and what k-means and Gaussian kernels use.
 *
 * @example
 * ```ts
 * import { squaredEuclideanDistance } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(squaredEuclideanDistance.compute(tensor([0, 0]), tensor([3, 4]), {})) // => 25
 * ```
 */
export const squaredEuclideanDistance: DistanceEntry = entry(
  'squaredEuclidean',
  'sqeuclidean',
  'The squared straight-line distance sum (a - b)^2: smooth at zero and not a metric.',
  false,
  NONE,
)

/**
 * The Manhattan (city block, L1) distance sum w_i |a_i - b_i| (scipy `cityblock`).
 *
 * @example
 * ```ts
 * import { manhattanDistance } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(manhattanDistance.compute(tensor([0, 0]), tensor([3, 4]), {})) // => 7
 * ```
 */
export const manhattanDistance: DistanceEntry = entry(
  'manhattan',
  'cityblock',
  'The city-block (L1) distance sum |a - b|, the path along the axes.',
  true,
  NONE,
)

/**
 * The Chebyshev (L-infinity) distance max_i w_i |a_i - b_i| (scipy `chebyshev`).
 *
 * @example
 * ```ts
 * import { chebyshevDistance } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(chebyshevDistance.compute(tensor([0, 0]), tensor([3, 4]), {})) // => 4
 * ```
 */
export const chebyshevDistance: DistanceEntry = entry(
  'chebyshev',
  'chebyshev',
  'The L-infinity distance max |a - b|, the largest coordinate difference.',
  true,
  NONE,
)

/**
 * The Minkowski distance (sum w_i |a_i - b_i|^p)^(1/p) (scipy `minkowski`): p = 1 is Manhattan, p = 2 Euclidean, and
 * it tends to Chebyshev as p grows. A metric for p >= 1.
 *
 * @example
 * ```ts
 * import { minkowskiDistance } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(minkowskiDistance.compute(tensor([0, 0]), tensor([3, 4]), { p: 1 })) // => 7
 * ```
 */
export const minkowskiDistance: DistanceEntry = entry(
  'minkowski',
  'minkowski',
  'The p-norm of the difference (sum |a - b|^p)^(1/p): Manhattan at p = 1, Euclidean at p = 2.',
  true,
  dict({ p: P }),
)

/**
 * The standardised Euclidean distance sqrt(sum (a_i - b_i)^2 / v_i), each coordinate scaled by its variance (scipy
 * `seuclidean`); `variance` defaults to the sample variances of the points in `distance` and `pairwiseDistances`.
 *
 * @example
 * ```ts
 * import { seuclideanDistance } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(seuclideanDistance.compute(tensor([0, 0]), tensor([3, 4]), { variance: [9, 16] })) // => 1.414214
 * ```
 */
export const seuclideanDistance: DistanceEntry = requiring(
  entry(
    'seuclidean',
    'seuclidean',
    'The Euclidean distance with each coordinate divided by its variance, sqrt(sum (a - b)^2 / v).',
    true,
    NONE,
  ),
  (o) => o.variance !== undefined,
  'needs the variances (distance and pairwiseDistances default them to the sample variances of the points)',
)

/**
 * The cosine distance 1 - a . b / (|a| |b|), one minus the cosine of the angle between the vectors (scipy `cosine`):
 * zero for parallel vectors, one for orthogonal, two for opposite; scale-free, so the choice for term vectors and
 * embeddings. Not a metric (its square root is).
 *
 * @example
 * ```ts
 * import { cosineDistance } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(cosineDistance.compute(tensor([1, 0]), tensor([0, 1]), {})) // => 1
 * ```
 */
export const cosineDistance: DistanceEntry = entry(
  'cosine',
  'cosine',
  'One minus the cosine of the angle between the vectors, 1 - a . b / (|a| |b|); scale-free.',
  false,
  NONE,
)

/**
 * The correlation distance 1 - r, one minus the Pearson correlation of the two vectors' coordinates (scipy
 * `correlation`): the cosine distance after centring each vector.
 *
 * @example
 * ```ts
 * import { correlationDistance } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(correlationDistance.compute(tensor([1, 2, 3]), tensor([2, 4, 6]), {})) // => 0
 * ```
 */
export const correlationDistance: DistanceEntry = entry(
  'correlation',
  'correlation',
  'One minus the Pearson correlation of the coordinates: the cosine distance of the centred vectors.',
  false,
  NONE,
)

/**
 * The Mahalanobis distance sqrt((a - b)^T S^-1 (a - b)), the Euclidean distance after whitening by a covariance S
 * (scipy `mahalanobis`): give `covariance` or `inverseCovariance`; `distance` and `pairwiseDistances` default to the
 * sample covariance of the points.
 *
 * @example
 * ```ts
 * import { mahalanobisDistance } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(mahalanobisDistance.compute(tensor([0, 0]), tensor([2, 0]), { covariance: [[4, 0], [0, 1]] })) // => 1
 * ```
 */
export const mahalanobisDistance: DistanceEntry = requiring(
  entry(
    'mahalanobis',
    'mahalanobis',
    'The Euclidean distance after whitening by a covariance, sqrt((a - b)^T S^-1 (a - b)).',
    true,
    NONE,
  ),
  (o) => o.covariance !== undefined || o.inverseCovariance !== undefined,
  'needs a covariance or inverseCovariance (distance and pairwiseDistances default to the sample covariance of the points)',
)

/**
 * The Hamming distance, the fraction of coordinates that differ (scipy `hamming`, on any values; not differentiable).
 *
 * @example
 * ```ts
 * import { hammingDistance } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(hammingDistance.compute(tensor([1, 0, 1, 1]), tensor([1, 1, 1, 0]), {})) // => 0.5
 * ```
 */
export const hammingDistance: DistanceEntry = entry(
  'hamming',
  'hamming',
  'The fraction of coordinates that differ.',
  true,
  NONE,
)

/**
 * The Jaccard distance on binary vectors: among the coordinates where either is non-zero, the fraction that differ
 * (scipy's boolean `jaccard`; not differentiable).
 *
 * @example
 * ```ts
 * import { jaccardDistance } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(jaccardDistance.compute(tensor([1, 1, 0, 0]), tensor([1, 0, 1, 0]), {})) // => 0.666667
 * ```
 */
export const jaccardDistance: DistanceEntry = entry(
  'jaccard',
  'jaccard',
  'On binary vectors, the fraction of the coordinates set in either that are not set in both.',
  true,
  NONE,
)

/**
 * The Canberra distance sum |a_i - b_i| / (|a_i| + |b_i|), a relative Manhattan distance sensitive near zero (scipy
 * `canberra`; terms with both coordinates zero count as zero).
 *
 * @example
 * ```ts
 * import { canberraDistance } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(canberraDistance.compute(tensor([1, 2]), tensor([3, 2]), {})) // => 0.5
 * ```
 */
export const canberraDistance: DistanceEntry = entry(
  'canberra',
  'canberra',
  'The relative city-block distance sum |a - b| / (|a| + |b|), sensitive to differences near zero.',
  true,
  NONE,
)

/**
 * The Bray-Curtis dissimilarity sum |a_i - b_i| / sum |a_i + b_i| (scipy `braycurtis`), in [0, 1] for non-negative
 * data such as counts and abundances. Not a metric.
 *
 * @example
 * ```ts
 * import { braycurtisDistance } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(braycurtisDistance.compute(tensor([1, 3]), tensor([3, 1]), {})) // => 0.5
 * ```
 */
export const braycurtisDistance: DistanceEntry = entry(
  'braycurtis',
  'braycurtis',
  'The Bray-Curtis dissimilarity sum |a - b| / sum |a + b|, in [0, 1] for non-negative data.',
  false,
  NONE,
)

/**
 * The haversine (great-circle) distance between `[latitude, longitude]` points on a sphere: the central angle, times
 * `radius` (default 1; 6371 for kilometres on Earth), with coordinates in degrees unless `degrees` is false
 * (sklearn `haversine_distances` takes radians).
 *
 * @example
 * ```ts
 * import { haversineDistance } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(haversineDistance.compute(tensor([0, 0]), tensor([0, 180]), {})) // => 3.141593
 * ```
 */
export const haversineDistance: DistanceEntry = {
  name: 'haversine',
  description: 'The great-circle distance between latitude-longitude points on a sphere of the given radius.',
  metric: true,
  parameters: dict({ degrees: DEGREES }),
  compute: (a, b, options) => {
    const scale = (options.degrees ?? true) ? RADIANS_PER_DEGREE : 1
    const radius = options.radius ?? 1
    const angle = distanceBetween(scale === 1 ? a : mul(a, scale), scale === 1 ? b : mul(b, scale), 'haversine')
    return radius === 1 ? angle : mul(angle, radius)
  },
  scalar: (dim, options) => {
    if (dim !== 2) throw new OmaiError('shape', 'haversine: points are [latitude, longitude]')
    const angle = metricFunction('haversine', dim)
    const scale = (options.degrees ?? true) ? RADIANS_PER_DEGREE : 1
    const radius = options.radius ?? 1
    if (scale === 1 && radius === 1) return angle
    const u = new Float64Array(2)
    const v = new Float64Array(2)
    return (a, ia, b, ib) => {
      u[0] = (a[ia] ?? 0) * scale
      u[1] = (a[ia + 1] ?? 0) * scale
      v[0] = (b[ib] ?? 0) * scale
      v[1] = (b[ib + 1] ?? 0) * scale
      return radius * angle(u, 0, v, 0)
    }
  },
}

/**
 * Every registered distance by name, so a page can list them, show their descriptions and draw their parameters.
 *
 * @index table Every registered distance by name.
 * @example
 * ```ts
 * import { distances } from 'omai-engine/distances'
 * distances.minkowski.parameters.spaces['p']?.kind // => 'real'
 * Object.keys(distances).length // => 14
 * ```
 */
export const distances: Readonly<Record<DistanceName, DistanceEntry>> = {
  euclidean: euclideanDistance,
  squaredEuclidean: squaredEuclideanDistance,
  manhattan: manhattanDistance,
  chebyshev: chebyshevDistance,
  minkowski: minkowskiDistance,
  seuclidean: seuclideanDistance,
  cosine: cosineDistance,
  correlation: correlationDistance,
  mahalanobis: mahalanobisDistance,
  hamming: hammingDistance,
  jaccard: jaccardDistance,
  canberra: canberraDistance,
  braycurtis: braycurtisDistance,
  haversine: haversineDistance,
}

/**
 * The registered distance names in table order, for a `choice` control or a loop over the registry.
 *
 * @example
 * ```ts
 * import { distanceNames } from 'omai-engine/distances'
 * import { choice } from 'omai-engine/space'
 * choice(distanceNames, { default: 'euclidean' }).options[0] // => 'euclidean'
 * ```
 */
export const distanceNames: readonly DistanceName[] = Object.keys(distances) as DistanceName[]

/**
 * A distance and its knobs as one tagged-union space: a value names the distance under `name` and carries its
 * parameters beside it (`{ name: 'minkowski', p: 3 }`), ready to pass to `distance` or `pairwiseDistances`.
 *
 * @example
 * ```ts
 * import { distanceSpace } from 'omai-engine/distances'
 * import { defaults } from 'omai-engine/space'
 * defaults(distanceSpace) // => { name: 'euclidean' }
 * ```
 */
export const distanceSpace: UnionSpace = union(
  Object.fromEntries(distanceNames.map((name) => [name, distances[name].parameters])),
  { tag: 'name', default: 'euclidean' },
)

/**
 * The registered distance of a name or alias.
 *
 * @param name - A registered name (`'euclidean'`, `'manhattan'`, ...) or an alias (`'l2'`, `'l1'`, `'cityblock'`,
 *   `'sqeuclidean'`, `'linf'`).
 * @returns The entry.
 * @throws OmaiError `value` for a name that is not registered.
 * @example
 * ```ts
 * import { distanceEntry } from 'omai-engine/distances'
 * distanceEntry('l1').name // => 'manhattan'
 * distanceEntry('nope') // => throws value
 * ```
 */
export function distanceEntry(name: string): DistanceEntry {
  const canonical = Object.hasOwn(distanceAliases, name) ? distanceAliases[name as DistanceAlias] : name
  if (!Object.hasOwn(distances, canonical))
    throw new OmaiError('value', `unknown distance '${name}'; the registered names are ${distanceNames.join(', ')}`, {
      name,
    })
  return distances[canonical as DistanceName]
}
