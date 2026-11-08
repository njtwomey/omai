/**
 * The registry of named similarities, the mirror of the distance registry: cosine, dot product, Pearson correlation,
 * the Gaussian (RBF) and Laplacian kernels of the Euclidean and Manhattan distances, and the inverse distance
 * 1 / (1 + d) over any registered distance; each a plain record with its description, range, knobs as a `DictSpace`
 * and a differentiable `compute`, built on the distance entries. The conversions between distances and similarities
 * (inverse, exponential, Gaussian, linear; and back, including the chord distance of a normalised kernel) are here
 * too, for turning any matrix of one kind into the other.
 */

import { OmaiError } from '../base/index.ts'
import {
  choice,
  type DictSpace,
  dict,
  type RealSpace,
  real,
  type Space,
  type UnionSpace,
  union,
} from '../space/index.ts'
import {
  add,
  div,
  exp,
  maximum,
  mul,
  neg,
  promoteFloat,
  reciprocal,
  sqrt,
  sub,
  sum,
  type TensorLike,
  type Value,
} from '../tensor/index.ts'
import {
  type DistanceAlias,
  type DistanceName,
  type DistanceOptions,
  distanceEntry,
  distanceNames,
  distances,
  manhattanDistance,
  squaredEuclideanDistance,
} from './distances.ts'

/** The parameters of the named similarities, which include those of the distance an entry is built on. */
export interface SimilarityOptions extends DistanceOptions {
  /** The scale gamma of the Gaussian exp(-gamma d^2) and Laplacian exp(-gamma d) similarities (default 1). */
  readonly gamma?: number
  /**
   * The bandwidth sigma, an alternative to gamma: gamma = 1 / (2 sigma^2) for the Gaussian and 1 / sigma for the
   * Laplacian (gamma wins when both are given).
   */
  readonly bandwidth?: number
  /** The distance the inverse-distance similarity is built on (default euclidean), with that distance's options beside it. */
  readonly distance?: DistanceName | DistanceAlias
  /** The scale of the inverse-distance similarity 1 / (1 + d / scale) (default 1). */
  readonly scale?: number
}

/** The names of the registered similarities. */
export type SimilarityName = 'cosine' | 'dot' | 'pearson' | 'gaussian' | 'laplacian' | 'inverseDistance'

/** Familiar aliases accepted wherever a similarity is named. */
export type SimilarityAlias = 'rbf' | 'linear' | 'correlation'

/** The aliases of {@link SimilarityAlias} resolved to registered names. */
export const similarityAliases: Readonly<Record<SimilarityAlias, SimilarityName>> = {
  rbf: 'gaussian',
  linear: 'dot',
  correlation: 'pearson',
}

/**
 * A registered similarity: a plain record a page can list, describe and draw controls for, and the computation
 * behind it. `compute` takes the parameters already resolved (`similarity` and `pairwiseSimilarities` fill the
 * data-dependent defaults of a seuclidean or mahalanobis inner distance from the points first).
 *
 * @concept distances
 */
export interface SimilarityEntry {
  readonly name: SimilarityName
  /** One sentence: the formula and what the similarity is for. */
  readonly description: string
  /** The interval the values lie in. */
  readonly range: readonly [number, number]
  /** The knobs a page can draw. */
  readonly parameters: DictSpace
  /**
   * The similarity over the coordinate (last) axis of `a` and `b` `[..., d]` broadcast together, differentiable in
   * both: rows against rows give `[n]`, `[m, 1, d]` against `[1, n, d]` the matrix `[m, n]`.
   */
  readonly compute: (a: Value, b: Value, options: SimilarityOptions) => Value
}

const GAMMA: RealSpace = real(1e-3, 1e3, { log: true, default: 1, label: 'gamma', description: 'The scale gamma' })
const SCALE: RealSpace = real(1e-3, 1e3, { log: true, default: 1, label: 'scale', description: 'The distance scale' })
const NONE: DictSpace = dict({})

function gammaOf(options: SimilarityOptions, kernel: 'gaussian' | 'laplacian'): number {
  if (options.gamma !== undefined) return options.gamma
  if (options.bandwidth !== undefined)
    return kernel === 'gaussian' ? 1 / (2 * options.bandwidth * options.bandwidth) : 1 / options.bandwidth
  return 1
}

/**
 * The cosine similarity a . b / (|a| |b|), the cosine of the angle between the vectors (sklearn `cosine_similarity`):
 * 1 for parallel, 0 for orthogonal, -1 for opposite; scale-free.
 *
 * @example
 * ```ts
 * import { cosineSimilarity } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(cosineSimilarity.compute(tensor([1, 0]), tensor([1, 1]), {})) // => 0.707107
 * ```
 */
export const cosineSimilarity: SimilarityEntry = {
  name: 'cosine',
  description: 'The cosine of the angle between the vectors, a . b / (|a| |b|); 1 parallel, 0 orthogonal, -1 opposite.',
  range: [-1, 1],
  parameters: NONE,
  compute: (a, b, options) => sub(1, distances.cosine.compute(a, b, options)),
}

/**
 * The dot product a . b (sklearn `linear_kernel`): the linear kernel, unbounded and scale-dependent.
 *
 * @example
 * ```ts
 * import { dotSimilarity } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(dotSimilarity.compute(tensor([1, 2]), tensor([3, 4]), {})) // => 11
 * ```
 */
export const dotSimilarity: SimilarityEntry = {
  name: 'dot',
  description: 'The dot product a . b, the linear kernel.',
  range: [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY],
  parameters: NONE,
  compute: (a, b) => sum(mul(a, b), -1),
}

/**
 * The Pearson correlation of the two vectors' coordinates (numpy `corrcoef`): the cosine similarity of the centred
 * vectors, one minus the correlation distance.
 *
 * @example
 * ```ts
 * import { pearsonSimilarity } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(pearsonSimilarity.compute(tensor([1, 2, 3]), tensor([3, 2, 1]), {})) // => -1
 * ```
 */
export const pearsonSimilarity: SimilarityEntry = {
  name: 'pearson',
  description: 'The Pearson correlation of the coordinates: the cosine similarity of the centred vectors.',
  range: [-1, 1],
  parameters: NONE,
  compute: (a, b, options) => sub(1, distances.correlation.compute(a, b, options)),
}

/**
 * The Gaussian (RBF) similarity exp(-gamma |a - b|^2) (sklearn `rbf_kernel`), 1 at zero distance and falling to 0;
 * `gamma` or a `bandwidth` sigma with gamma = 1 / (2 sigma^2) (kernels' `rbfKernel` is the same function with a
 * differentiable length scale).
 *
 * @example
 * ```ts
 * import { gaussianSimilarity } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(gaussianSimilarity.compute(tensor([0, 0]), tensor([1, 0]), { gamma: 1 })) // => 0.367879
 * ```
 */
export const gaussianSimilarity: SimilarityEntry = {
  name: 'gaussian',
  description: 'The Gaussian (RBF) kernel exp(-gamma |a - b|^2): 1 at zero distance, falling to 0.',
  range: [0, 1],
  parameters: dict({ gamma: GAMMA }),
  compute: (a, b, options) => exp(mul(squaredEuclideanDistance.compute(a, b, options), -gammaOf(options, 'gaussian'))),
}

/**
 * The Laplacian similarity exp(-gamma |a - b|_1) (sklearn `laplacian_kernel`), the exponential of the Manhattan
 * distance; `gamma` or a `bandwidth` sigma with gamma = 1 / sigma.
 *
 * @example
 * ```ts
 * import { laplacianSimilarity } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(laplacianSimilarity.compute(tensor([0, 0]), tensor([1, 1]), { gamma: 0.5 })) // => 0.367879
 * ```
 */
export const laplacianSimilarity: SimilarityEntry = {
  name: 'laplacian',
  description: 'The Laplacian kernel exp(-gamma |a - b|_1), the exponential of the Manhattan distance.',
  range: [0, 1],
  parameters: dict({ gamma: GAMMA }),
  compute: (a, b, options) => exp(mul(manhattanDistance.compute(a, b, options), -gammaOf(options, 'laplacian'))),
}

/** The knobs of every distance, each active only when `distance` names an entry that has it. */
function innerParameters(): Readonly<Record<string, Space>> {
  const users = new Map<string, { readonly space: Space; readonly names: DistanceName[] }>()
  for (const name of distanceNames) {
    for (const [key, space] of Object.entries(distances[name].parameters.spaces)) {
      const found = users.get(key)
      if (found === undefined) users.set(key, { space, names: [name] })
      else found.names.push(name)
    }
  }
  return Object.fromEntries(
    [...users].map(([key, { space, names }]) => [key, { ...space, when: { key: 'distance', in: names } }]),
  )
}

/**
 * The inverse distance 1 / (1 + d / scale) over any registered distance (default Euclidean), named by `distance`
 * with that distance's own options beside it: 1 at zero distance, falling towards 0, in (0, 1].
 *
 * @example
 * ```ts
 * import { inverseDistanceSimilarity } from 'omai-engine/distances'
 * import { tensor, toArray } from 'omai-engine/tensor'
 * toArray(inverseDistanceSimilarity.compute(tensor([0, 0]), tensor([3, 4]), { distance: 'manhattan' })) // => 0.125
 * ```
 */
export const inverseDistanceSimilarity: SimilarityEntry = {
  name: 'inverseDistance',
  description: 'The inverse distance 1 / (1 + d / scale) over any registered distance: 1 at zero, falling towards 0.',
  range: [0, 1],
  parameters: dict({
    distance: choice(distanceNames, { default: 'euclidean', label: 'distance', description: 'The distance inverted' }),
    scale: SCALE,
    ...innerParameters(),
  }),
  compute: (a, b, options) => {
    const d = distanceEntry(options.distance ?? 'euclidean').compute(a, b, options)
    const scale = options.scale ?? 1
    return reciprocal(add(scale === 1 ? d : div(d, scale), 1))
  },
}

/**
 * Every registered similarity by name, so a page can list them, show their descriptions and draw their parameters.
 *
 * @example
 * ```ts
 * import { similarities } from 'omai-engine/distances'
 * similarities.gaussian.range // => [0, 1]
 * Object.keys(similarities).length // => 6
 * ```
 */
export const similarities: Readonly<Record<SimilarityName, SimilarityEntry>> = {
  cosine: cosineSimilarity,
  dot: dotSimilarity,
  pearson: pearsonSimilarity,
  gaussian: gaussianSimilarity,
  laplacian: laplacianSimilarity,
  inverseDistance: inverseDistanceSimilarity,
}

/**
 * The registered similarity names in table order, for a `choice` control or a loop over the registry.
 *
 * @example
 * ```ts
 * import { similarityNames } from 'omai-engine/distances'
 * similarityNames.includes('gaussian') // => true
 * ```
 */
export const similarityNames: readonly SimilarityName[] = Object.keys(similarities) as SimilarityName[]

/**
 * A similarity and its knobs as one tagged-union space: a value names the similarity under `name` and carries its
 * parameters beside it (`{ name: 'gaussian', gamma: 0.5 }`), ready to pass to `similarity` or `pairwiseSimilarities`.
 *
 * @example
 * ```ts
 * import { similaritySpace } from 'omai-engine/distances'
 * import { defaults } from 'omai-engine/space'
 * defaults(similaritySpace) // => { name: 'cosine' }
 * ```
 */
export const similaritySpace: UnionSpace = union(
  Object.fromEntries(similarityNames.map((name) => [name, similarities[name].parameters])),
  { tag: 'name', default: 'cosine' },
)

/**
 * The registered similarity of a name or alias.
 *
 * @param name - A registered name (`'cosine'`, `'gaussian'`, ...) or an alias (`'rbf'`, `'linear'`,
 *   `'correlation'`).
 * @returns The entry.
 * @throws OmaiError `value` for a name that is not registered.
 * @example
 * ```ts
 * import { similarityEntry } from 'omai-engine/distances'
 * similarityEntry('rbf').name // => 'gaussian'
 * ```
 */
export function similarityEntry(name: string): SimilarityEntry {
  const canonical = Object.hasOwn(similarityAliases, name) ? similarityAliases[name as SimilarityAlias] : name
  if (!Object.hasOwn(similarities, canonical))
    throw new OmaiError(
      'value',
      `unknown similarity '${name}'; the registered names are ${similarityNames.join(', ')}`,
      { name },
    )
  return similarities[canonical as SimilarityName]
}

/** How {@link distanceToSimilarity} maps a distance d, with a scale s. */
export type DistanceToSimilarityForm = 'inverse' | 'exponential' | 'gaussian' | 'linear'

/** Options of {@link distanceToSimilarity}. */
export interface DistanceToSimilarityOptions {
  /** `inverse` 1 / (1 + d / s) (default), `exponential` exp(-d / s), `gaussian` exp(-d^2 / (2 s^2)), `linear` max(0, 1 - d / s). */
  readonly form?: DistanceToSimilarityForm
  /** The scale s (default 1). */
  readonly scale?: number
}

/**
 * Turn distances into similarities, elementwise on any shape (a vector, or a pairwise matrix): 1 at zero distance and
 * decreasing, by one of four forms. The Gaussian form of a Euclidean matrix is the RBF kernel matrix; the linear form
 * of a cosine distance is the cosine similarity for scale 1.
 *
 * @param d - Distances, any shape.
 * @param options - The form and scale.
 * @returns The similarities, the same shape, differentiable.
 * @throws OmaiError `value` for an unknown form.
 * @concept distances
 * @example
 * ```ts
 * import { distanceToSimilarity } from 'omai-engine/distances'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(distanceToSimilarity([0, 1, 3])) // => [1, 0.5, 0.25]
 * toArray(distanceToSimilarity([0, 2], { form: 'linear', scale: 4 })) // => [1, 0.5]
 * ```
 */
export function distanceToSimilarity(d: TensorLike, options: DistanceToSimilarityOptions = {}): Value {
  const [x] = promoteFloat(d) as [Value]
  const scale = options.scale ?? 1
  const scaled = scale === 1 ? x : div(x, scale)
  switch (options.form ?? 'inverse') {
    case 'inverse':
      return reciprocal(add(scaled, 1))
    case 'exponential':
      return exp(neg(scaled))
    case 'gaussian':
      return exp(mul(mul(scaled, scaled), -0.5))
    case 'linear':
      return maximum(sub(1, scaled), 0)
    default:
      throw new OmaiError('value', `distanceToSimilarity: unknown form '${String(options.form)}'`)
  }
}

/** How {@link similarityToDistance} maps a similarity s. */
export type SimilarityToDistanceForm = 'linear' | 'inverse' | 'chord'

/** Options of {@link similarityToDistance}. */
export interface SimilarityToDistanceOptions {
  /**
   * `linear` 1 - s (default; cosine similarity to cosine distance), `inverse` 1 / s - 1 (undoing the inverse form),
   * `chord` sqrt(2 - 2 s), the Euclidean distance in the feature space of a normalised kernel (k(x, x) = 1: Gaussian,
   * Laplacian, cosine).
   */
  readonly form?: SimilarityToDistanceForm
}

/**
 * Turn similarities into distances, elementwise on any shape: 0 at full similarity and increasing. The chord form is
 * the kernel-induced distance of a normalised kernel, sqrt(k(x, x) + k(y, y) - 2 k(x, y)) with unit diagonal, so the
 * chord of a Gaussian similarity matrix is a true metric between the points.
 *
 * @param s - Similarities, any shape.
 * @param options - The form.
 * @returns The distances, the same shape, differentiable.
 * @throws OmaiError `value` for an unknown form.
 * @concept distances
 * @example
 * ```ts
 * import { similarityToDistance } from 'omai-engine/distances'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(similarityToDistance([1, 0.5, 0.25], { form: 'inverse' })) // => [0, 1, 3]
 * toArray(similarityToDistance([1, 0, -1], { form: 'chord' })) // => [0, 1.414214, 2]
 * ```
 */
export function similarityToDistance(s: TensorLike, options: SimilarityToDistanceOptions = {}): Value {
  const [x] = promoteFloat(s) as [Value]
  switch (options.form ?? 'linear') {
    case 'linear':
      return sub(1, x)
    case 'inverse':
      return sub(reciprocal(x), 1)
    case 'chord':
      return sqrt(maximum(sub(2, mul(x, 2)), 0))
    default:
      throw new OmaiError('value', `similarityToDistance: unknown form '${String(options.form)}'`)
  }
}
