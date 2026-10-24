import { WhiteboxError } from '../base/index.ts'
import { entr, relEntr, xlogy } from '../numerics/special/index.ts'
import {
  abs,
  add,
  div,
  equal,
  greater,
  log,
  mul,
  neg,
  sqrt,
  sub,
  sum,
  type TensorLike,
  type Value,
  where,
} from '../tensor/index.ts'
import { float, inBase, normaliseAlong } from './_shared.ts'

/** Options of the entropies and divergences of probability vectors. */
export interface DiscreteOptions {
  /** Logarithm base: e (nats) by default, 2 for bits. */
  readonly base?: number
  /** The axis that holds the outcomes (default -1); the other axes are batch axes. */
  readonly axis?: number
  /** Normalise the inputs to sum to one along the axis first, as scipy.stats.entropy (default true). */
  readonly normalise?: boolean
}

/** Options of the functions of joint probability tables (every axis is a variable). */
export interface TableOptions {
  /** Logarithm base: e (nats) by default, 2 for bits. */
  readonly base?: number
}

function prepared(x: TensorLike, options: DiscreteOptions): Value {
  const v = float(x)
  return options.normalise === false ? v : normaliseAlong(v, options.axis ?? -1)
}

function table(x: TensorLike, rank: number | undefined, name: string): Value {
  const v = float(x)
  if (rank !== undefined && v.shape.length !== rank)
    throw new WhiteboxError('shape', `${name}: expected a table of rank ${rank}, got shape [${v.shape.join(', ')}]`)
  return normaliseAlong(v)
}

/** The entropy of the marginal of a normalised table on the given axes (all others summed out). */
function marginalEntropy(p: Value, keep: readonly number[]): Value {
  const rank = p.shape.length
  const drop = Array.from({ length: rank }, (_, i) => i).filter((i) => !keep.includes(i))
  return sum(entr(drop.length === 0 ? p : sum(p, drop)))
}

/**
 * The Shannon entropy H(p) = -sum p log p of a probability vector (scipy.stats.entropy with one argument): the inputs
 * are normalised along the axis first, zero entries contribute nothing, and the result is differentiable.
 *
 * @param p - Probabilities or unnormalised weights; the outcomes lie along `options.axis`.
 * @param options - Base, axis and whether to normalise.
 * @returns The entropy, one value per batch entry.
 * @concept entropy
 * @example
 * ```ts
 * import { entropy } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * item(entropy([1, 1, 1, 1], { base: 2 })) // => 2
 * ```
 */
export function entropy(p: TensorLike, options: DiscreteOptions = {}): Value {
  return inBase(sum(entr(prepared(p, options)), options.axis ?? -1), options.base)
}

/**
 * The binary entropy function h(p) = -p log p - (1 - p) log(1 - p), elementwise.
 *
 * @param p - Probabilities in [0, 1].
 * @param options - The logarithm base.
 * @returns h(p), differentiable on (0, 1).
 * @concept entropy
 * @example
 * ```ts
 * import { binaryEntropy } from 'whitebox-engine/information'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(binaryEntropy([0, 0.5, 1], { base: 2 })) // => [0, 1, 0]
 * ```
 */
export function binaryEntropy(p: TensorLike, options: TableOptions = {}): Value {
  const v = float(p)
  return inBase(add(entr(v), entr(sub(1, v))), options.base)
}

/**
 * The joint entropy H(X1, ..., Xk) of a joint probability table (one axis per variable), normalised over all entries.
 *
 * @param joint - A table of probabilities or counts.
 * @param options - The logarithm base.
 * @returns The joint entropy.
 * @concept entropy
 * @example
 * ```ts
 * import { jointEntropy } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * item(jointEntropy([[1, 1], [1, 1]], { base: 2 })) // => 2
 * ```
 */
export function jointEntropy(joint: TensorLike, options: TableOptions = {}): Value {
  return inBase(sum(entr(table(joint, undefined, 'jointEntropy'))), options.base)
}

/**
 * The conditional entropy H(rest | given) = H(all) - H(given) of a joint table, where `given` names the axes of the
 * conditioning variables and the remaining axes are the variables whose uncertainty is measured.
 *
 * @param joint - A table of probabilities or counts, one axis per variable.
 * @param given - The conditioning axes (default the last axis).
 * @param options - The logarithm base.
 * @returns H(rest | given).
 * @concept entropy
 * @example
 * ```ts
 * import { conditionalEntropy } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * // X is a copy of Y: knowing Y leaves no uncertainty.
 * item(conditionalEntropy([[1, 0], [0, 1]])) // => 0
 * ```
 */
export function conditionalEntropy(
  joint: TensorLike,
  given: number | readonly number[] = -1,
  options: TableOptions = {},
): Value {
  const p = table(joint, undefined, 'conditionalEntropy')
  const rank = p.shape.length
  const axes = (typeof given === 'number' ? [given] : given).map((a) => (a < 0 ? a + rank : a))
  if (axes.some((a) => a < 0 || a >= rank)) throw new WhiteboxError('value', 'conditionalEntropy: axis out of range')
  return inBase(sub(sum(entr(p)), marginalEntropy(p, axes)), options.base)
}

/**
 * The cross-entropy H(p, q) = -sum p log q = H(p) + KL(p || q). Infinite when q vanishes where p does not.
 *
 * @param p - The true probabilities.
 * @param q - The model probabilities.
 * @param options - Base, axis and whether to normalise both inputs.
 * @returns The cross-entropy.
 * @concept cross-entropy
 * @example
 * ```ts
 * import { crossEntropy } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * item(crossEntropy([1, 0], [0.5, 0.5], { base: 2 })) // => 1
 * ```
 */
export function crossEntropy(p: TensorLike, q: TensorLike, options: DiscreteOptions = {}): Value {
  return inBase(neg(sum(xlogy(prepared(p, options), prepared(q, options)), options.axis ?? -1)), options.base)
}

/**
 * The relative entropy (Kullback-Leibler divergence) KL(p || q) = sum p log(p / q) of probability vectors, as
 * scipy.stats.entropy(p, q): non-negative, zero only when p = q, infinite when q vanishes where p does not. For
 * distribution objects use `dists.klDivergence`.
 *
 * @param p - The first probability vector.
 * @param q - The second.
 * @param options - Base, axis and whether to normalise.
 * @returns KL(p || q), differentiable where both are positive.
 * @concept kl-divergence
 * @example
 * ```ts
 * import { relativeEntropy } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * item(relativeEntropy([0.5, 0.5], [0.25, 0.75], { base: 2 })) // => 0.20751874963942185
 * ```
 */
export function relativeEntropy(p: TensorLike, q: TensorLike, options: DiscreteOptions = {}): Value {
  return inBase(sum(relEntr(prepared(p, options), prepared(q, options)), options.axis ?? -1), options.base)
}

/** Options of the Jensen-Shannon divergence. */
export interface JensenShannonOptions extends DiscreteOptions {
  /** Mixture weights of p and q (default [0.5, 0.5]). */
  readonly weights?: readonly [number, number]
}

/**
 * The Jensen-Shannon divergence JS(p, q) = w1 KL(p || m) + w2 KL(q || m) with m = w1 p + w2 q: symmetric (for equal
 * weights), finite and at most log 2.
 *
 * @param p - The first probability vector.
 * @param q - The second.
 * @param options - Base, axis, normalisation and mixture weights.
 * @returns The divergence.
 * @concept jensen-shannon
 * @example
 * ```ts
 * import { jensenShannonDivergence } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * item(jensenShannonDivergence([1, 0], [0, 1], { base: 2 })) // => 1
 * ```
 */
export function jensenShannonDivergence(p: TensorLike, q: TensorLike, options: JensenShannonOptions = {}): Value {
  const [w1, w2] = options.weights ?? [0.5, 0.5]
  const a = prepared(p, options)
  const b = prepared(q, options)
  const m = add(mul(a, w1), mul(b, w2))
  const axis = options.axis ?? -1
  return inBase(add(mul(sum(relEntr(a, m), axis), w1), mul(sum(relEntr(b, m), axis), w2)), options.base)
}

/**
 * The Jensen-Shannon distance sqrt(JS(p, q)) (scipy.spatial.distance.jensenshannon), a metric on distributions.
 *
 * @param p - The first probability vector.
 * @param q - The second.
 * @param options - Base, axis and normalisation.
 * @returns The distance.
 * @concept jensen-shannon
 * @example
 * ```ts
 * import { jensenShannonDistance } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * item(jensenShannonDistance([1, 0], [0, 1], { base: 2 })) // => 1
 * ```
 */
export function jensenShannonDistance(p: TensorLike, q: TensorLike, options: DiscreteOptions = {}): Value {
  return sqrt(jensenShannonDivergence(p, q, options))
}

/**
 * The total variation distance TV(p, q) = (1 / 2) sum |p - q|, the largest difference in probability that the two
 * assign to any event.
 *
 * @param p - The first probability vector.
 * @param q - The second.
 * @param options - Axis and normalisation (the base is ignored).
 * @returns The distance in [0, 1].
 * @concept total-variation
 * @example
 * ```ts
 * import { totalVariation } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * item(totalVariation([0.5, 0.5], [1, 0])) // => 0.5
 * ```
 */
export function totalVariation(p: TensorLike, q: TensorLike, options: DiscreteOptions = {}): Value {
  return mul(sum(abs(sub(prepared(p, options), prepared(q, options))), options.axis ?? -1), 0.5)
}

/**
 * The mutual information I(X; Y) = KL(p(x, y) || p(x) p(y)) of a two-way table of probabilities or counts (for a
 * contingency table of counts this is sklearn.metrics.mutual_info_score).
 *
 * @param joint - A `[nx, ny]` table.
 * @param options - The logarithm base.
 * @returns I(X; Y) >= 0.
 * @concept mutual-information
 * @example
 * ```ts
 * import { mutualInformation } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * item(mutualInformation([[1, 0], [0, 1]], { base: 2 })) // => 1
 * ```
 */
export function mutualInformation(joint: TensorLike, options: TableOptions = {}): Value {
  const p = table(joint, 2, 'mutualInformation')
  return inBase(sum(relEntr(p, mul(sum(p, 1, true), sum(p, 0, true)))), options.base)
}

/**
 * The conditional mutual information I(X; Y | Z) = sum p(x, y, z) log(p(z) p(x, y, z) / (p(x, z) p(y, z))) of a
 * three-way table.
 *
 * @param joint - A `[nx, ny, nz]` table of probabilities or counts.
 * @param options - The logarithm base.
 * @returns I(X; Y | Z) >= 0.
 * @concept mutual-information
 * @example
 * ```ts
 * import { conditionalMutualInformation } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * // X = Y = Z: once Z is known, X tells nothing more about Y.
 * item(conditionalMutualInformation([[[1, 0], [0, 0]], [[0, 0], [0, 1]]])) // => 0
 * ```
 */
export function conditionalMutualInformation(joint: TensorLike, options: TableOptions = {}): Value {
  const p = table(joint, 3, 'conditionalMutualInformation')
  const pz = sum(p, [0, 1], true)
  const positive = greater(pz, 0)
  const q = div(mul(sum(p, 1, true), sum(p, 0, true)), where(positive, pz, 1))
  return inBase(sum(relEntr(p, q)), options.base)
}

/**
 * The interaction information (co-information) of the k variables of a joint table, by inclusion-exclusion over the
 * marginal entropies: I(X1; ...; Xk) = -sum over non-empty subsets T of (-1)^|T| H(X_T). For two variables it is the
 * mutual information; for three it is I(X; Y) - I(X; Y | Z) (McGill's sign: negative means synergy, as for XOR).
 *
 * @param joint - A table with one axis per variable (rank 2 or more).
 * @param options - The logarithm base.
 * @returns The co-information (of either sign).
 * @concept mutual-information
 * @example
 * ```ts
 * import { interactionInformation } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * // Z = X xor Y with fair independent bits: pure synergy of one bit.
 * Math.round(item(interactionInformation([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], { base: 2 })) * 1e9) / 1e9 // => -1
 * ```
 */
export function interactionInformation(joint: TensorLike, options: TableOptions = {}): Value {
  const p = table(joint, undefined, 'interactionInformation')
  const rank = p.shape.length
  if (rank < 2) throw new WhiteboxError('shape', 'interactionInformation: needs at least two variables')
  let total: Value = float(0)
  for (let mask = 1; mask < 1 << rank; mask++) {
    const keep = Array.from({ length: rank }, (_, i) => i).filter((i) => (mask >> i) & 1)
    const h = marginalEntropy(p, keep)
    total = keep.length % 2 === 1 ? add(total, h) : sub(total, h)
  }
  return inBase(total, options.base)
}

/**
 * The pointwise mutual information log(p(x, y) / (p(x) p(y))) of every cell of a two-way table (-inf where p(x, y) = 0).
 *
 * @param joint - A `[nx, ny]` table of probabilities or counts.
 * @param options - The logarithm base.
 * @returns The PMI table.
 * @concept mutual-information
 * @example
 * ```ts
 * import { pointwiseMutualInformation } from 'whitebox-engine/information'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(pointwiseMutualInformation([[1, 0], [0, 1]], { base: 2 })) // => [[1, -Infinity], [-Infinity, 1]]
 * ```
 */
export function pointwiseMutualInformation(joint: TensorLike, options: TableOptions = {}): Value {
  const p = table(joint, 2, 'pointwiseMutualInformation')
  const positive = greater(p, 0)
  const ratio = div(where(positive, p, 1), where(positive, mul(sum(p, 1, true), sum(p, 0, true)), 1))
  return inBase(where(positive, log(ratio), Number.NEGATIVE_INFINITY), options.base)
}

/**
 * The normalised pointwise mutual information PMI(x, y) / (-log p(x, y)) of every cell (Bouma 2009): -1 for cells that
 * never occur, 0 for independence and 1 for complete co-occurrence.
 *
 * @param joint - A `[nx, ny]` table of probabilities or counts.
 * @returns The NPMI table in [-1, 1] (independent of the base).
 * @concept mutual-information
 * @example
 * ```ts
 * import { normalisedPointwiseMutualInformation } from 'whitebox-engine/information'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(normalisedPointwiseMutualInformation([[1, 0], [0, 1]])) // => [[1, -1], [-1, 1]]
 * ```
 */
export function normalisedPointwiseMutualInformation(joint: TensorLike): Value {
  const p = table(joint, 2, 'normalisedPointwiseMutualInformation')
  const inside = greater(p, 0)
  const certain = equal(p, 1)
  const denominator = neg(log(where(inside, where(certain, 0.5, p), 0.5)))
  const npmi = div(pointwiseMutualInformation(p), denominator)
  return where(inside, where(certain, 1, npmi), -1)
}

/**
 * The information gain of a split, H(classes) - sum over children of (n_c / n) H(classes in child c): the mutual
 * information between the class and the branch taken, as used to grow decision trees (ID3, C4.5).
 *
 * @param counts - A `[children, classes]` table of class counts in each child.
 * @param options - The logarithm base.
 * @returns The gain >= 0.
 * @concept information-gain
 * @example
 * ```ts
 * import { informationGain } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * // A perfect split of a balanced two-class node gains one bit.
 * item(informationGain([[5, 0], [0, 5]], { base: 2 })) // => 1
 * ```
 */
export function informationGain(counts: TensorLike, options: TableOptions = {}): Value {
  const p = table(counts, 2, 'informationGain')
  const classes = marginalEntropy(p, [1])
  const children = marginalEntropy(p, [0])
  return inBase(sub(classes, sub(sum(entr(p)), children)), options.base)
}

/**
 * The gain ratio of C4.5: the information gain divided by the split information H(children), which penalises splits into
 * many small children; 0 when the split information is 0.
 *
 * @param counts - A `[children, classes]` table of class counts in each child.
 * @returns The gain ratio in [0, 1] (independent of the base).
 * @concept information-gain
 * @example
 * ```ts
 * import { gainRatio } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * item(gainRatio([[5, 0], [0, 5]])) // => 1
 * ```
 */
export function gainRatio(counts: TensorLike): Value {
  const p = table(counts, 2, 'gainRatio')
  const split = marginalEntropy(p, [0])
  const positive = greater(split, 0)
  return where(positive, div(informationGain(p), where(positive, split, 1)), 0)
}
