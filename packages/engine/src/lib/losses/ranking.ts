/**
 * Learning-to-rank losses over lists of scored items: pairwise losses (margin ranking, RankNet, pairwise hinge, BPR for
 * implicit feedback, and LambdaRank, weighted by the NDCG change of each swap) and listwise ones (ListNet, the ranking
 * softmax, ListMLE's Plackett-Luce likelihood and ApproxNDCG, a smooth surrogate of NDCG).
 */

import { MdmlError, type Reduction } from '../base/index.ts'
import { logSigmoid, softplus } from '../numerics/special/index.ts'
import {
  abs,
  add,
  arange,
  argsort,
  cast,
  cumsum,
  div,
  exp,
  expandDims,
  greater,
  lessEqual,
  log,
  log2,
  logSoftmax,
  max,
  mul,
  neg,
  pow,
  promoteFloat,
  relu,
  sigmoid,
  softmax,
  sort,
  stopGradient,
  sub,
  sum,
  type TensorLike,
  takeAlongAxis,
  type Value,
  where,
} from '../tensor/index.ts'
import { checkScalar, type LossOptions, reduce } from './_reduce.ts'

/** Options shared by the learning-to-rank losses: lists lie along the last axis. */
export interface RankingOptions {
  readonly reduction?: Reduction
  /** Valid items `[..., L]` (padding is false); default all valid. */
  readonly mask?: TensorLike
}

/** The gain of a relevance grade: 2^rel - 1 ('exponential', the usual NDCG gain) or rel ('linear'). */
export type Gain = 'exponential' | 'linear'

const listInputs = (scores: TensorLike, relevance: TensorLike, mask?: TensorLike): [Value, Value, Value] => {
  const [s, r] = promoteFloat(scores, relevance) as [Value, Value]
  if (s.shape.length === 0) throw new MdmlError('shape', 'ranking losses need lists along the last axis')
  const m = mask === undefined ? add(mul(r, 0), 1) : cast(promoteFloat(mask)[0] as Value, 'float64')
  return [s, r, m]
}

/** The pairs (i, j) of a list with rel_i > rel_j, both valid, as a 0/1 matrix `[..., L, L]`. */
const preferred = (r: Value, m: Value): Value =>
  mul(cast(greater(expandDims(r, -1), expandDims(r, -2)), 'float64'), mul(expandDims(m, -1), expandDims(m, -2)))

/** Score differences s_i - s_j `[..., L, L]`. */
const differences = (s: Value): Value => sub(expandDims(s, -1), expandDims(s, -2))

/** Reduce pairwise terms: 'mean' over the preferred pairs, 'sum' over them, 'none' the masked `[..., L, L]` matrix. */
const reducePairs = (terms: Value, pairs: Value, reduction: Reduction): Value =>
  reduction === 'none' ? mul(terms, pairs) : reduce(terms, { reduction, weight: pairs })

/**
 * Margin ranking loss max(0, -y (x1 - x2) + margin) (torch.nn.functional.margin_ranking_loss): y = 1 means x1 should
 * rank above x2.
 *
 * @param x1 - First scores.
 * @param x2 - Second scores.
 * @param y - +1 or -1 per pair.
 * @param options - margin (default 0), reduction and weights.
 * @returns The loss, differentiable away from the hinge.
 * @example
 * ```ts
 * import { marginRanking } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(marginRanking([1], [2], [1], { margin: 0.5 })) // => 1.5
 * ```
 */
export function marginRanking(
  x1: TensorLike,
  x2: TensorLike,
  y: TensorLike,
  options: LossOptions & { readonly margin?: number } = {},
): Value {
  const [a, b, t] = promoteFloat(x1, x2, y) as [Value, Value, Value]
  return reduce(relu(add(mul(neg(t), sub(a, b)), options.margin ?? 0)), options)
}

/**
 * RankNet (Burges et al., 2005): the logistic loss log(1 + exp(-sigma (s_i - s_j))) over every pair of a list with
 * rel_i > rel_j, the cross-entropy of the pairwise preference probability.
 *
 * @param scores - Scores `[..., L]`.
 * @param relevance - Relevance grades `[..., L]`.
 * @param options - sigma (default 1), mask and reduction ('mean' over preferred pairs).
 * @returns The loss, differentiable in the scores.
 * @concept learning-to-rank
 * @example
 * ```ts
 * import { rankNet } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(rankNet([0, 0], [1, 0])) === Math.LN2 // => true
 * ```
 */
export function rankNet(
  scores: TensorLike,
  relevance: TensorLike,
  options: RankingOptions & { readonly sigma?: number } = {},
): Value {
  const [s, r, m] = listInputs(scores, relevance, options.mask)
  const terms = softplus(mul(differences(s), -(options.sigma ?? 1)))
  return reducePairs(terms, preferred(r, m), options.reduction ?? 'mean')
}

/**
 * Pairwise hinge loss max(0, margin - (s_i - s_j)) over every pair of a list with rel_i > rel_j (the RankSVM loss).
 *
 * @param scores - Scores `[..., L]`.
 * @param relevance - Relevance grades `[..., L]`.
 * @param options - margin (default 1), mask and reduction ('mean' over preferred pairs).
 * @returns The loss, differentiable away from the hinge.
 * @example
 * ```ts
 * import { pairwiseHinge } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(pairwiseHinge([0.25, 0], [1, 0])) // => 0.75
 * ```
 */
export function pairwiseHinge(
  scores: TensorLike,
  relevance: TensorLike,
  options: RankingOptions & { readonly margin?: number } = {},
): Value {
  const [s, r, m] = listInputs(scores, relevance, options.mask)
  const terms = relu(sub(options.margin ?? 1, differences(s)))
  return reducePairs(terms, preferred(r, m), options.reduction ?? 'mean')
}

/**
 * Bayesian personalised ranking (Rendle et al., 2009): -log sigma(s_pos - s_neg), for implicit feedback where each
 * observed item should outscore sampled unobserved ones. Negatives broadcast against positives (`[N]` against
 * `[N, K]` after expanding).
 *
 * @param positive - Scores of observed items.
 * @param negative - Scores of sampled negatives (broadcastable).
 * @param options - Reduction and weights.
 * @returns The loss, differentiable.
 * @example
 * ```ts
 * import { bpr } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(bpr([1], [1])) === Math.LN2 // => true
 * ```
 */
export function bpr(positive: TensorLike, negative: TensorLike, options: LossOptions = {}): Value {
  const [p, n] = promoteFloat(positive, negative) as [Value, Value]
  return reduce(neg(logSigmoid(sub(p, n))), options)
}

/** The gains of relevance grades. */
const gains = (r: Value, gain: Gain): Value => (gain === 'linear' ? r : sub(pow(2, r), 1))

/** 1 / log2(1 + rank) for 1-based ranks, zero past the cutoff k. */
const discounts = (ranks: Value, k: number | undefined): Value => {
  const d = div(1, log2(add(ranks, 1)))
  return k === undefined ? d : where(lessEqual(ranks, k), d, 0)
}

/** The ideal DCG of each list `[...]`: gains sorted in decreasing order, discounted (and truncated at k). */
const idealDcg = (g: Value, m: Value, k: number | undefined): Value => {
  const sorted = sort(mul(g, m), -1, true)
  const n = sorted.shape[sorted.shape.length - 1] ?? 0
  return sum(mul(sorted, discounts(add(arange(n), 1), k)), -1)
}

/** 1-based ranks of the scores in each list, largest first (padding ranked last). */
const ranksOf = (s: Value, m: Value): Value => {
  const masked = where(greater(m, 0), s, Number.NEGATIVE_INFINITY)
  return add(cast(argsort(argsort(stopGradient(masked), -1, true), -1), 'float64'), 1)
}

/**
 * LambdaRank (Burges et al., 2006): RankNet's pairwise logistic loss with each pair weighted by |delta NDCG|, the change
 * in NDCG(@k) from swapping the two items in the current ranking, so that gradients focus on the top of the list. The
 * weights are constants (no gradient flows through the ranking).
 *
 * @param scores - Scores `[..., L]`.
 * @param relevance - Relevance grades `[..., L]`.
 * @param options - sigma (default 1), k (NDCG cutoff), gain, mask and reduction ('mean' averages each list's weighted
 *   sum over lists, 'none' returns the per-list sums).
 * @returns The loss, differentiable in the scores.
 * @example
 * ```ts
 * import { lambdaRank } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(lambdaRank([0, 0], [1, 0])) - Math.LN2 * (1 - 1 / Math.log2(3))) < 1e-15 // => true
 * ```
 */
export function lambdaRank(
  scores: TensorLike,
  relevance: TensorLike,
  options: RankingOptions & { readonly sigma?: number; readonly k?: number; readonly gain?: Gain } = {},
): Value {
  const [s, r, m] = listInputs(scores, relevance, options.mask)
  const g = gains(r, options.gain ?? 'exponential')
  const d = discounts(ranksOf(s, m), options.k)
  const ideal = idealDcg(g, m, options.k)
  const swap = mul(abs(differences(g)), abs(differences(d)))
  const safeIdeal = where(greater(ideal, 0), ideal, 1)
  const delta = div(swap, expandDims(expandDims(safeIdeal, -1), -1))
  const terms = mul(mul(softplus(mul(differences(s), -(options.sigma ?? 1))), delta), preferred(r, m))
  return reduce(sum(terms, [-2, -1]), { reduction: options.reduction ?? 'mean' })
}

/** Log-softmax over the valid items of each list (padding at -inf, masked to zero by the callers). */
const maskedLogSoftmax = (x: Value, m: Value): Value =>
  logSoftmax(where(greater(m, 0), x, Number.NEGATIVE_INFINITY), -1)

/**
 * ListNet (Cao et al., 2007), top-one form: the cross-entropy between softmax(relevance / T) and softmax(scores).
 *
 * @param scores - Scores `[..., L]`.
 * @param relevance - Relevance grades `[..., L]`.
 * @param options - temperature T of the target distribution (default 1), mask and reduction (over lists).
 * @returns The loss, differentiable in the scores.
 * @example
 * ```ts
 * import { listNet } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(listNet([0, 0], [1, 1])) - Math.LN2) < 1e-15 // => true
 * ```
 */
export function listNet(
  scores: TensorLike,
  relevance: TensorLike,
  options: RankingOptions & { readonly temperature?: number } = {},
): Value {
  checkScalar('listNet', 'temperature', options.temperature, 'positive')
  const [s, r, m] = listInputs(scores, relevance, options.mask)
  const target = mul(softmax(where(greater(m, 0), div(r, options.temperature ?? 1), Number.NEGATIVE_INFINITY), -1), m)
  const logp = where(greater(m, 0), maskedLogSoftmax(s, m), 0)
  return reduce(neg(sum(mul(target, logp), -1)), { reduction: options.reduction ?? 'mean' })
}

/**
 * Ranking softmax loss (optax.losses.ranking_softmax_loss): -sum_i rel_i log softmax(s)_i, the listwise
 * cross-entropy with unnormalised relevance weights.
 *
 * @param scores - Scores `[..., L]`.
 * @param relevance - Relevance labels `[..., L]`.
 * @param options - Mask and reduction (over lists).
 * @returns The loss, differentiable in the scores.
 * @example
 * ```ts
 * import { softmaxRanking } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(softmaxRanking([0, 0], [1, 0])) === Math.LN2 // => true
 * ```
 */
export function softmaxRanking(scores: TensorLike, relevance: TensorLike, options: RankingOptions = {}): Value {
  const [s, r, m] = listInputs(scores, relevance, options.mask)
  const logp = where(greater(m, 0), maskedLogSoftmax(s, m), 0)
  return reduce(neg(sum(mul(mul(r, m), logp), -1)), { reduction: options.reduction ?? 'mean' })
}

/**
 * ListMLE (Xia et al., 2008): the Plackett-Luce negative log-likelihood of the order given by the relevance,
 * sum_i [log sum_{j >= i} exp(s_pi(j)) - s_pi(i)], with ties broken by position (a stable sort).
 *
 * @param scores - Scores `[..., L]`.
 * @param relevance - Relevance grades `[..., L]` defining the target order.
 * @param options - Mask (padding is placed last and skipped) and reduction (over lists).
 * @returns The loss, differentiable in the scores.
 * @example
 * ```ts
 * import { listMle } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(listMle([0, 0], [1, 0])) === Math.LN2 // => true
 * ```
 */
export function listMle(scores: TensorLike, relevance: TensorLike, options: RankingOptions = {}): Value {
  const [s, r, m] = listInputs(scores, relevance, options.mask)
  // Padding sorts last because its key is -inf.
  const key = where(greater(m, 0), r, Number.NEGATIVE_INFINITY)
  const order = argsort(key, -1, true)
  const ordered = takeAlongAxis(s, order, -1)
  const valid = takeAlongAxis(m, order, -1)
  const shift = stopGradient(max(where(greater(valid, 0), ordered, Number.NEGATIVE_INFINITY), -1, true))
  const tail = cumsum(mul(exp(sub(ordered, shift)), valid), -1, true)
  const safeTail = where(greater(valid, 0), tail, 1)
  const terms = mul(sub(add(log(safeTail), shift), ordered), valid)
  return reduce(sum(terms, -1), { reduction: options.reduction ?? 'mean' })
}

/**
 * ApproxNDCG (Qin, Liu and Li, 2010; TF-Ranking's ApproxNDCGLoss): minus the NDCG computed with smooth ranks
 * 1 + sum_{j != i} sigma(alpha (s_j - s_i)), a differentiable surrogate of NDCG itself. Lists without relevant items
 * contribute zero.
 *
 * @param scores - Scores `[..., L]`.
 * @param relevance - Relevance grades `[..., L]`.
 * @param options - alpha (sharpness of the smooth rank, default 10), gain, mask and reduction (over lists).
 * @returns The loss (in [-1, 0]), differentiable in the scores.
 * @example
 * ```ts
 * import { approxNdcg } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(approxNdcg([100, 0], [1, 0])) // => -1
 * ```
 */
export function approxNdcg(
  scores: TensorLike,
  relevance: TensorLike,
  options: RankingOptions & { readonly alpha?: number; readonly gain?: Gain } = {},
): Value {
  checkScalar('approxNdcg', 'alpha', options.alpha, 'positive')
  const [s, r, m] = listInputs(scores, relevance, options.mask)
  const alpha = options.alpha ?? 10
  const pairMask = mul(expandDims(m, -1), expandDims(m, -2))
  // sigma(alpha (s_j - s_i)) summed over j != i; the diagonal contributes sigma(0) = 1/2, removed below.
  const smooth = sum(mul(sigmoid(mul(differences(s), -alpha)), pairMask), -1)
  const ranks = add(sub(smooth, mul(m, 0.5)), 1)
  const g = mul(gains(r, options.gain ?? 'exponential'), m)
  const dcg = sum(div(g, log2(add(ranks, 1))), -1)
  const ideal = idealDcg(g, m, undefined)
  const ndcg = where(greater(ideal, 0), div(dcg, where(greater(ideal, 0), ideal, 1)), 0)
  return reduce(neg(ndcg), { reduction: options.reduction ?? 'mean' })
}
