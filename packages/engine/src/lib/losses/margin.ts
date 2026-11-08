/**
 * Margin losses on a score f and a label y in {-1, +1}: the hinge (SVM) and squared hinge, logistic, exponential
 * (AdaBoost's), modified Huber and perceptron losses, plus the Crammer-Singer and Weston-Watkins multi-class hinge
 * losses.
 */

import type { Reduction } from '../base/index.ts'
import { softplus } from '../numerics/special/index.ts'
import {
  add,
  cast,
  exp,
  expandDims,
  less,
  max,
  mul,
  neg,
  oneHot,
  pow,
  promoteFloat,
  relu,
  square,
  squeeze,
  sub,
  sum,
  type TensorLike,
  take,
  takeAlongAxis,
  type Value,
  where,
} from '../tensor/index.ts'
import { checkScalar, type LossOptions, reduce } from './_reduce.ts'

/** The margins y f of binary scores with labels in {-1, +1}. */
const margins = (scores: TensorLike, labels: TensorLike): Value => {
  const [f, y] = promoteFloat(scores, labels) as [Value, Value]
  return mul(y, f)
}

/**
 * Hinge loss max(0, margin - y f) for labels in {-1, +1} (optax.losses.hinge_loss; the SVM loss).
 *
 * @param scores - Decision values f.
 * @param labels - Labels in {-1, +1}.
 * @param options - margin (default 1), reduction and weights.
 * @returns The loss, differentiable away from the hinge.
 * @concept losses
 * @example
 * ```ts
 * import { hinge } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(hinge([0.25, 2], [1, -1], { reduction: 'sum' })) // => 3.75
 * ```
 */
export function hinge(
  scores: TensorLike,
  labels: TensorLike,
  options: LossOptions & { readonly margin?: number } = {},
): Value {
  checkScalar('hinge', 'margin', options.margin, 'nonNegative')
  return reduce(relu(sub(options.margin ?? 1, margins(scores, labels))), options)
}

/**
 * Squared hinge loss max(0, margin - y f)^2 for labels in {-1, +1} (the L2-SVM loss).
 *
 * @param scores - Decision values f.
 * @param labels - Labels in {-1, +1}.
 * @param options - margin (default 1), reduction and weights.
 * @returns The loss, differentiable.
 * @example
 * ```ts
 * import { squaredHinge } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(squaredHinge([0.5], [1])) // => 0.25
 * ```
 */
export function squaredHinge(
  scores: TensorLike,
  labels: TensorLike,
  options: LossOptions & { readonly margin?: number } = {},
): Value {
  checkScalar('squaredHinge', 'margin', options.margin, 'nonNegative')
  return reduce(square(relu(sub(options.margin ?? 1, margins(scores, labels)))), options)
}

/**
 * Logistic loss log(1 + exp(-y f)) for labels in {-1, +1} (torch.nn.functional.soft_margin_loss; logistic
 * regression's loss in margin form), computed stably.
 *
 * @param scores - Decision values f.
 * @param labels - Labels in {-1, +1}.
 * @param options - Reduction and weights.
 * @returns The loss, differentiable.
 * @example
 * ```ts
 * import { logistic } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(logistic([0], [1])) === Math.LN2 // => true
 * ```
 */
export function logistic(scores: TensorLike, labels: TensorLike, options: LossOptions = {}): Value {
  return reduce(softplus(neg(margins(scores, labels))), options)
}

/**
 * Exponential loss exp(-y f) for labels in {-1, +1}, the loss AdaBoost minimises stagewise.
 *
 * @param scores - Decision values f.
 * @param labels - Labels in {-1, +1}.
 * @param options - Reduction and weights.
 * @returns The loss, differentiable.
 * @example
 * ```ts
 * import { exponential } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(exponential([0], [1])) // => 1
 * ```
 */
export function exponential(scores: TensorLike, labels: TensorLike, options: LossOptions = {}): Value {
  return reduce(exp(neg(margins(scores, labels))), options)
}

/**
 * Modified Huber loss (Zhang, 2004; sklearn's SGDClassifier `modified_huber`): max(0, 1 - z)^2 for z = y f >= -1 and
 * -4 z below, a smooth hinge that is robust to outliers and yields probability estimates.
 *
 * @param scores - Decision values f.
 * @param labels - Labels in {-1, +1}.
 * @param options - Reduction and weights.
 * @returns The loss, differentiable.
 * @example
 * ```ts
 * import { modifiedHuber } from 'omai-engine/losses'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(modifiedHuber([-2, 0], [1, 1], { reduction: 'none' })) // => [8, 1]
 * ```
 */
export function modifiedHuber(scores: TensorLike, labels: TensorLike, options: LossOptions = {}): Value {
  const z = margins(scores, labels)
  return reduce(where(less(z, -1), mul(z, -4), square(relu(sub(1, z)))), options)
}

/**
 * Perceptron loss max(0, -y f) for labels in {-1, +1} (optax.losses.perceptron_loss).
 *
 * @param scores - Decision values f.
 * @param labels - Labels in {-1, +1}.
 * @param options - Reduction and weights.
 * @returns The loss, differentiable away from zero.
 * @example
 * ```ts
 * import { perceptron } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(perceptron([0.5, -2], [1, 1], { reduction: 'sum' })) // => 2
 * ```
 */
export function perceptron(scores: TensorLike, labels: TensorLike, options: LossOptions = {}): Value {
  return reduce(relu(neg(margins(scores, labels))), options)
}

/** Options of the multi-class margin losses. */
export interface MulticlassMarginOptions {
  /** The margin (default 1). */
  readonly margin?: number
  readonly reduction?: Reduction
}

/**
 * Crammer-Singer multi-class hinge loss max(0, margin + max_{j != y} s_j - s_y) (optax.losses.multiclass_hinge_loss,
 * sklearn.metrics.hinge_loss for several classes): only the strongest wrong class is penalised.
 *
 * @param scores - Class scores `[..., C]`.
 * @param target - Class indices `[...]`.
 * @param options - margin (default 1) and reduction.
 * @returns The loss, differentiable away from ties.
 * @example
 * ```ts
 * import { crammerSinger } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(crammerSinger([[2, 1.5, 0]], [0])) // => 0.5
 * ```
 */
export function crammerSinger(scores: TensorLike, target: TensorLike, options: MulticlassMarginOptions = {}): Value {
  checkScalar('crammerSinger', 'margin', options.margin, 'nonNegative')
  const [s] = promoteFloat(scores) as [Value]
  const c = s.shape[s.shape.length - 1] ?? 0
  const index = cast(promoteFloat(target)[0] as Value, 'int32')
  const hot = oneHot(index, c)
  const own = squeeze(takeAlongAxis(s, expandDims(index, -1), -1), -1)
  const strongest = max(add(s, mul(sub(1, hot), options.margin ?? 1)), -1)
  return reduce(sub(strongest, own), { reduction: options.reduction ?? 'mean' })
}

/**
 * Weston-Watkins multi-class hinge loss sum_{j != y} max(0, margin - s_y + s_j)^p / C
 * (torch.nn.functional.multi_margin_loss): every wrong class within the margin is penalised.
 *
 * @param scores - Class scores `[N, C]`.
 * @param target - Class indices `[N]`.
 * @param options - margin (default 1), p (1 or 2, default 1), classWeight `[C]` (scales a sample by its target's
 *   weight) and reduction.
 * @returns The loss, differentiable away from the hinges.
 * @example
 * ```ts
 * import { westonWatkins } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(westonWatkins([[1, 1.5]], [0])) // => 0.75
 * ```
 */
export function westonWatkins(
  scores: TensorLike,
  target: TensorLike,
  options: MulticlassMarginOptions & { readonly p?: 1 | 2; readonly classWeight?: TensorLike } = {},
): Value {
  checkScalar('westonWatkins', 'margin', options.margin, 'nonNegative')
  const [s] = promoteFloat(scores) as [Value]
  const c = s.shape[s.shape.length - 1] ?? 0
  const index = cast(promoteFloat(target)[0] as Value, 'int32')
  const hot = oneHot(index, c)
  const own = takeAlongAxis(s, expandDims(index, -1), -1)
  let violations = relu(add(sub(s, own), options.margin ?? 1))
  if ((options.p ?? 1) === 2) violations = pow(violations, 2)
  let terms = mul(sum(mul(violations, sub(1, hot)), -1), 1 / c)
  if (options.classWeight !== undefined) terms = mul(terms, take(promoteFloat(options.classWeight)[0] as Value, index))
  return reduce(terms, { reduction: options.reduction ?? 'mean' })
}
