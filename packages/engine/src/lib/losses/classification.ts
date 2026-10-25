/**
 * Classification losses, torch.nn.functional style: binary cross-entropy on logits (computed stably) and on
 * probabilities, softmax cross-entropy with class-index or probability targets and label smoothing options, negative
 * log-likelihood of log-probabilities, the multi-label soft-margin loss, and sigmoid and softmax focal losses, which
 * down-weight easy examples.
 */

import { MdmlError } from '../base/index.ts'
import { logSigmoid } from '../numerics/special/index.ts'
import {
  add,
  cast,
  div,
  equal,
  exp,
  expandDims,
  log,
  logSoftmax,
  maximum,
  mul,
  neg,
  notEqual,
  pow,
  promoteFloat,
  sigmoid,
  squeeze,
  sub,
  sum,
  type TensorLike,
  take,
  takeAlongAxis,
  type Value,
  where,
} from '../tensor/index.ts'
import { type LossOptions, type Reduction, reduce } from './_reduce.ts'

/**
 * Binary cross-entropy on logits, -[y log sigma(x) + (1 - y) log(1 - sigma(x))], computed stably from the logits
 * (torch.nn.functional.binary_cross_entropy_with_logits).
 *
 * @param logits - Logits.
 * @param target - Targets in [0, 1].
 * @param options - posWeight (weight of positive terms, broadcast over the last axis), labelSmoothing (targets become
 *   y (1 - eps) + eps / 2, as Keras and optax), reduction and weights.
 * @returns The loss, differentiable.
 * @concept losses
 * @example
 * ```ts
 * import { binaryCrossEntropyWithLogits } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(binaryCrossEntropyWithLogits([0], [1])) === Math.LN2 // => true
 * ```
 */
export function binaryCrossEntropyWithLogits(
  logits: TensorLike,
  target: TensorLike,
  options: LossOptions & { readonly posWeight?: TensorLike; readonly labelSmoothing?: number } = {},
): Value {
  const [x, t] = promoteFloat(logits, target) as [Value, Value]
  const y = smoothBinary(t, options.labelSmoothing)
  const positive =
    options.posWeight === undefined ? logSigmoid(x) : mul(logSigmoid(x), promoteFloat(options.posWeight)[0] as Value)
  return reduce(neg(add(mul(y, positive), mul(sub(1, y), logSigmoid(neg(x))))), options)
}

const smoothBinary = (y: Value, eps: number | undefined): Value =>
  eps === undefined || eps === 0 ? y : add(mul(y, 1 - eps), eps / 2)

/**
 * Binary cross-entropy on probabilities, with log clamped at -100 as torch does
 * (torch.nn.functional.binary_cross_entropy).
 *
 * @param probs - Probabilities.
 * @param target - Targets in [0, 1].
 * @param options - labelSmoothing (as {@link binaryCrossEntropyWithLogits}), reduction and weights.
 * @returns The loss, differentiable.
 * @example
 * ```ts
 * import { binaryCrossEntropy } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(binaryCrossEntropy([0.5], [0])) === Math.LN2 // => true
 * ```
 */
export function binaryCrossEntropy(
  probs: TensorLike,
  target: TensorLike,
  options: LossOptions & { readonly labelSmoothing?: number } = {},
): Value {
  const [p, t] = promoteFloat(probs, target) as [Value, Value]
  const y = smoothBinary(t, options.labelSmoothing)
  // torch clamps each log at -100 so that a probability of exactly 0 or 1 gives a finite loss.
  const clampedLog = (v: Value): Value => maximum(log(v), -100)
  return reduce(neg(add(mul(y, clampedLog(p)), mul(sub(1, y), clampedLog(sub(1, p))))), options)
}

/**
 * Sigmoid focal loss (Lin et al., 2017; torchvision.ops.sigmoid_focal_loss): the binary cross-entropy scaled by
 * (1 - p_t)^gamma, which down-weights easy examples, and by alpha_t = alpha y + (1 - alpha)(1 - y).
 *
 * @param logits - Logits.
 * @param target - Targets in [0, 1].
 * @param options - alpha (default 0.25; negative disables it), gamma (default 2), reduction and weights.
 * @returns The loss, differentiable.
 * @concept losses
 * @example
 * ```ts
 * import { focal } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(focal([0], [1], { alpha: -1, gamma: 1 })) === Math.LN2 / 2 // => true
 * ```
 */
export function focal(
  logits: TensorLike,
  target: TensorLike,
  options: LossOptions & { readonly alpha?: number; readonly gamma?: number } = {},
): Value {
  const [x, y] = promoteFloat(logits, target) as [Value, Value]
  const alpha = options.alpha ?? 0.25
  const gamma = options.gamma ?? 2
  const ce = neg(add(mul(y, logSigmoid(x)), mul(sub(1, y), logSigmoid(neg(x)))))
  const p = sigmoid(x)
  const pt = add(mul(p, y), mul(sub(1, p), sub(1, y)))
  let terms = gamma === 0 ? ce : mul(ce, pow(sub(1, pt), gamma))
  if (alpha >= 0) terms = mul(terms, add(mul(y, alpha), mul(sub(1, y), 1 - alpha)))
  return reduce(terms, options)
}

/**
 * Multi-class (softmax) focal loss: -alpha_y (1 - p_y)^gamma log p_y from logits along the last axis, with integer
 * class targets.
 *
 * @param logits - Logits `[..., C]`.
 * @param target - Class indices `[...]`.
 * @param options - gamma (default 2), alpha (per-class weights `[C]`), ignoreIndex and reduction ('mean' averages over
 *   the kept targets).
 * @returns The loss, differentiable in the logits.
 * @example
 * ```ts
 * import { categoricalFocal } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(categoricalFocal([[0, 0]], [1], { gamma: 1 })) === Math.LN2 / 2 // => true
 * ```
 */
export function categoricalFocal(
  logits: TensorLike,
  target: TensorLike,
  options: {
    readonly gamma?: number
    readonly alpha?: TensorLike
    readonly ignoreIndex?: number
    readonly reduction?: Reduction
  } = {},
): Value {
  const [z] = promoteFloat(logits) as [Value]
  const gamma = options.gamma ?? 2
  const t = promoteFloat(target)[0] as Value
  const ignore = options.ignoreIndex
  const index = cast(ignore === undefined ? t : where(equal(t, ignore), 0, t), 'int32')
  const logp = squeeze(takeAlongAxis(logSoftmax(z, -1), expandDims(index, -1), -1), -1)
  let terms = neg(gamma === 0 ? logp : mul(pow(sub(1, exp(logp)), gamma), logp))
  if (options.alpha !== undefined) terms = mul(terms, take(promoteFloat(options.alpha)[0] as Value, index))
  if (ignore === undefined) return reduce(terms, { reduction: options.reduction ?? 'mean' })
  const keep = cast(notEqual(t, ignore), 'float64')
  terms = mul(terms, keep)
  return (options.reduction ?? 'mean') === 'mean' ? div(sum(terms), sum(keep)) : reduce(terms, options)
}

/**
 * Multi-label soft-margin loss (torch.nn.functional.multilabel_soft_margin_loss): the binary cross-entropy on logits
 * averaged over the classes of each sample, then reduced over samples.
 *
 * @param logits - Logits `[N, C]`.
 * @param target - Binary targets `[N, C]`.
 * @param options - classWeight `[C]` and reduction (over samples).
 * @returns The loss, differentiable.
 * @example
 * ```ts
 * import { multiLabelSoftMargin } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(multiLabelSoftMargin([[0, 0]], [[1, 0]])) === Math.LN2 // => true
 * ```
 */
export function multiLabelSoftMargin(
  logits: TensorLike,
  target: TensorLike,
  options: { readonly classWeight?: TensorLike; readonly reduction?: Reduction } = {},
): Value {
  const [x, y] = promoteFloat(logits, target) as [Value, Value]
  let terms = neg(add(mul(y, logSigmoid(x)), mul(sub(1, y), logSigmoid(neg(x)))))
  if (options.classWeight !== undefined) terms = mul(terms, promoteFloat(options.classWeight)[0] as Value)
  const c = x.shape[x.shape.length - 1] ?? 1
  return reduce(div(sum(terms, -1), c), { reduction: options.reduction ?? 'mean' })
}

/** Options of {@link crossEntropy}. */
export interface CrossEntropyOptions {
  readonly reduction?: Reduction
  /** Per-class weights `[C]`; a weighted mean divides by the total weight of the targets' classes. */
  readonly classWeight?: TensorLike
  /** Mix targets with the uniform distribution by this amount. */
  readonly labelSmoothing?: number
  /** Targets with this class index contribute nothing (and do not count in the mean). */
  readonly ignoreIndex?: number
}

/**
 * Categorical cross-entropy from logits along the last axis, with integer class targets or class-probability targets
 * (torch.nn.functional.cross_entropy, classes last).
 *
 * @param logits - Logits `[..., C]`.
 * @param target - Class indices `[...]` or probabilities `[..., C]`.
 * @param options - Class weights, label smoothing, ignore index and reduction.
 * @returns The loss, differentiable in the logits.
 * @concept losses
 * @example
 * ```ts
 * import { crossEntropy } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(crossEntropy([[0, 0]], [1])) - Math.LN2) < 1e-15 // => true
 * ```
 */
export function crossEntropy(logits: TensorLike, target: TensorLike, options: CrossEntropyOptions = {}): Value {
  const [z] = promoteFloat(logits) as [Value]
  const logp = logSoftmax(z, -1)
  const c = z.shape[z.shape.length - 1] ?? 0
  const eps = options.labelSmoothing ?? 0
  const weights = options.classWeight === undefined ? undefined : (promoteFloat(options.classWeight)[0] as Value)
  const t = promoteFloat(target)[0] as Value
  const probabilityTargets = t.shape.length === z.shape.length
  if (probabilityTargets) {
    const smoothed = eps === 0 ? t : add(mul(t, 1 - eps), eps / c)
    const terms = neg(sum(mul(weights === undefined ? smoothed : mul(smoothed, weights), logp), -1))
    return reduce(terms, { reduction: options.reduction ?? 'mean' })
  }
  if (t.shape.length !== z.shape.length - 1)
    throw new MdmlError('shape', 'crossEntropy: targets must be [...] or [..., C]')
  const ignore = options.ignoreIndex
  const keep = ignore === undefined ? undefined : cast(sub(1, cast(equal(t, ignore), 'float64')), 'float64')
  // Ignored targets gather class 0 (any valid class) and are then masked out.
  const index = cast(ignore === undefined ? t : where(equal(t, ignore), 0, t), 'int32')
  const picked = squeeze(takeAlongAxis(logp, expandDims(index, -1), -1), -1)
  // With smoothing: (1 - eps) (-log p_y) + eps mean_c (-log p_c), each weighted by its class weight.
  const nll = neg(weights === undefined ? picked : mul(picked, take(weights, index)))
  const uniform = neg(div(sum(weights === undefined ? logp : mul(logp, weights), -1), c))
  let terms = eps === 0 ? nll : add(mul(nll, 1 - eps), mul(uniform, eps))
  if (keep !== undefined) terms = mul(terms, keep)
  const reduction = options.reduction ?? 'mean'
  if (reduction !== 'mean') return reduce(terms, { reduction })
  // torch's weighted mean divides by the total weight of the (kept) targets' classes.
  let denominator: Value = weights === undefined ? add(mul(t, 0), 1) : take(weights, index)
  if (keep !== undefined) denominator = mul(denominator, keep)
  return div(sum(terms), sum(denominator))
}

/**
 * Negative log-likelihood of class targets given log-probabilities (torch.nn.functional.nll_loss, classes last).
 *
 * @param logProbs - Log-probabilities `[..., C]`.
 * @param target - Class indices `[...]`.
 * @param options - Class weights, ignore index and reduction (a weighted mean divides by the kept targets' weights).
 * @returns The loss, differentiable in the log-probabilities.
 * @example
 * ```ts
 * import { nllLoss } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(nllLoss([[-1, -2]], [1])) // => 2
 * ```
 */
export function nllLoss(
  logProbs: TensorLike,
  target: TensorLike,
  options: Omit<CrossEntropyOptions, 'labelSmoothing'> = {},
): Value {
  const [lp] = promoteFloat(logProbs) as [Value]
  const t = promoteFloat(target)[0] as Value
  const ignore = options.ignoreIndex
  const index = cast(ignore === undefined ? t : where(equal(t, ignore), 0, t), 'int32')
  const weights = options.classWeight === undefined ? undefined : (promoteFloat(options.classWeight)[0] as Value)
  let w: Value | undefined = weights === undefined ? undefined : take(weights, index)
  if (ignore !== undefined) {
    const keep = cast(notEqual(t, ignore), 'float64')
    w = w === undefined ? keep : mul(w, keep)
  }
  const terms = neg(squeeze(takeAlongAxis(lp, expandDims(index, -1), -1), -1))
  return reduce(terms, { reduction: options.reduction ?? 'mean', ...(w === undefined ? {} : { weight: w }) })
}
