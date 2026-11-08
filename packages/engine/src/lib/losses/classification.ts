/**
 * Classification losses, torch.nn.functional style: binary cross-entropy on logits (computed stably) and on
 * probabilities, softmax cross-entropy with class-index or probability targets and label smoothing options, negative
 * log-likelihood of log-probabilities, the multi-label soft-margin loss, and sigmoid and softmax focal losses, which
 * down-weight easy examples; and the losses made for noisy labels: generalised, truncated and symmetric cross-entropy,
 * the bootstrapped losses, the probability MAE, the forward and backward corrected losses given a noise transition
 * matrix, and the bi-tempered logistic loss.
 */

import { OmaiError, type Reduction } from '../base/index.ts'
import { inv } from '../numerics/linalg/index.ts'
import { logSigmoid } from '../numerics/special/index.ts'
import {
  abs,
  add,
  argmax,
  cast,
  div,
  equal,
  exp,
  expandDims,
  greater,
  log,
  logSoftmax,
  matmul,
  max,
  maximum,
  mul,
  neg,
  notEqual,
  oneHot,
  pow,
  promoteFloat,
  reciprocal,
  relu,
  sigmoid,
  squeeze,
  stopGradient,
  sub,
  sum,
  type TensorLike,
  take,
  takeAlongAxis,
  transpose,
  type Value,
  where,
} from '../tensor/index.ts'
import { checkScalar, checkWeights, type LossOptions, reduce } from './_reduce.ts'

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
 * import { binaryCrossEntropyWithLogits } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(binaryCrossEntropyWithLogits([0], [1])) === Math.LN2 // => true
 * ```
 */
export function binaryCrossEntropyWithLogits(
  logits: TensorLike,
  target: TensorLike,
  options: LossOptions & { readonly posWeight?: TensorLike; readonly labelSmoothing?: number } = {},
): Value {
  checkScalar('binaryCrossEntropyWithLogits', 'labelSmoothing', options.labelSmoothing, 'unit')
  const [x, t] = promoteFloat(logits, target) as [Value, Value]
  const y = smoothBinary(t, options.labelSmoothing)
  const positive =
    options.posWeight === undefined ? logSigmoid(x) : mul(logSigmoid(x), promoteFloat(options.posWeight)[0] as Value)
  return reduce(neg(add(mul(y, positive), mul(sub(1, y), logSigmoid(neg(x))))), options)
}

/** exp(-100): below this probability torch's binary cross-entropy clamps its logarithm at -100. */
const LOG_FLOOR = Math.exp(-100)

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
 * import { binaryCrossEntropy } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(binaryCrossEntropy([0.5], [0])) === Math.LN2 // => true
 * ```
 */
export function binaryCrossEntropy(
  probs: TensorLike,
  target: TensorLike,
  options: LossOptions & { readonly labelSmoothing?: number } = {},
): Value {
  checkScalar('binaryCrossEntropy', 'labelSmoothing', options.labelSmoothing, 'unit')
  const [p, t] = promoteFloat(probs, target) as [Value, Value]
  const y = smoothBinary(t, options.labelSmoothing)
  // torch clamps each log at -100 so that a probability of exactly 0 or 1 gives a finite loss: the clamp is on the
  // input of the log (log(max(p, e^-100)) is the same value), so the gradient there is 0, not infinity times 0.
  const clampedLog = (v: Value): Value => log(maximum(v, LOG_FLOOR))
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
 * import { focal } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(focal([0], [1], { alpha: -1, gamma: 1 })) === Math.LN2 / 2 // => true
 * ```
 */
export function focal(
  logits: TensorLike,
  target: TensorLike,
  options: LossOptions & { readonly alpha?: number; readonly gamma?: number } = {},
): Value {
  const [x, y] = promoteFloat(logits, target) as [Value, Value]
  checkScalar('focal', 'gamma', options.gamma, 'nonNegative')
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
 * import { categoricalFocal } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
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
  checkScalar('categoricalFocal', 'gamma', options.gamma, 'nonNegative')
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
 * import { multiLabelSoftMargin } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
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
 * @index function Categorical cross-entropy from logits.
 * @example
 * ```ts
 * import { crossEntropy } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(crossEntropy([[0, 0]], [1])) - Math.LN2) < 1e-15 // => true
 * ```
 */
export function crossEntropy(logits: TensorLike, target: TensorLike, options: CrossEntropyOptions = {}): Value {
  checkScalar('crossEntropy', 'labelSmoothing', options.labelSmoothing, 'unit')
  checkWeights('crossEntropy', 'classWeight', options.classWeight)
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
    throw new OmaiError('shape', 'crossEntropy: targets must be [...] or [..., C]')
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
 * import { nllLoss } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
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

// --- Losses for noisy labels ---------------------------------------------------------------------------------------

/** The class probabilities and the probability of each target class, for the noise-robust losses below. */
function classProbabilities(
  logits: TensorLike,
  target: TensorLike,
): { readonly logp: Value; readonly p: Value; readonly picked: Value; readonly pickedLog: Value; readonly c: number } {
  const [z] = promoteFloat(logits) as [Value]
  const c = z.shape[z.shape.length - 1] ?? 0
  const t = promoteFloat(target)[0] as Value
  if (t.shape.length !== z.shape.length - 1) throw new OmaiError('shape', 'targets must be class indices [...]')
  const logp = logSoftmax(z, -1)
  const index = expandDims(cast(t, 'int32'), -1)
  const pickedLog = squeeze(takeAlongAxis(logp, index, -1), -1)
  return { logp, p: exp(logp), picked: exp(pickedLog), pickedLog, c }
}

/**
 * Generalised cross-entropy (Zhang and Sabuncu, 2018): the Box-Cox transform (1 - p_y^q) / q of the probability of the
 * target class, which is the cross-entropy as q goes to 0 and half the probability MAE at q = 1. Large q bounds the
 * loss (so a mislabelled example cannot dominate) at the price of slower fitting.
 *
 * @param logits - Logits `[..., C]`.
 * @param target - Class indices `[...]`.
 * @param options - q in (0, 1] (default 0.7) and reduction.
 * @returns The loss, differentiable in the logits.
 * @concept noise-robust-losses
 * @example
 * ```ts
 * import { generalisedCrossEntropy } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(generalisedCrossEntropy([[0, 0]], [0], { q: 1 })) - 0.5) < 1e-15 // => true
 * ```
 */
export function generalisedCrossEntropy(
  logits: TensorLike,
  target: TensorLike,
  options: { readonly q?: number; readonly reduction?: Reduction } = {},
): Value {
  const q = options.q ?? 0.7
  if (!(q > 0 && q <= 1)) throw new OmaiError('value', 'generalisedCrossEntropy: q must lie in (0, 1]')
  const { pickedLog } = classProbabilities(logits, target)
  return reduce(div(sub(1, exp(mul(pickedLog, q))), q), { reduction: options.reduction ?? 'mean' })
}

/**
 * Truncated generalised cross-entropy (Zhang and Sabuncu, 2018): the generalised loss with the probability of the
 * target class floored at k, so an example whose label the model gives probability below k contributes a constant and
 * no gradient (it is treated as noise and ignored).
 *
 * @param logits - Logits `[..., C]`.
 * @param target - Class indices `[...]`.
 * @param options - q (default 0.7), the threshold k in (0, 1) (default 0.5) and reduction.
 * @returns The loss, differentiable in the logits (zero gradient below the threshold).
 * @example
 * ```ts
 * import { truncatedCrossEntropy } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(truncatedCrossEntropy([[0, 5]], [0], { q: 1, k: 0.5 })) - 0.5) < 1e-15 // => true
 * ```
 */
export function truncatedCrossEntropy(
  logits: TensorLike,
  target: TensorLike,
  options: { readonly q?: number; readonly k?: number; readonly reduction?: Reduction } = {},
): Value {
  const q = options.q ?? 0.7
  const k = options.k ?? 0.5
  if (!(q > 0 && q <= 1)) throw new OmaiError('value', 'truncatedCrossEntropy: q must lie in (0, 1]')
  const { picked } = classProbabilities(logits, target)
  const floor = (1 - k ** q) / q
  const terms = where(greater(picked, k), div(sub(1, pow(picked, q)), q), floor)
  return reduce(terms, { reduction: options.reduction ?? 'mean' })
}

/**
 * Symmetric cross-entropy (Wang et al., 2019): alpha times the cross-entropy plus beta times the reverse
 * cross-entropy, -sum_k p_k log q_k with q the one-hot label whose zero entries are clipped to exp(A) (so the reverse
 * term is -A (1 - p_y), bounded and noise-tolerant).
 *
 * @param logits - Logits `[..., C]`.
 * @param target - Class indices `[...]`.
 * @param options - alpha (default 0.1), beta (default 1), the log of the clipped zero, A (default log 1e-4) and reduction.
 * @returns The loss, differentiable in the logits.
 * @example
 * ```ts
 * import { symmetricCrossEntropy } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(symmetricCrossEntropy([[0, 0]], [0], { alpha: 1, beta: 1, A: -2 })) - (Math.LN2 + 1)) < 1e-15 // => true
 * ```
 */
export function symmetricCrossEntropy(
  logits: TensorLike,
  target: TensorLike,
  options: {
    readonly alpha?: number
    readonly beta?: number
    readonly A?: number
    readonly reduction?: Reduction
  } = {},
): Value {
  const { pickedLog, picked } = classProbabilities(logits, target)
  const a = options.A ?? Math.log(1e-4)
  const terms = add(mul(neg(pickedLog), options.alpha ?? 0.1), mul(sub(1, picked), -a * (options.beta ?? 1)))
  return reduce(terms, { reduction: options.reduction ?? 'mean' })
}

/**
 * The probability MAE (Ghosh et al., 2017), sum_k |1[k = y] - p_k| = 2 (1 - p_y): symmetric in the sense that its sum
 * over all labels is constant, which makes it tolerant of symmetric label noise (and slow to train).
 *
 * @param logits - Logits `[..., C]`.
 * @param target - Class indices `[...]`.
 * @param options - reduction.
 * @returns The loss, differentiable in the logits.
 * @example
 * ```ts
 * import { probabilityMae } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(probabilityMae([[0, 0]], [1])) // => 1
 * ```
 */
export function probabilityMae(
  logits: TensorLike,
  target: TensorLike,
  options: { readonly reduction?: Reduction } = {},
): Value {
  const { p, c } = classProbabilities(logits, target)
  const hot = oneHot(cast(promoteFloat(target)[0] as Value, 'int32'), c)
  return reduce(sum(abs(sub(hot, p)), -1), { reduction: options.reduction ?? 'mean' })
}

/**
 * Bootstrapped cross-entropy (Reed et al., 2015): the target is a convex combination beta y + (1 - beta) q of the
 * given label y and the model's own belief q (soft: its probabilities; hard: its arg-max class, a constant), so the
 * model partly trusts itself against labels it disagrees with.
 *
 * @param logits - Logits `[..., C]`.
 * @param target - Class indices `[...]`.
 * @param options - beta (default 0.8), mode ('soft' or 'hard', default 'soft') and reduction.
 * @returns The loss, differentiable in the logits (the model's own belief is held constant).
 * @example
 * ```ts
 * import { bootstrapped } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(bootstrapped([[0, 0]], [0], { beta: 1 })) - Math.LN2) < 1e-15 // => true
 * ```
 */
export function bootstrapped(
  logits: TensorLike,
  target: TensorLike,
  options: { readonly beta?: number; readonly mode?: 'soft' | 'hard'; readonly reduction?: Reduction } = {},
): Value {
  const beta = options.beta ?? 0.8
  const { logp, p, c } = classProbabilities(logits, target)
  const hot = oneHot(cast(promoteFloat(target)[0] as Value, 'int32'), c)
  const belief =
    (options.mode ?? 'soft') === 'soft' ? stopGradient(p) : oneHot(cast(stopGradient(argmax(p, -1)), 'int32'), c)
  const mixed = add(mul(hot, beta), mul(belief, 1 - beta))
  return reduce(neg(sum(mul(mixed, logp), -1)), { reduction: options.reduction ?? 'mean' })
}

/**
 * Forward-corrected cross-entropy (Patrini et al., 2017): given the noise transition matrix T (T[i][j] = P(observed
 * label j | true class i)) the model's clean probabilities p are pushed through it, and the loss is
 * -log (T^T p)_y of the observed label; minimising it recovers the clean posterior.
 *
 * @param logits - Logits `[N, C]`.
 * @param target - Observed (noisy) class indices `[N]`.
 * @param transition - Row-stochastic transition matrix `[C, C]`.
 * @param options - reduction.
 * @returns The loss, differentiable in the logits.
 * @example
 * ```ts
 * import { forwardCorrected } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(forwardCorrected([[0, 0]], [1], [[1, 0], [0.5, 0.5]])) - Math.log(4)) < 1e-15 // => true
 * ```
 */
export function forwardCorrected(
  logits: TensorLike,
  target: TensorLike,
  transition: TensorLike,
  options: { readonly reduction?: Reduction } = {},
): Value {
  const { p } = classProbabilities(logits, target)
  const t = promoteFloat(transition)[0] as Value
  const noisy = matmul(p, t)
  const index = expandDims(cast(promoteFloat(target)[0] as Value, 'int32'), -1)
  const picked = squeeze(takeAlongAxis(noisy, index, -1), -1)
  return reduce(neg(log(picked)), { reduction: options.reduction ?? 'mean' })
}

/**
 * Backward-corrected cross-entropy (Patrini et al., 2017): the per-class loss vector l_j = -log p_j is mapped through
 * the inverse transition matrix, and the entry of the observed label is the loss, (T^-1 l)_y, an unbiased estimate of
 * the clean loss (and negative for some inputs, which is why the forward correction is the stabler of the two).
 *
 * @param logits - Logits `[N, C]`.
 * @param target - Observed (noisy) class indices `[N]`.
 * @param transition - Row-stochastic, invertible transition matrix `[C, C]`.
 * @param options - reduction.
 * @returns The loss, differentiable in the logits.
 * @example
 * ```ts
 * import { backwardCorrected, crossEntropy } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(backwardCorrected([[1, 2]], [1], [[1, 0], [0, 1]])) === item(crossEntropy([[1, 2]], [1])) // => true
 * ```
 */
export function backwardCorrected(
  logits: TensorLike,
  target: TensorLike,
  transition: TensorLike,
  options: { readonly reduction?: Reduction } = {},
): Value {
  const { logp } = classProbabilities(logits, target)
  const tinv = inv(promoteFloat(transition)[0] as Value)
  const corrected = matmul(neg(logp), transpose(tinv))
  const index = expandDims(cast(promoteFloat(target)[0] as Value, 'int32'), -1)
  return reduce(squeeze(takeAlongAxis(corrected, index, -1), -1), { reduction: options.reduction ?? 'mean' })
}

/**
 * The tempered logarithm log_t(x) = (x^(1-t) - 1) / (1 - t), the logarithm at t = 1.
 *
 * @param x - The argument, positive.
 * @param t - The temperature; `1` is the ordinary logarithm.
 * @returns The tempered logarithm of each element, differentiable.
 * @example
 * ```ts
 * import { logTemperature } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(logTemperature(5, 0)) // => 4
 * item(logTemperature(1, 1)) // => 0
 * ```
 */
export function logTemperature(x: TensorLike, t: number): Value {
  const [v] = promoteFloat(x) as [Value]
  return t === 1 ? log(v) : div(sub(pow(v, 1 - t), 1), 1 - t)
}

/**
 * The tempered exponential exp_t(x) = [1 + (1 - t) x]_+^(1 / (1 - t)), the exponential at t = 1.
 *
 * @param x - The argument.
 * @param t - The temperature; `1` is the ordinary exponential.
 * @returns The tempered exponential of each element, differentiable.
 * @example
 * ```ts
 * import { expTemperature } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(expTemperature(3, 0)) // => 4
 * item(expTemperature(0, 1)) // => 1
 * ```
 */
export function expTemperature(x: TensorLike, t: number): Value {
  const [v] = promoteFloat(x) as [Value]
  return t === 1 ? exp(v) : pow(relu(add(mul(v, 1 - t), 1)), 1 / (1 - t))
}

/**
 * The tempered softmax of Amid et al. (2019): exp_t(a - lambda(a)) with the normaliser lambda chosen so the
 * probabilities sum to one, found by their fixed-point iteration (exact for t = 1, a close approximation after a few
 * iterations otherwise, the same one the reference implementation takes).
 *
 * @param activations - Activations `[..., C]`.
 * @param t - The temperature (t = 1 is the softmax; t > 1 gives heavier tails).
 * @param iterations - Fixed-point iterations (default 5).
 * @returns Probabilities `[..., C]`, differentiable.
 * @example
 * ```ts
 * import { temperedSoftmax } from 'omai-engine/losses'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(temperedSoftmax([0, 0], 1)) // => [0.5, 0.5]
 * ```
 */
export function temperedSoftmax(activations: TensorLike, t: number, iterations = 5): Value {
  const [a] = promoteFloat(activations) as [Value]
  if (t === 1) return exp(logSoftmax(a, -1))
  const shifted = sub(a, max(a, -1, true))
  let normalised = shifted
  for (let i = 0; i < iterations; i++) {
    const partition = sum(expTemperature(normalised, t), -1, true)
    normalised = mul(shifted, pow(partition, 1 - t))
  }
  const partition = sum(expTemperature(normalised, t), -1, true)
  const normaliser = add(neg(logTemperature(reciprocal(partition), t)), max(a, -1, true))
  return expTemperature(sub(a, normaliser), t)
}

/**
 * The bi-tempered logistic loss (Amid et al., 2019): the Bregman divergence between the one-hot label and the
 * tempered softmax of the activations, with temperature t1 < 1 on the loss (bounded, so noisy labels stop mattering)
 * and t2 > 1 on the softmax (heavy tails, so outliers stop mattering). t1 = t2 = 1 is the cross-entropy.
 *
 * @param activations - Activations `[N, C]`.
 * @param target - Class indices `[N]`.
 * @param options - t1 (default 0.8), t2 (default 1.2), fixed-point iterations (default 5) and reduction.
 * @returns The loss, differentiable in the activations.
 * @example
 * ```ts
 * import { biTemperedLogistic, crossEntropy } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(biTemperedLogistic([[1, 2, 0]], [1], { t1: 1, t2: 1 })) - item(crossEntropy([[1, 2, 0]], [1]))) < 1e-14 // => true
 * ```
 */
export function biTemperedLogistic(
  activations: TensorLike,
  target: TensorLike,
  options: {
    readonly t1?: number
    readonly t2?: number
    readonly iterations?: number
    readonly reduction?: Reduction
  } = {},
): Value {
  const t1 = options.t1 ?? 0.8
  const t2 = options.t2 ?? 1.2
  const [a] = promoteFloat(activations) as [Value]
  const c = a.shape[a.shape.length - 1] ?? 0
  const hot = oneHot(cast(promoteFloat(target)[0] as Value, 'int32'), c)
  const p = temperedSoftmax(a, t2, options.iterations ?? 5)
  // For a one-hot label y the terms y (log_t1 y - log_t1 p) - (y^(2-t1) - p^(2-t1)) / (2-t1) reduce to the following.
  const picked = sum(mul(hot, p), -1)
  const terms =
    t1 === 1 ? neg(log(picked)) : add(neg(logTemperature(picked, t1)), div(sub(sum(pow(p, 2 - t1), -1), 1), 2 - t1))
  return reduce(terms, { reduction: options.reduction ?? 'mean' })
}
