import { WhiteboxError } from '../base/index.ts'
import { logSigmoid } from '../numerics/special/index.ts'
import {
  add,
  cast,
  div,
  equal,
  expandDims,
  log,
  logSoftmax,
  maximum,
  mul,
  neg,
  promoteFloat,
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
 * @param options - posWeight (weight of positive terms), reduction and weights.
 * @returns The loss, differentiable.
 * @concept losses
 * @example
 * ```ts
 * import { binaryCrossEntropyWithLogits } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(binaryCrossEntropyWithLogits([0], [1])) === Math.LN2 // => true
 * ```
 */
export function binaryCrossEntropyWithLogits(
  logits: TensorLike,
  target: TensorLike,
  options: LossOptions & { readonly posWeight?: TensorLike } = {},
): Value {
  const [x, y] = promoteFloat(logits, target) as [Value, Value]
  const positive =
    options.posWeight === undefined ? logSigmoid(x) : mul(logSigmoid(x), promoteFloat(options.posWeight)[0] as Value)
  return reduce(neg(add(mul(y, positive), mul(sub(1, y), logSigmoid(neg(x))))), options)
}

/**
 * Binary cross-entropy on probabilities, with log clamped at -100 as torch does
 * (torch.nn.functional.binary_cross_entropy).
 *
 * @param probs - Probabilities.
 * @param target - Targets in [0, 1].
 * @param options - Reduction and weights.
 * @returns The loss, differentiable.
 * @example
 * ```ts
 * import { binaryCrossEntropy } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(binaryCrossEntropy([0.5], [0])) === Math.LN2 // => true
 * ```
 */
export function binaryCrossEntropy(probs: TensorLike, target: TensorLike, options: LossOptions = {}): Value {
  const [p, y] = promoteFloat(probs, target) as [Value, Value]
  // torch clamps each log at -100 so that a probability of exactly 0 or 1 gives a finite loss.
  const clampedLog = (v: Value): Value => maximum(log(v), -100)
  return reduce(neg(add(mul(y, clampedLog(p)), mul(sub(1, y), clampedLog(sub(1, p))))), options)
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
 * import { crossEntropy } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
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
    throw new WhiteboxError('shape', 'crossEntropy: targets must be [...] or [..., C]')
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
 * @param options - Reduction.
 * @returns The loss, differentiable in the log-probabilities.
 * @example
 * ```ts
 * import { nllLoss } from 'whitebox-engine/losses'
 * import { item } from 'whitebox-engine/tensor'
 * item(nllLoss([[-1, -2]], [1])) // => 2
 * ```
 */
export function nllLoss(
  logProbs: TensorLike,
  target: TensorLike,
  options: { readonly reduction?: Reduction } = {},
): Value {
  const [lp] = promoteFloat(logProbs) as [Value]
  const index = cast(promoteFloat(target)[0] as Value, 'int32')
  return reduce(neg(squeeze(takeAlongAxis(lp, expandDims(index, -1), -1), -1)), options)
}
