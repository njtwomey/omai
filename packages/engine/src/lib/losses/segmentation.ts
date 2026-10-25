/**
 * Segmentation losses that optimise overlap directly rather than per-pixel accuracy, MONAI style with classes on the
 * last axis: soft Dice, Tversky (trading false positives against false negatives) and focal Tversky, and the Lovasz
 * hinge and Lovasz-softmax losses, convex surrogates of the IoU.
 */

import { MdmlError } from '../base/index.ts'
import {
  abs,
  add,
  argsort,
  asTensor,
  cast,
  concat,
  cumsum,
  div,
  equal,
  expandDims,
  mul,
  notEqual,
  oneHot,
  pow,
  promoteFloat,
  relu,
  reshape,
  sigmoid,
  sliceAxis,
  softmax,
  square,
  sub,
  sum,
  type TensorLike,
  take,
  takeAlongAxis,
  type Value,
  where,
} from '../tensor/index.ts'
import { type Reduction, reduce } from './_reduce.ts'

/** Options of the overlap losses (Dice, Tversky, focal Tversky); classes lie along the last axis, as MONAI's. */
export interface OverlapOptions {
  /** Applied to the input first: 'sigmoid', 'softmax' (over classes) or 'none' (probabilities given; default). */
  readonly activation?: 'none' | 'sigmoid' | 'softmax'
  /** Keep class 0 (default true); false drops it, as for a background class. */
  readonly includeBackground?: boolean
  /** Added to numerator and denominator (default 1e-5). */
  readonly smooth?: number
  /** Pool the batch before taking the ratio (default false: one ratio per sample and class). */
  readonly batch?: boolean
  /** Index targets with this class are excluded from every sum. */
  readonly ignoreIndex?: number
  /** Per-class weights `[C]` (after dropping the background) multiplying each class's loss. */
  readonly classWeight?: TensorLike
  /** 'mean' over samples and classes, 'sum', or 'none' (`[N, C]`, or `[C]` with `batch`). */
  readonly reduction?: Reduction
}

/** Soft true-positive, false-positive and false-negative counts per sample and class (or per class, pooled). */
interface Counts {
  readonly tp: Value
  readonly fp: Value
  readonly fn: Value
  readonly p2: Value
  readonly t2: Value
}

function counts(input: TensorLike, target: TensorLike, options: OverlapOptions): Counts {
  const [x] = promoteFloat(input) as [Value]
  if (x.shape.length < 2) throw new MdmlError('shape', 'overlap losses expect [N, ..., C] inputs')
  const c = x.shape[x.shape.length - 1] ?? 1
  const act = options.activation ?? 'none'
  let p = act === 'sigmoid' ? sigmoid(x) : act === 'softmax' ? softmax(x, -1) : x
  const t0 = asTensor(target)
  let keep: Value | undefined
  let t: Value
  if (t0.shape.length === x.shape.length - 1) {
    const labels = promoteFloat(t0)[0] as Value
    const ignore = options.ignoreIndex
    if (ignore !== undefined) keep = expandDims(cast(notEqual(labels, ignore), 'float64'), -1)
    t = oneHot(cast(ignore === undefined ? labels : where(equal(labels, ignore), 0, labels), 'int32'), c)
  } else {
    t = promoteFloat(t0)[0] as Value
  }
  if (keep !== undefined) {
    p = mul(p, keep)
    t = mul(t, keep)
  }
  if (options.includeBackground === false) {
    p = sliceAxis(p, -1, 1, c)
    t = sliceAxis(t, -1, 1, c)
  }
  const rank = p.shape.length
  const spatial = Array.from({ length: rank - 2 }, (_, i) => i + 1)
  const axes = options.batch === true ? [0, ...spatial] : spatial
  const total = (v: Value): Value => (axes.length === 0 ? v : sum(v, axes))
  return {
    tp: total(mul(p, t)),
    fp: total(mul(p, sub(1, t))),
    fn: total(mul(sub(1, p), t)),
    p2: total(square(p)),
    t2: total(square(t)),
  }
}

function finish(score: Value, options: OverlapOptions): Value {
  const weighted = options.classWeight === undefined ? score : mul(score, promoteFloat(options.classWeight)[0] as Value)
  return reduce(weighted, { reduction: options.reduction ?? 'mean' })
}

/**
 * Soft Dice loss (Milletari et al., 2016; monai.losses.DiceLoss with channels last): 1 - (2 TP + s) / (sum p + sum t
 * + s) per sample and class, from probabilities (or logits with an activation) against index or one-hot targets.
 *
 * @param input - Probabilities or logits `[N, ..., C]`.
 * @param target - Class indices `[N, ...]` or one-hot / soft targets `[N, ..., C]`.
 * @param options - Activation, background, smoothing, batch pooling, squared denominators (V-Net), ignore index,
 *   class weights and reduction.
 * @returns The loss, differentiable in the input.
 * @concept segmentation
 * @example
 * ```ts
 * import { dice } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(dice([[[1, 0], [0, 1]]], [[0, 1]], { smooth: 0 })) // => 0
 * ```
 */
export function dice(
  input: TensorLike,
  target: TensorLike,
  options: OverlapOptions & { readonly squared?: boolean } = {},
): Value {
  const k = counts(input, target, options)
  const s = options.smooth ?? 1e-5
  const denominator = options.squared === true ? add(k.p2, k.t2) : add(add(add(k.tp, k.fp), k.tp), k.fn)
  return finish(sub(1, div(add(mul(k.tp, 2), s), add(denominator, s))), options)
}

/**
 * Tversky loss (Salehi et al., 2017; monai.losses.TverskyLoss): 1 - (TP + s) / (TP + alpha FP + beta FN + s), which
 * trades false positives against false negatives; alpha = beta = 0.5 gives Dice.
 *
 * @param input - Probabilities or logits `[N, ..., C]`.
 * @param target - Class indices `[N, ...]` or one-hot targets `[N, ..., C]`.
 * @param options - alpha (default 0.5, weights FP), beta (default 0.5, weights FN) and the overlap options.
 * @returns The loss, differentiable in the input.
 * @example
 * ```ts
 * import { tversky } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(tversky([[[0.5, 0.5]]], [[0]], { smooth: 0, alpha: 1, beta: 0, reduction: 'sum' })) // => 1
 * ```
 */
export function tversky(
  input: TensorLike,
  target: TensorLike,
  options: OverlapOptions & { readonly alpha?: number; readonly beta?: number } = {},
): Value {
  return finish(tverskyLoss(counts(input, target, options), options), options)
}

function tverskyLoss(k: Counts, options: OverlapOptions & { readonly alpha?: number; readonly beta?: number }): Value {
  const s = options.smooth ?? 1e-5
  const denominator = add(add(add(k.tp, mul(k.fp, options.alpha ?? 0.5)), mul(k.fn, options.beta ?? 0.5)), s)
  return sub(1, div(add(k.tp, s), denominator))
}

/**
 * Focal Tversky loss (Abraham and Khan, 2019): (1 - TI)^gamma for the Tversky index TI, which focuses training on
 * poorly segmented classes; gamma = 0.75 corresponds to the paper's 1 / (4/3).
 *
 * @param input - Probabilities or logits `[N, ..., C]`.
 * @param target - Class indices `[N, ...]` or one-hot targets `[N, ..., C]`.
 * @param options - gamma (default 0.75), alpha (default 0.7), beta (default 0.3) and the overlap options.
 * @returns The loss, differentiable in the input (where TI < 1).
 * @example
 * ```ts
 * import { focalTversky } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(focalTversky([[[0.75]]], [[[1]]], { smooth: 0, alpha: 0, beta: 1, gamma: 2 })) // => 0.0625
 * ```
 */
export function focalTversky(
  input: TensorLike,
  target: TensorLike,
  options: OverlapOptions & { readonly alpha?: number; readonly beta?: number; readonly gamma?: number } = {},
): Value {
  const k = counts(input, target, options)
  const score = tverskyLoss(k, { ...options, alpha: options.alpha ?? 0.7, beta: options.beta ?? 0.3 })
  return finish(pow(relu(score), options.gamma ?? 0.75), options)
}

/** The gradient of the Lovasz extension of the Jaccard loss for ground truth sorted by decreasing error. */
function lovaszGrad(gtSorted: Value): Value {
  const n = gtSorted.shape[gtSorted.shape.length - 1] ?? 0
  const gts = sum(gtSorted, -1, true)
  const intersection = sub(gts, cumsum(gtSorted, -1))
  const union = add(gts, cumsum(sub(1, gtSorted), -1))
  const jaccard = sub(1, div(intersection, union))
  if (n === 1) return jaccard
  const head = sliceAxis(jaccard, -1, 0, 1)
  const rest = sub(sliceAxis(jaccard, -1, 1, n), sliceAxis(jaccard, -1, 0, n - 1))
  return concat([head, rest], -1)
}

/**
 * Lovasz hinge loss (Berman, Rannen Triki and Blaschko, 2018): the Lovasz extension of the Jaccard (IoU) loss applied
 * to the hinge errors 1 - s y of binary logits, a convex surrogate that optimises IoU directly. Per image (default)
 * the loss is computed on each sample and averaged; otherwise all pixels are pooled.
 *
 * @param logits - Binary logits `[N, ...]`.
 * @param labels - Binary labels in {0, 1} `[N, ...]`.
 * @param options - perImage (default true) and ignoreIndex (pixels with this label are dropped).
 * @returns The loss, differentiable in the logits (almost everywhere).
 * @example
 * ```ts
 * import { lovaszHinge } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(lovaszHinge([[2, -2]], [[1, 0]])) // => 0
 * ```
 */
export function lovaszHinge(
  logits: TensorLike,
  labels: TensorLike,
  options: { readonly perImage?: boolean; readonly ignoreIndex?: number } = {},
): Value {
  const [x, y] = promoteFloat(logits, labels) as [Value, Value]
  const n = x.shape[0] ?? 1
  const perImage = options.perImage ?? true
  const rows = perImage ? n : 1
  const flatX = reshape(x, [rows, -1])
  const flatY = reshape(y, [rows, -1])
  if (options.ignoreIndex !== undefined) {
    // Drop ignored pixels by giving them zero error and zero ground truth past the end of the sort.
    const keep = cast(notEqual(flatY, options.ignoreIndex), 'float64')
    return lovaszHingeRows(mul(flatX, keep), mul(where(equal(flatY, options.ignoreIndex), 0, flatY), keep), keep)
  }
  return lovaszHingeRows(flatX, flatY, undefined)
}

/** Sort errors (decreasing) with their ground truth; ignored pixels go last with zero error and no ground truth. */
function sortErrors(errors: Value, gt: Value, keep: Value | undefined): { readonly errors: Value; readonly gt: Value } {
  const keyed = keep === undefined ? errors : where(equal(keep, 0), Number.NEGATIVE_INFINITY, errors)
  const order = argsort(keyed, -1, true)
  const sorted = takeAlongAxis(keyed, order, -1)
  const clean = keep === undefined ? sorted : where(equal(takeAlongAxis(keep, order, -1), 0), 0, sorted)
  return { errors: clean, gt: takeAlongAxis(gt, order, -1) }
}

function lovaszHingeRows(x: Value, y: Value, keep: Value | undefined): Value {
  const s = sortErrors(sub(1, mul(x, sub(mul(y, 2), 1))), y, keep)
  return reduce(sum(mul(relu(s.errors), lovaszGrad(s.gt)), -1), { reduction: 'mean' })
}

/**
 * Lovasz-softmax loss (Berman et al., 2018): for each class, the Lovasz extension of the Jaccard loss of the errors
 * |1[y = c] - p_c|, averaged over the classes present in the labels (or all classes). The multi-class counterpart of
 * {@link lovaszHinge}.
 *
 * @param probs - Class probabilities `[N, ..., C]`.
 * @param labels - Class indices `[N, ...]`.
 * @param options - classes ('present', default, or 'all'), perImage (default false) and ignoreIndex.
 * @returns The loss, differentiable in the probabilities.
 * @example
 * ```ts
 * import { lovaszSoftmax } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(lovaszSoftmax([[[1, 0], [0, 1]]], [[0, 1]])) // => 0
 * ```
 */
export function lovaszSoftmax(
  probs: TensorLike,
  labels: TensorLike,
  options: { readonly classes?: 'present' | 'all'; readonly perImage?: boolean; readonly ignoreIndex?: number } = {},
): Value {
  const [p] = promoteFloat(probs) as [Value]
  const c = p.shape[p.shape.length - 1] ?? 1
  const n = p.shape[0] ?? 1
  const rows = options.perImage === true ? n : 1
  const flatP = reshape(p, [rows, -1, c])
  const labelValues = promoteFloat(labels)[0] as Value
  const flatY = reshape(labelValues, [rows, -1])
  const ignore = options.ignoreIndex
  const keep = ignore === undefined ? undefined : cast(notEqual(flatY, ignore), 'float64')
  const safeY = ignore === undefined ? flatY : where(equal(flatY, ignore), 0, flatY)
  const perRow: Value[] = []
  for (let r = 0; r < rows; r++) {
    const pr = take(flatP, [r], 0)
    const yr = take(safeY, [r], 0)
    const kr = keep === undefined ? undefined : take(keep, [r], 0)
    const losses: Value[] = []
    const present: Value[] = []
    for (let k = 0; k < c; k++) {
      let fg = cast(equal(yr, k), 'float64')
      if (kr !== undefined) fg = mul(fg, kr)
      const s = sortErrors(abs(sub(fg, reshape(sliceAxis(pr, -1, k, k + 1), fg.shape))), fg, kr)
      losses.push(reshape(sum(mul(s.errors, lovaszGrad(s.gt))), [1]))
      present.push(reshape(options.classes === 'all' ? asTensor(1) : cast(notEqual(sum(fg), 0), 'float64'), [1]))
    }
    const l = concat(losses)
    const w = concat(present)
    perRow.push(reshape(div(sum(mul(l, w)), sum(w)), [1]))
  }
  return reduce(concat(perRow), { reduction: 'mean' })
}
