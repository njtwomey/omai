/**
 * Contrastive and metric-learning losses, which train embeddings so that similar pairs lie close and dissimilar ones
 * far apart: InfoNCE, SimCLR's NT-Xent, supervised contrastive and CLIP's symmetric image-text loss, the triplet,
 * contrastive, cosine and hinge embedding losses, and the additive-margin softmax heads of face recognition (ArcFace,
 * CosFace); and the self-supervised objectives without negatives or with decorrelation, Barlow Twins and VICReg, with
 * the alignment and uniformity terms of Wang and Isola that explain what a contrastive loss optimises.
 *
 * @index function Contrastive and metric-learning losses such as InfoNCE and triplet.
 */

import type { Reduction } from '../base/index.ts'
import { logaddexp } from '../numerics/special/index.ts'
import {
  add,
  arange,
  asTensor,
  cast,
  clip,
  concat,
  diagonal,
  div,
  equal,
  exp,
  expandDims,
  eye,
  greater,
  log,
  logSoftmax,
  logsumexp,
  matmul,
  maximum,
  mean,
  minimum,
  mul,
  neg,
  oneHot,
  pow,
  promoteFloat,
  relu,
  sqrt,
  square,
  sub,
  sum,
  type TensorLike,
  transpose,
  type Value,
  variance,
  where,
} from '../tensor/index.ts'
import { checkScalar, type LossOptions, reduce } from './_reduce.ts'
import { l2Normalise, pairwiseDistance } from './_vectors.ts'
import { crossEntropy } from './classification.ts'

/** Options of {@link infoNce}. */
export interface InfoNceOptions {
  /** Softmax temperature (default 0.1). */
  readonly temperature?: number
  /** Explicit negatives `[M, D]` shared by every query, or `[N, M, D]` per query; otherwise the other keys. */
  readonly negatives?: TensorLike
  /** Compare cosine similarities (default true) rather than raw dot products. */
  readonly normalise?: boolean
  readonly reduction?: Reduction
}

/**
 * InfoNCE (van den Oord et al., 2018): the cross-entropy of picking each query's positive key among negatives,
 * -log exp(q . k+ / T) / sum_k exp(q . k / T). Without explicit negatives the other keys in the batch are the
 * negatives (the in-batch form of MoCo and CPC).
 *
 * @param query - Queries `[N, D]`.
 * @param positive - Positive keys `[N, D]`.
 * @param options - temperature, negatives, normalisation and reduction.
 * @returns The loss, differentiable in queries, keys and negatives.
 * @concept contrastive-learning
 * @example
 * ```ts
 * import { infoNce } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(infoNce([[1, 0], [0, 1]], [[1, 0], [0, 1]], { temperature: 1 })) - Math.log(1 + Math.exp(-1))) < 1e-15 // => true
 * ```
 */
export function infoNce(query: TensorLike, positive: TensorLike, options: InfoNceOptions = {}): Value {
  checkScalar('infoNce', 'temperature', options.temperature, 'positive')
  let [q, k] = promoteFloat(query, positive) as [Value, Value]
  const normalise = options.normalise ?? true
  if (normalise) {
    q = l2Normalise(q)
    k = l2Normalise(k)
  }
  const t = options.temperature ?? 0.1
  const reduction = options.reduction ?? 'mean'
  const n = q.shape[0] ?? 0
  if (options.negatives === undefined) {
    return crossEntropy(div(matmul(q, transpose(k)), t), arange(n), { reduction })
  }
  let negs = promoteFloat(options.negatives)[0] as Value
  if (normalise) negs = l2Normalise(negs)
  const pos = expandDims(sum(mul(q, k), -1), 1)
  const negLogits = negs.shape.length === 2 ? matmul(q, transpose(negs)) : sum(mul(expandDims(q, 1), negs), -1)
  const logits = div(concat([pos, negLogits], 1), t)
  return crossEntropy(logits, mul(arange(n), 0), { reduction })
}

/**
 * NT-Xent, the normalised temperature-scaled cross-entropy of SimCLR (Chen et al., 2020), in the label form of
 * optax.losses.ntxent: every pair of embeddings with the same label is a positive pair, scored against the anchor's
 * negatives (embeddings with other labels), and the loss is averaged over positive pairs. For two augmented views of
 * N examples, stack them and use labels [0..N-1, 0..N-1].
 *
 * @param embeddings - Embeddings `[N, D]`.
 * @param labels - Integer labels `[N]`.
 * @param options - temperature (default 0.07).
 * @returns The scalar loss, differentiable in the embeddings.
 * @example
 * ```ts
 * import { ntXent } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * const loss = ntXent([[1, 0], [1, 0], [0, 1], [0, 1]], [0, 0, 1, 1], { temperature: 1 })
 * Math.abs(item(loss) - Math.log(1 + 2 * Math.exp(-1))) < 1e-15 // => true
 * ```
 */
export function ntXent(
  embeddings: TensorLike,
  labels: TensorLike,
  options: { readonly temperature?: number } = {},
): Value {
  const [e] = promoteFloat(embeddings) as [Value]
  const n = e.shape[0] ?? 0
  const z = l2Normalise(e, 2.220446049250313e-16)
  checkScalar('ntXent', 'temperature', options.temperature, 'positive')
  const sim = div(matmul(z, transpose(z)), options.temperature ?? 0.07)
  const y = asTensor(labels)
  const same = equal(expandDims(y, 1), expandDims(y, 0))
  const positives = sub(cast(same, 'float64'), eye(n))
  // For each positive pair (i, p): s_ip - log(exp(s_ip) + sum over the negatives a of i of exp(s_ia)). Non-negatives
  // enter the log-sum-exp at a huge negative value (not -inf, which a row without negatives would turn into NaN).
  const negLse = logsumexp(where(same, -1e300, sim), 1, true)
  const logRatio = sub(sim, logaddexp(sim, negLse))
  return neg(div(sum(mul(logRatio, positives)), sum(positives)))
}

/**
 * Triplet margin loss max(0, d(a, p) - d(a, n) + margin) (torch.nn.functional.triplet_margin_loss), with the
 * pairwise distance ||x - y + eps||_p and the optional distance swap (Balntas et al., 2016).
 *
 * @param anchor - Anchors `[N, D]`.
 * @param positive - Positives `[N, D]`.
 * @param negative - Negatives `[N, D]`.
 * @param options - margin (default 1), p (default 2), eps (default 1e-6), swap, reduction and weights.
 * @returns The loss, differentiable away from the hinge.
 * @example
 * ```ts
 * import { tripletMargin } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(tripletMargin([[0, 0]], [[3, 4]], [[0, 1]], { eps: 0 })) // => 5
 * ```
 */
export function tripletMargin(
  anchor: TensorLike,
  positive: TensorLike,
  negative: TensorLike,
  options: LossOptions & {
    readonly margin?: number
    readonly p?: number
    readonly eps?: number
    readonly swap?: boolean
  } = {},
): Value {
  checkScalar('tripletMargin', 'margin', options.margin, 'nonNegative')
  const [a, p, n] = promoteFloat(anchor, positive, negative) as [Value, Value, Value]
  const norm = options.p ?? 2
  const eps = options.eps ?? 1e-6
  const dPos = pairwiseDistance(a, p, norm, eps)
  let dNeg = pairwiseDistance(a, n, norm, eps)
  if (options.swap === true) dNeg = minimum(dNeg, pairwiseDistance(p, n, norm, eps))
  return reduce(relu(add(sub(dPos, dNeg), options.margin ?? 1)), options)
}

/**
 * Contrastive loss (Hadsell, Chopra and LeCun, 2006): 0.5 d^2 for similar pairs and 0.5 max(0, margin - d)^2 for
 * dissimilar ones, with d the Euclidean distance between the two embeddings.
 *
 * @param x1 - First embeddings `[N, D]`.
 * @param x2 - Second embeddings `[N, D]`.
 * @param similar - 1 for similar pairs, 0 for dissimilar `[N]`.
 * @param options - margin (default 1), reduction and weights.
 * @returns The loss, differentiable (away from coincident dissimilar pairs).
 * @example
 * ```ts
 * import { contrastive } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(contrastive([[0, 0], [0, 0]], [[0, 2], [0, 0.5]], [1, 0], { reduction: 'sum' })) // => 2.125
 * ```
 */
export function contrastive(
  x1: TensorLike,
  x2: TensorLike,
  similar: TensorLike,
  options: LossOptions & { readonly margin?: number } = {},
): Value {
  checkScalar('contrastive', 'margin', options.margin, 'nonNegative')
  const [a, b, y] = promoteFloat(x1, x2, similar) as [Value, Value, Value]
  const d2 = sum(square(sub(a, b)), -1)
  const d = sqrt(d2)
  const far = square(relu(sub(options.margin ?? 1, d)))
  return reduce(mul(add(mul(y, d2), mul(sub(1, y), far)), 0.5), options)
}

/**
 * Cosine embedding loss (torch.nn.functional.cosine_embedding_loss): 1 - cos(x1, x2) for y = 1 and
 * max(0, cos(x1, x2) - margin) for y = -1.
 *
 * @param x1 - First embeddings `[N, D]`.
 * @param x2 - Second embeddings `[N, D]`.
 * @param y - +1 (similar) or -1 (dissimilar) `[N]`.
 * @param options - margin (default 0), reduction and weights.
 * @returns The loss, differentiable.
 * @example
 * ```ts
 * import { cosineEmbedding } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(cosineEmbedding([[1, 0]], [[0, 1]], [1])) // => 1
 * ```
 */
export function cosineEmbedding(
  x1: TensorLike,
  x2: TensorLike,
  y: TensorLike,
  options: LossOptions & { readonly margin?: number } = {},
): Value {
  const [a, b, t] = promoteFloat(x1, x2, y) as [Value, Value, Value]
  // torch adds 1e-12 to each squared norm.
  const cos = div(sum(mul(a, b), -1), sqrt(mul(add(sum(square(a), -1), 1e-12), add(sum(square(b), -1), 1e-12))))
  return reduce(where(greater(t, 0), sub(1, cos), relu(sub(cos, options.margin ?? 0))), options)
}

/**
 * Hinge embedding loss (torch.nn.functional.hinge_embedding_loss): x for y = 1 and max(0, margin - x) for y = -1,
 * where x is usually a distance between a pair.
 *
 * @param x - Distances.
 * @param y - +1 or -1.
 * @param options - margin (default 1), reduction and weights.
 * @returns The loss, differentiable away from the hinge.
 * @example
 * ```ts
 * import { hingeEmbedding } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(hingeEmbedding([0.25, 2], [-1, 1], { reduction: 'sum' })) // => 2.75
 * ```
 */
export function hingeEmbedding(
  x: TensorLike,
  y: TensorLike,
  options: LossOptions & { readonly margin?: number } = {},
): Value {
  checkScalar('hingeEmbedding', 'margin', options.margin, 'nonNegative')
  const [d, t] = promoteFloat(x, y) as [Value, Value]
  return reduce(where(greater(t, 0), d, relu(sub(options.margin ?? 1, d))), options)
}

/**
 * Supervised contrastive loss (Khosla et al., 2020, the L_out form): for each anchor, minus the mean over its
 * positives (same label, not itself) of log exp(z_i . z_p / T) / sum_{a != i} exp(z_i . z_a / T), scaled by
 * T / baseTemperature. Anchors without positives are skipped; the result averages over the others.
 *
 * @param features - Embeddings `[N, D]`.
 * @param labels - Integer labels `[N]`.
 * @param options - temperature (default 0.07), baseTemperature (default 0.07), normalise (default true).
 * @returns The scalar loss, differentiable in the features.
 * @example
 * ```ts
 * import { supervisedContrastive } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * const loss = supervisedContrastive([[1, 0], [1, 0], [0, 1]], [0, 0, 1], { temperature: 1, baseTemperature: 1 })
 * Math.abs(item(loss) - Math.log(1 + Math.exp(-1))) < 1e-15 // => true
 * ```
 */
export function supervisedContrastive(
  features: TensorLike,
  labels: TensorLike,
  options: { readonly temperature?: number; readonly baseTemperature?: number; readonly normalise?: boolean } = {},
): Value {
  checkScalar('supervisedContrastive', 'temperature', options.temperature, 'positive')
  checkScalar('supervisedContrastive', 'baseTemperature', options.baseTemperature, 'positive')
  const [f] = promoteFloat(features) as [Value]
  const z = (options.normalise ?? true) ? l2Normalise(f) : f
  const t = options.temperature ?? 0.07
  const n = z.shape[0] ?? 0
  const sim = div(matmul(z, transpose(z)), t)
  // Self-similarities leave the denominator.
  const others = sub(1, eye(n))
  const logits = where(greater(others, 0), sim, Number.NEGATIVE_INFINITY)
  const logProb = where(greater(others, 0), logSoftmax(logits, 1), 0)
  const y = asTensor(labels)
  const positives = mul(cast(equal(expandDims(y, 1), expandDims(y, 0)), 'float64'), others)
  const counts = sum(positives, 1)
  const anchors = cast(greater(counts, 0), 'float64')
  const meanLogProb = div(sum(mul(logProb, positives), 1), maximum(counts, 1))
  const perAnchor = mul(neg(meanLogProb), t / (options.baseTemperature ?? 0.07))
  return div(sum(mul(perAnchor, anchors)), sum(anchors))
}

/**
 * The symmetric contrastive loss of CLIP (Radford et al., 2021): the mean of the image-to-text and text-to-image
 * cross-entropies over the logit matrix scale * cos(image_i, text_j), with matching pairs on the diagonal.
 *
 * @param imageEmbeddings - Image embeddings `[N, D]`.
 * @param textEmbeddings - Text embeddings `[N, D]`.
 * @param options - logitScale (default 1 / 0.07) and normalise (default true).
 * @returns The scalar loss, differentiable in both embeddings (and in a traced logit scale).
 * @example
 * ```ts
 * import { clipLoss } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * const loss = clipLoss([[1, 0], [0, 1]], [[1, 0], [0, 1]], { logitScale: 1 })
 * Math.abs(item(loss) - Math.log(1 + Math.exp(-1))) < 1e-15 // => true
 * ```
 */
export function clipLoss(
  imageEmbeddings: TensorLike,
  textEmbeddings: TensorLike,
  options: { readonly logitScale?: TensorLike; readonly normalise?: boolean } = {},
): Value {
  let [img, txt] = promoteFloat(imageEmbeddings, textEmbeddings) as [Value, Value]
  if (options.normalise ?? true) {
    img = l2Normalise(img)
    txt = l2Normalise(txt)
  }
  const n = img.shape[0] ?? 0
  const logits = mul(matmul(img, transpose(txt)), options.logitScale ?? 1 / 0.07)
  const labels = arange(n)
  return mul(add(crossEntropy(logits, labels), crossEntropy(transpose(logits), labels)), 0.5)
}

/** Options of the additive-margin softmax heads. */
export interface MarginSoftmaxOptions {
  /** The feature scale s (default 64). */
  readonly scale?: number
  /** The margin m (default 0.5 for ArcFace, 0.35 for CosFace). */
  readonly margin?: number
}

/** Cosines between L2-normalised embeddings `[N, D]` and class centres `[C, D]`. */
const cosines = (embeddings: TensorLike, weights: TensorLike): Value => {
  const [x, w] = promoteFloat(embeddings, weights) as [Value, Value]
  return matmul(l2Normalise(x), transpose(l2Normalise(w)))
}

const targetMask = (labels: TensorLike, c: number): Value => oneHot(asTensor(labels), c)

/**
 * ArcFace logits (Deng et al., 2019): s cos(theta + m) for the target class and s cos(theta) for the others, with
 * theta the angle between the normalised embedding and class centre. As in the reference implementation, past
 * theta + m > pi the target logit becomes cos(theta) - m sin(m) to stay monotonic (or, with `easyMargin`, the margin
 * applies only where cos(theta) > 0).
 *
 * @param embeddings - Embeddings `[N, D]`.
 * @param weights - Class centres `[C, D]`.
 * @param labels - Class indices `[N]`.
 * @param options - scale (default 64), margin (default 0.5), easyMargin.
 * @returns Logits `[N, C]` for a softmax cross-entropy, differentiable.
 * @example
 * ```ts
 * import { arcFaceLogits } from 'mdml-engine/losses'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(arcFaceLogits([[1, 0]], [[1, 0], [0, 1]], [1], { scale: 1, margin: 0 })) // => [[1, 0]]
 * ```
 */
export function arcFaceLogits(
  embeddings: TensorLike,
  weights: TensorLike,
  labels: TensorLike,
  options: MarginSoftmaxOptions & { readonly easyMargin?: boolean } = {},
): Value {
  checkScalar('arcFaceLogits', 'scale', options.scale, 'positive')
  checkScalar('arcFaceLogits', 'margin', options.margin, 'nonNegative')
  const cos = cosines(embeddings, weights)
  const c = cos.shape[1] ?? 0
  const m = options.margin ?? 0.5
  const sin = sqrt(clip(sub(1, square(cos)), 0, 1))
  let phi = sub(mul(cos, Math.cos(m)), mul(sin, Math.sin(m)))
  phi =
    options.easyMargin === true
      ? where(greater(cos, 0), phi, cos)
      : where(greater(cos, Math.cos(Math.PI - m)), phi, sub(cos, Math.sin(Math.PI - m) * m))
  const hot = targetMask(labels, c)
  return mul(add(mul(hot, phi), mul(sub(1, hot), cos)), options.scale ?? 64)
}

/**
 * CosFace logits (Wang et al., 2018; large-margin cosine loss): s (cos(theta) - m) for the target class and
 * s cos(theta) for the others.
 *
 * @param embeddings - Embeddings `[N, D]`.
 * @param weights - Class centres `[C, D]`.
 * @param labels - Class indices `[N]`.
 * @param options - scale (default 64), margin (default 0.35).
 * @returns Logits `[N, C]`, differentiable.
 * @example
 * ```ts
 * import { cosFaceLogits } from 'mdml-engine/losses'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(cosFaceLogits([[1, 0]], [[1, 0], [0, 1]], [0], { scale: 2, margin: 0.25 })) // => [[1.5, 0]]
 * ```
 */
export function cosFaceLogits(
  embeddings: TensorLike,
  weights: TensorLike,
  labels: TensorLike,
  options: MarginSoftmaxOptions = {},
): Value {
  checkScalar('cosFaceLogits', 'scale', options.scale, 'positive')
  checkScalar('cosFaceLogits', 'margin', options.margin, 'nonNegative')
  const cos = cosines(embeddings, weights)
  const hot = targetMask(labels, cos.shape[1] ?? 0)
  return mul(sub(cos, mul(hot, options.margin ?? 0.35)), options.scale ?? 64)
}

/**
 * ArcFace loss: the softmax cross-entropy of {@link arcFaceLogits}.
 *
 * @param embeddings - Embeddings `[N, D]`.
 * @param weights - Class centres `[C, D]`.
 * @param labels - Class indices `[N]`.
 * @param options - scale, margin, easyMargin and reduction.
 * @returns The loss, differentiable in embeddings and centres.
 * @example
 * ```ts
 * import { arcFace } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(arcFace([[1, 0]], [[1, 0], [-1, 0]], [0], { scale: 1, margin: 0 })) - Math.log(1 + Math.exp(-2))) < 1e-15 // => true
 * ```
 */
export function arcFace(
  embeddings: TensorLike,
  weights: TensorLike,
  labels: TensorLike,
  options: MarginSoftmaxOptions & { readonly easyMargin?: boolean; readonly reduction?: Reduction } = {},
): Value {
  return crossEntropy(arcFaceLogits(embeddings, weights, labels, options), labels, {
    reduction: options.reduction ?? 'mean',
  })
}

/**
 * CosFace loss: the softmax cross-entropy of {@link cosFaceLogits}.
 *
 * @param embeddings - Embeddings `[N, D]`.
 * @param weights - Class centres `[C, D]`.
 * @param labels - Class indices `[N]`.
 * @param options - scale, margin and reduction.
 * @returns The loss, differentiable in embeddings and centres.
 * @example
 * ```ts
 * import { cosFace } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(cosFace([[1, 0]], [[1, 0], [-1, 0]], [0], { scale: 1, margin: 0 })) - Math.log(1 + Math.exp(-2))) < 1e-15 // => true
 * ```
 */
export function cosFace(
  embeddings: TensorLike,
  weights: TensorLike,
  labels: TensorLike,
  options: MarginSoftmaxOptions & { readonly reduction?: Reduction } = {},
): Value {
  return crossEntropy(cosFaceLogits(embeddings, weights, labels, options), labels, {
    reduction: options.reduction ?? 'mean',
  })
}

/** The sum of the squared off-diagonal entries of a square matrix. */
function offDiagonalSquares(m: Value): Value {
  return sub(sum(square(m)), sum(square(diagonal(m))))
}

/**
 * Barlow Twins (Zbontar et al., 2021): standardise each embedding dimension over the batch, form the
 * cross-correlation matrix c between the two views' embeddings, and push it to the identity: the diagonal to one
 * (invariance) and the off-diagonal to zero (redundancy reduction, weighted by lambda). The standardisation is the
 * batch normalisation (no affine) of the reference code.
 *
 * @param z1 - Embeddings of the first view `[N, D]`.
 * @param z2 - Embeddings of the second view `[N, D]`.
 * @param options - lambda, the off-diagonal weight (default 0.0051), eps of the standardisation (default 1e-5).
 * @returns The scalar loss, differentiable in both views.
 * @concept self-supervised-learning
 * @example
 * ```ts
 * import { barlowTwins } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * const z = [[1, -1], [-1, 1], [1, 1], [-1, -1]]
 * item(barlowTwins(z, z, { eps: 0 })) // => 0
 * ```
 */
export function barlowTwins(
  z1: TensorLike,
  z2: TensorLike,
  options: { readonly lambda?: number; readonly eps?: number } = {},
): Value {
  const [a, b] = promoteFloat(z1, z2) as [Value, Value]
  const n = a.shape[0] ?? 1
  const eps = options.eps ?? 1e-5
  const standardise = (z: Value): Value =>
    div(sub(z, mean(z, 0, true)), sqrt(add(variance(z, 0, { keepDims: true }), eps)))
  const c = div(matmul(transpose(standardise(a)), standardise(b)), n)
  const onDiagonal = sum(square(sub(diagonal(c), 1)))
  return add(onDiagonal, mul(offDiagonalSquares(c), options.lambda ?? 0.0051))
}

/** Options of {@link vicreg}. */
export interface VicregOptions {
  /** Weight of the invariance (mean squared error between views); default 25. */
  readonly invariance?: number
  /** Weight of the variance hinge; default 25. */
  readonly variance?: number
  /** Weight of the covariance penalty; default 1. */
  readonly covariance?: number
  /** Target standard deviation of every dimension (default 1). */
  readonly gamma?: number
  readonly eps?: number
}

/**
 * VICReg (Bardes et al., 2022): variance-invariance-covariance regularisation. The invariance term is the mean
 * squared error between the two views' embeddings; the variance term hinges each dimension's standard deviation (over
 * the batch) at gamma, which prevents collapse; the covariance term penalises the off-diagonal entries of each view's
 * covariance matrix, which decorrelates the dimensions.
 *
 * @param z1 - Embeddings of the first view `[N, D]`.
 * @param z2 - Embeddings of the second view `[N, D]`.
 * @param options - The three weights, gamma and eps.
 * @returns The scalar loss, differentiable in both views.
 * @example
 * ```ts
 * import { vicreg } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(vicreg([[1, 1], [-1, -1]], [[1, 1], [-1, -1]], { eps: 0, covariance: 0 })) // => 0
 * ```
 */
export function vicreg(z1: TensorLike, z2: TensorLike, options: VicregOptions = {}): Value {
  const [a, b] = promoteFloat(z1, z2) as [Value, Value]
  const [n, d] = [a.shape[0] ?? 1, a.shape[1] ?? 1]
  const eps = options.eps ?? 1e-4
  const gamma = options.gamma ?? 1
  const invarianceTerm = mean(square(sub(a, b)))
  const centre = (z: Value): Value => sub(z, mean(z, 0, true))
  const hinge = (z: Value): Value => mean(relu(sub(gamma, sqrt(add(variance(z, 0, { ddof: 1 }), eps)))))
  const covariancePenalty = (z: Value): Value => {
    const c = div(matmul(transpose(centre(z)), centre(z)), n - 1)
    return div(offDiagonalSquares(c), d)
  }
  return add(
    add(mul(invarianceTerm, options.invariance ?? 25), mul(add(hinge(a), hinge(b)), (options.variance ?? 25) / 2)),
    mul(add(covariancePenalty(a), covariancePenalty(b)), options.covariance ?? 1),
  )
}

/**
 * Alignment of positive pairs (Wang and Isola, 2020): the mean over pairs of ||x - y||^alpha, small when positives map
 * together. Pair it with {@link uniformity}: a contrastive loss asymptotically optimises the sum of the two.
 *
 * @param x - Normalised embeddings `[N, D]`.
 * @param y - Their positives `[N, D]`.
 * @param options - alpha (default 2).
 * @returns The scalar alignment.
 * @example
 * ```ts
 * import { alignment } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * item(alignment([[1, 0]], [[0, 1]])) // => 2
 * ```
 */
export function alignment(x: TensorLike, y: TensorLike, options: { readonly alpha?: number } = {}): Value {
  checkScalar('alignment', 'alpha', options.alpha, 'positive')
  const [a, b] = promoteFloat(x, y) as [Value, Value]
  const alpha = options.alpha ?? 2
  const sq = sum(square(sub(a, b)), -1)
  return mean(alpha === 2 ? sq : pow(sq, alpha / 2))
}

/**
 * Uniformity on the sphere (Wang and Isola, 2020): log of the mean Gaussian potential exp(-t ||x_i - x_j||^2) over all
 * pairs, low when the embeddings spread evenly over the hypersphere.
 *
 * @param x - Normalised embeddings `[N, D]`.
 * @param options - t (default 2).
 * @returns The scalar uniformity (more negative is more uniform).
 * @example
 * ```ts
 * import { uniformity } from 'mdml-engine/losses'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(uniformity([[1, 0], [-1, 0]], { t: 1 })) + 4) < 1e-12 // => true
 * ```
 */
export function uniformity(x: TensorLike, options: { readonly t?: number } = {}): Value {
  checkScalar('uniformity', 't', options.t, 'positive')
  const [a] = promoteFloat(x) as [Value]
  const n = a.shape[0] ?? 0
  const gram = matmul(a, transpose(a))
  const sq = sum(square(a), -1, true)
  const dist2 = maximum(sub(add(sq, transpose(sq)), mul(gram, 2)), 0)
  const upper = sub(1, eye(n))
  const potentials = mul(exp(mul(dist2, -(options.t ?? 2))), upper)
  return log(div(sum(potentials), sum(upper)))
}
