/**
 * Losses that measure the distance between distributions: torch's `klDiv`, the KL and Jensen-Shannon divergences
 * between categorical distributions, knowledge distillation from a teacher's softened outputs, the maximum mean
 * discrepancy (a kernel two-sample distance), the entropic optimal transport loss computed by Sinkhorn iterations and
 * the sliced Wasserstein distance, which averages one-dimensional optimal transport over random projections.
 */

import { OmaiError, type Reduction } from '../base/index.ts'
import { xlogy } from '../numerics/special/index.ts'
import { type Key, normal } from '../random/index.ts'
import {
  abs,
  add,
  div,
  equal,
  exp,
  expandDims,
  isTensor,
  log,
  logSoftmax,
  logsumexp,
  matmul,
  mean,
  mul,
  pow,
  promoteFloat,
  relu,
  softmax,
  sort,
  sqrt,
  square,
  stopGradient,
  sub,
  sum,
  type TensorLike,
  tensor,
  toFlat,
  transpose,
  type Value,
  where,
} from '../tensor/index.ts'
import { checkScalar, reduce } from './_reduce.ts'
import { crossEntropy } from './classification.ts'

/**
 * torch.nn.functional.kl_div: the pointwise KL terms target (log target - input) for log-probability inputs, with
 * torch's reductions including 'batchmean' (the sum divided by the batch size, the true KL per sample; 'mean'
 * averages over every element, as torch does).
 *
 * @param input - Log-probabilities of the approximating distribution.
 * @param target - Probabilities (or log-probabilities with `logTarget`) of the reference distribution.
 * @param options - reduction (default 'mean') and logTarget.
 * @returns The loss, differentiable in the input (and target).
 * @concept kl-divergence
 * @example
 * ```ts
 * import { klDiv } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(klDiv([[Math.log(0.5), Math.log(0.5)]], [[1, 0]], { reduction: 'batchmean' })) === Math.LN2 // => true
 * ```
 */
export function klDiv(
  input: TensorLike,
  target: TensorLike,
  options: { readonly reduction?: Reduction | 'batchmean'; readonly logTarget?: boolean } = {},
): Value {
  const [x, t] = promoteFloat(input, target) as [Value, Value]
  const terms = options.logTarget === true ? mul(exp(t), sub(t, x)) : sub(xlogy(t, t), mul(t, x))
  const reduction = options.reduction ?? 'mean'
  if (reduction === 'batchmean') return div(sum(terms), x.shape.length === 0 ? 1 : (x.shape[0] ?? 1))
  return reduce(terms, { reduction })
}

/** Probability inputs as log-probabilities along the last axis. */
const _logProbs = (x: Value, logits: boolean): Value => (logits ? logSoftmax(x, -1) : log(x))
const probs = (x: Value, logits: boolean): Value => (logits ? softmax(x, -1) : x)

/**
 * KL(p || q) = sum_i p_i log(p_i / q_i) between categorical distributions along the last axis
 * (scipy.stats.entropy(p, q)), from probabilities or logits.
 *
 * @param p - The reference distribution `[..., K]`.
 * @param q - The approximating distribution `[..., K]`.
 * @param options - logits (inputs are logits; default false) and reduction over the leading axes.
 * @returns The divergence, differentiable.
 * @example
 * ```ts
 * import { klDivergence } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(klDivergence([1, 0], [0.5, 0.5])) === Math.LN2 // => true
 * ```
 */
export function klDivergence(
  p: TensorLike,
  q: TensorLike,
  options: { readonly logits?: boolean; readonly reduction?: Reduction } = {},
): Value {
  const [a, b] = promoteFloat(p, q) as [Value, Value]
  const logits = options.logits ?? false
  const pa = probs(a, logits)
  const terms = logits ? mul(pa, sub(logSoftmax(a, -1), logSoftmax(b, -1))) : sub(xlogy(pa, pa), xlogy(pa, b))
  return reduce(sum(terms, -1), { reduction: options.reduction ?? 'mean' })
}

/**
 * Jensen-Shannon divergence 0.5 KL(p || m) + 0.5 KL(q || m) with m = (p + q) / 2, symmetric and bounded by log 2
 * (the square of scipy.spatial.distance.jensenshannon).
 *
 * @param p - First distribution `[..., K]`.
 * @param q - Second distribution `[..., K]`.
 * @param options - logits, base of the logarithm (default e) and reduction over the leading axes.
 * @returns The divergence, differentiable.
 * @example
 * ```ts
 * import { jensenShannon } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(jensenShannon([1, 0], [0, 1], { base: 2 })) // => 1
 * ```
 */
export function jensenShannon(
  p: TensorLike,
  q: TensorLike,
  options: { readonly logits?: boolean; readonly base?: number; readonly reduction?: Reduction } = {},
): Value {
  const [a, b] = promoteFloat(p, q) as [Value, Value]
  const logits = options.logits ?? false
  const pa = probs(a, logits)
  const pb = probs(b, logits)
  const m = mul(add(pa, pb), 0.5)
  const kl = (x: Value): Value => sum(sub(xlogy(x, x), xlogy(x, m)), -1)
  let js = mul(add(kl(pa), kl(pb)), 0.5)
  if (options.base !== undefined) js = div(js, Math.log(options.base))
  return reduce(js, { reduction: options.reduction ?? 'mean' })
}

/** Options of {@link distillation}. */
export interface DistillationOptions {
  /** Softmax temperature T (default 1); the soft term is scaled by T^2 to keep its gradients comparable. */
  readonly temperature?: number
  /** Weight of the soft (teacher) term when hard labels are given (default 0.5). */
  readonly alpha?: number
  /** Hard class labels `[N]`: adds (1 - alpha) times the cross-entropy with them. */
  readonly target?: TensorLike
}

/**
 * Knowledge distillation (Hinton, Vinyals and Dean, 2015): T^2 KL(softmax(teacher / T) || softmax(student / T)) per
 * sample, averaged over the batch, optionally mixed with the cross-entropy on hard labels:
 * alpha T^2 KL + (1 - alpha) CE(student, y).
 *
 * @param studentLogits - Student logits `[N, C]`.
 * @param teacherLogits - Teacher logits `[N, C]` (treated as constants).
 * @param options - temperature, alpha and hard labels.
 * @returns The scalar loss, differentiable in the student logits.
 * @example
 * ```ts
 * import { distillation } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(distillation([[1, 2]], [[1, 2]], { temperature: 3 })) // => 0
 * ```
 */
export function distillation(
  studentLogits: TensorLike,
  teacherLogits: TensorLike,
  options: DistillationOptions = {},
): Value {
  checkScalar('distillation', 'temperature', options.temperature, 'positive')
  const [s, t] = promoteFloat(studentLogits, teacherLogits) as [Value, Value]
  const temp = options.temperature ?? 1
  const teacher = stopGradient(logSoftmax(div(t, temp), -1))
  const student = logSoftmax(div(s, temp), -1)
  const soft = mul(klDiv(student, teacher, { reduction: 'batchmean', logTarget: true }), temp * temp)
  if (options.target === undefined) return soft
  const alpha = options.alpha ?? 0.5
  return add(mul(soft, alpha), mul(crossEntropy(s, options.target), 1 - alpha))
}

/** Squared Euclidean distances between rows `[N, M]`. */
const squaredDistances = (x: Value, y: Value): Value => sum(square(sub(expandDims(x, 1), expandDims(y, 0))), -1)

/** Options of {@link mmd}. */
export interface MmdOptions {
  /** 'gaussian' exp(-d^2 / (2 s^2)) (default), 'laplacian' exp(-d / s), or 'energy' (-d, the energy distance). */
  readonly kernel?: 'gaussian' | 'laplacian' | 'energy'
  /** Bandwidths s; several are summed (multi-scale MMD). Default: the median heuristic on the pooled sample. */
  readonly bandwidths?: readonly number[]
  /** The unbiased U-statistic (diagonals excluded) rather than the biased V-statistic (default false). */
  readonly unbiased?: boolean
}

/**
 * Squared maximum mean discrepancy (Gretton et al., 2012) between samples x and y:
 * E k(x, x') + E k(y, y') - 2 E k(x, y), a differentiable two-sample distance for training generative models and
 * domain adaptation. The default bandwidth is the median pairwise distance of the pooled sample (no gradient).
 *
 * @param x - Samples `[N, D]`.
 * @param y - Samples `[M, D]`.
 * @param options - kernel, bandwidths and estimator.
 * @returns The scalar MMD^2, differentiable in both samples.
 * @example
 * ```ts
 * import { mmd } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * item(mmd([[0], [1]], [[0], [1]], { bandwidths: [1] })) // => 0
 * ```
 */
export function mmd(x: TensorLike, y: TensorLike, options: MmdOptions = {}): Value {
  const [a, b] = promoteFloat(x, y) as [Value, Value]
  const kind = options.kernel ?? 'gaussian'
  const dxx = squaredDistances(a, a)
  const dyy = squaredDistances(b, b)
  const dxy = squaredDistances(a, b)
  const bandwidths = options.bandwidths ?? (kind === 'energy' ? [1] : [medianDistance(dxx, dyy, dxy)])
  // sqrt of a clamped squared distance; exact zeros (the diagonal) get a zero gradient instead of NaN.
  const dist = (d2: Value): Value => {
    const pos = where(equal(d2, 0), 1, d2)
    return where(equal(d2, 0), 0, sqrt(maxZero(pos)))
  }
  const kernel = (d2: Value): Value => {
    if (kind === 'energy') return mul(dist(d2), -1)
    let total: Value | undefined
    for (const s of bandwidths) {
      const k = kind === 'gaussian' ? exp(div(maxZero(d2), -2 * s * s)) : exp(div(dist(d2), -s))
      total = total === undefined ? k : add(total, k)
    }
    return total ?? tensor(0)
  }
  const n = a.shape[0] ?? 1
  const m = b.shape[0] ?? 1
  const kxx = kernel(dxx)
  const kyy = kernel(dyy)
  const kxy = kernel(dxy)
  if (options.unbiased === true) {
    const offDiagonal = (k: Value, size: number): Value => div(sub(sum(k), sumDiagonal(k, size)), size * (size - 1))
    return sub(add(offDiagonal(kxx, n), offDiagonal(kyy, m)), mul(mean(kxy), 2))
  }
  return sub(add(mean(kxx), mean(kyy)), mul(mean(kxy), 2))
}

const maxZero = (v: Value): Value => relu(v)

function sumDiagonal(k: Value, size: number): Value {
  const eye = tensor(Array.from({ length: size }, (_, i) => Array.from({ length: size }, (_, j) => (i === j ? 1 : 0))))
  return sum(mul(k, eye))
}

/** The median pairwise distance of the pooled sample (over distinct pairs), as a plain number. */
function medianDistance(dxx: Value, dyy: Value, dxy: Value): number {
  const values: number[] = []
  const collect = (d: Value, square_: boolean): void => {
    const c = stopGradient(d)
    if (!isTensor(c)) return
    const flat = toFlat(c)
    const rows = d.shape[0] ?? 0
    const cols = d.shape[1] ?? 0
    for (let i = 0; i < rows; i++) {
      for (let j = square_ ? i + 1 : 0; j < cols; j++) values.push(Math.sqrt(Math.max(flat[i * cols + j] ?? 0, 0)))
    }
  }
  collect(dxx, true)
  collect(dyy, true)
  collect(dxy, false)
  if (values.length === 0)
    throw new OmaiError('value', 'mmd: the median heuristic needs concrete samples; pass bandwidths')
  values.sort((p, q) => p - q)
  const mid = values.length >> 1
  const med = values.length % 2 === 1 ? (values[mid] ?? 1) : ((values[mid - 1] ?? 1) + (values[mid] ?? 1)) / 2
  return med > 0 ? med : 1
}

/** Options of {@link sinkhorn}. */
export interface SinkhornOptions {
  /** Entropic regularisation epsilon (default 0.1). */
  readonly epsilon?: number
  /** Sample weights a `[N]` and b `[M]` (default uniform). */
  readonly a?: TensorLike
  readonly b?: TensorLike
  /** Iteration cap (default 1000) and marginal tolerance for stopping early (default 1e-9; checked on concrete values). */
  readonly maxIter?: number
  /** The iteration cap under the name `optim` uses (`maxIterations` wins when both are given). */
  readonly maxIterations?: number
  readonly tol?: number
  /** The debiased Sinkhorn divergence OT(x, y) - (OT(x, x) + OT(y, y)) / 2 (default false). */
  readonly debiased?: boolean
}

/** Log-domain Sinkhorn on a cost matrix; returns the transport cost <P, C>. */
function sinkhornCost(cost: Value, logA: Value, logB: Value, eps: number, maxIter: number, tol: number): Value {
  const logK = div(cost, -eps)
  let logU = mul(logA, 0)
  let logV = mul(logB, 0)
  for (let it = 0; it < maxIter; it++) {
    logV = sub(logB, logsumexp(add(logK, expandDims(logU, 1)), 0))
    logU = sub(logA, logsumexp(add(logK, expandDims(logV, 0)), 1))
    if (it % 10 === 9 && isTensor(logU) && isTensor(logV)) {
      // The row marginals hold exactly after the u-update; check the column marginals.
      const plan = exp(add(add(expandDims(logU, 1), logK), expandDims(logV, 0)))
      const err = toFlat(abs(sub(sum(plan, 0), exp(logB))))
      let worst = 0
      for (const e of err) worst = Math.max(worst, e)
      if (worst < tol) break
    }
  }
  const plan = exp(add(add(expandDims(logU, 1), logK), expandDims(logV, 0)))
  return sum(mul(plan, cost))
}

/**
 * Entropic optimal-transport loss between two weighted point clouds (Cuturi, 2013): the transport cost <P, C> of the
 * Sinkhorn plan for the squared Euclidean cost, computed in the log domain (as POT's `sinkhorn2` with
 * `method='sinkhorn_log'`), optionally debiased into the Sinkhorn divergence (POT's
 * `empirical_sinkhorn_divergence`). The iterations are unrolled, so the loss is differentiable in the points. That is
 * the difference from `sinkhorn` of `optim`, which solves one transport problem on a given cost matrix over plain
 * numbers (the plan, potentials and convergence flag, as a steppable `sinkhornAlgorithm`) and carries no gradient; the
 * two are different computations of one idea, not a copy of each other, and are exported under one name from each
 * folder.
 *
 * @param x - Points `[N, D]`.
 * @param y - Points `[M, D]`.
 * @param options - epsilon, weights, iteration limits and debiasing.
 * @returns The scalar loss.
 * @example
 * ```ts
 * import { sinkhorn } from 'omai-engine/losses'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(sinkhorn([[0], [1]], [[0], [1]], { epsilon: 0.01 }))) < 1e-12 // => true
 * ```
 */
export function sinkhorn(x: TensorLike, y: TensorLike, options: SinkhornOptions = {}): Value {
  checkScalar('sinkhorn', 'epsilon', options.epsilon, 'positive')
  const [p, q] = promoteFloat(x, y) as [Value, Value]
  const n = p.shape[0] ?? 1
  const m = q.shape[0] ?? 1
  const eps = options.epsilon ?? 0.1
  const maxIter = options.maxIterations ?? options.maxIter ?? 1000
  const tol = options.tol ?? 1e-9
  const weights = (w: TensorLike | undefined, k: number): Value =>
    w === undefined ? tensor(Array.from({ length: k }, () => Math.log(1 / k))) : log(promoteFloat(w)[0] as Value)
  const logA = weights(options.a, n)
  const logB = weights(options.b, m)
  const ot = (u: Value, v: Value, la: Value, lb: Value): Value =>
    sinkhornCost(maxZero(squaredDistances(u, v)), la, lb, eps, maxIter, tol)
  const xy = ot(p, q, logA, logB)
  if (options.debiased !== true) return xy
  return sub(xy, mul(add(ot(p, p, logA, logA), ot(q, q, logB, logB)), 0.5))
}

/**
 * The sliced Wasserstein distance (Rabin et al., 2012; Bonneel et al., 2015): project both samples onto random
 * directions on the sphere, solve the one-dimensional optimal transport in each by sorting, and average. Cheap
 * (O(L N log N)) and differentiable, so it trains generators where full transport is too slow. It is not
 * `slicedWasserstein` of `optim`, which returns the p-th root always (this returns the power `SW_p^p` unless `root` is
 * set), takes its directions as a `[d, L]` matrix or its own key option, and accepts weights and clouds of different
 * sizes; this one is one vectorised sort over all directions and needs equal sample counts, which is why the two are
 * kept as separate implementations.
 *
 * @param x - Samples `[N, D]`.
 * @param y - Samples `[N, D]` (the same count as x).
 * @param key - Random key for the directions.
 * @param options - projections (default 50), the order p (default 2) and `root` to return the p-th root SW_p rather
 *   than SW_p^p (default false: the power is smooth where the root has an infinite slope at zero).
 * @returns The scalar distance, differentiable in both samples.
 * @index function Sliced Wasserstein distance by random projections; returns the power, root with a flag.
 * @example
 * ```ts
 * import { slicedWasserstein } from 'omai-engine/losses'
 * import { key } from 'omai-engine/random'
 * import { item } from 'omai-engine/tensor'
 * item(slicedWasserstein([[0, 0], [1, 1]], [[1, 1], [0, 0]], key(0))) // => 0
 * ```
 */
export function slicedWasserstein(
  x: TensorLike,
  y: TensorLike,
  key: Key,
  options: { readonly projections?: number; readonly p?: number; readonly root?: boolean } = {},
): Value {
  const [a, b] = promoteFloat(x, y) as [Value, Value]
  if (a.shape[0] !== b.shape[0] || a.shape[1] !== b.shape[1])
    throw new OmaiError('shape', 'slicedWasserstein: the samples must have the same shape')
  const p = options.p ?? 2
  const directions = normal(key, [options.projections ?? 50, a.shape[1] ?? 1])
  const unit = div(directions, sqrt(sum(square(directions), -1, true)))
  const pa = sort(matmul(a, transpose(unit)), 0)
  const pb = sort(matmul(b, transpose(unit)), 0)
  const power = mean(pow(abs(sub(pa, pb)), p))
  return options.root === true ? pow(power, 1 / p) : power
}
