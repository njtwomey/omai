/**
 * Estimators of entropy and mutual information from samples rather than known distributions: plug-in estimates for
 * discrete data with the Miller-Madow bias correction, kernel density estimates for continuous data, Schreiber's
 * transfer entropy (directed information flow between time series), and the InfoNCE, NWJ and MINE variational lower
 * bounds computed from a critic's scores.
 */

import { logSumExpOf, MdmlError, meanOf } from '../base/index.ts'
import { cdist } from '../numerics/geometry/index.ts'
import { cholesky, choleskyLogdet, inv } from '../numerics/linalg/index.ts'
import {
  add,
  diagonal,
  div,
  exp,
  eye,
  greater,
  item,
  logsumexp,
  matmul,
  mean,
  mul,
  sub,
  sum,
  type TensorLike,
  toFlat,
  transpose,
  type Value,
  where,
} from '../tensor/index.ts'
import { concat, points, tensorOf } from './_samples.ts'
import { countsOf, entropyOfCounts, float, jointCodes, type Labels, labelCodes, logOfBase } from './_shared.ts'
import { ksgConditionalMutualInformation } from './neighbour-estimators.ts'

/** Options of the plug-in estimators. */
export interface PlugInOptions {
  /** Logarithm base (default e). */
  readonly base?: number
  /** `'millerMadow'` adds the first-order bias correction (m - 1) / (2 N) of each entropy (default `'none'`). */
  readonly correction?: 'none' | 'millerMadow'
}

function plugIn(codes: { codes: Int32Array; m: number }, correction: PlugInOptions['correction']): number {
  const counts = countsOf(codes.codes, codes.m)
  const h = entropyOfCounts(counts)
  return correction === 'millerMadow' ? h + (codes.m - 1) / (2 * codes.codes.length) : h
}

/**
 * The plug-in (maximum-likelihood) entropy of discrete samples, the entropy of their empirical frequencies, optionally
 * with the Miller-Madow correction (m - 1) / (2 N) for the downward bias (m observed values, N samples).
 *
 * @param samples - Discrete observations (numbers or strings).
 * @param options - Base and correction.
 * @returns The estimated entropy.
 * @concept entropy-estimation
 * @example
 * ```ts
 * import { plugInEntropy } from 'mdml-engine/information'
 * plugInEntropy(['a', 'b', 'a', 'b'], { base: 2 }) // => 1
 * plugInEntropy(['a', 'b', 'a', 'b'], { base: 2, correction: 'millerMadow' }) // => 1.1803368801111204
 * ```
 */
export function plugInEntropy(samples: Labels, options: PlugInOptions = {}): number {
  return plugIn(labelCodes(samples, 'plugInEntropy'), options.correction) / logOfBase(options.base)
}

/**
 * The plug-in mutual information of paired discrete samples, H(X) + H(Y) - H(X, Y) of the empirical frequencies (equal
 * to sklearn.metrics.mutual_info_score), optionally with each entropy Miller-Madow corrected.
 *
 * @param x - Discrete observations of X.
 * @param y - Paired observations of Y.
 * @param options - Base and correction.
 * @returns The estimated mutual information.
 * @concept mutual-information
 * @example
 * ```ts
 * import { plugInMutualInformation } from 'mdml-engine/information'
 * plugInMutualInformation([0, 0, 1, 1], ['a', 'a', 'b', 'b'], { base: 2 }) // => 1
 * ```
 */
export function plugInMutualInformation(x: Labels, y: Labels, options: PlugInOptions = {}): number {
  const a = labelCodes(x, 'plugInMutualInformation')
  const b = labelCodes(y, 'plugInMutualInformation')
  if (a.codes.length !== b.codes.length) throw new MdmlError('shape', 'plugInMutualInformation: lengths differ')
  const c = options.correction
  return (plugIn(a, c) + plugIn(b, c) - plugIn(jointCodes(a, b), c)) / logOfBase(options.base)
}

/** Options of the kernel estimators. */
export interface KernelEstimatorOptions {
  /** Bandwidth rule or factor multiplying the data covariance's square root, as scipy.stats.gaussian_kde (default `'scott'`). */
  readonly bandwidth?: 'scott' | 'silverman' | number
  /** Leave each sample out of its own density estimate (default false: the resubstitution estimate). */
  readonly leaveOneOut?: boolean
  /** Logarithm base (default e). */
  readonly base?: number
}

/**
 * The kernel (resubstitution) estimate of differential entropy, -mean log f(x_i), with f the Gaussian kernel density
 * estimate of scipy.stats.gaussian_kde: kernel covariance the sample covariance times the squared bandwidth factor
 * (Scott's n^(-1/(d+4)) by default). With `leaveOneOut` each sample is left out of its own estimate (Ahmad and Lin).
 *
 * @param x - Samples `[n]` or `[n, d]`.
 * @param options - Bandwidth, leave-one-out and base.
 * @returns The estimated entropy.
 * @concept entropy-estimation
 * @example
 * ```ts
 * import { kdeEntropy } from 'mdml-engine/information'
 * import { key, normal } from 'mdml-engine/random'
 * // Standard normal samples: the entropy is log(2 pi e) / 2 = 1.4189 nats.
 * Math.abs(kdeEntropy(normal(key(0), [400])) - 1.4189) < 0.1 // => true
 * ```
 */
export function kdeEntropy(x: TensorLike, options: KernelEstimatorOptions = {}): number {
  const p = points(x, 'kdeEntropy')
  const loo = options.leaveOneOut === true
  if (p.n < 2 + (loo ? 1 : 0)) throw new MdmlError('value', 'kdeEntropy: needs more samples')
  const bw = options.bandwidth ?? 'scott'
  const factor =
    typeof bw === 'number' ? bw : bw === 'scott' ? p.n ** (-1 / (p.d + 4)) : ((p.n * (p.d + 2)) / 4) ** (-1 / (p.d + 4))
  const t = tensorOf(p)
  const centred = sub(t, mean(t, 0, true))
  const covariance = mul(div(matmul(transpose(centred), centred), p.n - 1), factor * factor)
  const logNorm = 0.5 * (p.d * Math.log(2 * Math.PI) + item(choleskyLogdet(cholesky(covariance))))
  const d2 = toFlat(cdist(t, t, 'mahalanobis', { vi: inv(covariance) }))
  const logDensity = Array.from({ length: p.n }, (_, i) => {
    const terms: number[] = []
    for (let j = 0; j < p.n; j++) if (!(loo && i === j)) terms.push(-0.5 * (d2[i * p.n + j] ?? 0) ** 2)
    return logSumExpOf(terms) - Math.log(terms.length) - logNorm
  })
  return -meanOf(logDensity) / logOfBase(options.base)
}

/**
 * The kernel estimate of mutual information between continuous variables, H(X) + H(Y) - H(X, Y) with each entropy from
 * {@link kdeEntropy} (each with its own bandwidth by the same rule).
 *
 * @param x - Samples of X, `[n]` or `[n, dx]`.
 * @param y - Paired samples of Y.
 * @param options - Bandwidth, leave-one-out and base.
 * @returns The estimate.
 * @concept mutual-information
 * @example
 * ```ts
 * import { kdeMutualInformation } from 'mdml-engine/information'
 * const x = Array.from({ length: 50 }, (_, i) => Math.sin(i))
 * kdeMutualInformation(x, x.map((v, i) => v + 0.05 * Math.cos(3 * i))) > 1 // => true
 * ```
 */
export function kdeMutualInformation(x: TensorLike, y: TensorLike, options: KernelEstimatorOptions = {}): number {
  const px = points(x, 'kdeMutualInformation')
  const py = points(y, 'kdeMutualInformation')
  return (
    kdeEntropy(tensorOf(px), options) +
    kdeEntropy(tensorOf(py), options) -
    kdeEntropy(tensorOf(concat(px, py)), options)
  )
}

/** Options of {@link transferEntropy}. */
export interface TransferEntropyOptions {
  /** Length k of the target's own history (default 1). */
  readonly history?: number
  /** Length l of the source's history (default 1). */
  readonly sourceHistory?: number
  /** `'plugIn'` for discrete series (default), `'ksg'` for continuous ones (Frenzel-Pompe conditional MI). */
  readonly estimator?: 'plugIn' | 'ksg'
  /** Neighbours of the KSG estimator (default 3). */
  readonly neighbours?: number
  /** Logarithm base (default e). */
  readonly base?: number
}

/**
 * Schreiber's transfer entropy from a source series Y to a target series X,
 * T(Y -> X) = I(X_(t+1); Y_t^(l) | X_t^(k)): how much the source's recent past adds to predicting the target beyond
 * the target's own past. Plug-in frequencies for discrete series (as pyinform's `transfer_entropy` for l = 1), the
 * KSG conditional mutual information for continuous ones.
 *
 * @param source - The source series Y.
 * @param target - The target series X (same length).
 * @param options - History lengths, estimator, neighbours and base.
 * @returns The transfer entropy (>= 0 for the plug-in estimator).
 * @concept transfer-entropy
 * @example
 * ```ts
 * import { transferEntropy } from 'mdml-engine/information'
 * // The target copies the source one step later: one bit flows from source to target, none back.
 * const y = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
 * const x = [0, ...y.slice(0, -1)]
 * transferEntropy(y, x, { base: 2 }) > 0.8 // => true
 * transferEntropy(x, y, { base: 2 }) < 0.2 // => true
 * ```
 */
export function transferEntropy(source: TensorLike, target: TensorLike, options: TransferEntropyOptions = {}): number {
  const ys = toFlat(float(source))
  const xs = toFlat(float(target))
  if (ys.length !== xs.length) throw new MdmlError('shape', 'transferEntropy: series differ in length')
  const k = options.history ?? 1
  const l = options.sourceHistory ?? 1
  const start = Math.max(k, l) - 1
  const rows = xs.length - 1 - start
  if (rows < 2) throw new MdmlError('value', 'transferEntropy: series too short for the histories')
  const future: number[] = []
  const past: number[][] = []
  const sourcePast: number[][] = []
  for (let t = start; t < xs.length - 1; t++) {
    future.push(xs[t + 1] ?? 0)
    past.push(xs.slice(t - k + 1, t + 1))
    sourcePast.push(ys.slice(t - l + 1, t + 1))
  }
  if ((options.estimator ?? 'plugIn') === 'ksg') {
    return ksgConditionalMutualInformation(future, sourcePast, past, {
      k: options.neighbours ?? 3,
      ...(options.base === undefined ? {} : { base: options.base }),
    })
  }
  const code = (vs: readonly number[][]): { codes: Int32Array; m: number } =>
    labelCodes(
      vs.map((v) => v.join(',')),
      'transferEntropy',
    )
  const f = labelCodes(future, 'transferEntropy')
  const xp = code(past)
  const yp = code(sourcePast)
  const h = (...parts: { codes: Int32Array; m: number }[]): number => {
    const joint = jointCodes(...parts)
    return entropyOfCounts(countsOf(joint.codes, joint.m))
  }
  return (h(f, xp) - h(xp) - h(f, xp, yp) + h(xp, yp)) / logOfBase(options.base)
}

function scoreMatrix(scores: TensorLike, name: string): { s: Value; n: number } {
  const s = float(scores)
  const n = s.shape[0] ?? 0
  if (s.shape.length !== 2 || s.shape[1] !== n || n < 2)
    throw new MdmlError('shape', `${name}: scores must be a square [n, n] matrix with n >= 2`)
  return { s, n }
}

/**
 * The InfoNCE lower bound on mutual information (van den Oord et al. 2018; Poole et al. 2019) from a matrix of critic
 * scores f(x_i, y_j) whose diagonal holds the paired samples: log n + mean_i (f_ii - logsumexp_j f_ij). It never exceeds
 * log n. Differentiable in the scores, so a critic can be trained by maximising it.
 *
 * @param scores - Critic scores `[n, n]`, positives on the diagonal.
 * @returns The bound in nats.
 * @concept variational-mutual-information
 * @example
 * ```ts
 * import { infoNce } from 'mdml-engine/information'
 * import { item } from 'mdml-engine/tensor'
 * // A critic that cannot tell pairs apart gives 0; a perfect one approaches log n.
 * Math.abs(item(infoNce([[1, 1], [1, 1]]))) < 1e-12 // => true
 * item(infoNce([[50, 0], [0, 50]])) // => 0.6931471805599453
 * ```
 */
export function infoNce(scores: TensorLike): Value {
  const { s, n } = scoreMatrix(scores, 'infoNce')
  return add(mean(sub(diagonal(s), logsumexp(s, 1))), Math.log(n))
}

/**
 * The NWJ (Nguyen-Wainwright-Jordan, f-GAN KL) lower bound on mutual information from critic scores:
 * mean_i f_ii - e^(-1) mean_(i != j) e^(f_ij), the off-diagonal pairs standing in for the product of the marginals.
 * Unbiased but with high variance; tight when f = 1 + log(p(x, y) / (p(x) p(y))).
 *
 * @param scores - Critic scores `[n, n]`, positives on the diagonal.
 * @returns The bound in nats.
 * @concept variational-mutual-information
 * @example
 * ```ts
 * import { nwjBound } from 'mdml-engine/information'
 * import { item } from 'mdml-engine/tensor'
 * item(nwjBound([[1, 1], [1, 1]])) // => 0
 * ```
 */
export function nwjBound(scores: TensorLike): Value {
  const { s, n } = scoreMatrix(scores, 'nwjBound')
  const off = sub(1, eye(n))
  const marginal = div(sum(mul(exp(sub(s, 1)), off)), n * (n - 1))
  return sub(mean(diagonal(s)), marginal)
}

/**
 * The Donsker-Varadhan lower bound used by MINE (Belghazi et al. 2018) from critic scores:
 * mean_i f_ii - log mean_(i != j) e^(f_ij). Tighter than NWJ for the same critic; its minibatch gradient is biased, which
 * MINE corrects with a moving average of the denominator.
 *
 * @param scores - Critic scores `[n, n]`, positives on the diagonal.
 * @returns The bound in nats.
 * @concept variational-mutual-information
 * @example
 * ```ts
 * import { mineBound } from 'mdml-engine/information'
 * import { item } from 'mdml-engine/tensor'
 * item(mineBound([[3, 1], [1, 3]])) // => 2
 * ```
 */
export function mineBound(scores: TensorLike): Value {
  const { s, n } = scoreMatrix(scores, 'mineBound')
  const offDiagonal = where(greater(eye(n), 0), Number.NEGATIVE_INFINITY, s)
  return sub(mean(diagonal(s)), sub(logsumexp(offDiagonal), Math.log(n * (n - 1))))
}
