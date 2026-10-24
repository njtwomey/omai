import { logSumExpOf, meanOf, WhiteboxError } from '../base/index.ts'
import { nextAfter } from '../numerics/approximation/index.ts'
import { cdist } from '../numerics/geometry/index.ts'
import { cholesky, choleskyLogdet, inv } from '../numerics/linalg/index.ts'
import { kdTree, type SpatialTree } from '../numerics/spatial/index.ts'
import { digamma, lgamma } from '../numerics/special/index.ts'
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
import { countsOf, entropyOfCounts, float, jointCodes, type Labels, labelCodes, logOfBase } from './_shared.ts'

/** Samples as rows: `[n]` (one variable) or `[n, d]`. */
interface Points {
  readonly data: Float64Array
  readonly n: number
  readonly d: number
}

function points(x: TensorLike, name: string): Points {
  const v = float(x)
  if (v.shape.length !== 1 && v.shape.length !== 2)
    throw new WhiteboxError('shape', `${name}: samples must be [n] or [n, d]`)
  const n = v.shape[0] ?? 0
  return { data: Float64Array.from(toFlat(v)), n, d: v.shape.length === 1 ? 1 : (v.shape[1] ?? 0) }
}

function concat(...parts: readonly Points[]): Points {
  const n = parts[0]?.n ?? 0
  if (parts.some((p) => p.n !== n)) throw new WhiteboxError('shape', 'estimators: samples differ in number')
  const d = parts.reduce((s, p) => s + p.d, 0)
  const data = new Float64Array(n * d)
  for (let i = 0; i < n; i++) {
    let o = 0
    for (const p of parts) {
      for (let j = 0; j < p.d; j++) data[i * d + o + j] = p.data[i * p.d + j] ?? 0
      o += p.d
    }
  }
  return { data, n, d }
}

function tensorOf(p: Points): Value {
  return float(
    p.data.length === 0
      ? []
      : Array.from({ length: p.n }, (_, i) => Array.from(p.data.subarray(i * p.d, (i + 1) * p.d))),
  )
}

function row(p: Points, i: number): number[][] {
  return [Array.from(p.data.subarray(i * p.d, (i + 1) * p.d))]
}

/** Digamma of each number. */
function psi(xs: ArrayLike<number>): number[] {
  return toFlat(digamma(float(Float64Array.from(xs as ArrayLike<number>))))
}

/** Count, for each point, the others strictly within its radius (the point itself is included in the count). */
function countWithin(tree: SpatialTree, p: Points, radii: ArrayLike<number>, strict: boolean): number[] {
  return Array.from({ length: p.n }, (_, i) => {
    const r = radii[i] ?? 0
    return tree.countNeighbours(row(p, i), strict ? nextAfter(r, 0) : r)[0] ?? 0
  })
}

function checkK(k: number, n: number, name: string): void {
  if (!Number.isInteger(k) || k < 1 || k >= n)
    throw new WhiteboxError('value', `${name}: k must be an integer in [1, n - 1], got ${k} with n = ${n}`)
}

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
 * import { plugInEntropy } from 'whitebox-engine/information'
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
 * import { plugInMutualInformation } from 'whitebox-engine/information'
 * plugInMutualInformation([0, 0, 1, 1], ['a', 'a', 'b', 'b'], { base: 2 }) // => 1
 * ```
 */
export function plugInMutualInformation(x: Labels, y: Labels, options: PlugInOptions = {}): number {
  const a = labelCodes(x, 'plugInMutualInformation')
  const b = labelCodes(y, 'plugInMutualInformation')
  if (a.codes.length !== b.codes.length) throw new WhiteboxError('shape', 'plugInMutualInformation: lengths differ')
  const c = options.correction
  return (plugIn(a, c) + plugIn(b, c) - plugIn(jointCodes(a, b), c)) / logOfBase(options.base)
}

/** Options of the k-nearest-neighbour estimators. */
export interface NeighbourOptions {
  /** Neighbours (default 3). */
  readonly k?: number
  /** Logarithm base (default e). */
  readonly base?: number
}

/**
 * The Kozachenko-Leonenko k-nearest-neighbour estimate of differential entropy,
 * psi(N) - psi(k) + log c_d + (d / N) sum log eps_i, with eps_i the distance from sample i to its k-th neighbour (found
 * with a KD tree) and c_d the volume of the unit ball: 2^d for the max norm (the default, as NPEET) or
 * pi^(d/2) / Gamma(d/2 + 1) for the Euclidean norm. Repeated samples give zero distances; add a little jitter first.
 *
 * @param x - Samples `[n]` or `[n, d]` of a continuous variable.
 * @param options - Neighbours, base and the norm.
 * @returns The estimated entropy.
 * @concept entropy-estimation
 * @example
 * ```ts
 * import { kozachenkoLeonenkoEntropy } from 'whitebox-engine/information'
 * import { key, uniform } from 'whitebox-engine/random'
 * // Uniform samples on [0, 1]: the entropy is log 1 = 0.
 * Math.abs(kozachenkoLeonenkoEntropy(uniform(key(0), [500]))) < 0.1 // => true
 * ```
 */
export function kozachenkoLeonenkoEntropy(
  x: TensorLike,
  options: NeighbourOptions & { readonly metric?: 'chebyshev' | 'euclidean' } = {},
): number {
  const p = points(x, 'kozachenkoLeonenkoEntropy')
  const k = options.k ?? 3
  checkK(k, p.n, 'kozachenkoLeonenkoEntropy')
  const metric = options.metric ?? 'chebyshev'
  const tree = kdTree(tensorOf(p), { metric })
  const dist = toFlat(tree.query(tensorOf(p), k + 1).distances)
  const logs = Array.from({ length: p.n }, (_, i) => Math.log(dist[i * (k + 1) + k] ?? 0))
  const logVolume = metric === 'chebyshev' ? p.d * Math.LN2 : (p.d / 2) * Math.log(Math.PI) - item(lgamma(p.d / 2 + 1))
  const [psiN = 0, psiK = 0] = psi([p.n, k])
  return (psiN - psiK + logVolume + p.d * meanOf(logs)) / logOfBase(options.base)
}

/**
 * The Kraskov-Stoegbauer-Grassberger k-nearest-neighbour estimate of mutual information between continuous variables
 * (max norm in the joint space, KD trees for the searches). Algorithm 1 (the default, as NPEET and sklearn):
 * psi(k) + psi(N) - mean(psi(n_x + 1) + psi(n_y + 1)), counting the points strictly closer in each marginal than the
 * k-th joint neighbour; algorithm 2: psi(k) - 1/k + psi(N) - mean(psi(n_x) + psi(n_y)) with marginal radii.
 *
 * @param x - Samples `[n]` or `[n, dx]` of X.
 * @param y - Paired samples `[n]` or `[n, dy]` of Y.
 * @param options - Neighbours, base and the algorithm (1 or 2).
 * @returns The estimate (it can be slightly negative for independent variables).
 * @concept mutual-information
 * @example
 * ```ts
 * import { ksgMutualInformation } from 'whitebox-engine/information'
 * const x = Array.from({ length: 100 }, (_, i) => Math.sin(i))
 * // Y = X carries a lot of information; Y = a scrambled order carries little.
 * ksgMutualInformation(x, x.map((v) => 2 * v + 1)) > 2 // => true
 * ```
 */
export function ksgMutualInformation(
  x: TensorLike,
  y: TensorLike,
  options: NeighbourOptions & { readonly algorithm?: 1 | 2 } = {},
): number {
  const px = points(x, 'ksgMutualInformation')
  const py = points(y, 'ksgMutualInformation')
  const joint = concat(px, py)
  const n = joint.n
  const k = options.k ?? 3
  checkK(k, n, 'ksgMutualInformation')
  const jt = kdTree(tensorOf(joint), { metric: 'chebyshev' })
  const tx = kdTree(tensorOf(px), { metric: 'chebyshev' })
  const ty = kdTree(tensorOf(py), { metric: 'chebyshev' })
  const found = jt.query(tensorOf(joint), k + 1)
  const dist = toFlat(found.distances)
  const [psiN = 0, psiK = 0] = psi([n, k])
  if ((options.algorithm ?? 1) === 1) {
    const eps = Array.from({ length: n }, (_, i) => dist[i * (k + 1) + k] ?? 0)
    const nx = countWithin(tx, px, eps, true)
    const ny = countWithin(ty, py, eps, true)
    return (psiK + psiN - meanOf(psi(nx)) - meanOf(psi(ny))) / logOfBase(options.base)
  }
  const idx = toFlat(found.indices)
  const ex = new Float64Array(n)
  const ey = new Float64Array(n)
  const maxNorm = (p: Points, i: number, j: number): number => {
    let m = 0
    for (let c = 0; c < p.d; c++) m = Math.max(m, Math.abs((p.data[i * p.d + c] ?? 0) - (p.data[j * p.d + c] ?? 0)))
    return m
  }
  for (let i = 0; i < n; i++) {
    const neighbours = Array.from({ length: k + 1 }, (_, j) => idx[i * (k + 1) + j] ?? 0).filter((j) => j !== i)
    for (const j of neighbours.slice(0, k)) {
      ex[i] = Math.max(ex[i] ?? 0, maxNorm(px, i, j))
      ey[i] = Math.max(ey[i] ?? 0, maxNorm(py, i, j))
    }
  }
  const nx = countWithin(tx, px, ex, false).map((c) => c - 1)
  const ny = countWithin(ty, py, ey, false).map((c) => c - 1)
  return (psiK - 1 / k + psiN - meanOf(psi(nx)) - meanOf(psi(ny))) / logOfBase(options.base)
}

/**
 * The Frenzel-Pompe (KSG-style) k-nearest-neighbour estimate of conditional mutual information I(X; Y | Z),
 * psi(k) - mean(psi(n_xz + 1) + psi(n_yz + 1) - psi(n_z + 1)), as NPEET's `cmi`.
 *
 * @param x - Samples of X, `[n]` or `[n, dx]`.
 * @param y - Samples of Y.
 * @param z - Samples of the conditioning variable Z.
 * @param options - Neighbours and base.
 * @returns The estimate.
 * @concept mutual-information
 * @example
 * ```ts
 * import { ksgConditionalMutualInformation } from 'whitebox-engine/information'
 * const z = Array.from({ length: 100 }, (_, i) => Math.sin(i))
 * const x = z.map((v, i) => v + 0.1 * Math.cos(7 * i))
 * const y = z.map((v, i) => v + 0.1 * Math.cos(11 * i))
 * // X and Y share Z; given Z little is left.
 * ksgConditionalMutualInformation(x, y, z) < 0.3 // => true
 * ```
 */
export function ksgConditionalMutualInformation(
  x: TensorLike,
  y: TensorLike,
  z: TensorLike,
  options: NeighbourOptions = {},
): number {
  const px = points(x, 'ksgConditionalMutualInformation')
  const py = points(y, 'ksgConditionalMutualInformation')
  const pz = points(z, 'ksgConditionalMutualInformation')
  const joint = concat(px, py, pz)
  const k = options.k ?? 3
  checkK(k, joint.n, 'ksgConditionalMutualInformation')
  const dist = toFlat(kdTree(tensorOf(joint), { metric: 'chebyshev' }).query(tensorOf(joint), k + 1).distances)
  const eps = Array.from({ length: joint.n }, (_, i) => dist[i * (k + 1) + k] ?? 0)
  const avg = (p: Points): number =>
    meanOf(psi(countWithin(kdTree(tensorOf(p), { metric: 'chebyshev' }), p, eps, true)))
  const xz = concat(px, pz)
  const yz = concat(py, pz)
  const [psiK = 0] = psi([k])
  return (psiK - avg(xz) - avg(yz) + avg(pz)) / logOfBase(options.base)
}

/**
 * The k-nearest-neighbour estimate of the KL divergence KL(p || q) from samples x ~ p and y ~ q (Wang, Kulkarni and
 * Verdu 2009; as NPEET's `kldiv`): (d / n) sum log(nu_i / rho_i) + log(m / (n - 1)), with rho_i the max-norm distance
 * from x_i to its k-th neighbour among the other x and nu_i to its k-th neighbour among the y.
 *
 * @param x - Samples `[n]` or `[n, d]` from p.
 * @param y - Samples `[m]` or `[m, d]` from q.
 * @param options - Neighbours and base.
 * @returns The estimate (it can be slightly negative when p = q).
 * @concept kl-divergence
 * @example
 * ```ts
 * import { knnKlDivergence } from 'whitebox-engine/information'
 * import { key, normal } from 'whitebox-engine/random'
 * import { add } from 'whitebox-engine/tensor'
 * // N(0, 1) against N(1, 1): the divergence is 1/2.
 * Math.abs(knnKlDivergence(normal(key(0), [500]), add(normal(key(1), [500]), 1)) - 0.5) < 0.15 // => true
 * ```
 */
export function knnKlDivergence(x: TensorLike, y: TensorLike, options: NeighbourOptions = {}): number {
  const px = points(x, 'knnKlDivergence')
  const py = points(y, 'knnKlDivergence')
  if (px.d !== py.d) throw new WhiteboxError('shape', 'knnKlDivergence: samples differ in dimension')
  const k = options.k ?? 3
  checkK(k, Math.min(px.n, py.n + 1), 'knnKlDivergence')
  const tx = tensorOf(px)
  const rho = toFlat(kdTree(tx, { metric: 'chebyshev' }).query(tx, k + 1).distances)
  const nu = toFlat(kdTree(tensorOf(py), { metric: 'chebyshev' }).query(tx, k).distances)
  const logRatio = Array.from(
    { length: px.n },
    (_, i) => Math.log(nu[i * k + k - 1] ?? 0) - Math.log(rho[i * (k + 1) + k] ?? 0),
  )
  return (px.d * meanOf(logRatio) + Math.log(py.n / (px.n - 1))) / logOfBase(options.base)
}

/**
 * Ross's k-nearest-neighbour estimate of the mutual information between a continuous variable and a discrete one
 * (sklearn.feature_selection.mutual_info_classif): for each sample, the distance to its k-th neighbour among samples of
 * the same label sets a radius, and psi(N) + mean(psi(k)) - mean(psi(N_label)) - mean(psi(m)) counts all samples within
 * it. Samples whose label occurs once are ignored.
 *
 * @param x - Continuous samples `[n]` or `[n, d]`.
 * @param labels - Their discrete labels.
 * @param options - Neighbours (default 3), base and the norm (default max norm, the absolute difference in 1-D).
 * @returns The estimate.
 * @concept mutual-information
 * @example
 * ```ts
 * import { mixedMutualInformation } from 'whitebox-engine/information'
 * const x = [0.1, 0.2, 0.3, 0.4, 0.5, 5.1, 5.2, 5.3, 5.4, 5.5]
 * const y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
 * // Separated classes: close to log 2.
 * mixedMutualInformation(x, y) > 0.6 // => true
 * ```
 */
export function mixedMutualInformation(
  x: TensorLike,
  labels: Labels,
  options: NeighbourOptions & { readonly metric?: 'chebyshev' | 'euclidean' } = {},
): number {
  const p = points(x, 'mixedMutualInformation')
  const { codes, m } = labelCodes(labels, 'mixedMutualInformation')
  if (codes.length !== p.n) throw new WhiteboxError('shape', 'mixedMutualInformation: lengths differ')
  const metric = options.metric ?? 'chebyshev'
  const counts = countsOf(codes, m)
  const radius = new Float64Array(p.n)
  const kAll = new Float64Array(p.n)
  for (let c = 0; c < m; c++) {
    const members = Array.from({ length: p.n }, (_, i) => i).filter((i) => codes[i] === c)
    if (members.length < 2) continue
    const k = Math.min(options.k ?? 3, members.length - 1)
    const sub: Points = {
      data: Float64Array.from(members.flatMap((i) => Array.from(p.data.subarray(i * p.d, (i + 1) * p.d)))),
      n: members.length,
      d: p.d,
    }
    const dist = toFlat(kdTree(tensorOf(sub), { metric }).query(tensorOf(sub), k + 1).distances)
    members.forEach((i, j) => {
      radius[i] = dist[j * (k + 1) + k] ?? 0
      kAll[i] = k
    })
  }
  const kept = Array.from({ length: p.n }, (_, i) => i).filter((i) => (counts[codes[i] ?? 0] ?? 0) > 1)
  if (kept.length === 0) throw new WhiteboxError('value', 'mixedMutualInformation: every label occurs once')
  const q: Points = {
    data: Float64Array.from(kept.flatMap((i) => Array.from(p.data.subarray(i * p.d, (i + 1) * p.d)))),
    n: kept.length,
    d: p.d,
  }
  const mAll = countWithin(
    kdTree(tensorOf(q), { metric }),
    q,
    kept.map((i) => radius[i] ?? 0),
    true,
  )
  const [psiN = 0] = psi([kept.length])
  const value =
    psiN +
    meanOf(psi(kept.map((i) => kAll[i] ?? 0))) -
    meanOf(psi(kept.map((i) => counts[codes[i] ?? 0] ?? 0))) -
    meanOf(psi(mAll))
  return value / logOfBase(options.base)
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
 * import { kdeEntropy } from 'whitebox-engine/information'
 * import { key, normal } from 'whitebox-engine/random'
 * // Standard normal samples: the entropy is log(2 pi e) / 2 = 1.4189 nats.
 * Math.abs(kdeEntropy(normal(key(0), [400])) - 1.4189) < 0.1 // => true
 * ```
 */
export function kdeEntropy(x: TensorLike, options: KernelEstimatorOptions = {}): number {
  const p = points(x, 'kdeEntropy')
  const loo = options.leaveOneOut === true
  if (p.n < 2 + (loo ? 1 : 0)) throw new WhiteboxError('value', 'kdeEntropy: needs more samples')
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
 * import { kdeMutualInformation } from 'whitebox-engine/information'
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
 * import { transferEntropy } from 'whitebox-engine/information'
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
  if (ys.length !== xs.length) throw new WhiteboxError('shape', 'transferEntropy: series differ in length')
  const k = options.history ?? 1
  const l = options.sourceHistory ?? 1
  const start = Math.max(k, l) - 1
  const rows = xs.length - 1 - start
  if (rows < 2) throw new WhiteboxError('value', 'transferEntropy: series too short for the histories')
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
    throw new WhiteboxError('shape', `${name}: scores must be a square [n, n] matrix with n >= 2`)
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
 * import { infoNce } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
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
 * import { nwjBound } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
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
 * import { mineBound } from 'whitebox-engine/information'
 * import { item } from 'whitebox-engine/tensor'
 * item(mineBound([[3, 1], [1, 3]])) // => 2
 * ```
 */
export function mineBound(scores: TensorLike): Value {
  const { s, n } = scoreMatrix(scores, 'mineBound')
  const offDiagonal = where(greater(eye(n), 0), Number.NEGATIVE_INFINITY, s)
  return sub(mean(diagonal(s)), sub(logsumexp(offDiagonal), Math.log(n * (n - 1))))
}
