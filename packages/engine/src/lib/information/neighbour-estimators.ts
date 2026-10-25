/**
 * k-nearest-neighbour estimators for continuous data, which avoid binning by using distances to the k-th neighbour: the
 * Kozachenko-Leonenko entropy, the Kraskov-Stoegbauer-Grassberger (KSG) mutual information and its conditional form (as
 * NPEET), a k-NN estimate of the KL divergence, and Ross's estimator for a continuous and a discrete variable
 * (scikit-learn's mutual_info_classif).
 */

import { MdmlError, meanOf } from '../base/index.ts'
import { kdTree } from '../numerics/spatial/index.ts'
import { lgamma } from '../numerics/special/index.ts'
import { item, type TensorLike, toFlat } from '../tensor/index.ts'
import { checkK, concat, countWithin, type Points, points, psi, tensorOf } from './_samples.ts'
import { countsOf, type Labels, labelCodes, logOfBase } from './_shared.ts'

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
 * import { kozachenkoLeonenkoEntropy } from 'mdml-engine/information'
 * import { key, uniform } from 'mdml-engine/random'
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
 * import { ksgMutualInformation } from 'mdml-engine/information'
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
 * import { ksgConditionalMutualInformation } from 'mdml-engine/information'
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
 * import { knnKlDivergence } from 'mdml-engine/information'
 * import { key, normal } from 'mdml-engine/random'
 * import { add } from 'mdml-engine/tensor'
 * // N(0, 1) against N(1, 1): the divergence is 1/2.
 * Math.abs(knnKlDivergence(normal(key(0), [500]), add(normal(key(1), [500]), 1)) - 0.5) < 0.15 // => true
 * ```
 */
export function knnKlDivergence(x: TensorLike, y: TensorLike, options: NeighbourOptions = {}): number {
  const px = points(x, 'knnKlDivergence')
  const py = points(y, 'knnKlDivergence')
  if (px.d !== py.d) throw new MdmlError('shape', 'knnKlDivergence: samples differ in dimension')
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
 * import { mixedMutualInformation } from 'mdml-engine/information'
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
  if (codes.length !== p.n) throw new MdmlError('shape', 'mixedMutualInformation: lengths differ')
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
  if (kept.length === 0) throw new MdmlError('value', 'mixedMutualInformation: every label occurs once')
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
