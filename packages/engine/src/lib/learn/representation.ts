/**
 * Measures of the quality of learned representations: the alignment and uniformity of Wang and Isola (2020), the two
 * properties contrastive losses optimise (positive pairs close, the whole set spread evenly on the sphere), the effective
 * rank of Roy and Vetterli (2007), which counts the dimensions a representation actually uses (a collapse shows as a rank
 * near one), and centred kernel alignment (Kornblith et al. 2019), a similarity between two representations of the same
 * examples that is invariant to rotations and isotropic scaling.
 */

import { MdmlError } from '../base/index.ts'
import { svd } from '../numerics/linalg/index.ts'
import { asTensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { squaredDistances } from './_shared.ts'
import { matrixView } from './validation.ts'

/** Options of {@link alignment} and {@link uniformity}. */
export interface RepresentationOptions {
  /** L2-normalise the rows first, as the paper does (default true). */
  readonly normalise?: boolean
}

function rowsOf(who: string, x: TensorLike, normalise: boolean): { n: number; d: number; data: Float64Array } {
  const m = matrixView(x, who)
  const data = Float64Array.from(m.data)
  if (normalise) {
    for (let i = 0; i < m.n; i++) {
      let nrm = 0
      for (let j = 0; j < m.d; j++) nrm += (data[i * m.d + j] ?? 0) ** 2
      nrm = Math.sqrt(nrm) || 1
      for (let j = 0; j < m.d; j++) data[i * m.d + j] = (data[i * m.d + j] ?? 0) / nrm
    }
  }
  return { n: m.n, d: m.d, data }
}

/**
 * Alignment (Wang and Isola 2020): the mean over positive pairs of |f(x) - f(y)|^alpha, small when the representation
 * maps the pairs a method regards as the same to nearby points.
 *
 * @param x - Representations of the first element of each positive pair `[n, d]`.
 * @param y - Representations of the second element `[n, d]`.
 * @param alpha - The exponent (default 2).
 * @param options - Whether to normalise the rows (default true).
 * @returns The alignment (lower is better); 0 when the pairs coincide.
 * @concept representation-learning
 * @example
 * ```ts
 * import { alignment } from 'mdml-engine/learn'
 * alignment([[1, 0], [0, 1]], [[1, 0], [0, 1]]) // => 0
 * alignment([[1, 0]], [[0, 1]]) // => 2
 * ```
 */
export function alignment(x: TensorLike, y: TensorLike, alpha = 2, options: RepresentationOptions = {}): number {
  const a = rowsOf('alignment', x, options.normalise ?? true)
  const b = rowsOf('alignment', y, options.normalise ?? true)
  if (a.n !== b.n || a.d !== b.d)
    throw new MdmlError('shape', 'alignment: the two sets of representations must have the same shape')
  let total = 0
  for (let i = 0; i < a.n; i++) {
    let s = 0
    for (let j = 0; j < a.d; j++) s += ((a.data[i * a.d + j] ?? 0) - (b.data[i * a.d + j] ?? 0)) ** 2
    total += s ** (alpha / 2)
  }
  return total / a.n
}

/**
 * Uniformity (Wang and Isola 2020): log of the mean over pairs of exp(-t |f(x) - f(y)|^2), the log of the average pairwise
 * Gaussian potential. It is 0 when every point is the same (a collapse) and falls as the points spread; the lowest values
 * belong to the uniform distribution on the sphere.
 *
 * @param x - Representations `[n, d]`.
 * @param t - The temperature of the potential (default 2).
 * @param options - Whether to normalise the rows (default true).
 * @returns The uniformity (lower is more uniform).
 * @concept representation-learning
 * @example
 * ```ts
 * import { uniformity } from 'mdml-engine/learn'
 * uniformity([[1, 0], [1, 0], [1, 0]]) // => 0
 * ```
 */
export function uniformity(x: TensorLike, t = 2, options: RepresentationOptions = {}): number {
  const m = rowsOf('uniformity', x, options.normalise ?? true)
  if (m.n < 2) throw new MdmlError('value', 'uniformity: needs at least two points')
  const d2 = squaredDistances({ n: m.n, d: m.d, data: m.data })
  let total = 0
  let pairs = 0
  for (let i = 0; i < m.n; i++)
    for (let j = i + 1; j < m.n; j++) {
      total += Math.exp(-t * (d2[i * m.n + j] ?? 0))
      pairs++
    }
  return Math.log(total / pairs)
}

/**
 * The effective rank (Roy and Vetterli 2007; the quantity RankMe uses): exp of the entropy of the normalised singular
 * values p_i = s_i / sum s. It is the number of dimensions a representation uses, between 1 (a collapse onto a line) and
 * min(n, d) (all directions equally).
 *
 * @param x - Representations `[n, d]` (centre them first to measure the spread rather than the offset).
 * @returns The effective rank.
 * @concept representation-learning
 * @example
 * ```ts
 * import { effectiveRank } from 'mdml-engine/learn'
 * effectiveRank([[1, 0], [0, 1]]) // => 2
 * ```
 */
export function effectiveRank(x: TensorLike): number {
  matrixView(x, 'effectiveRank')
  const s = toFlat(svd(asTensor(x)).s)
  const total = s.reduce((a, b) => a + b, 0)
  if (!(total > 0)) throw new MdmlError('value', 'effectiveRank: the matrix is zero')
  let h = 0
  for (const v of s) {
    const p = v / total
    if (p > 0) h -= p * Math.log(p)
  }
  return Math.exp(h)
}

/** Options of {@link cka}. */
export interface CkaOptions {
  /** `'linear'` (default) or `'rbf'`. */
  readonly kernel?: 'linear' | 'rbf'
  /** For `'rbf'`: the bandwidth as a fraction of the median pairwise distance (default 0.5, Kornblith et al.). */
  readonly sigmaFraction?: number
}

/** A centred Gram matrix of the rows, flattened `[n, n]`. */
function centredGram(m: { n: number; d: number; data: Float64Array }, options: CkaOptions): Float64Array {
  const { n, d, data } = m
  let k: Float64Array
  if ((options.kernel ?? 'linear') === 'linear') {
    k = new Float64Array(n * n)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        let s = 0
        for (let c = 0; c < d; c++) s += (data[i * d + c] ?? 0) * (data[j * d + c] ?? 0)
        k[i * n + j] = s
      }
  } else {
    const d2 = squaredDistances({ n, d, data })
    const dists: number[] = []
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) dists.push(Math.sqrt(d2[i * n + j] ?? 0))
    dists.sort((a, b) => a - b)
    const median = dists[Math.floor(dists.length / 2)] ?? 1
    const sigma = (options.sigmaFraction ?? 0.5) * (median || 1)
    k = Float64Array.from(d2, (v) => Math.exp(-v / (2 * sigma * sigma)))
  }
  // Centre: H K H with H = I - 11'/n, by subtracting row means, column means and adding the grand mean.
  const rowMean = new Float64Array(n)
  let grand = 0
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) rowMean[i] = (rowMean[i] ?? 0) + (k[i * n + j] ?? 0) / n
    grand += (rowMean[i] ?? 0) / n
  }
  const out = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) out[i * n + j] = (k[i * n + j] ?? 0) - (rowMean[i] ?? 0) - (rowMean[j] ?? 0) + grand
  return out
}

/**
 * Centred kernel alignment (Kornblith, Norouzi, Lee and Hinton 2019): HSIC(K, L) / sqrt(HSIC(K, K) HSIC(L, L)) for the Gram
 * matrices K and L of two representations of the same n examples, with HSIC(K, L) = trace(K_c L_c). In [0, 1]; the linear
 * version is invariant to orthogonal transformations and isotropic scaling of either representation, so two networks that
 * learn the same features in different bases score 1.
 *
 * @param x - The first representation `[n, d1]`.
 * @param y - The second representation of the same examples `[n, d2]`.
 * @param options - The kernel.
 * @returns The similarity.
 * @concept representation-learning
 * @example
 * ```ts
 * import { cka } from 'mdml-engine/learn'
 * cka([[1, 0], [0, 1], [2, 1], [1, 3]], [[0, 2], [2, 0], [2, 4], [6, 2]]) // => 1
 * ```
 */
export function cka(x: TensorLike, y: TensorLike, options: CkaOptions = {}): number {
  const a = rowsOf('cka', x, false)
  const b = rowsOf('cka', y, false)
  if (a.n !== b.n) throw new MdmlError('shape', 'cka: the representations must describe the same examples')
  const k = centredGram(a, options)
  const l = centredGram(b, options)
  const hsic = (p: Float64Array, q: Float64Array): number => {
    let s = 0
    for (let i = 0; i < p.length; i++) s += (p[i] ?? 0) * (q[i] ?? 0)
    return s
  }
  return hsic(k, l) / Math.sqrt(hsic(k, k) * hsic(l, l))
}
