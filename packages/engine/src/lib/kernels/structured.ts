/**
 * Kernels for structured or special inputs: white noise and constant kernels, chi-square, histogram-intersection and
 * related kernels for histograms, string kernels (k-spectrum and Lodhi's subsequence kernel), set and Jaccard kernels,
 * diffusion and regularised Laplacian kernels between the nodes of a graph, and the Weisfeiler-Lehman subtree kernel
 * between graphs.
 */

import { OmaiError } from '../base/index.ts'
import {
  type Graph,
  heatKernel,
  isGraph,
  laplacian,
  normalisedLaplacian,
  weisfeilerLehmanAlgorithm,
} from '../graph/index.ts'
import { inv } from '../numerics/linalg/index.ts'
import {
  add,
  div,
  exp,
  expandDims,
  eye,
  fromStorage,
  mean,
  minimum,
  mul,
  neg,
  ones,
  prod,
  promoteFloat,
  sqrt,
  square,
  stack,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  type Value,
  zeros,
} from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { safeDivide } from './_safe.ts'
import { asRows, defineKernel, type Kernel } from './kernel.ts'

function param(x: TensorLike): Value {
  return promoteFloat(x)[0] as Value
}

/** Rows broadcast against each other: `[m, 1, d]` and `[1, n, d]`. */
function pairs(x: Value, y: Value | undefined): [Value, Value] {
  return [expandDims(x, 1), expandDims(y ?? x, 0)]
}

/** Options of {@link whiteNoiseKernel}. */
export interface WhiteNoiseOptions {
  /** The noise variance (default 1). */
  readonly noiseLevel?: TensorLike
}

/**
 * White noise, k(x, y) = sigma^2 [x and y are the same observation] (scikit-learn WhiteKernel): the Gram matrix of a
 * set is sigma^2 I and the cross-covariance between two sets is zero, as in scikit-learn, so it models observation
 * noise when added to another kernel.
 *
 * @param options - The noise level.
 * @returns The kernel, hyperparameters `{ noiseLevel }`.
 * @concept kernel-functions
 * @example
 * ```ts
 * import { whiteNoiseKernel } from 'omai-engine/kernels'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(whiteNoiseKernel({ noiseLevel: 0.5 }).k([[0], [1]])) // => [[0.5, 0], [0, 0.5]]
 * ```
 */
export function whiteNoiseKernel(options: WhiteNoiseOptions = {}): Kernel {
  return defineKernel({
    name: 'white',
    params: { noiseLevel: param(options.noiseLevel ?? 1) },
    stationary: true,
    matrix: (p, x, y) =>
      y === undefined ? mul(eye(x.shape[0] ?? 0), p.noiseLevel) : zeros([x.shape[0] ?? 0, y.shape[0] ?? 0]),
    diag: (p, x) => mul(ones([x.shape[0] ?? 0]), p.noiseLevel),
  })
}

/** Options of {@link constantKernel}. */
export interface ConstantOptions {
  /** The constant c >= 0 (default 1). */
  readonly value?: TensorLike
}

/**
 * The constant kernel k(x, y) = c (scikit-learn ConstantKernel): a random constant offset with variance c, or, as a
 * factor, an output scale.
 *
 * @param options - The constant.
 * @returns The kernel, hyperparameters `{ value }`.
 * @concept kernel-functions
 * @example
 * ```ts
 * import { constantKernel } from 'omai-engine/kernels'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(constantKernel({ value: 2 }).k([[0], [5]], [[1]])) // => [[2], [2]]
 * ```
 */
export function constantKernel(options: ConstantOptions = {}): Kernel {
  return defineKernel({
    name: 'constant',
    params: { value: param(options.value ?? 1) },
    stationary: true,
    matrix: (p, x, y) => mul(ones([x.shape[0] ?? 0, (y ?? x).shape[0] ?? 0]), p.value),
    diag: (p, x) => mul(ones([x.shape[0] ?? 0]), p.value),
  })
}

/**
 * The chi-square distance between rows of non-negative features (histograms),
 * d(x, y) = sum_i (x_i - y_i)^2 / (x_i + y_i), with 0 / 0 taken as 0. scikit-learn's additive_chi2_kernel is -d.
 *
 * @param x - Rows `[m, d]` (non-negative).
 * @param y - Rows `[n, d]` (default x).
 * @returns `[m, n]`.
 * @concept histogram-kernels
 * @example
 * ```ts
 * import { chi2Distance } from 'omai-engine/kernels'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(chi2Distance([[1, 0]], [[0, 1], [1, 0]])) // => [[2, 0]]
 * ```
 */
export function chi2Distance(x: TensorLike, y?: TensorLike): Value {
  const a = asRows(x, 'chi2Distance').rows
  const b = y === undefined ? undefined : asRows(y, 'chi2Distance').rows
  return chi2(a, b)
}

function chi2(x: Value, y: Value | undefined): Value {
  const [a, b] = pairs(x, y)
  return sum(safeDivide(square(sub(a, b)), add(a, b)), 2)
}

/** Options of {@link chi2Kernel}. */
export interface Chi2Options {
  /** The scale gamma (default 1, as scikit-learn). */
  readonly gamma?: TensorLike
}

/**
 * The exponential chi-square kernel k(x, y) = exp(-gamma sum_i (x_i - y_i)^2 / (x_i + y_i)) for histograms
 * (scikit-learn chi2_kernel), popular for bag-of-features image descriptors. Inputs must be non-negative.
 *
 * @param options - gamma.
 * @returns The kernel, hyperparameters `{ gamma }`.
 * @concept histogram-kernels
 * @reference J. Zhang et al., Local features and kernels for classification of texture and object categories (IJCV
 *   2007).
 * @example
 * ```ts
 * import { chi2Kernel } from 'omai-engine/kernels'
 * import { item } from 'omai-engine/tensor'
 * item(chi2Kernel({ gamma: 0.5 }).k([1, 0], [0, 1])) === Math.exp(-1) // => true
 * ```
 */
export function chi2Kernel(options: Chi2Options = {}): Kernel {
  return defineKernel({
    name: 'chi2',
    params: { gamma: param(options.gamma ?? 1) },
    matrix: (p, x, y) => exp(neg(mul(chi2(x, y), p.gamma))),
    diag: (_p, x) => ones([x.shape[0] ?? 0]),
  })
}

/**
 * The additive chi-square kernel k(x, y) = sum_i 2 x_i y_i / (x_i + y_i) for non-negative features: positive definite
 * (unlike scikit-learn's additive_chi2_kernel, which is minus the chi-square distance), and the kernel that the
 * additive chi-square sampler approximates.
 *
 * @returns The kernel (no hyperparameters).
 * @concept histogram-kernels
 * @reference A. Vedaldi and A. Zisserman, Efficient additive kernels via explicit feature maps (TPAMI 2012).
 * @example
 * ```ts
 * import { additiveChi2Kernel } from 'omai-engine/kernels'
 * import { item } from 'omai-engine/tensor'
 * item(additiveChi2Kernel().k([1, 2], [1, 0])) // => 1
 * ```
 */
export function additiveChi2Kernel(): Kernel {
  const value = (x: Value, y: Value | undefined): Value => {
    const [a, b] = pairs(x, y)
    return sum(safeDivide(mul(mul(a, b), 2), add(a, b)), 2)
  }
  return defineKernel({
    name: 'additive-chi2',
    params: {},
    matrix: (_p, x, y) => value(x, y),
    diag: (_p, x) => sum(x, 1),
  })
}

/** Options of {@link skewedChi2Kernel}. */
export interface SkewedChi2Options {
  /** The skewedness c > 0 (default 1); inputs must exceed -c. */
  readonly skewedness?: TensorLike
}

/**
 * The skewed chi-square kernel k(x, y) = prod_i 2 sqrt(x_i + c) sqrt(y_i + c) / (x_i + y_i + 2 c), the kernel that
 * scikit-learn's SkewedChi2Sampler approximates.
 *
 * @param options - The skewedness c.
 * @returns The kernel, hyperparameters `{ skewedness }`.
 * @concept histogram-kernels
 * @reference F. Li, C. Ionescu and C. Sminchisescu, Random Fourier approximations for skewed multiplicative histogram
 *   kernels (DAGM 2010).
 * @example
 * ```ts
 * import { skewedChi2Kernel } from 'omai-engine/kernels'
 * import { item } from 'omai-engine/tensor'
 * item(skewedChi2Kernel({ skewedness: 1 }).k([0], [3])) // => 0.8
 * ```
 */
export function skewedChi2Kernel(options: SkewedChi2Options = {}): Kernel {
  return defineKernel({
    name: 'skewed-chi2',
    params: { skewedness: param(options.skewedness ?? 1) },
    matrix: (p, x, y) => {
      const [a, b] = pairs(add(x, p.skewedness), y === undefined ? undefined : add(y, p.skewedness))
      return prod(div(mul(sqrt(mul(a, b)), 2), add(a, b)), 2)
    },
    diag: (_p, x) => ones([x.shape[0] ?? 0]),
  })
}

/**
 * The histogram intersection kernel k(x, y) = sum_i min(x_i, y_i) for non-negative features: positive definite, and
 * the overlap of two histograms.
 *
 * @returns The kernel (no hyperparameters).
 * @concept histogram-kernels
 * @reference M. J. Swain and D. H. Ballard, Color indexing (IJCV 1991); S. Maji, A. C. Berg and J. Malik,
 *   Classification using intersection kernel support vector machines is efficient (CVPR 2008).
 * @example
 * ```ts
 * import { histogramIntersectionKernel } from 'omai-engine/kernels'
 * import { item } from 'omai-engine/tensor'
 * item(histogramIntersectionKernel().k([3, 1, 0], [1, 2, 4])) // => 2
 * ```
 */
export function histogramIntersectionKernel(): Kernel {
  return defineKernel({
    name: 'histogram-intersection',
    params: {},
    matrix: (_p, x, y) => sum(minimum(...pairs(x, y)), 2),
    diag: (_p, x) => sum(x, 1),
  })
}

/** Normalise a kernel matrix to unit self-similarity: K_ij / sqrt(K_ii K_jj) (zero where a self-similarity is 0). */
function cosineNormalise(k: Float64Array, m: number, n: number, da: ArrayLike<number>, db: ArrayLike<number>): void {
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const d = Math.sqrt((da[i] ?? 0) * (db[j] ?? 0))
      k[i * n + j] = d === 0 ? 0 : (k[i * n + j] ?? 0) / d
    }
  }
}

/** Fill a kernel matrix from a pairwise function, with optional cosine normalisation. */
function pairwise<T>(
  a: readonly T[],
  b: readonly T[] | undefined,
  f: (s: T, t: T) => number,
  normalise: boolean,
): Tensor<'float64'> {
  const other = b ?? a
  const m = a.length
  const n = other.length
  const out = new Float64Array(m * n)
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[i * n + j] = f(a[i] as T, other[j] as T)
  if (normalise) {
    const da = a.map((s) => f(s, s))
    const db = b === undefined ? da : other.map((t) => f(t, t))
    cosineNormalise(out, m, n, da, db)
  }
  return fromStorage(out, [m, n], 'float64')
}

/** Options of {@link spectrumKernel}. */
export interface SpectrumOptions {
  /** The substring length k (default 3). */
  readonly k?: number
  /** Normalise to unit self-similarity (default false). */
  readonly normalise?: boolean
}

function kmers(s: string, k: number): Map<string, number> {
  const counts = new Map<string, number>()
  for (let i = 0; i + k <= s.length; i++) {
    const w = s.slice(i, i + k)
    counts.set(w, (counts.get(w) ?? 0) + 1)
  }
  return counts
}

/**
 * The k-spectrum string kernel: the inner product of the counts of every length-k substring (k-mer),
 * k(s, t) = sum_u count_u(s) count_u(t).
 *
 * @param a - Strings.
 * @param b - Strings (default a).
 * @param options - k and normalisation.
 * @returns `[a.length, b.length]`.
 * @throws OmaiError `domain` for k < 1.
 * @concept string-kernels
 * @reference C. Leslie, E. Eskin and W. S. Noble, The spectrum kernel (PSB 2002).
 * @example
 * ```ts
 * import { spectrumKernel } from 'omai-engine/kernels'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(spectrumKernel(['abab'], ['bab', 'aaa'], { k: 2 })) // => [[3, 0]]
 * ```
 */
export function spectrumKernel(
  a: readonly string[],
  b?: readonly string[],
  options: SpectrumOptions = {},
): Tensor<'float64'> {
  const k = options.k ?? 3
  if (!(Number.isInteger(k) && k >= 1)) throw new OmaiError('domain', `spectrumKernel: k must be >= 1, got ${k}`)
  const features = new Map<string, Map<string, number>>()
  const of = (s: string): Map<string, number> => {
    let f = features.get(s)
    if (f === undefined) {
      f = kmers(s, k)
      features.set(s, f)
    }
    return f
  }
  const dot = (s: string, t: string): number => {
    let total = 0
    const fs = of(s)
    for (const [w, c] of of(t)) total += c * (fs.get(w) ?? 0)
    return total
  }
  return pairwise(a, b, dot, options.normalise ?? false)
}

/** Options of {@link subsequenceKernel}. */
export interface SubsequenceOptions {
  /** The subsequence length n (default 2). */
  readonly n?: number
  /** The gap decay lambda in (0, 1] (default 0.5). */
  readonly lambda?: number
  /** Normalise to unit self-similarity (default true, as Lodhi et al.). */
  readonly normalise?: boolean
}

/** The string subsequence kernel K_n(s, t) by the dynamic programme of Lodhi et al., O(n |s| |t|). */
function ssk(s: string, t: string, n: number, lambda: number): number {
  const ls = s.length
  const lt = t.length
  if (Math.min(ls, lt) < n) return 0
  const w = lt + 1
  // kp[p * w + q] = K'_{i}(s[:p], t[:q]), starting from K'_0 = 1.
  let kp = new Float64Array((ls + 1) * w).fill(1)
  for (let i = 1; i < n; i++) {
    const next = new Float64Array((ls + 1) * w)
    for (let p = i; p <= ls; p++) {
      let kpp = 0
      for (let q = i; q <= lt; q++) {
        const match = s[p - 1] === t[q - 1] ? lambda * (kp[(p - 1) * w + (q - 1)] ?? 0) : 0
        kpp = lambda * (kpp + match)
        next[p * w + q] = lambda * (next[(p - 1) * w + q] ?? 0) + kpp
      }
    }
    kp = next
  }
  let total = 0
  for (let p = n; p <= ls; p++) {
    for (let q = n; q <= lt; q++) {
      if (s[p - 1] === t[q - 1]) total += lambda * lambda * (kp[(p - 1) * w + (q - 1)] ?? 0)
    }
  }
  return total
}

/**
 * The string subsequence kernel of Lodhi et al.: every common subsequence of length n counts, discounted by
 * lambda^(its total span in both strings), so contiguous matches weigh most. K_n(s, t) =
 * sum_u sum_{i: s[i] = u} sum_{j: t[j] = u} lambda^(l(i) + l(j)), computed by dynamic programming in O(n |s| |t|).
 *
 * @param a - Strings.
 * @param b - Strings (default a).
 * @param options - Subsequence length, decay and normalisation.
 * @returns `[a.length, b.length]`.
 * @throws OmaiError `domain` for n < 1 or lambda outside (0, 1].
 * @concept string-kernels
 * @reference H. Lodhi et al., Text classification using string kernels (JMLR 2, 2002).
 * @example
 * ```ts
 * import { subsequenceKernel } from 'omai-engine/kernels'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(subsequenceKernel(['cat'], ['cart'], { n: 2, lambda: 1, normalise: false })) // => [[3]]
 * ```
 */
export function subsequenceKernel(
  a: readonly string[],
  b?: readonly string[],
  options: SubsequenceOptions = {},
): Tensor<'float64'> {
  const n = options.n ?? 2
  const lambda = options.lambda ?? 0.5
  if (!(Number.isInteger(n) && n >= 1)) throw new OmaiError('domain', `subsequenceKernel: n must be >= 1, got ${n}`)
  if (!(lambda > 0 && lambda <= 1))
    throw new OmaiError('domain', `subsequenceKernel: lambda must be in (0, 1], got ${lambda}`)
  return pairwise(a, b, (s, t) => ssk(s, t, n, lambda), options.normalise ?? true)
}

/** Options of {@link setKernel}. */
export interface SetKernelOptions {
  /** Average over pairs (the mean embedding, default) or sum them. */
  readonly reduce?: 'mean' | 'sum'
  /** Normalise to unit self-similarity (default false). */
  readonly normalise?: boolean
}

/**
 * The set (multi-instance) kernel between sets of points: K(A, B) = mean over a in A, b in B of k(a, b), the inner
 * product of the kernel mean embeddings (with `reduce: 'sum'`, the sum kernel of Gärtner et al.). Differentiable in
 * the points and in the base kernel's hyperparameters.
 *
 * @param kernel - The base kernel on points.
 * @param a - Sets, each `[n_i, d]`.
 * @param b - Sets (default a).
 * @param options - Reduction and normalisation.
 * @returns `[a.length, b.length]`.
 * @concept set-kernels
 * @reference T. Gärtner et al., Multi-instance kernels (ICML 2002); A. Smola et al., A Hilbert space embedding for
 *   distributions (ALT 2007).
 * @example
 * ```ts
 * import { linearKernel, setKernel } from 'omai-engine/kernels'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(setKernel(linearKernel(), [[[1], [3]]], [[[2]]])) // => [[4]]
 * ```
 */
export function setKernel(
  kernel: Kernel,
  a: readonly TensorLike[],
  b?: readonly TensorLike[],
  options: SetKernelOptions = {},
): Value {
  const reduce = options.reduce ?? 'mean'
  const rowsOf = (s: TensorLike): Value => asRows(s, 'setKernel').rows
  const left = a.map(rowsOf)
  const right = b === undefined ? left : b.map(rowsOf)
  const value = (s: Value, t: Value): Value => {
    const k = kernel.k(s, t)
    return reduce === 'mean' ? mean(k) : sum(k)
  }
  const matrix = stack(left.map((s) => stack(right.map((t) => value(s, t)))))
  if (options.normalise !== true) return matrix
  const da = stack(left.map((s) => value(s, s)))
  const db = b === undefined ? da : stack(right.map((t) => value(t, t)))
  return safeDivide(matrix, sqrt(mul(expandDims(da, 1), expandDims(db, 0))))
}

/**
 * The Jaccard (Tanimoto) kernel between finite sets, |A ∩ B| / |A ∪ B| (1 for two empty sets), positive definite.
 *
 * @param a - Sets of labels.
 * @param b - Sets (default a).
 * @returns `[a.length, b.length]`.
 * @concept set-kernels
 * @reference L. Ralaivola et al., Graph kernels for chemical informatics (Neural Networks 18, 2005).
 * @example
 * ```ts
 * import { jaccardKernel } from 'omai-engine/kernels'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(jaccardKernel([['a', 'b', 'c']], [['b', 'c', 'd'], []])) // => [[0.5, 0]]
 * ```
 */
export function jaccardKernel(
  a: readonly (readonly (string | number)[])[],
  b?: readonly (readonly (string | number)[])[],
): Tensor<'float64'> {
  const sets = (xs: readonly (readonly (string | number)[])[]): Set<string | number>[] => xs.map((x) => new Set(x))
  const left = sets(a)
  const right = b === undefined ? left : sets(b)
  const similarity = (s: Set<string | number>, t: Set<string | number>): number => {
    let common = 0
    for (const v of s) if (t.has(v)) common++
    const union = s.size + t.size - common
    return union === 0 ? 1 : common / union
  }
  return pairwise(left, b === undefined ? undefined : right, similarity, false)
}

/** Options of the graph node kernels. */
export interface GraphKernelOptions {
  /** Use the symmetric normalised Laplacian (default false: the combinatorial one). */
  readonly normalised?: boolean
}

function laplacianOf(g: Graph | TensorLike, options: GraphKernelOptions): Value {
  if (isGraph(g)) return options.normalised === true ? normalisedLaplacian(g) : laplacian(g)
  return param(g)
}

/**
 * The diffusion (heat) kernel on the nodes of a graph, K = exp(-beta L) (Kondor and Lafferty), the graph analogue of
 * the RBF kernel; it is the graph domain's heat kernel, here as a kernel between nodes.
 *
 * @param g - A graph or a Laplacian `[n, n]`.
 * @param beta - The diffusion time (bandwidth).
 * @param options - Which Laplacian of a graph.
 * @returns `[n, n]`, differentiable in beta and in a given Laplacian.
 * @concept graph-kernels
 * @reference R. I. Kondor and J. Lafferty, Diffusion kernels on graphs and other discrete structures (ICML 2002).
 * @example
 * ```ts
 * import { diffusionKernel } from 'omai-engine/kernels'
 * import { pathGraph } from 'omai-engine/graph'
 * import { sum, item } from 'omai-engine/tensor'
 * Math.round(item(sum(diffusionKernel(pathGraph(3), 0.7))) * 1e9) / 1e9 // => 3
 * ```
 */
export function diffusionKernel(g: Graph | TensorLike, beta: TensorLike, options: GraphKernelOptions = {}): Value {
  return heatKernel(laplacianOf(g, options), beta)
}

/**
 * The regularised Laplacian kernel on the nodes of a graph, K = (I + sigma^2 L)^(-1) (Smola and Kondor): the
 * covariance of a Gaussian field penalising differences across edges.
 *
 * @param g - A graph or a Laplacian `[n, n]`.
 * @param sigma2 - The regularisation sigma^2 >= 0.
 * @param options - Which Laplacian of a graph.
 * @returns `[n, n]`, differentiable in sigma2 and in a given Laplacian.
 * @concept graph-kernels
 * @reference A. J. Smola and R. Kondor, Kernels and regularization on graphs (COLT 2003).
 * @example
 * ```ts
 * import { regularisedLaplacianKernel } from 'omai-engine/kernels'
 * import { pathGraph } from 'omai-engine/graph'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(regularisedLaplacianKernel(pathGraph(2), 0.5)) // => [[0.75, 0.25], [0.25, 0.75]]
 * ```
 */
export function regularisedLaplacianKernel(
  g: Graph | TensorLike,
  sigma2: TensorLike,
  options: GraphKernelOptions = {},
): Value {
  const l = laplacianOf(g, options)
  return inv(add(eye(l.shape[0] ?? 0), mul(l, param(sigma2))))
}

/** Options of {@link wlSubtreeFeatures} and {@link wlSubtreeKernel}. */
export interface WlSubtreeOptions {
  /** The number of refinement rounds h (default 3). */
  readonly iterations?: number
  /** Initial node labels, one array per graph (default: one constant label, so round 1 separates degrees). */
  readonly nodeLabels?: readonly ArrayLike<string | number>[]
  /** Initial node labels of the second set of graphs. */
  readonly otherNodeLabels?: readonly ArrayLike<string | number>[]
  /** Normalise to unit self-similarity (default false). */
  readonly normalise?: boolean
}

/**
 * The Weisfeiler-Lehman subtree features of a graph: how often each colour occurs after 0, 1, ..., h rounds of WL
 * refinement (keys `round:colour`). Colours are hashes, so features of different graphs are comparable.
 *
 * @param g - The graph.
 * @param options - Rounds and initial node labels (the first entry of `nodeLabels`).
 * @returns Colour counts.
 * @concept graph-kernels
 * @example
 * ```ts
 * import { wlSubtreeFeatures } from 'omai-engine/kernels'
 * import { pathGraph } from 'omai-engine/graph'
 * [...wlSubtreeFeatures(pathGraph(3), { iterations: 1 }).values()].sort() // => [1, 2, 3]
 * ```
 */
export function wlSubtreeFeatures(g: Graph, options: WlSubtreeOptions = {}): Map<string, number> {
  const iterations = options.iterations ?? 3
  if (!(Number.isInteger(iterations) && iterations >= 0))
    throw new OmaiError('domain', `wlSubtreeFeatures: iterations must be >= 0, got ${iterations}`)
  // Unlabelled graphs start from one constant label (Shervashidze et al.), so round 1 is the degree colouring. It is
  // not empty, which would make a degree-0 and a degree-1 node look alike to the refinement.
  const labels = options.nodeLabels?.[0] ?? Array.from({ length: g.n }, () => '*')
  const algorithm = weisfeilerLehmanAlgorithm(g, { nodeLabels: labels })
  // Refine for exactly h rounds: the kernel counts later rounds even after the colouring is stable.
  const { done: _stop, ...rounds } = algorithm
  const { history } = runAlgorithm(rounds, { maxSteps: iterations }).state
  const counts = new Map<string, number>()
  history.forEach((colours, round) => {
    for (const c of colours) {
      const key = `${round}:${c}`
      counts.set(key, (counts.get(key) ?? 0) + 1)
    }
  })
  return counts
}

/**
 * The Weisfeiler-Lehman subtree kernel between graphs: the inner product of their WL subtree features,
 * k(G, G') = sum over rounds 0..h and colours c of count_c(G) count_c(G').
 *
 * @param a - Graphs.
 * @param b - Graphs (default a).
 * @param options - Rounds, initial labels per graph, normalisation.
 * @returns `[a.length, b.length]`.
 * @concept graph-kernels
 * @reference N. Shervashidze et al., Weisfeiler-Lehman graph kernels (JMLR 12, 2011).
 * @example
 * ```ts
 * import { wlSubtreeKernel } from 'omai-engine/kernels'
 * import { pathGraph } from 'omai-engine/graph'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(wlSubtreeKernel([pathGraph(3)], undefined, { iterations: 1 })) // => [[14]]
 * ```
 */
export function wlSubtreeKernel(
  a: readonly Graph[],
  b?: readonly Graph[],
  options: WlSubtreeOptions = {},
): Tensor<'float64'> {
  const featuresOf = (graphs: readonly Graph[], labels: WlSubtreeOptions['nodeLabels']): Map<string, number>[] =>
    graphs.map((g, i) => {
      const own = labels?.[i]
      return wlSubtreeFeatures(g, {
        ...(options.iterations === undefined ? {} : { iterations: options.iterations }),
        ...(own === undefined ? {} : { nodeLabels: [own] }),
      })
    })
  const left = featuresOf(a, options.nodeLabels)
  const right = b === undefined ? undefined : featuresOf(b, options.otherNodeLabels ?? options.nodeLabels)
  const dot = (s: Map<string, number>, t: Map<string, number>): number => {
    let total = 0
    for (const [c, k] of t) total += k * (s.get(c) ?? 0)
    return total
  }
  return pairwise(left, right, dot, options.normalise ?? false)
}
