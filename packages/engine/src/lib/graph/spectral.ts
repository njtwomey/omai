/**
 * Graph signal processing and spectral methods: treating values on nodes as signals and analysing them in the
 * eigenbasis of the graph Laplacian. The graph Fourier transform and spectral filters, the heat kernel and diffusion,
 * Chebyshev polynomial filters without eigendecomposition, spectral graph wavelets, message-passing propagation,
 * Dirichlet energy, the Fiedler vector and spectral embeddings (dense or by traced subspace iteration).
 */

import { type Json, MdmlError } from '../base/index.ts'
import { chebyshevCoefficients } from '../numerics/approximation/index.ts'
import { ddot, dnrm2, eigh, qr } from '../numerics/linalg/index.ts'
import { eigsh } from '../numerics/sparse/index.ts'
import { foldIn, normal } from '../random/index.ts'
import {
  add,
  concreteOrThrow,
  div,
  exp,
  expandDims,
  eye,
  fromStorage,
  greater,
  matmul,
  mul,
  neg,
  promoteFloat,
  sliceAxis,
  sqrt,
  stack,
  sub,
  sum,
  swapaxes,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
  where,
} from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { type Graph, type GraphRunOptions, isGraph } from './graph.ts'
import { laplacian, laplacianCsr, normalisedAdjacency, normalisedLaplacian } from './matrices.ts'

/**
 * A graph shift operator: a graph (its combinatorial Laplacian, or the symmetric normalised one with `normalised`) or a
 * symmetric matrix `[..., n, n]` used as given (a Laplacian, possibly traced for differentiation).
 */
export type ShiftOperator = Graph | TensorLike

/** Options naming which Laplacian of a graph to use. */
export interface LaplacianChoice {
  /** Use the symmetric normalised Laplacian of a graph (default false). */
  readonly normalised?: boolean
  readonly weighted?: boolean
}

function operator(l: ShiftOperator, options: LaplacianChoice = {}): Value {
  if (isGraph(l)) {
    const choice = options.weighted === undefined ? {} : { weighted: options.weighted }
    return options.normalised === true ? normalisedLaplacian(l, choice) : laplacian(l, choice)
  }
  return promoteFloat(l)[0] as Value
}

/** Signals as columns: a vector `[n]` becomes `[n, 1]`, and the result is reshaped back. */
function asColumns(x: TensorLike): { columns: Value; vector: boolean } {
  const [v] = promoteFloat(x) as [Value]
  return v.shape.length === 1 ? { columns: expandDims(v, -1), vector: true } : { columns: v, vector: false }
}

function back(y: Value, vector: boolean): Value {
  return vector ? sum(y, -1) : y
}

/** The graph Fourier transform of a signal. */
export interface GraphFourier {
  /** The spectral coefficients U^T x, `[n]` or `[n, f]`. */
  readonly coefficients: Value
  /** The graph frequencies: the Laplacian eigenvalues, ascending. */
  readonly frequencies: Value
  /** The Fourier basis U: the eigenvectors as columns. */
  readonly basis: Value
}

/**
 * The graph Fourier transform (Shuman et al., 2013): expand a signal on the nodes in the eigenvectors of the
 * Laplacian, x_hat = U^T x, low frequencies being the smooth modes. Differentiable in the signal and, where the
 * eigenvalues are distinct, in the operator.
 *
 * @param l - A graph or a Laplacian `[n, n]`.
 * @param x - A signal `[n]`, or several as columns `[n, f]`.
 * @param options - Which Laplacian of a graph.
 * @returns The coefficients, frequencies and basis.
 * @concept graph-signal-processing
 * @reference D. I. Shuman et al., The emerging field of signal processing on graphs, IEEE Signal Processing Magazine
 *   30 (2013).
 * @example
 * ```ts
 * import { cycleGraph, graphFourierTransform } from 'mdml-engine/graph'
 * import { toArray } from 'mdml-engine/tensor'
 * const f = graphFourierTransform(cycleGraph(4), [1, 1, 1, 1])
 * toArray(f.coefficients).map((c) => Math.round(Math.abs(c) * 1e9) / 1e9) // => [2, 0, 0, 0]
 * ```
 */
export function graphFourierTransform(l: ShiftOperator, x: TensorLike, options: LaplacianChoice = {}): GraphFourier {
  const { eigenvalues, eigenvectors } = eigh(operator(l, options))
  const { columns, vector } = asColumns(x)
  return {
    coefficients: back(matmul(swapaxes(eigenvectors, -1, -2), columns), vector),
    frequencies: eigenvalues,
    basis: eigenvectors,
  }
}

/**
 * The inverse graph Fourier transform x = U x_hat.
 *
 * @param basis - The Fourier basis from {@link graphFourierTransform}.
 * @param coefficients - Spectral coefficients `[n]` or `[n, f]`.
 * @returns The signal.
 * @concept graph-signal-processing
 * @example
 * ```ts
 * import { graphFourierTransform, inverseGraphFourierTransform, pathGraph } from 'mdml-engine/graph'
 * import { toArray } from 'mdml-engine/tensor'
 * const f = graphFourierTransform(pathGraph(3), [3, 1, 2])
 * toArray(inverseGraphFourierTransform(f.basis, f.coefficients)).map((v) => Math.round(v * 1e9) / 1e9) // => [3, 1, 2]
 * ```
 */
export function inverseGraphFourierTransform(basis: TensorLike, coefficients: TensorLike): Value {
  const { columns, vector } = asColumns(coefficients)
  return back(matmul(promoteFloat(basis)[0] as Value, columns), vector)
}

/**
 * Filter a signal in the graph spectral domain: y = U h(Lambda) U^T x for a frequency response h applied to the
 * eigenvalues. Differentiable in the signal, in parameters inside h, and in the operator where the eigenvalues are
 * distinct.
 *
 * @param l - A graph or a Laplacian `[n, n]`.
 * @param x - A signal `[n]` or `[n, f]`.
 * @param response - The frequency response, applied elementwise to the eigenvalues.
 * @param options - Which Laplacian of a graph.
 * @returns The filtered signal.
 * @concept graph-signal-processing
 * @example
 * ```ts
 * import { pathGraph, spectralFilter } from 'mdml-engine/graph'
 * import { onesLike, toArray } from 'mdml-engine/tensor'
 * toArray(spectralFilter(pathGraph(3), [1, 2, 3], (w) => onesLike(w))).map((v) => Math.round(v * 1e9) / 1e9) // => [1, 2, 3]
 * ```
 */
export function spectralFilter(
  l: ShiftOperator,
  x: TensorLike,
  response: (eigenvalues: Value) => Value,
  options: LaplacianChoice = {},
): Value {
  const { eigenvalues, eigenvectors } = eigh(operator(l, options))
  const { columns, vector } = asColumns(x)
  const spectrum = mul(matmul(swapaxes(eigenvectors, -1, -2), columns), expandDims(response(eigenvalues), -1))
  return back(matmul(eigenvectors, spectrum), vector)
}

/**
 * The heat kernel exp(-t L) (Kondor and Lafferty, 2002): the operator of diffusion for time t, a low-pass filter whose
 * entries are the diffused mass between nodes. Differentiable in t and, where the eigenvalues are distinct, in L.
 *
 * @param l - A graph or a Laplacian `[n, n]`.
 * @param t - The diffusion time (a number or a scalar value).
 * @param options - Which Laplacian of a graph.
 * @returns `[n, n]`.
 * @concept graph-signal-processing
 * @reference R. I. Kondor and J. Lafferty, Diffusion kernels on graphs and other discrete structures (ICML 2002).
 * @example
 * ```ts
 * import { heatKernel, pathGraph } from 'mdml-engine/graph'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(heatKernel(pathGraph(3), 0)).map((r) => r.map((v) => Math.round(v * 1e9) / 1e9)) // => [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
 * ```
 */
export function heatKernel(l: ShiftOperator, t: TensorLike, options: LaplacianChoice = {}): Value {
  const { eigenvalues, eigenvectors } = eigh(operator(l, options))
  const scaled = mul(eigenvectors, expandDims(exp(neg(mul(eigenvalues, t))), -2))
  return matmul(scaled, swapaxes(eigenvectors, -1, -2))
}

/**
 * Diffuse a signal by the heat kernel: exp(-t L) x.
 *
 * @param l - A graph or a Laplacian `[n, n]`.
 * @param x - A signal `[n]` or `[n, f]`.
 * @param t - The diffusion time.
 * @param options - Which Laplacian of a graph.
 * @returns The diffused signal (its total is conserved for a combinatorial Laplacian).
 * @concept graph-signal-processing
 * @example
 * ```ts
 * import { heatDiffusion, pathGraph } from 'mdml-engine/graph'
 * import { sum, item } from 'mdml-engine/tensor'
 * Math.round(item(sum(heatDiffusion(pathGraph(4), [1, 0, 0, 0], 2))) * 1e9) / 1e9 // => 1
 * ```
 */
export function heatDiffusion(l: ShiftOperator, x: TensorLike, t: TensorLike, options: LaplacianChoice = {}): Value {
  const { columns, vector } = asColumns(x)
  return back(matmul(heatKernel(l, t, options), columns), vector)
}

/**
 * A Chebyshev polynomial filter (Defferrard et al., 2016; Hammond et al., 2011): y = sum_k theta_k T_k(L~) x with
 * L~ = 2 L / lambdaMax - I, computed by the three-term recurrence without any eigendecomposition. Order K needs K sparse
 * products and reaches exactly the K-hop neighbourhood. Differentiable in L, x and the coefficients, and vectorisable.
 *
 * @param l - A graph or a Laplacian `[n, n]`.
 * @param x - A signal `[n]` or `[n, f]`.
 * @param coefficients - theta_0 .. theta_K `[K + 1]`.
 * @param options - `lambdaMax` (default 2, the bound for a normalised Laplacian; give an upper bound such as twice the
 *   largest degree for a combinatorial one), and which Laplacian of a graph (default here: the normalised one).
 * @returns The filtered signal.
 * @concept graph-signal-processing
 * @reference M. Defferrard, X. Bresson and P. Vandergheynst, Convolutional neural networks on graphs with fast
 *   localized spectral filtering (NeurIPS 2016).
 * @example
 * ```ts
 * import { chebyshevFilter, normalisedLaplacian, pathGraph } from 'mdml-engine/graph'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(chebyshevFilter(normalisedLaplacian(pathGraph(3)), [1, 2, 3], [2])) // => [2, 4, 6]
 * ```
 */
export function chebyshevFilter(
  l: ShiftOperator,
  x: TensorLike,
  coefficients: TensorLike,
  options: LaplacianChoice & { readonly lambdaMax?: number } = {},
): Value {
  const op = operator(l, { normalised: true, ...options })
  const lambdaMax = options.lambdaMax ?? 2
  const n = op.shape[op.shape.length - 1] ?? 0
  const scaled = sub(mul(op, 2 / lambdaMax), eye(n))
  const [theta] = promoteFloat(coefficients) as [Value]
  const order = theta.shape[theta.shape.length - 1] ?? 0
  if (order === 0) throw new MdmlError('value', 'chebyshevFilter: give at least one coefficient')
  const { columns, vector } = asColumns(x)
  const coef = (k: number): Value => sum(sliceAxis(theta, -1, k, k + 1), -1)
  let previous = columns
  let current = matmul(scaled, columns)
  let y = mul(columns, coef(0))
  if (order > 1) y = add(y, mul(current, coef(1)))
  for (let k = 2; k < order; k++) {
    const next = sub(mul(matmul(scaled, current), 2), previous)
    previous = current
    current = next
    y = add(y, mul(current, coef(k)))
  }
  return back(y, vector)
}

/**
 * The Chebyshev coefficients of a frequency response on [0, lambdaMax], for {@link chebyshevFilter}: interpolation at
 * order + 1 Chebyshev points (via `chebyshevCoefficients` of `mdml-engine/numerics/approximation`), so a smooth
 * response such as the heat kernel e^{-t lambda} is approximated without diagonalising the Laplacian.
 *
 * @param response - The response h(lambda).
 * @param order - The polynomial order K.
 * @param lambdaMax - The upper end of the spectrum (default 2).
 * @returns theta_0 .. theta_K.
 * @concept graph-signal-processing
 * @example
 * ```ts
 * import { chebyshevFilterCoefficients } from 'mdml-engine/graph'
 * chebyshevFilterCoefficients((x) => 3, 2).map((c) => Math.round(c * 1e12) / 1e12) // => [3, 0, 0]
 * ```
 */
export function chebyshevFilterCoefficients(
  response: (lambda: number) => number,
  order: number,
  lambdaMax = 2,
): number[] {
  if (!(Number.isInteger(order) && order >= 0)) throw new MdmlError('value', 'chebyshevFilterCoefficients: bad order')
  if (order === 0) return [response(lambdaMax / 2)]
  const values = Array.from({ length: order + 1 }, (_, j) =>
    response((lambdaMax / 2) * (1 + Math.cos((j * Math.PI) / order))),
  )
  return chebyshevCoefficients(values)
}

/** The kernel of {@link graphWavelets}: a band-pass g with g(0) = 0. */
export type WaveletKernel = 'mexican-hat' | 'heat-band' | ((x: Value) => Value)

function kernelFunction(kernel: WaveletKernel): (x: Value) => Value {
  if (typeof kernel === 'function') return kernel
  // g(x) = x e^{-x} (PyGSP MexicanHat); 'heat-band' is the difference of two heat kernels e^{-x} - e^{-2x}.
  return kernel === 'mexican-hat' ? (x) => mul(x, exp(neg(x))) : (x) => sub(exp(neg(x)), exp(mul(x, -2)))
}

/**
 * Spectral graph wavelets (Hammond, Vandergheynst and Gribonval, 2011): for each scale s the operator
 * psi_s = U g(s Lambda) U^T of a band-pass kernel g, whose column i is the wavelet centred at node i; with a signal,
 * the wavelet coefficients psi_s x per scale. Differentiable in the signal and the scales.
 *
 * @param l - A graph or a Laplacian `[n, n]`.
 * @param scales - The scales `[S]`.
 * @param options - The kernel (default mexican hat x e^{-x}), a signal `[n]` or `[n, f]` to transform, and which
 *   Laplacian of a graph.
 * @returns `[S, n, n]` wavelet operators, or `[S, n]` / `[S, n, f]` coefficients when a signal is given.
 * @concept graph-wavelets
 * @reference D. K. Hammond, P. Vandergheynst and R. Gribonval, Wavelets on graphs via spectral graph theory, Applied
 *   and Computational Harmonic Analysis 30 (2011).
 * @example
 * ```ts
 * import { cycleGraph, graphWavelets } from 'mdml-engine/graph'
 * graphWavelets(cycleGraph(6), [1, 2]).shape // => [2, 6, 6]
 * graphWavelets(cycleGraph(6), [1, 2], { signal: [1, 0, 0, 0, 0, 0] }).shape // => [2, 6]
 * ```
 */
export function graphWavelets(
  l: ShiftOperator,
  scales: TensorLike,
  options: LaplacianChoice & { readonly kernel?: WaveletKernel; readonly signal?: TensorLike } = {},
): Value {
  const { eigenvalues, eigenvectors } = eigh(operator(l, options))
  const g = kernelFunction(options.kernel ?? 'mexican-hat')
  const [s] = promoteFloat(scales) as [Value]
  const count = s.shape[0] ?? 0
  const ut = swapaxes(eigenvectors, -1, -2)
  const columns = options.signal === undefined ? undefined : asColumns(options.signal)
  const out: Value[] = []
  for (let i = 0; i < count; i++) {
    const response = g(mul(eigenvalues, sum(sliceAxis(s, 0, i, i + 1), 0)))
    const psi = matmul(mul(eigenvectors, expandDims(response, -2)), ut)
    out.push(columns === undefined ? psi : back(matmul(psi, columns.columns), columns.vector))
  }
  return stack(out, 0)
}

/**
 * Message passing with a normalised adjacency: `steps` rounds of x <- Â x (the propagation of SGC, Wu et al. 2019),
 * or with `teleport` alpha the personalised-PageRank propagation of APPNP (Gasteiger et al. 2019),
 * x <- (1 - alpha) Â x + alpha x_0. Edge weights weight the messages. Differentiable in the adjacency, the features and
 * alpha.
 *
 * @param a - A graph or an adjacency matrix `[n, n]`.
 * @param x - Node features `[n]` or `[n, f]`.
 * @param options - `steps` (default 1), `teleport` (default 0), and the normalisation of {@link normalisedAdjacency}
 *   (`selfLoops`, `kind`, `weighted`).
 * @returns The propagated features.
 * @concept message-passing
 * @reference F. Wu et al., Simplifying graph convolutional networks (ICML 2019); J. Gasteiger, A. Bojchevski and
 *   S. Günnemann, Predict then propagate (ICLR 2019).
 * @example
 * ```ts
 * import { pathGraph, propagate } from 'mdml-engine/graph'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(propagate(pathGraph(3), [3, 0, 3], { kind: 'rw', selfLoops: false })) // => [0, 3, 0]
 * ```
 */
export function propagate(
  a: Graph | TensorLike,
  x: TensorLike,
  options: {
    readonly steps?: number
    readonly teleport?: TensorLike
    readonly selfLoops?: boolean
    readonly kind?: 'sym' | 'rw'
    readonly weighted?: boolean
  } = {},
): Value {
  const { steps = 1, teleport, ...normalisation } = options
  if (!(Number.isInteger(steps) && steps >= 0)) throw new MdmlError('value', 'propagate: steps must be >= 0')
  const op = normalisedAdjacency(a, normalisation)
  const { columns, vector } = asColumns(x)
  let h = columns
  for (let k = 0; k < steps; k++) {
    const next = matmul(op, h)
    h = teleport === undefined ? next : add(mul(next, sub(1, teleport)), mul(columns, teleport))
  }
  return back(h, vector)
}

/**
 * The Dirichlet energy x^T L x (summed over columns for `[n, f]`): with the combinatorial Laplacian, the sum over
 * edges of w_ij (x_i - x_j)^2, a signal's roughness and the quantity graph regularisers penalise. Differentiable in
 * both arguments.
 *
 * @param l - A graph or a Laplacian `[n, n]`.
 * @param x - A signal `[n]` or `[n, f]`.
 * @param options - Which Laplacian of a graph.
 * @returns A scalar.
 * @concept graph-signal-processing
 * @example
 * ```ts
 * import { dirichletEnergy, pathGraph } from 'mdml-engine/graph'
 * import { item } from 'mdml-engine/tensor'
 * item(dirichletEnergy(pathGraph(3), [0, 1, 3])) // => 5
 * ```
 */
export function dirichletEnergy(l: ShiftOperator, x: TensorLike, options: LaplacianChoice = {}): Value {
  const { columns } = asColumns(x)
  return sum(mul(columns, matmul(operator(l, options), columns)), [-2, -1])
}

/**
 * The Fiedler vector (NetworkX fiedler_vector): the eigenvector of the second-smallest eigenvalue of the Laplacian
 * (the algebraic connectivity), unit length with its largest-magnitude entry positive. Dense by default, or by the
 * Lanczos solver of `mdml-engine/numerics/sparse` for large graphs.
 *
 * @param g - A connected graph with at least two nodes.
 * @param options - `normalised` (default false), `weighted` (default true) and `method` (`'dense'` or `'lanczos'`).
 * @returns The vector.
 * @throws MdmlError with code `value` for fewer than two nodes.
 * @concept spectral-clustering
 * @example
 * ```ts
 * import { fiedlerVector, pathGraph } from 'mdml-engine/graph'
 * Array.from(fiedlerVector(pathGraph(3)), (v) => Math.round(v * 1e4) / 1e4 + 0) // => [0.7071, 0, -0.7071]
 * ```
 */
export function fiedlerVector(
  g: Graph,
  options: LaplacianChoice & { readonly method?: 'dense' | 'lanczos' } = {},
): Float64Array {
  if (g.n < 2) throw new MdmlError('value', 'fiedlerVector: the graph needs at least two nodes')
  if (options.method === 'lanczos' && g.n > 2) {
    const l = laplacianCsr(g, options)
    const r = eigsh(l, 2, { which: 'SA' })
    const v = toFlat(r.vectors)
    return Float64Array.from({ length: g.n }, (_, i) => v[i * 2 + 1] ?? 0)
  }
  const { eigenvectors } = eigh(operator(g, options))
  const u = toFlat(concreteOrThrow(eigenvectors as Tensor, 'fiedlerVector'))
  return Float64Array.from({ length: g.n }, (_, i) => u[i * g.n + 1] ?? 0)
}

/**
 * A spectral embedding of the nodes (Ng, Jordan and Weiss, 2001; Belkin and Niyogi, 2003): the eigenvectors of the
 * k smallest eigenvalues of the symmetric normalised Laplacian (or the combinatorial one), optionally skipping the
 * trivial first and normalising each row to unit length, the input to k-means in spectral clustering. Differentiable
 * in a dense adjacency where the eigenvalues are distinct.
 *
 * @param a - A graph or a symmetric adjacency matrix `[n, n]`.
 * @param k - The embedding dimension.
 * @param options - `normalised` (default true), `dropFirst` (default false), `rowNormalise` (default false) and
 *   `weighted`.
 * @returns `[n, k]`.
 * @concept spectral-clustering
 * @example
 * ```ts
 * import { barbellGraph, spectralEmbedding } from 'mdml-engine/graph'
 * spectralEmbedding(barbellGraph(4, 0), 2).shape // => [8, 2]
 * ```
 */
export function spectralEmbedding(
  a: Graph | TensorLike,
  k: number,
  options: LaplacianChoice & { readonly dropFirst?: boolean; readonly rowNormalise?: boolean } = {},
): Value {
  const choice = options.weighted === undefined ? {} : { weighted: options.weighted }
  const l = (options.normalised ?? true) ? normalisedLaplacian(a, choice) : laplacian(a, choice)
  const n = l.shape[l.shape.length - 1] ?? 0
  const first = options.dropFirst === true ? 1 : 0
  if (!(Number.isInteger(k) && k >= 1 && k + first <= n)) {
    throw new MdmlError('value', `spectralEmbedding: k must be in [1, ${n - first}]`)
  }
  const y = sliceAxis(eigh(l).eigenvectors, -1, first, first + k)
  if (options.rowNormalise !== true) return y
  // Unit rows (a zero row stays zero).
  const norms = sum(mul(y, y), -1, true)
  return div(y, sqrt(where(greater(norms, 0), norms, 1)))
}

/** The state of {@link spectralEmbeddingAlgorithm}: an orthonormal basis and its Ritz values. */
export interface SubspaceState {
  /** The basis `[n, k]`, row-major. */
  readonly basis: Float64Array
  /** Ritz values of the Laplacian, ascending. */
  readonly values: Float64Array
  /** The largest residual norm |L v - lambda v| over the Ritz pairs. */
  readonly residual: number
}

/**
 * The spectral embedding as a traced subspace iteration: repeatedly multiply a k-column basis by M = 2I - L (whose
 * largest eigenvalues are the smallest of the normalised Laplacian L, all in [0, 2]), orthonormalise by QR, and rotate
 * by the Rayleigh-Ritz eigenvectors of the projected k-by-k matrix. Convergence is geometric in the eigengap ratio;
 * suits large sparse graphs where dense diagonalisation is too costly.
 *
 * @param g - The graph.
 * @param k - The number of eigenvectors (including the trivial one).
 * @param options - `tol` on the residual (default 1e-8) and `weighted`.
 * @returns The algorithm.
 * @concept spectral-clustering
 * @example
 * ```ts
 * import { barbellGraph, spectralEmbeddingAlgorithm } from 'mdml-engine/graph'
 * spectralEmbeddingAlgorithm(barbellGraph(4, 0), 2).name // => 'spectral-subspace-iteration'
 * ```
 */
export function spectralEmbeddingAlgorithm(
  g: Graph,
  k: number,
  options: { readonly tol?: number; readonly weighted?: boolean } = {},
): Algorithm<SubspaceState> {
  const n = g.n
  if (!(Number.isInteger(k) && k >= 1 && k <= n)) throw new MdmlError('value', `spectral iteration: k in [1, ${n}]`)
  const tol = options.tol ?? 1e-8
  const l = laplacianCsr(g, {
    normalised: true,
    ...(options.weighted === undefined ? {} : { weighted: options.weighted }),
  })
  const lv = toFlat(l.values)
  const applyL = (x: Float64Array, cols: number): Float64Array => {
    const y = new Float64Array(n * cols)
    for (let r = 0; r < n; r++) {
      for (let t = l.indptr[r] ?? 0; t < (l.indptr[r + 1] ?? 0); t++) {
        const c = l.indices[t] ?? 0
        const w = lv[t] ?? 0
        for (let j = 0; j < cols; j++) y[r * cols + j] = (y[r * cols + j] ?? 0) + w * (x[c * cols + j] ?? 0)
      }
    }
    return y
  }
  const rayleighRitz = (q: Float64Array): SubspaceState => {
    const lq = applyL(q, k)
    // T = Q^T L Q, then rotate by its eigenvectors.
    const t = new Float64Array(k * k)
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        let s = 0
        for (let r = 0; r < n; r++) s += (q[r * k + i] ?? 0) * (lq[r * k + j] ?? 0)
        t[i * k + j] = s
      }
    }
    for (let i = 0; i < k; i++)
      for (let j = 0; j < i; j++) t[i * k + j] = t[j * k + i] = ((t[i * k + j] ?? 0) + (t[j * k + i] ?? 0)) / 2
    const { eigenvalues, eigenvectors } = eigh(fromStorage(t, [k, k], 'float64'))
    const w = toFlat(concreteOrThrow(eigenvalues as Tensor, 'spectral iteration'))
    const z = toFlat(concreteOrThrow(eigenvectors as Tensor, 'spectral iteration'))
    const basis = new Float64Array(n * k)
    const lb = new Float64Array(n * k)
    for (let r = 0; r < n; r++) {
      for (let j = 0; j < k; j++) {
        let b = 0
        let c = 0
        for (let i = 0; i < k; i++) {
          b += (q[r * k + i] ?? 0) * (z[i * k + j] ?? 0)
          c += (lq[r * k + i] ?? 0) * (z[i * k + j] ?? 0)
        }
        basis[r * k + j] = b
        lb[r * k + j] = c
      }
    }
    let residual = 0
    for (let j = 0; j < k; j++) {
      const res = Float64Array.from(
        { length: n },
        (_, r) => (lb[r * k + j] ?? 0) - (w[j] ?? 0) * (basis[r * k + j] ?? 0),
      )
      residual = Math.max(residual, dnrm2(res))
    }
    return { basis, values: Float64Array.from(w), residual }
  }
  const orthonormal = (x: Float64Array): Float64Array =>
    Float64Array.from(toFlat(concreteOrThrow(qr(fromStorage(x, [n, k], 'float64')).q as Tensor, 'spectral iteration')))
  return {
    name: 'spectral-subspace-iteration',
    config: { k, tol } as Json,
    schema: channels({
      metrics: { residual: { description: 'largest Ritz residual |L v - lambda v|', direction: 'lower' } },
      tensors: { values: { role: 'state', description: 'Ritz values of the normalised Laplacian' } },
    }),
    init: (key) => rayleighRitz(orthonormal(Float64Array.from(normal(foldIn(key, 0), [n * k]).data))),
    step: (s) => {
      const lb = applyL(s.basis, k)
      const m = Float64Array.from(s.basis, (b, i) => 2 * b - (lb[i] ?? 0))
      return rayleighRitz(orthonormal(m))
    },
    done: (s) => (s.residual <= tol ? 'converged' : undefined),
    observe: (s) => ({ metrics: { residual: s.residual }, tensors: { values: { role: 'state', value: s.values } } }),
  }
}

/**
 * The spectral embedding of a graph by traced subspace iteration ({@link spectralEmbeddingAlgorithm}): the k
 * eigenvectors of the smallest eigenvalues of the normalised Laplacian, without dense diagonalisation.
 *
 * @param g - The graph.
 * @param k - The number of eigenvectors (including the trivial one).
 * @param options - `tol` (default 1e-8), `maxIter` (default 10000), `weighted`, and the seed, key and tracing.
 * @returns The eigenvalues (ascending) and the eigenvectors `[n, k]` as columns, each with its largest-magnitude entry
 *   positive.
 * @throws MdmlError with code `convergence` when the iteration does not converge.
 * @concept spectral-clustering
 * @example
 * ```ts
 * import { barbellGraph, iterativeSpectralEmbedding } from 'mdml-engine/graph'
 * Math.round(iterativeSpectralEmbedding(barbellGraph(4, 0), 2).values[0] * 1e6) / 1e6 // => 0
 * ```
 */
export function iterativeSpectralEmbedding(
  g: Graph,
  k: number,
  options: { readonly tol?: number; readonly maxIter?: number; readonly weighted?: boolean } & GraphRunOptions = {},
): { readonly values: Float64Array; readonly vectors: Tensor<'float64'> } {
  const { tol, maxIter = 10000, weighted, ...run } = options
  const algorithm = spectralEmbeddingAlgorithm(g, k, {
    ...(tol === undefined ? {} : { tol }),
    ...(weighted === undefined ? {} : { weighted }),
  })
  const result = runAlgorithm(algorithm, { ...run, maxSteps: maxIter })
  if (result.status !== 'converged') {
    throw new MdmlError('convergence', `iterativeSpectralEmbedding: no convergence in ${maxIter} iterations`)
  }
  const n = g.n
  const b = Float64Array.from(result.state.basis)
  for (let j = 0; j < k; j++) {
    const column = Float64Array.from({ length: n }, (_, r) => b[r * k + j] ?? 0)
    let big = 0
    for (let r = 0; r < n; r++) if (Math.abs(column[r] ?? 0) > Math.abs(column[big] ?? 0)) big = r
    const sign = (column[big] ?? 0) < 0 ? -1 : 1
    const norm = Math.sqrt(ddot(column, column)) || 1
    for (let r = 0; r < n; r++) b[r * k + j] = (sign * (column[r] ?? 0)) / norm
  }
  return { values: result.state.values, vectors: fromStorage(b, [n, k], 'float64') }
}
