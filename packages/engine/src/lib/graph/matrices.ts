import { WhiteboxError } from '../base/index.ts'
import { eigh } from '../numerics/linalg/index.ts'
import { type CsrMatrix, csr, primalValues } from '../numerics/sparse/index.ts'
import {
  add,
  concreteOrThrow,
  diagonal,
  div,
  expandDims,
  eye,
  fromStorage,
  greater,
  matmul,
  mul,
  promoteFloat,
  sliceAxis,
  sqrt,
  stack,
  sub,
  sum,
  type Tensor,
  type TensorLike,
  type Value,
  where,
} from '../tensor/index.ts'
import { type EdgeInput, type Graph, graph, isGraph } from './graph.ts'

/** A graph, or its (possibly weighted, possibly batched `[..., n, n]`) dense adjacency matrix. */
export type GraphOrMatrix = Graph | TensorLike

/**
 * The dense adjacency matrix (NetworkX to_numpy_array): `A[u, v]` is the weight of the edge from u to v (symmetric
 * when undirected; a self-loop's weight sits on the diagonal once).
 *
 * @param g - The graph.
 * @param options - `weighted` (default true; false puts 1 for every edge).
 * @returns `[n, n]`.
 * @concept graph-matrices
 * @example
 * ```ts
 * import { adjacencyMatrix, graph } from 'whitebox-engine/graph'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(adjacencyMatrix(graph([[0, 1, 2], [1, 2]]))) // => [[0, 2, 0], [2, 0, 1], [0, 1, 0]]
 * ```
 */
export function adjacencyMatrix(g: Graph, options: { readonly weighted?: boolean } = {}): Tensor<'float64'> {
  const weighted = options.weighted ?? true
  const n = g.n
  const a = new Float64Array(n * n)
  for (const e of g.edges) {
    const w = weighted ? e.weight : 1
    a[e.source * n + e.target] = w
    if (!g.directed) a[e.target * n + e.source] = w
  }
  return fromStorage(a, [n, n], 'float64')
}

/**
 * The sparse adjacency matrix in CSR form (NetworkX to_scipy_sparse_array), for large graphs and the sparse
 * eigensolvers.
 *
 * @param g - The graph.
 * @param options - `weighted` (default true).
 * @returns The `[n, n]` CSR matrix.
 * @concept graph-matrices
 * @example
 * ```ts
 * import { adjacencyCsr, pathGraph } from 'whitebox-engine/graph'
 * import { nnz } from 'whitebox-engine/numerics/sparse'
 * nnz(adjacencyCsr(pathGraph(4))) // => 6
 * ```
 */
export function adjacencyCsr(g: Graph, options: { readonly weighted?: boolean } = {}): CsrMatrix {
  const weighted = options.weighted ?? true
  const values = weighted ? Float64Array.from(g.out.weights) : new Float64Array(g.out.indices.length).fill(1)
  return csr({
    indptr: g.out.indptr,
    indices: g.out.indices,
    values: fromStorage(values, [values.length], 'float64'),
    shape: [g.n, g.n],
  })
}

/**
 * A graph from an adjacency matrix (NetworkX from_numpy_array, from_scipy_sparse_array): every non-zero entry is an
 * edge with that weight. Undirected (the default) reads each pair once, taking `A[i, j]` for `i <= j` and falling back
 * to `A[j, i]`.
 *
 * @param a - A dense `[n, n]` matrix or a CSR matrix.
 * @param options - `directed` (default false) and node labels.
 * @returns The graph.
 * @throws WhiteboxError with code `shape` when the matrix is not square.
 * @concept graph-matrices
 * @example
 * ```ts
 * import { fromAdjacency } from 'whitebox-engine/graph'
 * fromAdjacency([[0, 2], [2, 0]]).edges // => [{ source: 0, target: 1, weight: 2 }]
 * fromAdjacency([[0, 1], [0, 0]], { directed: true }).edges.length // => 1
 * ```
 */
export function fromAdjacency(
  a: TensorLike | CsrMatrix,
  options: { readonly directed?: boolean; readonly labels?: readonly string[] } = {},
): Graph {
  const directed = options.directed ?? false
  let n: number
  let entries: [number, number, number][]
  if (typeof a === 'object' && a !== null && 'indptr' in a) {
    const m = a as CsrMatrix
    n = m.shape[0]
    if (m.shape[1] !== n) throw new WhiteboxError('shape', 'fromAdjacency: the matrix must be square')
    const v = primalValues(m)
    entries = Array.from(m.indices, (j, k): [number, number, number] => [m.rows[k] ?? 0, j, v[k] ?? 0])
  } else {
    const t = concreteOrThrow(promoteFloat(a as TensorLike)[0] as Tensor, 'fromAdjacency')
    n = t.shape[0] ?? 0
    if (t.shape.length !== 2 || t.shape[1] !== n) {
      throw new WhiteboxError('shape', 'fromAdjacency: the matrix must be square')
    }
    entries = []
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) entries.push([i, j, t.data[i * n + j] ?? 0])
  }
  const lookup = new Map<number, number>()
  for (const [i, j, w] of entries) if (w !== 0) lookup.set(i * n + j, w)
  const edges: EdgeInput[] = []
  for (const [i, j, w] of entries) {
    if (w === 0) continue
    if (directed) edges.push([i, j, w])
    else if (i <= j) edges.push([i, j, w])
    else if (!lookup.has(j * n + i)) edges.push([j, i, w])
  }
  return graph(edges, { directed, nodes: options.labels ?? n })
}

/**
 * The incidence matrix `[n, m]` (NetworkX incidence_matrix): unoriented, 1 at both ends of each edge; oriented, -1 at
 * the source and +1 at the target, so `B B^T` is the Laplacian of an unweighted graph. Self-loop columns are empty, as
 * in NetworkX. With `weighted`, entries are scaled by the edge weight.
 *
 * @param g - The graph.
 * @param options - `oriented` (default false) and `weighted` (default false).
 * @returns `[n, m]`, columns in edge order.
 * @concept graph-matrices
 * @example
 * ```ts
 * import { incidenceMatrix, pathGraph } from 'whitebox-engine/graph'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(incidenceMatrix(pathGraph(3), { oriented: true })) // => [[-1, 0], [1, -1], [0, 1]]
 * ```
 */
export function incidenceMatrix(
  g: Graph,
  options: { readonly oriented?: boolean; readonly weighted?: boolean } = {},
): Tensor<'float64'> {
  const oriented = options.oriented ?? false
  const m = g.edges.length
  const b = new Float64Array(g.n * m)
  for (const [k, e] of g.edges.entries()) {
    if (e.source === e.target) continue
    const w = options.weighted === true ? e.weight : 1
    b[e.source * m + k] = oriented ? -w : w
    b[e.target * m + k] = w
  }
  return fromStorage(b, [g.n, m], 'float64')
}

/** The dense adjacency of a graph or matrix as a float value (differentiable when a matrix is given). */
function dense(a: GraphOrMatrix, weighted = true): Value {
  if (isGraph(a)) return adjacencyMatrix(a, { weighted })
  const [v] = promoteFloat(a) as [Value]
  const n = v.shape[v.shape.length - 1]
  if (v.shape.length < 2 || v.shape[v.shape.length - 2] !== n) {
    throw new WhiteboxError('shape', 'graph matrices need a square adjacency matrix [..., n, n]')
  }
  return v
}

function sizeOf(a: Value): number {
  return a.shape[a.shape.length - 1] ?? 0
}

/** 1 / d (or 1 / sqrt(d)) where d > 0 and 0 elsewhere, with finite gradients at zero degrees. */
function safeInverse(d: Value, root: boolean): Value {
  const positive = greater(d, 0)
  const safe = where(positive, d, 1)
  return where(positive, div(1, root ? sqrt(safe) : safe), 0)
}

/**
 * The degree matrix D = diag(row sums of A) (weighted degrees; out-degrees for a directed graph).
 *
 * @param a - A graph or an adjacency matrix `[..., n, n]` (differentiable).
 * @param options - `weighted` (default true) when a graph is given.
 * @returns `[..., n, n]`.
 * @concept graph-matrices
 * @example
 * ```ts
 * import { degreeMatrix, pathGraph } from 'whitebox-engine/graph'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(degreeMatrix(pathGraph(3))) // => [[1, 0, 0], [0, 2, 0], [0, 0, 1]]
 * ```
 */
export function degreeMatrix(a: GraphOrMatrix, options: { readonly weighted?: boolean } = {}): Value {
  const adj = dense(a, options.weighted ?? true)
  return mul(eye(sizeOf(adj)), expandDims(sum(adj, -1), -1))
}

/**
 * The combinatorial Laplacian L = D - A (NetworkX laplacian_matrix, scipy.sparse.csgraph.laplacian): positive
 * semi-definite for non-negative weights, with the constant vector in its null space and as many zero eigenvalues as
 * components. Differentiable in a dense adjacency.
 *
 * @param a - A graph or an adjacency matrix `[..., n, n]`.
 * @param options - `weighted` (default true) when a graph is given.
 * @returns `[..., n, n]`.
 * @concept graph-laplacian
 * @example
 * ```ts
 * import { laplacian, pathGraph } from 'whitebox-engine/graph'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(laplacian(pathGraph(3))) // => [[1, -1, 0], [-1, 2, -1], [0, -1, 1]]
 * ```
 */
export function laplacian(a: GraphOrMatrix, options: { readonly weighted?: boolean } = {}): Value {
  const adj = dense(a, options.weighted ?? true)
  return sub(mul(eye(sizeOf(adj)), expandDims(sum(adj, -1), -1)), adj)
}

/**
 * The symmetric normalised Laplacian D^{-1/2} (D - A) D^{-1/2} = I - D^{-1/2} A D^{-1/2} (NetworkX
 * normalized_laplacian_matrix, scipy's laplacian with `normed=True`); rows and columns of isolated nodes are zero. Its
 * eigenvalues lie in [0, 2].
 *
 * @param a - A graph or an adjacency matrix `[..., n, n]` (differentiable).
 * @param options - `weighted` (default true) when a graph is given.
 * @returns `[..., n, n]`.
 * @concept graph-laplacian
 * @example
 * ```ts
 * import { completeGraph, normalisedLaplacian } from 'whitebox-engine/graph'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(normalisedLaplacian(completeGraph(2))) // => [[1, -1], [-1, 1]]
 * ```
 */
export function normalisedLaplacian(a: GraphOrMatrix, options: { readonly weighted?: boolean } = {}): Value {
  const adj = dense(a, options.weighted ?? true)
  const s = safeInverse(sum(adj, -1), true)
  const l = sub(mul(eye(sizeOf(adj)), expandDims(sum(adj, -1), -1)), adj)
  return mul(mul(l, expandDims(s, -1)), expandDims(s, -2))
}

/**
 * The random-walk Laplacian D^{-1} (D - A) = I - P, with P the transition matrix (rows of isolated nodes are zero).
 * Similar to the symmetric normalised Laplacian, so it has the same eigenvalues.
 *
 * @param a - A graph or an adjacency matrix `[..., n, n]` (differentiable).
 * @param options - `weighted` (default true) when a graph is given.
 * @returns `[..., n, n]`.
 * @concept graph-laplacian
 * @example
 * ```ts
 * import { pathGraph, randomWalkLaplacian } from 'whitebox-engine/graph'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(randomWalkLaplacian(pathGraph(3)))[1] // => [-0.5, 1, -0.5]
 * ```
 */
export function randomWalkLaplacian(a: GraphOrMatrix, options: { readonly weighted?: boolean } = {}): Value {
  const adj = dense(a, options.weighted ?? true)
  const l = sub(mul(eye(sizeOf(adj)), expandDims(sum(adj, -1), -1)), adj)
  return mul(l, expandDims(safeInverse(sum(adj, -1), false), -1))
}

/**
 * The random-walk transition matrix P = D^{-1} A (row-stochastic), optionally lazy (`laziness` I + (1 - laziness) P).
 * A node without out-edges (a dangling node) jumps uniformly (default), stays put (`'self'`), or keeps a zero row.
 *
 * @param a - A graph or an adjacency matrix `[..., n, n]` (differentiable).
 * @param options - `laziness` in [0, 1] (default 0), the `dangling` rule, and `weighted` (default true).
 * @returns `[..., n, n]`.
 * @concept random-walks
 * @example
 * ```ts
 * import { pathGraph, transitionMatrix } from 'whitebox-engine/graph'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(transitionMatrix(pathGraph(3))) // => [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
 * toArray(transitionMatrix(pathGraph(2), { laziness: 0.5 })) // => [[0.5, 0.5], [0.5, 0.5]]
 * ```
 */
export function transitionMatrix(
  a: GraphOrMatrix,
  options: {
    readonly laziness?: number
    readonly dangling?: 'uniform' | 'self' | 'zero'
    readonly weighted?: boolean
  } = {},
): Value {
  const adj = dense(a, options.weighted ?? true)
  const n = sizeOf(adj)
  const d = sum(adj, -1, true)
  const positive = greater(d, 0)
  const rule = options.dangling ?? 'uniform'
  const fallback =
    rule === 'uniform'
      ? fromStorage(new Float64Array(n * n).fill(1 / n), [n, n], 'float64')
      : rule === 'self'
        ? eye(n)
        : fromStorage(new Float64Array(n * n), [n, n], 'float64')
  const p = where(positive, div(adj, where(positive, d, 1)), fallback)
  const lazy = options.laziness ?? 0
  if (!(lazy >= 0 && lazy <= 1)) throw new WhiteboxError('value', 'transitionMatrix: laziness must be in [0, 1]')
  return lazy === 0 ? p : add(mul(eye(n), lazy), mul(p, 1 - lazy))
}

/**
 * The modularity matrix B = A - k k^T / 2m (NetworkX modularity_matrix): its leading eigenvectors split a graph into
 * communities, and Q = s^T B s / 4m for a two-way split s in {-1, 1}^n.
 *
 * @param a - A graph or a symmetric adjacency matrix `[..., n, n]` (differentiable).
 * @param options - `weighted` (default true) when a graph is given.
 * @returns `[..., n, n]`.
 * @concept modularity
 * @example
 * ```ts
 * import { completeGraph, modularityMatrix } from 'whitebox-engine/graph'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(modularityMatrix(completeGraph(2))) // => [[-0.5, 0.5], [0.5, -0.5]]
 * ```
 */
export function modularityMatrix(a: GraphOrMatrix, options: { readonly weighted?: boolean } = {}): Value {
  const adj = dense(a, options.weighted ?? true)
  const k = sum(adj, -1)
  const twoM = sum(k, -1, true)
  return sub(adj, div(mul(expandDims(k, -1), expandDims(k, -2)), expandDims(twoM, -1)))
}

/**
 * The normalised adjacency used for message passing (Kipf and Welling, 2017): with self-loops added (by default),
 * Ã = A + I and D̃ its degrees, the symmetric form D̃^{-1/2} Ã D̃^{-1/2} or the random-walk (mean-aggregation) form
 * D̃^{-1} Ã. Rows of isolated nodes without self-loops are zero. Differentiable in a dense adjacency.
 *
 * @param a - A graph or an adjacency matrix `[..., n, n]`.
 * @param options - `selfLoops` (default true), `kind` (`'sym'`, default, or `'rw'`) and `weighted` (default true).
 * @returns `[..., n, n]`.
 * @concept message-passing
 * @reference T. N. Kipf and M. Welling, Semi-supervised classification with graph convolutional networks (ICLR 2017).
 * @example
 * ```ts
 * import { normalisedAdjacency, pathGraph } from 'whitebox-engine/graph'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(normalisedAdjacency(pathGraph(2), { kind: 'rw' })) // => [[0.5, 0.5], [0.5, 0.5]]
 * (toArray(normalisedAdjacency(pathGraph(2))) as number[][])[0]?.map((v) => Math.round(v * 1e12) / 1e12) // => [0.5, 0.5]
 * toArray(normalisedAdjacency(pathGraph(3), { kind: 'rw', selfLoops: false }))[1] // => [0.5, 0, 0.5]
 * ```
 */
export function normalisedAdjacency(
  a: GraphOrMatrix,
  options: { readonly selfLoops?: boolean; readonly kind?: 'sym' | 'rw'; readonly weighted?: boolean } = {},
): Value {
  const base = dense(a, options.weighted ?? true)
  const adj = (options.selfLoops ?? true) ? add(base, eye(sizeOf(base))) : base
  const d = sum(adj, -1)
  if (options.kind === 'rw') return mul(adj, expandDims(safeInverse(d, false), -1))
  const s = safeInverse(d, true)
  return mul(mul(adj, expandDims(s, -1)), expandDims(s, -2))
}

/**
 * Laplacian positional encodings (Dwivedi et al., 2020): the eigenvectors of the symmetric normalised Laplacian for the
 * k smallest eigenvalues after the first, as node features `[n, k]`. Each column has its largest-magnitude entry
 * positive (eigenvectors are defined up to sign; models usually randomise it). Differentiable where the eigenvalues
 * are distinct.
 *
 * @param a - A graph or an adjacency matrix `[n, n]`.
 * @param k - The number of encodings (at most n - 1).
 * @param options - `weighted` (default true) when a graph is given.
 * @returns `[n, k]`.
 * @concept positional-encodings
 * @reference V. P. Dwivedi et al., Benchmarking graph neural networks (2020).
 * @example
 * ```ts
 * import { cycleGraph, laplacianPositionalEncoding } from 'whitebox-engine/graph'
 * laplacianPositionalEncoding(cycleGraph(6), 2).shape // => [6, 2]
 * ```
 */
export function laplacianPositionalEncoding(
  a: GraphOrMatrix,
  k: number,
  options: { readonly weighted?: boolean } = {},
): Value {
  const l = normalisedLaplacian(a, options)
  const n = sizeOf(l)
  if (!(Number.isInteger(k) && k >= 1 && k < n)) {
    throw new WhiteboxError('value', `laplacianPositionalEncoding: k must be in [1, ${n - 1}]`)
  }
  return sliceAxis(eigh(l).eigenvectors, -1, 1, k + 1)
}

/**
 * Random-walk structural encodings (Dwivedi et al., 2022): for each node, the probabilities of returning to itself
 * after 1 .. k steps of the random walk, `[diag(P), diag(P^2), ..., diag(P^k)]`, as node features `[n, k]`.
 * Differentiable in a dense adjacency.
 *
 * @param a - A graph or an adjacency matrix `[n, n]`.
 * @param k - The number of steps.
 * @param options - `weighted` (default true) when a graph is given.
 * @returns `[n, k]`.
 * @concept positional-encodings
 * @reference V. P. Dwivedi et al., Graph neural networks with learnable structural and positional representations
 *   (ICLR 2022).
 * @example
 * ```ts
 * import { cycleGraph, randomWalkPositionalEncoding } from 'whitebox-engine/graph'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(randomWalkPositionalEncoding(cycleGraph(4), 2))[0] // => [0, 0.5]
 * ```
 */
export function randomWalkPositionalEncoding(
  a: GraphOrMatrix,
  k: number,
  options: { readonly weighted?: boolean } = {},
): Value {
  if (!(Number.isInteger(k) && k >= 1)) throw new WhiteboxError('value', 'randomWalkPositionalEncoding: k must be >= 1')
  const p = transitionMatrix(a, { dangling: 'zero', ...options })
  const columns: Value[] = []
  let power = p
  for (let i = 0; i < k; i++) {
    columns.push(diagonal(power, 0, -2, -1))
    if (i + 1 < k) power = matmul(power, p)
  }
  return stack(columns, -1)
}

/**
 * The sparse combinatorial or symmetric normalised Laplacian in CSR form, for spectral methods on large graphs (with
 * the Lanczos solver of `whitebox-engine/numerics/sparse`).
 *
 * @param g - The graph.
 * @param options - `normalised` (default false) and `weighted` (default true).
 * @returns The `[n, n]` CSR matrix.
 * @concept graph-laplacian
 * @example
 * ```ts
 * import { laplacianCsr, pathGraph } from 'whitebox-engine/graph'
 * import { toDense } from 'whitebox-engine/numerics/sparse'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(toDense(laplacianCsr(pathGraph(2)))) // => [[1, -1], [-1, 1]]
 * ```
 */
export function laplacianCsr(
  g: Graph,
  options: { readonly normalised?: boolean; readonly weighted?: boolean } = {},
): CsrMatrix {
  const weighted = options.weighted ?? true
  const n = g.n
  const a = g.out
  const deg = new Float64Array(n)
  for (let u = 0; u < n; u++) {
    for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++)
      deg[u] = (deg[u] ?? 0) + (weighted ? (a.weights[k] ?? 0) : 1)
  }
  const scale = Float64Array.from(deg, (d) => (options.normalised === true ? (d > 0 ? 1 / Math.sqrt(d) : 0) : 1))
  const indptr = new Int32Array(n + 1)
  const indices: number[] = []
  const values: number[] = []
  for (let u = 0; u < n; u++) {
    const row = new Map<number, number>([[u, deg[u] ?? 0]])
    for (let k = a.indptr[u] ?? 0; k < (a.indptr[u + 1] ?? 0); k++) {
      const v = a.indices[k] ?? 0
      row.set(v, (row.get(v) ?? 0) - (weighted ? (a.weights[k] ?? 0) : 1))
    }
    for (const v of [...row.keys()].sort((p, q) => p - q)) {
      const x = (row.get(v) ?? 0) * (scale[u] ?? 0) * (scale[v] ?? 0)
      if (x === 0) continue
      indices.push(v)
      values.push(x)
    }
    indptr[u + 1] = indices.length
  }
  return csr({
    indptr,
    indices,
    values: fromStorage(Float64Array.from(values), [values.length], 'float64'),
    shape: [n, n],
  })
}
