/**
 * t-SNE (van der Maaten and Hinton 2008; scikit-learn's `TSNE`): neighbour probabilities calibrated to a perplexity in
 * the data are matched by a heavy-tailed Student-t similarity in the embedding, by gradient descent on their KL
 * divergence with early exaggeration, momentum and gains, one step per iteration. The gradient is exact over all pairs
 * or Barnes-Hut (a sparse P over the nearest neighbours, the repulsion summarised by a space-partitioning tree).
 */

import { MdmlError, maxOf } from '../base/index.ts'
import { type DistanceAlias, type DistanceName, distanceMatrix } from '../distances/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { normal } from '../random/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { type TensorLike, toArray } from '../tensor/index.ts'
import { type Algorithm, channels, stateFromJson, stateToJson } from '../trace/index.ts'
import { type EmbeddingParams, type EmbeddingVerbs, embeddingVerbs, pcaStart, trainingRows } from './_support.ts'

const MACHINE_EPSILON = Number.EPSILON

/**
 * Conditional probabilities p(j | i) with each row's precision beta found by bisection so that the row's entropy is
 * log(perplexity) (sklearn's `_binary_search_perplexity`, natural logarithms).
 *
 * @param sqdist - Squared distances `[n, m]` (rows of the neighbours' distances, or all of them).
 * @param perplexity - The target perplexity.
 * @param self - For each row, the column to leave out (its own index in a full matrix), or -1 to leave none out.
 * @returns The conditional probabilities and each row's beta (precision 1 / (2 sigma^2)).
 * @example
 * ```ts
 * import { conditionalProbabilities } from 'mdml-engine/manifold'
 * const { p } = conditionalProbabilities([[0, 1, 4], [1, 0, 1], [4, 1, 0]], 1.5)
 * p[0]?.reduce((a, b) => a + b, 0) // => 1
 * ```
 */
export function conditionalProbabilities(
  sqdist: readonly (readonly number[])[],
  perplexity: number,
  self?: (row: number) => number,
): { readonly p: number[][]; readonly beta: number[] } {
  const desired = Math.log(perplexity)
  const betas: number[] = []
  const p = sqdist.map((row, i) => {
    const skip = self === undefined ? i : self(i)
    let beta = 1
    let lo = Number.NEGATIVE_INFINITY
    let hi = Number.POSITIVE_INFINITY
    let pi = new Array<number>(row.length).fill(0)
    for (let step = 0; step < 100; step++) {
      pi = row.map((d, j) => (j === skip ? 0 : Math.exp(-d * beta)))
      let sum = pi.reduce((a, v) => a + v, 0)
      if (sum === 0) sum = 1e-8
      let sumDist = 0
      pi = pi.map((v, j) => {
        const q = v / sum
        sumDist += (row[j] ?? 0) * q
        return q
      })
      const diff = Math.log(sum) + beta * sumDist - desired
      if (Math.abs(diff) <= 1e-5) break
      if (diff > 0) {
        lo = beta
        beta = hi === Number.POSITIVE_INFINITY ? beta * 2 : (beta + hi) / 2
      } else {
        hi = beta
        beta = lo === Number.NEGATIVE_INFINITY ? beta / 2 : (beta + lo) / 2
      }
    }
    betas.push(beta)
    return pi
  })
  return { p, beta: betas }
}

/** Squared distances among the rows under a named metric (squared for every metric, as sklearn). */
function squaredDistances(x: TensorLike, metric: DistanceName | DistanceAlias): { n: number; rows: number[][] } {
  const { n, data } = distanceMatrix(x, metric)
  const squared = metric === 'euclidean' || metric === 'l2' ? data.map((v) => v * v) : data.map((v) => v * v)
  return { n, rows: Array.from({ length: n }, (_, i) => Array.from(squared.subarray(i * n, (i + 1) * n))) }
}

/**
 * t-SNE's joint probabilities: (p(j|i) + p(i|j)) / (2n) floored at machine epsilon (sklearn's `_joint_probabilities`).
 *
 * @param x - Points `[n, d]`.
 * @param perplexity - The perplexity.
 * @param metric - The distance (default euclidean); squared before calibration.
 * @returns The joint probabilities `[n, n]` (zero diagonal) and the calibrated bandwidths sigma_i.
 * @example
 * ```ts
 * import { jointProbabilities } from 'mdml-engine/manifold'
 * const { p } = jointProbabilities([[0], [1], [2], [10]], 1.5)
 * p.flat().reduce((a, b) => a + b, 0) // => 1
 * ```
 */
export function jointProbabilities(
  x: TensorLike,
  perplexity: number,
  metric: DistanceName | DistanceAlias = 'euclidean',
): { readonly p: number[][]; readonly sigmas: number[] } {
  const { rows } = squaredDistances(x, metric)
  const { p: cond, beta } = conditionalProbabilities(rows, perplexity)
  const sym = cond.map((r, i) => r.map((v, j) => v + (cond[j]?.[i] ?? 0)))
  const total = Math.max(
    sym.reduce((a, r) => a + r.reduce((s, v) => s + v, 0), 0),
    MACHINE_EPSILON,
  )
  return {
    p: sym.map((r, i) => r.map((v, j) => (i === j ? 0 : Math.max(v / total, MACHINE_EPSILON)))),
    sigmas: beta.map((b) => Math.sqrt(1 / (2 * b))),
  }
}

/** A sparse symmetric matrix by rows: for row i the entries `indptr[i] .. indptr[i + 1]` of `indices` and `values`. */
export interface SparseRows {
  readonly n: number
  readonly indptr: number[]
  readonly indices: number[]
  readonly values: number[]
}

/**
 * The sparse joint probabilities of Barnes-Hut t-SNE (sklearn's `_joint_probabilities_nn`): conditional probabilities
 * over the `3 perplexity + 1` nearest neighbours of each row, symmetrised and normalised.
 *
 * @param x - Points `[n, d]`.
 * @param perplexity - The perplexity.
 * @param metric - The distance (default euclidean); squared before calibration.
 * @returns The sparse joint probabilities and each row's calibrated bandwidth.
 * @example
 * ```ts
 * import { sparseJointProbabilities } from 'mdml-engine/manifold'
 * const { p } = sparseJointProbabilities([[0], [1], [2], [3], [10], [11], [12]], 1.5)
 * p.values.reduce((a, b) => a + b, 0) // => 1
 * ```
 */
export function sparseJointProbabilities(
  x: TensorLike,
  perplexity: number,
  metric: DistanceName | DistanceAlias = 'euclidean',
): { readonly p: SparseRows; readonly sigmas: number[] } {
  const { n, rows } = squaredDistances(x, metric)
  const k = Math.min(n - 1, Math.floor(3 * perplexity + 1))
  const neighbours = rows.map((row, i) =>
    row
      .map((d, j) => ({ d, j }))
      .filter((e) => e.j !== i)
      .sort((a, b) => a.d - b.d || a.j - b.j)
      .slice(0, k),
  )
  const { p: cond, beta } = conditionalProbabilities(
    neighbours.map((nb) => nb.map((e) => e.d)),
    perplexity,
    () => -1,
  )
  const entries = new Map<number, number>()
  for (const [i, nb] of neighbours.entries())
    for (const [a, e] of nb.entries()) {
      const v = cond[i]?.[a] ?? 0
      entries.set(i * n + e.j, (entries.get(i * n + e.j) ?? 0) + v)
      entries.set(e.j * n + i, (entries.get(e.j * n + i) ?? 0) + v)
    }
  const total = Math.max(
    [...entries.values()].reduce((a, v) => a + v, 0),
    MACHINE_EPSILON,
  )
  const keys = [...entries.keys()].sort((a, b) => a - b)
  const indptr = new Array<number>(n + 1).fill(0)
  const indices: number[] = []
  const values: number[] = []
  for (const key of keys) {
    const i = Math.floor(key / n)
    indptr[i + 1] = (indptr[i + 1] ?? 0) + 1
    indices.push(key % n)
    values.push((entries.get(key) ?? 0) / total)
  }
  for (let i = 0; i < n; i++) indptr[i + 1] = (indptr[i + 1] ?? 0) + (indptr[i] ?? 0)
  return { p: { n, indptr, indices, values }, sigmas: beta.map((b) => Math.sqrt(1 / (2 * b))) }
}

/**
 * The exact KL divergence and gradient on flat buffers: `p` `[n * n]` row-major, `y` `[n * k]`, the gradient written into
 * `grad` `[n * k]` (unless it is undefined: the divergence alone) and the Student-t kernel into `kern` `[n * n]` (scratch,
 * reused across iterations). The gradient uses `exaggeration` times p (early exaggeration) while the divergence is of p
 * itself. Sums run in the order of sklearn's `_kl_divergence`.
 */
function klAndGradient(
  p: Float64Array,
  y: Float64Array,
  n: number,
  k: number,
  dof: number,
  exaggeration: number,
  kern: Float64Array,
  grad: Float64Array | undefined,
): number {
  const power = (dof + 1) / -2
  let total = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i === j) {
        kern[i * n + j] = 0
        continue
      }
      let d2 = 0
      for (let a = 0; a < k; a++) {
        const diff = (y[i * k + a] ?? 0) - (y[j * k + a] ?? 0)
        d2 += diff * diff
      }
      const v = (1 + d2 / dof) ** power
      kern[i * n + j] = v
      total += v
    }
  let kl = 0
  grad?.fill(0)
  const c = (2 * (dof + 1)) / dof
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i === j) continue
      const kij = kern[i * n + j] ?? 0
      const q = Math.max(kij / total, MACHINE_EPSILON)
      const pij = p[i * n + j] ?? 0
      kl += pij * Math.log(Math.max(pij, MACHINE_EPSILON) / q)
      if (grad === undefined) continue
      const coef = (exaggeration === 1 ? pij : pij * exaggeration) - q
      const ck = c * (coef * kij)
      for (let a = 0; a < k; a++)
        grad[i * k + a] = (grad[i * k + a] ?? 0) + ck * ((y[i * k + a] ?? 0) - (y[j * k + a] ?? 0))
    }
  return kl
}

/**
 * The exact t-SNE objective and gradient (sklearn's `_kl_divergence`) for a Student-t kernel with `dof` degrees of
 * freedom.
 *
 * @param p - Joint probabilities `[n, n]`.
 * @param y - The embedding `[n, k]`.
 * @param dof - Degrees of freedom (default 1).
 * @returns The KL divergence and its gradient `[n, k]`.
 * @example
 * ```ts
 * import { tsneGradient } from 'mdml-engine/manifold'
 * tsneGradient([[0, 0.5], [0.5, 0]], [[0, 0], [1, 0]]).kl // => 0
 * ```
 */
export function tsneGradient(
  p: readonly (readonly number[])[],
  y: readonly (readonly number[])[],
  dof = 1,
): { readonly kl: number; readonly grad: number[][] } {
  const n = y.length
  const k = y[0]?.length ?? 0
  const grad = new Float64Array(n * k)
  const kl = klAndGradient(
    Float64Array.from(p.flat()),
    Float64Array.from(y.flat()),
    n,
    k,
    dof,
    1,
    new Float64Array(n * n),
    grad,
  )
  return { kl, grad: Array.from({ length: n }, (_, i) => Array.from(grad.subarray(i * k, (i + 1) * k))) }
}

// ---------------------------------------------------------------------------------------------------------------
// Barnes-Hut

/** A space-partitioning tree over an embedding (a quadtree in 2-D, an octree in 3-D, ...), in flat arrays. */
interface SpacePartitionTree {
  readonly k: number
  readonly arity: number
  count: number[]
  centre: number[]
  half: number[]
  com: number[]
  child: number[]
  /** The single point of a leaf (-1 for none or an interior node). */
  point: number[]
}

function newNode(t: SpacePartitionTree, centre: ArrayLike<number>, half: number): number {
  const id = t.count.length
  t.count.push(0)
  t.half.push(half)
  t.point.push(-1)
  for (let a = 0; a < t.k; a++) {
    t.centre.push(centre[a] ?? 0)
    t.com.push(0)
  }
  for (let c = 0; c < t.arity; c++) t.child.push(-1)
  return id
}

function childOf(t: SpacePartitionTree, node: number, y: ArrayLike<number>): number {
  let which = 0
  const centre = new Array<number>(t.k)
  const half = (t.half[node] ?? 0) / 2
  for (let a = 0; a < t.k; a++) {
    const up = (y[a] ?? 0) >= (t.centre[node * t.k + a] ?? 0)
    if (up) which |= 1 << a
    centre[a] = (t.centre[node * t.k + a] ?? 0) + (up ? half : -half)
  }
  let id = t.child[node * t.arity + which] ?? -1
  if (id < 0) {
    id = newNode(t, centre, half)
    t.child[node * t.arity + which] = id
  }
  return id
}

function insert(t: SpacePartitionTree, node: number, i: number, y: Float64Array, depth: number): void {
  const k = t.k
  const before = t.count[node] ?? 0
  for (let a = 0; a < k; a++)
    t.com[node * k + a] = ((t.com[node * k + a] ?? 0) * before + (y[i * k + a] ?? 0)) / (before + 1)
  t.count[node] = before + 1
  if (before === 0) {
    t.point[node] = i
    return
  }
  const q = t.point[node] ?? -1
  if (q >= 0) {
    // A leaf holding point(s) must split, unless the new point coincides with them (or the tree is already deep).
    let same = true
    for (let a = 0; a < k; a++) if ((y[q * k + a] ?? 0) !== (y[i * k + a] ?? 0)) same = false
    if (same || depth > 40) return
    t.point[node] = -1
    const below = childOf(t, node, y.subarray(q * k, (q + 1) * k))
    for (let r = 0; r < before; r++) insert(t, below, q, y, depth + 1)
  }
  insert(t, childOf(t, node, y.subarray(i * k, (i + 1) * k)), i, y, depth + 1)
}

function buildTree(y: Float64Array, n: number, k: number): SpacePartitionTree {
  const lo = new Array<number>(k).fill(Number.POSITIVE_INFINITY)
  const hi = new Array<number>(k).fill(Number.NEGATIVE_INFINITY)
  for (let i = 0; i < n; i++)
    for (let a = 0; a < k; a++) {
      lo[a] = Math.min(lo[a] ?? 0, y[i * k + a] ?? 0)
      hi[a] = Math.max(hi[a] ?? 0, y[i * k + a] ?? 0)
    }
  const centre = lo.map((v, a) => (v + (hi[a] ?? 0)) / 2)
  const half = Math.max(maxOf(lo.map((v, a) => ((hi[a] ?? 0) - v) / 2)), 1e-12) * (1 + 1e-9)
  const t: SpacePartitionTree = { k, arity: 1 << k, count: [], centre: [], half: [], com: [], child: [], point: [] }
  const root = newNode(t, centre, half)
  for (let i = 0; i < n; i++) insert(t, root, i, y, 0)
  return t
}

/**
 * The Barnes-Hut t-SNE objective and gradient (sklearn's `_kl_divergence_bh`, Student-t with one degree of freedom): the
 * attraction over the nonzeros of the sparse P, the repulsion from a tree in which a cell far enough from a point (its
 * width over its distance below `theta`) acts as one particle at its centre of mass. With `theta` 0 it is exact.
 *
 * @param p - The sparse joint probabilities.
 * @param y - The embedding `[n, k]`, flat.
 * @param k - Dimensions.
 * @param theta - The opening angle (sklearn's `angle`, default 0.5).
 * @param exaggeration - Early exaggeration multiplying P in the gradient (default 1).
 * @param grad - Written when given.
 * @returns The KL divergence (of the unexaggerated P).
 * @example
 * ```ts
 * import { barnesHutKl } from 'mdml-engine/manifold'
 * const p = { n: 2, indptr: [0, 1, 2], indices: [1, 0], values: [0.5, 0.5] }
 * barnesHutKl(p, Float64Array.of(0, 0, 1, 0), 2, 0) // => 0
 * ```
 */
export function barnesHutKl(
  p: SparseRows,
  y: Float64Array,
  k: number,
  theta = 0.5,
  exaggeration = 1,
  grad?: Float64Array,
): number {
  const n = p.n
  const tree = buildTree(y, n, k)
  const neg = new Float64Array(n * k)
  const sums = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const stack = [0]
    let sq = 0
    while (stack.length > 0) {
      const node = stack.pop() as number
      const count = tree.count[node] ?? 0
      if (count === 0) continue
      const leaf =
        (tree.point[node] ?? -1) >= 0 ||
        tree.child.slice(node * tree.arity, (node + 1) * tree.arity).every((c) => c < 0)
      if (leaf && tree.point[node] === i && count === 1) continue
      let d2 = 0
      for (let a = 0; a < k; a++) d2 += ((y[i * k + a] ?? 0) - (tree.com[node * k + a] ?? 0)) ** 2
      const width = 2 * (tree.half[node] ?? 0)
      if (leaf || (theta > 0 && width * width < theta * theta * d2)) {
        const q = 1 / (1 + d2)
        sq += count * q
        for (let a = 0; a < k; a++)
          neg[i * k + a] = (neg[i * k + a] ?? 0) + count * q * q * ((y[i * k + a] ?? 0) - (tree.com[node * k + a] ?? 0))
        continue
      }
      for (let c = 0; c < tree.arity; c++) {
        const ch = tree.child[node * tree.arity + c] ?? -1
        if (ch >= 0) stack.push(ch)
      }
    }
    sums[i] = sq
  }
  const sumQ = sums.reduce((a, v) => a + v, 0)
  let kl = 0
  grad?.fill(0)
  for (let i = 0; i < n; i++)
    for (let e = p.indptr[i] ?? 0; e < (p.indptr[i + 1] ?? 0); e++) {
      const j = p.indices[e] ?? 0
      const pij = p.values[e] ?? 0
      let d2 = 0
      for (let a = 0; a < k; a++) d2 += ((y[i * k + a] ?? 0) - (y[j * k + a] ?? 0)) ** 2
      const q = 1 / (1 + d2)
      kl += pij * Math.log(Math.max(pij, MACHINE_EPSILON) / Math.max(q / sumQ, MACHINE_EPSILON))
      if (grad === undefined) continue
      for (let a = 0; a < k; a++)
        grad[i * k + a] = (grad[i * k + a] ?? 0) + pij * exaggeration * q * ((y[i * k + a] ?? 0) - (y[j * k + a] ?? 0))
    }
  if (grad !== undefined) for (let t = 0; t < n * k; t++) grad[t] = 4 * ((grad[t] ?? 0) - (neg[t] ?? 0) / sumQ)
  return kl
}

// ---------------------------------------------------------------------------------------------------------------
// The algorithm

/** The state of t-SNE's gradient descent. */
export interface TsneState {
  readonly y: number[][]
  readonly update: number[][]
  readonly gains: number[][]
  /** The KL divergence (of the unexaggerated P) at the positions the last step started from (the initial ones before it). */
  readonly kl: number
  /** The norm of the gain-scaled gradient of the last step (a large number before the first). */
  readonly gradNorm: number
}

/** Options of {@link tsneAlgorithm}. */
export interface TsneOptions {
  readonly learningRate: number
  /** Early exaggeration of P (default 12) for the first `exaggerationIter` steps (default 250), with momentum 0.5. */
  readonly earlyExaggeration?: number
  readonly exaggerationIter?: number
  /** Degrees of freedom of the Student-t kernel (default 1; the exact gradient only). */
  readonly dof?: number
  /** The opening angle of the Barnes-Hut tree, used when P is sparse (default 0.5). */
  readonly theta?: number
  /** Stop when the gradient norm falls below this, checked every 50 steps (default 1e-7; 0 never). */
  readonly minGradNorm?: number
}

/**
 * t-SNE's gradient descent with momentum and per-coordinate gains (sklearn's `_gradient_descent`) as a traced
 * {@link Algorithm}: early exaggeration of P with momentum 0.5 for the first `exaggerationIter` steps, then, with the
 * update and gains reset, momentum 0.8. The reported KL divergence is always of the unexaggerated P. With dense `p` the
 * gradient is exact; with sparse `p` it is Barnes-Hut. It observes the KL divergence and the `embedding`.
 *
 * The optimiser is deliberately not a solver slot. It is sklearn's two-stage schedule (an exaggerated objective with
 * momentum 0.5, then the true one with momentum 0.8, the update and gains reset between them) and its gains rule is
 * part of what the embedding is compared to; a generic `smoothSolvers` mode would run on the true KL alone, drop the
 * exaggeration that opens the clusters, and not reproduce the reference. The KL and the embedding are on the tape at
 * every step.
 *
 * @param p - Joint probabilities, dense `[n, n]` or sparse.
 * @param init - The initial embedding.
 * @param options - Learning rate, early exaggeration and its length, kernel degrees of freedom, angle and gradient tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { jointProbabilities, tsneAlgorithm } from 'mdml-engine/manifold'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const x = [[0], [0.1], [0.2], [5], [5.1], [5.2]]
 * const { p } = jointProbabilities(x, 2)
 * const init = [[0, 0.1], [0.1, 0], [-0.1, 0], [0.05, 0.1], [0, -0.1], [0.1, 0.1]]
 * runAlgorithm(tsneAlgorithm(p, init, { learningRate: 50, exaggerationIter: 20 }), { maxSteps: 60 }).state.kl < 0.5 // => true
 * ```
 */
export function tsneAlgorithm(
  p: readonly (readonly number[])[] | SparseRows,
  init: readonly (readonly number[])[],
  options: TsneOptions,
): Algorithm<TsneState> {
  const ex = options.earlyExaggeration ?? 12
  const exIter = options.exaggerationIter ?? 250
  const lr = options.learningRate
  const dof = options.dof ?? 1
  const theta = options.theta ?? 0.5
  const minGrad = options.minGradNorm ?? 1e-7
  const n = init.length
  const k = init[0]?.length ?? 0
  const sparse = !Array.isArray(p)
  const pFlat = sparse ? undefined : Float64Array.from((p as readonly (readonly number[])[]).flat())
  const kern = new Float64Array(sparse ? 0 : n * n)
  const grad = new Float64Array(n * k)
  const y = new Float64Array(n * k)
  const evaluate = (exaggeration: number, withGrad: boolean): number =>
    pFlat === undefined
      ? barnesHutKl(p as SparseRows, y, k, theta, exaggeration, withGrad ? grad : undefined)
      : klAndGradient(pFlat, y, n, k, dof, exaggeration, kern, withGrad ? grad : undefined)
  const load = (rows: readonly (readonly number[])[]): void => {
    for (const [i, r] of rows.entries()) for (const [a, v] of r.entries()) y[i * k + a] = v
  }
  return {
    name: sparse ? 'tsne-barnes-hut' : 'tsne',
    config: { learningRate: lr, earlyExaggeration: ex, exaggerationIter: exIter, dof, theta: sparse ? theta : null },
    schema: channels({
      metrics: {
        kl: { description: 'the KL divergence between the data and embedding affinities', direction: 'lower' },
        gradNorm: { description: 'the norm of the gain-scaled gradient (-1 before the first step)' },
      },
      tensors: { embedding: { role: 'state', description: 'the embedding', shape: ['n', 'k'] } },
    }),
    init: () => {
      load(init)
      return {
        y: init.map((r) => [...r]),
        update: init.map((r) => r.map(() => 0)),
        gains: init.map((r) => r.map(() => 1)),
        kl: evaluate(1, false),
        gradNorm: Number.MAX_VALUE,
      }
    },
    step: (s, ctx) => {
      const exploring = ctx.t < exIter
      const momentum = exploring ? 0.5 : 0.8
      // sklearn runs the two stages as two descents, each starting with no update and unit gains.
      const fresh = ctx.t === exIter
      load(s.y)
      const klBefore = evaluate(exploring ? ex : 1, true)
      let norm = 0
      const gains = s.gains.map((r, i) =>
        r.map((gv, a) => {
          const g = fresh ? 1 : gv
          const u = fresh ? 0 : (s.update[i]?.[a] ?? 0)
          return Math.max(u * (grad[i * k + a] ?? 0) < 0 ? g + 0.2 : g * 0.8, 0.01)
        }),
      )
      const update = s.update.map((r, i) =>
        r.map((u, a) => {
          const g = (gains[i]?.[a] ?? 1) * (grad[i * k + a] ?? 0)
          norm += g * g
          return momentum * (fresh ? 0 : u) - lr * g
        }),
      )
      const next = s.y.map((r, i) => r.map((v, a) => v + (update[i]?.[a] ?? 0)))
      let kl = klBefore
      if (exploring) {
        // While exploring the gradient is of the exaggerated P; the divergence reported is of P at the new positions.
        load(next)
        kl = evaluate(1, false)
      }
      return { y: next, update, gains, kl, gradNorm: Math.sqrt(norm) }
    },
    done: (s, t) => (minGrad > 0 && t % 50 === 0 && s.gradNorm <= minGrad ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { kl: s.kl, gradNorm: s.gradNorm === Number.MAX_VALUE ? -1 : s.gradNorm },
      tensors: { embedding: { role: 'state', value: s.y.flat(), shape: [n, k] } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as TsneState,
  }
}

/** Hyperparameters of {@link tsne}. */
export interface TsneHyper {
  readonly nComponents: number
  readonly perplexity: number
  readonly earlyExaggeration: number
  /** The step size (`null`: `'auto'`, max(n / exaggeration / 4, 50) as sklearn). */
  readonly learningRate: number | null
  readonly maxIter: number
  /** How many steps run with the exaggerated P. */
  readonly exaggerationIter: number
  /** The PCA components scaled to standard deviation 1e-4 (default), or small Gaussian noise. */
  readonly init: 'pca' | 'random'
  /** The exact gradient over all pairs, or Barnes-Hut. */
  readonly method: 'exact' | 'barnes-hut'
  readonly theta: number
  readonly metric: DistanceName | DistanceAlias
}

/**
 * Perplexity-calibrated affinities and the initial embedding of a t-SNE run: the dense or sparse joint probabilities,
 * the calibrated bandwidths and the start (PCA scaled to standard deviation 1e-4, or small Gaussian noise).
 *
 * @param x - The rows to embed.
 * @param h - The hyperparameters.
 * @param key - The key for the random start.
 * @returns The affinities, bandwidths, start and the learning rate.
 */
export function tsneSetup(
  x: TensorLike,
  h: TsneHyper,
  key: Parameters<typeof normal>[0],
): {
  readonly p: readonly (readonly number[])[] | SparseRows
  readonly sigmas: number[]
  readonly start: number[][]
  readonly learningRate: number
} {
  const m = matrixView(x, 'tsne', { minSamples: 4 })
  if (!(h.perplexity < m.n)) throw new MdmlError('value', `tsne: perplexity must be below n = ${m.n}`)
  let start: number[][]
  if (h.init === 'random')
    start = (toArray(normal(key, [m.n, h.nComponents])) as number[][]).map((r) => r.map((v) => v * 1e-4))
  else start = pcaStart(x, h.nComponents, 1e-4)
  const affinities =
    h.method === 'barnes-hut'
      ? sparseJointProbabilities(x, h.perplexity, h.metric)
      : jointProbabilities(x, h.perplexity, h.metric)
  return {
    p: affinities.p,
    sigmas: affinities.sigmas,
    start,
    learningRate: h.learningRate ?? Math.max(m.n / h.earlyExaggeration / 4, 50),
  }
}

/**
 * t-SNE (sklearn.manifold.TSNE) on the iterative base: perplexity-calibrated joint probabilities, then one step of
 * gradient descent on the KL divergence to a Student-t similarity per iteration, exact or Barnes-Hut. The embedding is
 * on the tape at every step; new rows are placed by their nearest training rows (t-SNE has no out-of-sample map).
 *
 * @index estimator t-SNE embedding, one gradient step at a time.
 * @example
 * ```ts
 * import { tsne } from 'mdml-engine/manifold'
 * const x = [[0, 0], [0.1, 0], [0, 0.1], [5, 5], [5.1, 5], [5, 5.1], [0.1, 0.1], [5.1, 5.1]]
 * tsne.with({ perplexity: 2, maxIter: 50, exaggerationIter: 10 }).fit({ x }).params.embedding.length // => 8
 * ```
 */
export const tsne: IterativeEstimator<TsneHyper, TsneState, EmbeddingParams, EmbeddingVerbs> = iterative<
  TsneHyper,
  TsneState,
  EmbeddingParams,
  EmbeddingVerbs
>({
  name: 'manifold/tsne',
  kind: 'transformer',
  defaults: {
    nComponents: 2,
    perplexity: 30,
    earlyExaggeration: 12,
    learningRate: null,
    maxIter: 1000,
    exaggerationIter: 250,
    init: 'pca',
    method: 'exact',
    theta: 0.5,
    metric: 'euclidean',
  },
  hyperSpace: (h) =>
    dict({
      perplexity: real(2, 100, { default: h.perplexity, label: 'perplexity' }),
      maxIter: int(50, 2000, { default: h.maxIter, label: 'iterations' }),
      earlyExaggeration: real(1, 50, { default: h.earlyExaggeration, label: 'early exaggeration' }),
      init: choice(['pca', 'random'] as const, { default: h.init, label: 'initialisation' }),
      method: choice(['exact', 'barnes-hut'] as const, { default: h.method }),
    }),
  algorithm: (data, h, key) => {
    const { p, start, learningRate } = tsneSetup(data.x, h, key)
    return tsneAlgorithm(p, start, {
      learningRate,
      earlyExaggeration: h.earlyExaggeration,
      exaggerationIter: h.exaggerationIter,
      theta: h.theta,
      dof: Math.max(h.nComponents - 1, 1),
    })
  },
  maxSteps: (h) => h.maxIter,
  params: (s, data) => ({ embedding: s.y, data: trainingRows(data.x, 'tsne'), objective: s.kl }),
  verbs: (p) => embeddingVerbs(p, 'tsne'),
})
