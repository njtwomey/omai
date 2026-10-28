/**
 * Locally linear embedding (Roweis and Saul 2000; scikit-learn's standard method): each row is rebuilt as a weighted
 * combination of its neighbours, and the embedding is the low-dimensional configuration that the same weights rebuild
 * best, found from the bottom eigenvectors of `(I - W)^T (I - W)`.
 */

import { defineEstimator, type Estimator, type Fitted, matrixView } from '../learn/index.ts'
import { linalg } from '../numerics/index.ts'
import { dict, int, real } from '../space/index.ts'
import { toArray, toFlat } from '../tensor/index.ts'
import { type EmbeddingParams, type EmbeddingVerbs, embeddingVerbs, trainingRows } from './_support.ts'
import { neighbourEdges } from './isomap.ts'

/**
 * The reconstruction weights of standard LLE (sklearn's `barycenter_weights`): for each row, the weights summing to one
 * that best rebuild it from its neighbours, through a regularised local Gram system.
 *
 * @param rows - Points `[n][d]`.
 * @param neighbours - The neighbour indices of each row.
 * @param reg - The regularisation relative to the trace.
 * @returns The weights per row, aligned with its neighbours.
 * @example
 * ```ts
 * import { barycentreWeights } from 'mdml-engine/manifold'
 * barycentreWeights([[0], [1], [2]], [[1, 2], [0, 2], [0, 1]], 1e-3)[1] // => [0.5, 0.5]
 * ```
 */
export function barycentreWeights(
  rows: readonly (readonly number[])[],
  neighbours: readonly (readonly number[])[],
  reg: number,
): number[][] {
  return rows.map((xi, i) => {
    const nb = neighbours[i] ?? []
    const z = nb.map((j) => (rows[j] ?? []).map((v, q) => v - (xi[q] ?? 0)))
    const c = z.map((a) => z.map((b) => a.reduce((s, v, q) => s + v * (b[q] ?? 0), 0)))
    const trace = c.reduce((s, r, a) => s + (r[a] ?? 0), 0)
    const r = trace > 0 ? reg * trace : reg
    const w = toFlat(
      linalg.solve(
        c.map((row, a) => row.map((v, b) => (a === b ? v + r : v))),
        nb.map(() => 1),
      ),
    )
    const total = w.reduce((s, v) => s + v, 0)
    return w.map((v) => v / total)
  })
}

/** Hyperparameters of {@link locallyLinearEmbedding}. */
export interface LleHyper {
  readonly nNeighbors: number
  readonly nComponents: number
  /** Regularisation of the local Gram matrices, relative to their trace. */
  readonly reg: number
}

/**
 * Locally linear embedding (sklearn.manifold.LocallyLinearEmbedding, standard method): each row's barycentric weights
 * over its neighbours, then the bottom eigenvectors of (I - W)^T (I - W) after the constant one. The objective is the
 * reconstruction error. TODO: the modified, Hessian and LTSA variants.
 *
 * @param hyper - Neighbours, dimension and regularisation.
 * @returns The estimator.
 * @example
 * ```ts
 * import { locallyLinearEmbedding } from 'mdml-engine/manifold'
 * const x = Array.from({ length: 12 }, (_, i) => [Math.cos(i / 2), Math.sin(i / 2), i / 6])
 * locallyLinearEmbedding({ nNeighbors: 4 }).fit({ x }).params.embedding.length // => 12
 * ```
 */
export function locallyLinearEmbedding(
  hyper: Partial<LleHyper> = {},
): Estimator<LleHyper, Fitted<EmbeddingParams> & EmbeddingVerbs> {
  return defineEstimator<LleHyper, EmbeddingParams, EmbeddingVerbs>(
    {
      name: 'manifold/lle',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nNeighbors: int(2, 50, { default: h.nNeighbors, label: 'neighbours' }),
          nComponents: int(1, 3, { default: h.nComponents, label: 'dimensions' }),
          reg: real(1e-6, 1, { log: true, default: h.reg, label: 'regularisation' }),
        }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'locallyLinearEmbedding', { minSamples: 3 })
        const k = h.nComponents
        const rows = trainingRows(data.x, 'locallyLinearEmbedding')
        const { indices } = neighbourEdges(data.x, h.nNeighbors)
        const w = barycentreWeights(rows, indices, h.reg)
        // M = (I - W)^T (I - W) with W sparse by rows.
        const iw = Array.from({ length: m.n }, (_, i) => {
          const row = new Array<number>(m.n).fill(0)
          row[i] = 1
          for (const [a, j] of (indices[i] ?? []).entries()) row[j] = (row[j] ?? 0) - (w[i]?.[a] ?? 0)
          return row
        })
        const mm = Array.from({ length: m.n }, (_, a) =>
          Array.from({ length: m.n }, (_, b) => iw.reduce((s, r) => s + (r[a] ?? 0) * (r[b] ?? 0), 0)),
        )
        const { eigenvalues, eigenvectors } = linalg.eigh(mm)
        const values = toFlat(eigenvalues)
        const vectors = toArray(eigenvectors) as number[][]
        return {
          embedding: vectors.map((r) => Array.from({ length: k }, (_, c) => r[c + 1] ?? 0)),
          data: rows,
          objective: values.slice(1, k + 1).reduce((a, v) => a + v, 0),
        }
      },
      methods: (p) => embeddingVerbs(p, 'lle'),
    },
    { nNeighbors: 5, nComponents: 2, reg: 1e-3, ...hyper },
  )
}
