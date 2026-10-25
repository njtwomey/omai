/**
 * Locally linear embedding (Roweis and Saul, 2000; scikit-learn's standard method): each row is rebuilt as a weighted
 * combination of its neighbours, and the embedding is the low-dimensional configuration that the same weights rebuild
 * best, found from the bottom eigenvectors of `(I - W)^T (I - W)`.
 */

import { embeddingSteps } from '../../lib/exposition/index.ts'
import { defineExplainedEstimator, type Estimator, EstimatorClass, matrixView } from '../../lib/learn/index.ts'
import { linalg } from '../../lib/numerics/index.ts'
import { dict, int, real } from '../../lib/space/index.ts'
import { toArray, toFlat } from '../../lib/tensor/index.ts'
import { neighbourEdges } from './isomap.ts'
import type { EmbeddingParams, FittedEmbedding } from './mds.ts'

/** Hyperparameters of locally linear embedding. */
export type LleHyper = {
  readonly nNeighbors?: number
  readonly nComponents?: number
  /** Regularisation of the local Gram matrices, relative to their trace (default 1e-3). */
  readonly reg?: number
}

/**
 * The reconstruction weights of standard LLE (sklearn's `barycenter_weights`): for each row, the weights summing to
 * one that best rebuild it from its neighbours, through a regularised local Gram system.
 *
 * @param rows - Points `[n][d]`.
 * @param neighbours - The neighbour indices of each row.
 * @param reg - The regularisation relative to the trace.
 * @returns The weights per row, aligned with its neighbours.
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

/**
 * Locally linear embedding (Roweis and Saul 2000; sklearn.manifold.LocallyLinearEmbedding, standard method): each
 * row's barycentric weights over its neighbours, then the bottom eigenvectors of (I - W)^T (I - W) after the
 * constant one. Its exposition is an `embeddingSteps` record with the neighbour graph weighted by the
 * reconstruction weights, and the reconstruction error as objective. TODO: modified, Hessian and LTSA variants.
 *
 * @param hyper - Neighbours, dimension and regularisation.
 * @returns The estimator.
 */
export function locallyLinearEmbedding(hyper: LleHyper = {}): Estimator<LleHyper, FittedEmbedding> {
  return defineExplainedEstimator<LleHyper, EmbeddingParams, { readonly nFeaturesIn: number }>(
    {
      name: 'locally-linear-embedding',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          nNeighbors: int(2, 50, { default: h.nNeighbors ?? 5, label: 'neighbours' }),
          nComponents: int(1, 3, { default: h.nComponents ?? 2, label: 'dimensions' }),
          reg: real(1e-6, 1, {
            label: 'regularisation',
            description: 'The regularisation of the local Gram matrices.',
            log: true,
            default: h.reg ?? 1e-3,
          }),
        }),
      fit: (data, h) => {
        const m = matrixView(data.x, 'locallyLinearEmbedding', { minSamples: 3 })
        const k = h.nComponents ?? 2
        const rows = Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
        const { indices } = neighbourEdges(data.x, h.nNeighbors ?? 5)
        const w = barycentreWeights(rows, indices, h.reg ?? 1e-3)
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
        const embedding = vectors.map((r) => Array.from({ length: k }, (_, c) => r[c + 1] ?? 0))
        const error = values.slice(1, k + 1).reduce((a, v) => a + v, 0)
        const graph = indices.flatMap((nb, i) => nb.map((j, a): [number, number, number] => [i, j, w[i]?.[a] ?? 0]))
        return {
          params: { embedding, objective: error, nIter: 0, nFeaturesIn: m.d },
          exposition: [
            embeddingSteps({ method: 'lle', positions: [embedding], iterations: [0], objective: [error], graph }),
          ],
        }
      },
      methods: (p) => ({ nFeaturesIn: p.nFeaturesIn }),
    },
    hyper,
  )
}

/** Locally linear embedding as a class. */
export class LocallyLinearEmbedding extends EstimatorClass<LleHyper, FittedEmbedding> {
  /**
   * Make an unfitted model.
   *
   * @param hyper - The hyperparameters of {@link locallyLinearEmbedding}.
   */
  constructor(hyper: LleHyper = {}) {
    super(locallyLinearEmbedding(hyper))
  }
}
