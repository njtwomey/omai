/**
 * What the embeddings of this folder share and nobody else needs: the parameters and verbs of a fitted embedding, whose
 * `transform` returns the training rows' embedding as fitted and places other rows at the distance-weighted mean of the
 * embeddings of their nearest training rows, and the observation of an embedding per step.
 */

import { matrixView } from '../learn/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'

/** Rows of an embedding. */
export type Embedding = number[][]

/** The fitted parameters of an embedding: the embedding of the training rows, the rows themselves and the objective. */
export type EmbeddingParams = {
  readonly embedding: number[][]
  /** The training rows, for placing new ones. */
  readonly data: number[][]
  /** The quantity the method optimises (stress, divergence, cross-entropy, error). */
  readonly objective: number
}

/** The verbs of a fitted embedding. */
export interface EmbeddingVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly nFeaturesIn: number
  readonly featureNamesOut: (input?: readonly string[]) => string[]
}

/**
 * The verbs of an embedding of the training rows: an exact training row gets its fitted embedding; any other row the
 * mean of the embeddings of its `k` nearest training rows weighted by inverse distance (an out-of-sample extension that
 * none of these methods has in closed form).
 *
 * @param p - The parameters.
 * @param prefix - The output feature prefix.
 * @param k - Neighbours used for new rows (default 5).
 * @returns The verbs.
 */
export function embeddingVerbs(p: EmbeddingParams, prefix: string, k = 5): EmbeddingVerbs {
  const d = p.data[0]?.length ?? 0
  const e = p.embedding[0]?.length ?? 0
  return {
    transform: (x) => {
      const m = matrixView(x, 'transform', { nFeatures: d })
      const out = new Float64Array(m.n * e)
      for (let i = 0; i < m.n; i++) {
        const near: { j: number; dist: number }[] = []
        for (const [j, row] of p.data.entries()) {
          let s = 0
          for (let q = 0; q < d; q++) s += ((m.data[i * d + q] ?? 0) - (row[q] ?? 0)) ** 2
          near.push({ j, dist: Math.sqrt(s) })
        }
        near.sort((a, b) => a.dist - b.dist || a.j - b.j)
        const top = near.slice(0, Math.min(k, near.length))
        if ((top[0]?.dist ?? 1) === 0) {
          for (let c = 0; c < e; c++) out[i * e + c] = p.embedding[top[0]?.j ?? 0]?.[c] ?? 0
          continue
        }
        const w = top.map((t) => 1 / t.dist)
        const total = w.reduce((a, v) => a + v, 0)
        for (let c = 0; c < e; c++)
          out[i * e + c] = top.reduce((a, t, r) => a + (w[r] ?? 0) * (p.embedding[t.j]?.[c] ?? 0), 0) / total
      }
      return fromStorage(out, [m.n, e], 'float64')
    },
    nFeaturesIn: d,
    featureNamesOut: () => Array.from({ length: e }, (_, c) => `${prefix}${c}`),
  }
}

/** The training rows of a matrix view as plain rows. */
export function trainingRows(x: TensorLike, who: string, minSamples = 1): number[][] {
  const m = matrixView(x, who, { minSamples })
  return Array.from({ length: m.n }, (_, i) => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d)))
}
