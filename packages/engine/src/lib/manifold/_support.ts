/**
 * What the embeddings of this folder share and nobody else needs: the parameters and verbs of a fitted embedding, whose
 * `transform` returns the training rows' embedding as fitted and places other rows at the distance-weighted mean of the
 * embeddings of their nearest training rows, the PCA start of the layouts, and the locally scaled neighbour lists that
 * PaCMAP and TriMap both build their pairs and triplets from.
 */

import { pca } from '../decomposition/index.ts'
import { distanceMatrix } from '../distances/index.ts'
import { matrixView } from '../learn/index.ts'
import { foldIn, type Key, randint } from '../random/index.ts'
import { fromStorage, type Tensor, type TensorLike, toArray, toFlat } from '../tensor/index.ts'

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

/**
 * The PCA start of an embedding: the first `nComponents` principal scores, scaled so that the first column has standard
 * deviation `scale` (t-SNE uses 1e-4, PaCMAP and TriMap 1e-2).
 *
 * @param x - The rows `[n, d]`.
 * @param nComponents - The embedding dimension.
 * @param scale - The standard deviation of the first column.
 * @returns The start `[n, nComponents]`.
 */
export function pcaStart(x: TensorLike, nComponents: number, scale: number): number[][] {
  const m = matrixView(x, 'pcaStart')
  const z = toArray(
    pca
      .with({ nComponents: Math.min(nComponents, m.d) })
      .fit({ x })
      .transform(x),
  ) as number[][]
  const col = z.map((r) => r[0] ?? 0)
  const mu = col.reduce((a, v) => a + v, 0) / col.length
  const sd = Math.sqrt(col.reduce((a, v) => a + (v - mu) ** 2, 0) / col.length) || 1
  return z.map((r) => r.map((v) => (v / sd) * scale))
}

/** The neighbour lists of {@link scaledNeighbours}. */
export interface ScaledNeighbours {
  /** The `extra` nearest other rows of each row, nearest first. */
  readonly nbrs: number[][]
  /** The Euclidean distance to each of them. */
  readonly dist: number[][]
  /** Each row's scale: the mean distance to its 4th to 6th nearest other rows (at least 1e-10). */
  readonly sigma: number[]
  /** The squared distance to each neighbour over the product of the two rows' scales: d^2 / (sigma_i sigma_j). */
  readonly scaled: number[][]
  /** The full Euclidean distance matrix, flat `[n * n]`. */
  readonly all: Float64Array
}

/**
 * The locally scaled neighbourhoods of PaCMAP and TriMap: for each row its `extra` nearest other rows, the scale
 * sigma_i (the mean distance to the 4th to 6th nearest, as the `pacmap` and `trimap` packages) and the scaled squared
 * distance d_ij^2 / (sigma_i sigma_j), which ranks the neighbours so that a dense and a sparse region are on one scale.
 *
 * @param x - The rows `[n, d]`.
 * @param extra - How many neighbours to keep per row (at most n - 1).
 * @returns The neighbours, distances, scales and scaled distances.
 */
export function scaledNeighbours(x: TensorLike, extra: number): ScaledNeighbours {
  const { n, data } = distanceMatrix(x)
  const nbrs: number[][] = []
  const dist: number[][] = []
  const sigma: number[] = []
  for (let i = 0; i < n; i++) {
    const order = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => (data[i * n + a] ?? 0) - (data[i * n + b] ?? 0) || a - b)
    const top = order.slice(0, extra)
    nbrs.push(top)
    dist.push(top.map((j) => data[i * n + j] ?? 0))
    const ring = top.slice(3, 6).map((j) => data[i * n + j] ?? 0)
    const use = ring.length > 0 ? ring : top.map((j) => data[i * n + j] ?? 0)
    sigma.push(Math.max(use.reduce((a, v) => a + v, 0) / Math.max(1, use.length), 1e-10))
  }
  const scaled = nbrs.map((row, i) => row.map((j, a) => (dist[i]?.[a] ?? 0) ** 2 / ((sigma[i] ?? 1) * (sigma[j] ?? 1))))
  return { nbrs, dist, sigma, scaled, all: data }
}

/**
 * An endless stream of random row indices in `[0, n)` from a key, drawn block by block so that the stream is a pure
 * function of the key; callers take as many as they need and reject the ones they cannot use.
 */
export function* drawIndices(k: Key, n: number): Generator<number, never, undefined> {
  for (let block = 0; ; block++) yield* Array.from(toFlat(randint(foldIn(k, block), [64], 0, n)))
}
