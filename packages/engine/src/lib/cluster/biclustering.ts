/**
 * The estimators of spectral biclustering (scikit-learn's `SpectralCoclustering` and `SpectralBiclustering`) on the
 * iterative base, over the algorithms of `bicluster-algorithms.ts`: biclusters along the diagonal of a non-negative
 * matrix, or a checkerboard of them. The fitted value holds the labels and the bicluster indicators (`rows_` and
 * `columns_`) with scikit-learn's `get_indices`, `get_shape` and `get_submatrix`; a session's `bicluster/labels` and
 * `bicluster/order` paths give the partition at any step, and the score against a planted truth is `consensusScore`.
 */

import { OmaiError, toRows } from '../base/index.ts'
import { type IterativeEstimator, iterative, matrixView } from '../learn/index.ts'
import { choice, dict, int } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { type Contract, defineContract, derived } from '../trace/index.ts'
import {
  type BiclusteringNormalisation,
  type BiclusteringState,
  type BiclusterSvd,
  type Start,
  spectralBiclusteringAlgorithm,
  spectralCoclusteringAlgorithm,
} from './bicluster-algorithms.ts'
import type { Biclusters } from './bicluster-score.ts'
import type { LloydState } from './kmeans.ts'

const rowsOfMatrix = (x: TensorLike, who: string): number[][] => {
  const m = matrixView(x, who, { minSamples: 2 })
  return toRows(m.data, m.n, m.d)
}

/** The biclusters of a labelling, as indicator rows: bicluster `r * kc + c` is the rows labelled `r` and columns labelled `c`. */
function checkerboard(
  rowLabels: readonly number[],
  columnLabels: readonly number[],
  kr: number,
  kc: number,
): Biclusters {
  const pairs = Array.from({ length: kr * kc }, (_, b) => [Math.floor(b / kc), b % kc] as const)
  return {
    rows: pairs.map(([r]) => rowLabels.map((l) => l === r)),
    columns: pairs.map(([, c]) => columnLabels.map((l) => l === c)),
  }
}

/** The fitted parameters of the spectral biclusterers. */
export type BiclusterParams = {
  readonly rowLabels: number[]
  readonly columnLabels: number[]
  /** The bicluster indicators (scikit-learn's `rows_` and `columns_`). */
  readonly rows: boolean[][]
  readonly columns: boolean[][]
  readonly nRowClusters: number
  readonly nColumnClusters: number
}

/** The verbs of a fitted biclusterer (scikit-learn's `BiclusterMixin`). */
export interface BiclusterVerbs {
  /** The number of columns of the matrix the biclusters were found in. */
  readonly nFeaturesIn: number
  /** The row and column indices of bicluster `i`. */
  readonly getIndices: (i: number) => { readonly rows: number[]; readonly columns: number[] }
  /** The number of rows and of columns of bicluster `i`. */
  readonly getShape: (i: number) => [number, number]
  /** The submatrix of `x` that is bicluster `i`. */
  readonly getSubmatrix: (i: number, x: TensorLike) => Tensor
}

const biclusterVerbs = (p: BiclusterParams): BiclusterVerbs => {
  const indices = (i: number): { rows: number[]; columns: number[] } => {
    const rows = p.rows[i]
    const columns = p.columns[i]
    if (rows === undefined || columns === undefined)
      throw new OmaiError('value', `bicluster ${i} does not exist (${p.rows.length})`)
    return {
      rows: rows.flatMap((b, r) => (b ? [r] : [])),
      columns: columns.flatMap((b, c) => (b ? [c] : [])),
    }
  }
  return {
    nFeaturesIn: p.columns[0]?.length ?? 0,
    getIndices: indices,
    getShape: (i) => {
      const { rows, columns } = indices(i)
      return [rows.length, columns.length]
    },
    getSubmatrix: (i, x) => {
      const { rows, columns } = indices(i)
      const m = rowsOfMatrix(x, 'getSubmatrix')
      return fromStorage(
        Float64Array.from(rows.flatMap((r) => columns.map((c) => m[r]?.[c] ?? 0))),
        [rows.length, columns.length],
        'float64',
      )
    },
  }
}

/** The paths the biclusterers add to the iterative base's. */
export type BiclusterPath = 'bicluster/labels' | 'bicluster/order'

const orderBy = (labels: readonly number[]): number[] =>
  labels
    .map((l, i) => [l, i] as const)
    .sort((a, b) => a[0] - b[0] || a[1] - b[1])
    .map(([, i]) => i)

const labelPaths = <S>(of: (s: S) => { rows: number[]; columns: number[] }): Contract<S, BiclusterPath> =>
  defineContract({
    'bicluster/labels': derived<S>({
      description: 'the row and column labels at this step',
      compute: (s) => of(s),
    }),
    'bicluster/order': derived<S>({
      description: 'the rows and columns sorted by label, which brings the matrix to block form',
      compute: (s) => {
        const { rows, columns } = of(s)
        return { rows: orderBy(rows), columns: orderBy(columns) }
      },
    }),
  })

/** Hyperparameters of {@link spectralCoclustering}. */
export interface SpectralCoclusteringHyper {
  readonly nClusters: number
  readonly svdMethod: BiclusterSvd
  readonly nSvdVecs: number | null
  readonly nInit: number
  readonly init: Start
}

/**
 * Spectral co-clustering (sklearn.cluster.SpectralCoclustering) on the iterative base: biclusters along the diagonal of
 * a non-negative matrix, every row and every column in exactly one. The embedding is computed once and the k-means on
 * it plays one iteration per step (`bicluster/labels` and `bicluster/order` give the partition at any step). A
 * checkerboard is {@link spectralBiclustering}'s. It differs from scikit-learn in taking the exact SVD by default and in
 * not offering the mini-batch k-means.
 *
 * @index estimator Spectral co-clustering of a matrix into diagonal biclusters.
 * @example
 * ```ts
 * import { consensusScore, spectralCoclustering } from 'omai-engine/cluster'
 * const x = [[9, 8, 0, 1], [8, 9, 1, 0], [0, 1, 9, 8], [1, 0, 8, 9]]
 * const f = spectralCoclustering.with({ nClusters: 2 }).fit({ x })
 * consensusScore(f.params, { rows: [[1, 1, 0, 0], [0, 0, 1, 1]], columns: [[1, 1, 0, 0], [0, 0, 1, 1]] }) // => 1
 * ```
 */
export const spectralCoclustering: IterativeEstimator<
  SpectralCoclusteringHyper,
  LloydState,
  BiclusterParams,
  BiclusterVerbs,
  BiclusterPath
> = iterative<SpectralCoclusteringHyper, LloydState, BiclusterParams, BiclusterVerbs, BiclusterPath>({
  name: 'cluster/spectral-coclustering',
  kind: 'other',
  defaults: { nClusters: 3, svdMethod: 'exact', nSvdVecs: null, nInit: 10, init: 'k-means++' },
  hyperSpace: (h) =>
    dict({
      nClusters: int(2, 50, { default: h.nClusters, label: 'biclusters' }),
      svdMethod: choice(['exact', 'randomized'] as const, { default: h.svdMethod, label: 'SVD method' }),
      nInit: int(1, 50, { default: h.nInit, label: 'k-means starts' }),
    }),
  algorithm: (data, h, key) =>
    spectralCoclusteringAlgorithm(
      rowsOfMatrix(data.x, 'spectralCoclustering'),
      h.nClusters,
      { svdMethod: h.svdMethod, nSvdVecs: h.nSvdVecs, nInit: h.nInit, init: h.init },
      key,
    ),
  maxSteps: () => 300,
  contract: (data): Contract<LloydState, BiclusterPath> => {
    const n = matrixView(data.x, 'spectralCoclustering').n
    return labelPaths<LloydState>((s) => ({ rows: s.labels.slice(0, n), columns: s.labels.slice(n) }))
  },
  params: (s, data, h) => {
    const n = matrixView(data.x, 'spectralCoclustering').n
    const rowLabels = s.labels.slice(0, n)
    const columnLabels = s.labels.slice(n)
    const indicators = (labels: number[]): boolean[][] =>
      Array.from({ length: h.nClusters }, (_, c) => labels.map((l) => l === c))
    return {
      rowLabels,
      columnLabels,
      rows: indicators(rowLabels),
      columns: indicators(columnLabels),
      nRowClusters: h.nClusters,
      nColumnClusters: h.nClusters,
    }
  },
  verbs: (p) => biclusterVerbs(p),
})

/** Hyperparameters of {@link spectralBiclustering}. */
export interface SpectralBiclusteringHyper {
  /** Row and column cluster counts, or one number for both. */
  readonly nClusters: number | [number, number]
  readonly method: BiclusteringNormalisation
  readonly nComponents: number
  readonly nBest: number
  readonly svdMethod: BiclusterSvd
  readonly nSvdVecs: number | null
  readonly nInit: number
  readonly init: Start
}

/**
 * Spectral biclustering (sklearn.cluster.SpectralBiclustering) on the iterative base: a checkerboard of
 * `nClusters[0] x nClusters[1]` biclusters, every row in one row cluster and every column in one column cluster, found
 * as {@link spectralBiclusteringAlgorithm} (the rows' k-means plays first, then the columns'). The biclusters are
 * numbered row cluster major, as scikit-learn's `rows_` and `columns_`.
 *
 * @index estimator Spectral biclustering of a matrix into a checkerboard of biclusters.
 * @example
 * ```ts
 * import { spectralBiclustering } from 'omai-engine/cluster'
 * const x = [[1, 1, 5, 5], [1, 1, 5, 5], [8, 8, 2, 2], [8, 8, 2, 2]]
 * spectralBiclustering.with({ nClusters: [2, 2], nComponents: 2, nBest: 1 }).fit({ x }).params.rows.length // => 4
 * ```
 */
export const spectralBiclustering: IterativeEstimator<
  SpectralBiclusteringHyper,
  BiclusteringState,
  BiclusterParams,
  BiclusterVerbs,
  BiclusterPath
> = iterative<SpectralBiclusteringHyper, BiclusteringState, BiclusterParams, BiclusterVerbs, BiclusterPath>({
  name: 'cluster/spectral-biclustering',
  kind: 'other',
  defaults: {
    nClusters: 3,
    method: 'bistochastic',
    nComponents: 6,
    nBest: 3,
    svdMethod: 'exact',
    nSvdVecs: null,
    nInit: 10,
    init: 'k-means++',
  },
  hyperSpace: (h) =>
    dict({
      ...(typeof h.nClusters === 'number'
        ? { nClusters: int(1, 50, { default: h.nClusters, label: 'row and column clusters' }) }
        : {}),
      method: choice(['bistochastic', 'scale', 'log'] as const, { default: h.method, label: 'normalisation' }),
      nComponents: int(1, 50, { default: h.nComponents, label: 'singular vectors' }),
      nBest: int(1, 50, { default: h.nBest, label: 'vectors kept' }),
      nInit: int(1, 50, { default: h.nInit, label: 'k-means starts' }),
    }),
  algorithm: (data, h, key) =>
    spectralBiclusteringAlgorithm(
      rowsOfMatrix(data.x, 'spectralBiclustering'),
      h.nClusters,
      {
        method: h.method,
        nComponents: h.nComponents,
        nBest: h.nBest,
        svdMethod: h.svdMethod,
        nSvdVecs: h.nSvdVecs,
        nInit: h.nInit,
        init: h.init,
      },
      key,
    ),
  maxSteps: () => 600,
  contract: (): Contract<BiclusteringState, BiclusterPath> =>
    labelPaths<BiclusteringState>((s) => ({ rows: s.rows.labels, columns: s.columns.labels })),
  params: (s, _data, h) => {
    const [kr, kc] = typeof h.nClusters === 'number' ? [h.nClusters, h.nClusters] : h.nClusters
    const { rows, columns } = checkerboard(s.rows.labels, s.columns.labels, kr, kc)
    return {
      rowLabels: s.rows.labels,
      columnLabels: s.columns.labels,
      rows: rows as boolean[][],
      columns: columns as boolean[][],
      nRowClusters: kr,
      nColumnClusters: kc,
    }
  },
  verbs: (p) => biclusterVerbs(p),
})
