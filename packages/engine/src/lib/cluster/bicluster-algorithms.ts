/**
 * The algorithms of spectral biclustering of a data matrix, rows and columns together (scikit-learn's
 * `SpectralCoclustering` and `SpectralBiclustering`). Co-clustering (Dhillon 2001) finds blocks along the diagonal by
 * k-means on the scaled singular vectors of rows and columns jointly; biclustering (Kluger et al. 2003) finds a
 * checkerboard by choosing the singular vectors that are most piecewise constant and clustering the rows, then the
 * columns, in their span. The normalisation and the SVD are done once when the algorithm is made, and the k-means runs
 * as traced Lloyd steps (rows then columns for the checkerboard), so the partition forming can be played.
 */

import { OmaiError, toRows } from '../base/index.ts'
import { linalg } from '../numerics/index.ts'
import { foldIn, type Key } from '../random/index.ts'
import { fromStorage, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { bistochasticNormalize, logNormalize, scaleNormalize } from './bicluster-normalise.ts'
import { initialCentres, type LloydState, lloydAlgorithm, type Rows } from './kmeans.ts'

/** How the leading singular vectors are found: the exact SVD (scikit-learn's `arpack`), or the randomised one. */
export type BiclusterSvd = 'exact' | 'randomized'

/**
 * The singular vector pairs `[nDiscard, nSv)` of a matrix: left vectors as the columns of `u` `[n, m]`, right vectors
 * as the columns of `v` `[d, m]`.
 */
function singularVectors(
  a: Rows,
  nSv: number,
  nDiscard: number,
  method: BiclusterSvd,
  oversample: number | null,
  key: Key,
): { u: number[][]; v: number[][] } {
  const n = a.length
  const d = a[0]?.length ?? 0
  if (nSv > Math.min(n, d))
    throw new OmaiError('value', `biclustering: ${nSv} singular vectors are needed but the matrix is ${n} x ${d}`)
  const t = fromStorage(Float64Array.from(a.flat()), [n, d], 'float64')
  const r =
    method === 'exact'
      ? linalg.svd(t)
      : linalg.randomizedSvd(t, nSv, { key, ...(oversample === null ? {} : { oversample }) })
  const rank = method === 'exact' ? Math.min(n, d) : nSv
  const u = toRows(toFlat(r.u), n, rank)
  const vh = toRows(toFlat(r.vh), rank, d)
  const cols = Array.from({ length: nSv - nDiscard }, (_, c) => c + nDiscard)
  return {
    u: u.map((row) => cols.map((c) => row[c] ?? 0)),
    v: Array.from({ length: d }, (_, j) => cols.map((c) => vh[c]?.[j] ?? 0)),
  }
}

/** How the k-means centres start. */
export type Start = 'k-means++' | 'random'

/**
 * k-means with several starts, as scikit-learn's `n_init`: every start runs to convergence and the one with the lowest
 * inertia wins. Returns the winning start's initial centres (so a traced run can replay it) and its result.
 */
function bestKMeans(
  points: Rows,
  k: number,
  init: Start,
  nInit: number,
  key: Key,
): { init: number[][]; result: LloydState } {
  let best: { init: number[][]; result: LloydState } | undefined
  for (let r = 0; r < nInit; r++) {
    const start = initialCentres(points, k, init, foldIn(key, r))
    const result = runAlgorithm(lloydAlgorithm(points, start), { maxSteps: 300 }).state
    if (best === undefined || result.inertia < best.result.inertia) best = { init: start, result }
  }
  return best as { init: number[][]; result: LloydState }
}

/** Options of {@link spectralCoclusteringAlgorithm} and {@link spectralBiclusteringAlgorithm}. */
export interface SpectralBiclusterOptions {
  readonly svdMethod?: BiclusterSvd
  /** Oversampling of the randomised SVD (default 10). */
  readonly nSvdVecs?: number | null
  /** Starts of every k-means; the lowest inertia wins (default 10). */
  readonly nInit?: number
  /** How the k-means centres start (default k-means++). */
  readonly init?: Start
}

const resolved = (o: SpectralBiclusterOptions): Required<SpectralBiclusterOptions> => ({
  svdMethod: o.svdMethod ?? 'exact',
  nSvdVecs: o.nSvdVecs ?? null,
  nInit: o.nInit ?? 10,
  init: o.init ?? 'k-means++',
})

/**
 * Spectral co-clustering (Dhillon 2001; scikit-learn's `SpectralCoclustering`) as a traced {@link Algorithm}: scale
 * rows and columns, take the singular vectors from the second to the `1 + ceil(log2 k)`-th, stack the rows' vectors
 * (scaled by `1 / sqrt(row sum)`) over the columns' (scaled likewise), and run Lloyd's algorithm on those
 * `n + d` points, one iteration per step. The labels it observes are the rows' then the columns' (the first `n` are
 * the rows'); each cluster is a bicluster along the diagonal.
 *
 * @param x - The non-negative data matrix `[n, d]` (shifted to non-negative if it is not).
 * @param nClusters - The number of biclusters (at least 2, at most `n`).
 * @param options - SVD method, starts and initialisation.
 * @param key - Randomness for the starts (and the randomised SVD).
 * @returns The algorithm, a Lloyd run over the stacked embedding.
 * @throws OmaiError `value` for fewer than 2 clusters, more than `n`, or a zero row or column.
 * @example
 * ```ts
 * import { spectralCoclusteringAlgorithm } from 'omai-engine/cluster'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[9, 8, 0, 1], [8, 9, 1, 0], [0, 1, 9, 8], [1, 0, 8, 9]]
 * runAlgorithm(spectralCoclusteringAlgorithm(x, 2, {}, key(0)), { maxSteps: 50 }).state.labels.length // => 8
 * ```
 */
export function spectralCoclusteringAlgorithm(
  x: Rows,
  nClusters: number,
  options: SpectralBiclusterOptions,
  key: Key,
): Algorithm<LloydState> {
  const o = resolved(options)
  const n = x.length
  if (!(Number.isInteger(nClusters) && nClusters >= 2 && nClusters <= n))
    throw new OmaiError('value', `spectralCoclustering: nClusters must be an integer in 2..${n}, got ${nClusters}`)
  const { matrix, rowScale, columnScale } = scaleNormalize(x)
  const nSv = 1 + Math.ceil(Math.log2(nClusters))
  const { u, v } = singularVectors(matrix, nSv, 1, o.svdMethod, o.nSvdVecs, foldIn(key, 0))
  const z = [
    ...u.map((row, i) => row.map((c) => (rowScale[i] ?? 0) * c)),
    ...v.map((row, j) => row.map((c) => (columnScale[j] ?? 0) * c)),
  ]
  const { init } = bestKMeans(z, nClusters, o.init, o.nInit, foldIn(key, 1))
  return { ...lloydAlgorithm(z, init), name: 'spectral-coclustering' }
}

/** Which normalisation precedes the SVD of {@link spectralBiclustering}. */
export type BiclusteringNormalisation = 'bistochastic' | 'scale' | 'log'

/** Options of {@link spectralBiclusteringAlgorithm}. */
export interface SpectralBiclusteringOptions extends SpectralBiclusterOptions {
  readonly method?: BiclusteringNormalisation
  /** Singular vectors considered (default 6). */
  readonly nComponents?: number
  /** Of those, the most piecewise-constant ones kept (default 3). */
  readonly nBest?: number
}

/** The state of {@link spectralBiclusteringAlgorithm}: the row stage, then the column stage. */
export interface BiclusteringState {
  /** Which clustering is running: 0 the rows, 1 the columns. */
  readonly stage: 0 | 1
  readonly rows: LloydState
  readonly columns: LloydState
}

/** The vectors (rows of `vectors`) most nearly piecewise constant, by 1-D k-means, the nearest `nBest` of them. */
function bestPiecewise(
  vectors: Rows,
  nBest: number,
  k: number,
  o: Required<SpectralBiclusterOptions>,
  key: Key,
): number[][] {
  const scored = vectors.map((vec, i) => {
    const { result } = bestKMeans(
      vec.map((c) => [c]),
      k,
      o.init,
      o.nInit,
      foldIn(key, i),
    )
    const piecewise = result.labels.map((l) => result.centres[l]?.[0] ?? 0)
    return { i, distance: Math.sqrt(vec.reduce((s, c, q) => s + (c - (piecewise[q] ?? 0)) ** 2, 0)) }
  })
  return scored
    .sort((a, b) => a.distance - b.distance || a.i - b.i)
    .slice(0, nBest)
    .map(({ i }) => [...(vectors[i] ?? [])])
}

/** `data` (`[m, p]`) times the columns of the vectors (`[nBest][p]`): `[m, nBest]`. */
const project = (data: Rows, vectors: Rows): number[][] =>
  data.map((row) => vectors.map((vec) => vec.reduce((s, c, q) => s + c * (row[q] ?? 0), 0)))

const transposed = (a: Rows): number[][] => (a[0] ?? []).map((_, j) => a.map((r) => r[j] ?? 0))

/**
 * Spectral biclustering (Kluger et al. 2003; scikit-learn's `SpectralBiclustering`) as a traced {@link Algorithm}: the
 * matrix is normalised (bistochastic, scaled or log), its leading singular vectors found, the `nBest` most piecewise
 * constant of the rows' and of the columns' chosen (by 1-D k-means), and then the rows are clustered in the span of
 * the columns' chosen vectors and the columns in the span of the rows', each by Lloyd's algorithm one iteration per
 * step (the rows first). Rows in `r` groups and columns in `c` groups give the `r c` biclusters of a checkerboard.
 *
 * @param x - The data matrix `[n, d]`.
 * @param nClusters - Row and column cluster counts, or one number for both.
 * @param options - Normalisation, vectors considered and kept, SVD method, starts.
 * @param key - Randomness for the starts (and the randomised SVD).
 * @returns The algorithm.
 * @throws OmaiError `value` for a count outside `1..n`, `nBest` above `nComponents`, or too few singular vectors.
 * @example
 * ```ts
 * import { spectralBiclusteringAlgorithm } from 'omai-engine/cluster'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const x = [[1, 1, 5, 5], [1, 1, 5, 5], [8, 8, 2, 2], [8, 8, 2, 2]]
 * const out = runAlgorithm(spectralBiclusteringAlgorithm(x, [2, 2], { nComponents: 2, nBest: 1 }, key(0)), { maxSteps: 50 })
 * out.state.rows.labels.length // => 4
 * ```
 */
export function spectralBiclusteringAlgorithm(
  x: Rows,
  nClusters: number | readonly [number, number],
  options: SpectralBiclusteringOptions,
  key: Key,
): Algorithm<BiclusteringState> {
  const o = resolved(options)
  const method = options.method ?? 'bistochastic'
  const nComponents = options.nComponents ?? 6
  const nBest = options.nBest ?? 3
  const n = x.length
  const d = x[0]?.length ?? 0
  const [kr, kc] = typeof nClusters === 'number' ? [nClusters, nClusters] : nClusters
  for (const [k, size, what] of [
    [kr, n, 'row'],
    [kc, d, 'column'],
  ] as const)
    if (!(Number.isInteger(k) && k >= 1 && k <= size))
      throw new OmaiError(
        'value',
        `spectralBiclustering: the ${what} cluster count must be an integer in 1..${size}, got ${k}`,
      )
  if (nBest > nComponents)
    throw new OmaiError('value', `spectralBiclustering: nBest ${nBest} exceeds nComponents ${nComponents}`)
  const normalised =
    method === 'bistochastic'
      ? bistochasticNormalize(x)
      : method === 'scale'
        ? scaleNormalize(x).matrix
        : logNormalize(x)
  const discard = method === 'log' ? 0 : 1
  const { u, v } = singularVectors(normalised, nComponents + discard, discard, o.svdMethod, o.nSvdVecs, foldIn(key, 0))
  const bestRows = bestPiecewise(transposed(u), nBest, kr, o, foldIn(key, 1))
  const bestCols = bestPiecewise(transposed(v), nBest, kc, o, foldIn(key, 2))
  const data = x.map((r) => [...r])
  const rowPoints = project(data, bestCols)
  const columnPoints = project(transposed(data), bestRows)
  const rowStart = bestKMeans(rowPoints, kr, o.init, o.nInit, foldIn(key, 3)).init
  const columnStart = bestKMeans(columnPoints, kc, o.init, o.nInit, foldIn(key, 4)).init
  const rowAlg = lloydAlgorithm(rowPoints, rowStart)
  const columnAlg = lloydAlgorithm(columnPoints, columnStart)
  return {
    name: 'spectral-biclustering',
    config: { method, nComponents, nBest, kr, kc },
    schema: channels({
      metrics: {
        rowInertia: {
          description: 'squared distance of the rows to their centres, in the span of the column vectors',
          direction: 'lower',
        },
        columnInertia: {
          description: 'squared distance of the columns to their centres, in the span of the row vectors',
          direction: 'lower',
        },
        stage: { description: '0 while the rows are being clustered, 1 while the columns are' },
      },
      tensors: {
        rowLabels: { role: 'state', description: 'the row cluster of each row', shape: ['n'] },
        columnLabels: { role: 'state', description: 'the column cluster of each column', shape: ['d'] },
      },
    }),
    init: () => ({ stage: 0, rows: rowAlg.init(key), columns: columnAlg.init(key) }) as BiclusteringState,
    step: (s, ctx) => {
      if (s.stage === 0) {
        const rows = rowAlg.step(s.rows, ctx)
        return { ...s, rows, stage: rowAlg.done?.(rows, ctx.t) === undefined ? 0 : 1 }
      }
      return { ...s, columns: columnAlg.step(s.columns, ctx) }
    },
    done: (s, t) => (s.stage === 1 ? columnAlg.done?.(s.columns, t) : undefined),
    observe: (s) => ({
      metrics: { rowInertia: s.rows.inertia, columnInertia: s.columns.inertia, stage: s.stage },
      tensors: {
        rowLabels: { role: 'state', value: s.rows.labels, shape: [n] },
        columnLabels: { role: 'state', value: s.columns.labels, shape: [d] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as BiclusteringState,
  }
}
