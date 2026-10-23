import { WhiteboxError } from '../../lib/base/index.ts'
import { qr, svd } from '../../lib/numerics/linalg/index.ts'
import { foldIn, type Key, multinomial, normal, permutation, uniform } from '../../lib/random/index.ts'
import { fromStorage, matmul, mul, type Tensor, tensor, transpose } from '../../lib/tensor/index.ts'
import { matrix } from './_util.ts'

/**
 * The singular value profile of sklearn's `make_low_rank_matrix`: a bell-shaped head of width `effectiveRank` and an
 * exponential tail, `(1 - t) exp(-(i / r)^2) + t exp(-0.1 i / r)`.
 *
 * @param n - The number of singular values.
 * @param effectiveRank - The width of the head.
 * @param tailStrength - The relative weight of the tail in [0, 1].
 * @returns The singular values, descending.
 * @example
 * ```ts
 * import { lowRankProfile } from 'whitebox-engine/methods/datasets'
 * lowRankProfile(2, 1, 0) // => [1, 0.36787944117144233]
 * ```
 */
export function lowRankProfile(n: number, effectiveRank: number, tailStrength: number): number[] {
  return Array.from(
    { length: n },
    (_, i) =>
      (1 - tailStrength) * Math.exp(-((i / effectiveRank) ** 2)) + tailStrength * Math.exp((-0.1 * i) / effectiveRank),
  )
}

/** Options of {@link makeLowRankMatrix}. */
export interface LowRankOptions {
  readonly nSamples?: number
  readonly nFeatures?: number
  /** The approximate number of singular vectors that explain most of the data (default 10). */
  readonly effectiveRank?: number
  /** The relative importance of the fat noisy tail of the singular profile (default 0.5). */
  readonly tailStrength?: number
}

/**
 * A mostly low-rank matrix with bell-shaped singular values (sklearn.datasets.make_low_rank_matrix): `U diag(s) V^T`
 * with random orthonormal `U` and `V` (from QR factorisations of Gaussian matrices) and `s` from
 * {@link lowRankProfile}.
 *
 * @param key - The random key.
 * @param options - Shape, effective rank and tail strength.
 * @returns The matrix `[nSamples, nFeatures]` and its singular values.
 * @example
 * ```ts
 * import { makeLowRankMatrix } from 'whitebox-engine/methods/datasets'
 * import { key } from 'whitebox-engine/random'
 * makeLowRankMatrix(key(0), { nSamples: 6, nFeatures: 4, effectiveRank: 2 }).x.shape // => [6, 4]
 * ```
 */
export function makeLowRankMatrix(
  key: Key,
  options: LowRankOptions = {},
): { readonly x: Tensor; readonly singularValues: readonly number[] } {
  const n = options.nSamples ?? 100
  const d = options.nFeatures ?? 100
  const m = Math.min(n, d)
  const u = qr(normal(foldIn(key, 0), [n, m])).q
  const v = qr(normal(foldIn(key, 1), [d, m])).q
  const s = lowRankProfile(m, options.effectiveRank ?? 10, options.tailStrength ?? 0.5)
  const x = matmul(mul(u, tensor(s)), transpose(v)) as Tensor
  return { x, singularValues: s }
}

/** Options of {@link makeSparseCodedSignal}. */
export interface SparseCodedOptions {
  readonly nSamples: number
  readonly nComponents: number
  readonly nFeatures: number
  /** Non-zero coefficients per sample. */
  readonly nNonzeroCoefs: number
}

/**
 * A signal as a sparse combination of dictionary atoms (sklearn.datasets.make_sparse_coded_signal): unit-norm
 * Gaussian atoms, codes with `nNonzeroCoefs` standard normal entries per sample at random positions, and
 * `data = code dictionary`.
 *
 * @param key - The random key.
 * @param options - Sizes and sparsity.
 * @returns `data` `[nSamples, nFeatures]`, `dictionary` `[nComponents, nFeatures]` and `code` `[nSamples, nComponents]`.
 * @throws WhiteboxError with code `value` when the sparsity exceeds the number of components.
 * @example
 * ```ts
 * import { makeSparseCodedSignal } from 'whitebox-engine/methods/datasets'
 * import { key } from 'whitebox-engine/random'
 * const s = makeSparseCodedSignal(key(0), { nSamples: 3, nComponents: 5, nFeatures: 4, nNonzeroCoefs: 2 })
 * Array.from(s.code.data).filter((v) => v !== 0).length // => 6
 * ```
 */
export function makeSparseCodedSignal(
  key: Key,
  options: SparseCodedOptions,
): { readonly data: Tensor; readonly dictionary: Tensor; readonly code: Tensor } {
  const { nSamples, nComponents, nFeatures, nNonzeroCoefs } = options
  if (nNonzeroCoefs > nComponents)
    throw new WhiteboxError('value', 'makeSparseCodedSignal: more non-zero coefficients than components')
  const raw = normal(foldIn(key, 0), [nComponents, nFeatures]).data
  const dict = new Float64Array(nComponents * nFeatures)
  for (let c = 0; c < nComponents; c++) {
    let norm = 0
    for (let j = 0; j < nFeatures; j++) norm += (raw[c * nFeatures + j] ?? 0) ** 2
    norm = Math.sqrt(norm) || 1
    for (let j = 0; j < nFeatures; j++) dict[c * nFeatures + j] = (raw[c * nFeatures + j] ?? 0) / norm
  }
  const code = new Float64Array(nSamples * nComponents)
  const values = normal(foldIn(key, 1), [nSamples, nNonzeroCoefs]).data
  for (let i = 0; i < nSamples; i++) {
    const idx = permutation(foldIn(foldIn(key, 2), i), nComponents).data
    for (let j = 0; j < nNonzeroCoefs; j++) code[i * nComponents + (idx[j] ?? 0)] = values[i * nNonzeroCoefs + j] ?? 0
  }
  const dictionary = matrix(dict, nComponents, nFeatures)
  const codeT = matrix(code, nSamples, nComponents)
  return { data: matmul(codeT, dictionary) as Tensor, dictionary, code: codeT }
}

/**
 * A random symmetric positive-definite matrix (sklearn.datasets.make_spd_matrix): `U diag(1 + u) U^T` with `U` the
 * singular vectors of `A^T A` for uniform `A`, so its eigenvalues are `1 + u` with `u` uniform on [0, 1).
 *
 * @param key - The random key.
 * @param d - The dimension.
 * @returns The matrix and its eigenvalues (descending).
 * @example
 * ```ts
 * import { makeSpdMatrix } from 'whitebox-engine/methods/datasets'
 * import { key } from 'whitebox-engine/random'
 * makeSpdMatrix(key(0), 3).eigenvalues.every((e) => e >= 1 && e < 2) // => true
 * ```
 */
export function makeSpdMatrix(key: Key, d: number): { readonly x: Tensor; readonly eigenvalues: readonly number[] } {
  const a = uniform(foldIn(key, 0), [d, d])
  const { u } = svd(matmul(transpose(a), a))
  const e = Array.from(uniform(foldIn(key, 1), [d]).data, (v) => 1 + v).sort((p, q) => q - p)
  const x = matmul(mul(u, tensor(e)), transpose(u)) as Tensor
  // Symmetrise exactly.
  const sym = Float64Array.from(x.data as ArrayLike<number>)
  for (let i = 0; i < d; i++)
    for (let j = 0; j < i; j++) {
      const m = 0.5 * ((sym[i * d + j] ?? 0) + (sym[j * d + i] ?? 0))
      sym[i * d + j] = m
      sym[j * d + i] = m
    }
  return { x: fromStorage(sym, [d, d], 'float64'), eigenvalues: e }
}

/** Options of {@link makeSparseSpdMatrix}. */
export interface SparseSpdOptions {
  /** The probability that an off-diagonal Cholesky entry is zero (default 0.95). */
  readonly alpha?: number
  readonly smallestCoef?: number
  readonly largestCoef?: number
  /** Scale to a unit diagonal (a correlation-like precision; default false). */
  readonly normDiag?: boolean
}

/**
 * A sparse symmetric positive-definite precision matrix (sklearn.datasets.make_sparse_spd_matrix): a unit lower
 * triangular factor with sparse off-diagonal entries uniform in `[smallestCoef, largestCoef]`, rows and columns
 * permuted, and `precision = L^T L`. Sparse precisions are the truth for graphical-lasso studies.
 *
 * @param key - The random key.
 * @param d - The dimension.
 * @param options - Sparsity, coefficient range and diagonal normalisation.
 * @returns The precision matrix and its factor.
 * @example
 * ```ts
 * import { makeSparseSpdMatrix } from 'whitebox-engine/methods/datasets'
 * import { key } from 'whitebox-engine/random'
 * makeSparseSpdMatrix(key(0), 4, { normDiag: true }).precision.data[0] // => 1
 * ```
 */
export function makeSparseSpdMatrix(
  key: Key,
  d: number,
  options: SparseSpdOptions = {},
): { readonly precision: Tensor; readonly factor: Tensor } {
  const alpha = options.alpha ?? 0.95
  const lo = options.smallestCoef ?? 0.1
  const hi = options.largestCoef ?? 0.9
  const u = uniform(foldIn(key, 0), [d, d]).data
  const v = uniform(foldIn(key, 1), [d, d]).data
  const chol = new Float64Array(d * d)
  for (let i = 0; i < d; i++) {
    chol[i * d + i] = -1
    for (let j = 0; j < i; j++) if ((u[i * d + j] ?? 0) > alpha) chol[i * d + j] = lo + (hi - lo) * (v[i * d + j] ?? 0)
  }
  const perm = permutation(foldIn(key, 2), d).data
  const permuted = new Float64Array(d * d)
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++) permuted[i * d + j] = chol[(perm[i] ?? 0) * d + (perm[j] ?? 0)] ?? 0
  const factor = matrix(permuted, d, d)
  const prec = Float64Array.from((matmul(transpose(factor), factor) as Tensor).data as ArrayLike<number>)
  if (options.normDiag === true) {
    const diag = Array.from({ length: d }, (_, i) => Math.sqrt(prec[i * d + i] ?? 1))
    for (let i = 0; i < d; i++)
      for (let j = 0; j < d; j++) prec[i * d + j] = (prec[i * d + j] ?? 0) / ((diag[i] ?? 1) * (diag[j] ?? 1))
    for (let i = 0; i < d; i++) prec[i * d + i] = 1
  }
  return { precision: fromStorage(prec, [d, d], 'float64'), factor }
}

/** Options of {@link makeBiclusters} and {@link makeCheckerboard}. */
export interface BiclusterOptions {
  /** The standard deviation of added Gaussian noise (default 0). */
  readonly noise?: number
  readonly minval?: number
  readonly maxval?: number
  /** Shuffle rows and columns (default true). */
  readonly shuffle?: boolean
}

/** The planted structure of a bicluster matrix. */
export interface BiclusterTruth {
  readonly x: Tensor
  readonly rowLabels: readonly number[]
  readonly columnLabels: readonly number[]
  /** Bicluster indicators: `rows[b][i]` is true when row `i` belongs to bicluster `b`. */
  readonly rows: readonly (readonly boolean[])[]
  readonly columns: readonly (readonly boolean[])[]
}

function groupSizes(key: Key, n: number, k: number): number[] {
  return Array.from(multinomial(key, n, new Array<number>(k).fill(1 / k), 1).data)
}

function plant(
  key: Key,
  shape: readonly [number, number],
  rowLabels: number[],
  colLabels: number[],
  value: (r: number, c: number) => number,
  options: BiclusterOptions,
): { x: Tensor; rowLabels: number[]; colLabels: number[] } {
  const [nr, nc] = shape
  const noise = options.noise ?? 0
  const z = normal(foldIn(key, 10), [nr, nc]).data
  let rl = rowLabels
  let cl = colLabels
  if (options.shuffle !== false) {
    const pr = permutation(foldIn(key, 11), nr).data
    const pc = permutation(foldIn(key, 12), nc).data
    rl = Array.from(pr, (i) => rowLabels[i] ?? 0)
    cl = Array.from(pc, (j) => colLabels[j] ?? 0)
  }
  const x = new Float64Array(nr * nc)
  for (let i = 0; i < nr; i++)
    for (let j = 0; j < nc; j++) x[i * nc + j] = value(rl[i] ?? 0, cl[j] ?? 0) + noise * (z[i * nc + j] ?? 0)
  return { x: matrix(x, nr, nc), rowLabels: rl, colLabels: cl }
}

function labelsFromSizes(sizes: readonly number[]): number[] {
  return sizes.flatMap((s, c) => new Array<number>(s).fill(c))
}

/**
 * A matrix with constant blocks on a block diagonal (sklearn.datasets.make_biclusters): rows and columns split into
 * `nClusters` groups by a multinomial, bicluster `b` has value uniform in `[minval, maxval]`, plus noise.
 *
 * @param key - The random key.
 * @param shape - `[rows, columns]`.
 * @param nClusters - The number of biclusters.
 * @param options - Noise, value range and shuffling.
 * @returns The matrix and its planted biclusters.
 * @example
 * ```ts
 * import { makeBiclusters } from 'whitebox-engine/methods/datasets'
 * import { key } from 'whitebox-engine/random'
 * makeBiclusters(key(0), [6, 4], 2).rows.length // => 2
 * ```
 */
export function makeBiclusters(
  key: Key,
  shape: readonly [number, number],
  nClusters: number,
  options: BiclusterOptions = {},
): BiclusterTruth {
  const lo = options.minval ?? 10
  const hi = options.maxval ?? 100
  const consts = Array.from(uniform(foldIn(key, 0), [nClusters], { low: lo, high: hi }).data)
  const rows = labelsFromSizes(groupSizes(foldIn(key, 1), shape[0], nClusters))
  const cols = labelsFromSizes(groupSizes(foldIn(key, 2), shape[1], nClusters))
  const out = plant(key, shape, rows, cols, (r, c) => (r === c ? (consts[r] ?? 0) : 0), options)
  return {
    x: out.x,
    rowLabels: out.rowLabels,
    columnLabels: out.colLabels,
    rows: Array.from({ length: nClusters }, (_, b) => out.rowLabels.map((r) => r === b)),
    columns: Array.from({ length: nClusters }, (_, b) => out.colLabels.map((c) => c === b)),
  }
}

/**
 * A matrix with a checkerboard of constant blocks (sklearn.datasets.make_checkerboard): rows in `nClusters[0]` groups,
 * columns in `nClusters[1]` groups, and every (row group, column group) block its own value, plus noise.
 *
 * @param key - The random key.
 * @param shape - `[rows, columns]`.
 * @param nClusters - Row and column cluster counts.
 * @param options - Noise, value range and shuffling.
 * @returns The matrix and its planted biclusters (one per block, row-group major).
 * @example
 * ```ts
 * import { makeCheckerboard } from 'whitebox-engine/methods/datasets'
 * import { key } from 'whitebox-engine/random'
 * makeCheckerboard(key(0), [6, 6], [2, 3]).rows.length // => 6
 * ```
 */
export function makeCheckerboard(
  key: Key,
  shape: readonly [number, number],
  nClusters: readonly [number, number],
  options: BiclusterOptions = {},
): BiclusterTruth {
  const [kr, kc] = nClusters
  const lo = options.minval ?? 10
  const hi = options.maxval ?? 100
  const consts = Array.from(uniform(foldIn(key, 0), [kr * kc], { low: lo, high: hi }).data)
  const rows = labelsFromSizes(groupSizes(foldIn(key, 1), shape[0], kr))
  const cols = labelsFromSizes(groupSizes(foldIn(key, 2), shape[1], kc))
  const out = plant(key, shape, rows, cols, (r, c) => consts[r * kc + c] ?? 0, options)
  const blocks = Array.from({ length: kr * kc }, (_, b) => [Math.floor(b / kc), b % kc] as const)
  return {
    x: out.x,
    rowLabels: out.rowLabels,
    columnLabels: out.colLabels,
    rows: blocks.map(([r]) => out.rowLabels.map((l) => l === r)),
    columns: blocks.map(([, c]) => out.colLabels.map((l) => l === c)),
  }
}
