/**
 * Sparse coding and dictionary learning: sparse codes of samples over a dictionary (as scikit-learn's `sparse_encode`),
 * the K-SVD, method-of-optimal-directions and online (Mairal et al.) dictionary learners as traced algorithms, and the
 * measures of a dictionary that guarantee sparse recovery (mutual coherence, the Babel function, estimates of the
 * restricted isometry constant).
 */

import { type Json, OmaiError } from '../base/index.ts'
import { ddot, dnrm2, lstsq, svd } from '../numerics/linalg/index.ts'
import { foldIn, type Key, key as makeKey, permutation } from '../random/index.ts'
import { asTensor, fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, type RunStatus, runAlgorithm } from '../trace/index.ts'
import type { DenseMatrix } from './_vector.ts'
import { lassoCoordinateDescent, orthogonalMatchingPursuit } from './sparse.ts'

function matrix(a: TensorLike, name: string): DenseMatrix {
  const t = primalOf(asTensor(a)) as Tensor
  if (t.shape.length !== 2) throw new OmaiError('shape', `${name}: expected a matrix`)
  return { data: Float64Array.from(t.data), rows: t.shape[0] ?? 0, cols: t.shape[1] ?? 0 }
}

const row = (m: DenseMatrix, i: number): Float64Array => m.data.slice(i * m.cols, (i + 1) * m.cols)
const transposed = (m: DenseMatrix): DenseMatrix => {
  const out = new Float64Array(m.rows * m.cols)
  for (let i = 0; i < m.rows; i++) for (let j = 0; j < m.cols; j++) out[j * m.rows + i] = m.data[i * m.cols + j] ?? 0
  return { data: out, rows: m.cols, cols: m.rows }
}
const asTensorOf = (m: DenseMatrix): Tensor => fromStorage(m.data, [m.rows, m.cols], 'float64')

/** Options of {@link sparseEncode}. */
export interface SparseEncodeOptions {
  /** `omp` (orthogonal matching pursuit), `lassoCd` (coordinate descent) or `threshold` (soft thresholding). */
  readonly algorithm?: 'omp' | 'lassoCd' | 'threshold'
  /** Atoms per code for `omp` (default min(max(nFeatures / 10, 1), nComponents), sklearn's). */
  readonly nNonzeroCoefs?: number
  /** The L1 weight for `lassoCd` and the threshold for `threshold` (default 1). */
  readonly alpha?: number
  /** Coordinate descent tolerance and iteration cap (sklearn's 1e-4 and 1000). */
  readonly tol?: number
  readonly maxIterations?: number
}

/**
 * Sparse codes of samples over a dictionary (sklearn.decomposition.sparse_encode): each row of X is approximated by
 * code @ dictionary with a sparse code, by OMP, by the lasso min |x - D^T c|^2 / 2 + alpha |c|_1 (coordinate descent),
 * or by soft thresholding of the correlations.
 *
 * @param x - Samples `[nSamples, nFeatures]`.
 * @param dictionary - Atoms as rows `[nComponents, nFeatures]`.
 * @param options - Algorithm, sparsity, alpha and tolerances.
 * @returns The codes `[nSamples, nComponents]`.
 * @concept sparse-coding
 * @example
 * ```ts
 * import { sparseEncode } from 'omai-engine/optim'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(sparseEncode([[3, 0.5]], [[1, 0], [0, 1]], { algorithm: 'threshold', alpha: 1 })) // => [[2, 0]]
 * ```
 */
export function sparseEncode(x: TensorLike, dictionary: TensorLike, options: SparseEncodeOptions = {}): Tensor {
  const X = matrix(x, 'sparseEncode')
  const D = matrix(dictionary, 'sparseEncode')
  if (X.cols !== D.cols) throw new OmaiError('shape', 'sparseEncode: samples and atoms differ in length')
  const algorithm = options.algorithm ?? 'lassoCd'
  const alpha = options.alpha ?? 1
  const k = D.rows
  const out = new Float64Array(X.rows * k)
  const dt = asTensorOf(transposed(D))
  for (let i = 0; i < X.rows; i++) {
    const xi = row(X, i)
    let code: ArrayLike<number>
    if (algorithm === 'threshold') {
      code = Float64Array.from({ length: k }, (_, j) => {
        const c = ddot(row(D, j), xi)
        return Math.sign(c) * Math.max(Math.abs(c) - alpha, 0)
      })
    } else if (algorithm === 'omp') {
      const nnz = options.nNonzeroCoefs ?? Math.min(Math.max(Math.floor(X.cols / 10), 1), k)
      code = toFlat(orthogonalMatchingPursuit(dt, xi, { nNonzeroCoefs: nnz }).coef)
    } else {
      code = toFlat(
        lassoCoordinateDescent(dt, xi, alpha, {
          tol: options.tol ?? 1e-4,
          maxIterations: options.maxIterations ?? 1000,
        }).coef,
      )
    }
    out.set(Array.from(code), i * k)
  }
  return fromStorage(out, [X.rows, k], 'float64')
}

/** Options of the dictionary learners. */
export interface DictionaryLearningOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** The key for the initial atoms (random samples) and minibatches (default key(0)). */
  readonly key?: Key
  /** Initial atoms `[nComponents, nFeatures]` (rows are normalised). */
  readonly init?: TensorLike
  readonly maxIterations?: number
  /** Stop once the relative change of the reconstruction error falls to this (default 1e-8). */
  readonly tol?: number
}

/** State of the dictionary learners. */
export interface DictionaryState {
  /** Atoms as rows `[nComponents, nFeatures]`, unit norm. */
  readonly dictionary: Float64Array
  /** Codes `[nSamples, nComponents]`. */
  readonly code: Float64Array
  /** Frobenius norm of X - code dictionary. */
  readonly error: number
  readonly change: number
  /** Online learning's sufficient statistics A (k x k) and B (nFeatures x k). */
  readonly a?: Float64Array
  readonly b?: Float64Array
}

/** Result of the dictionary learners. */
export interface DictionaryResult {
  readonly dictionary: Tensor
  readonly code: Tensor
  readonly error: number
  readonly iterations: number
  readonly status: RunStatus
  /** The reconstruction error after each iteration. */
  readonly history: readonly number[]
}

const DICT_SCHEMA = channels({
  metrics: {
    error: { description: 'Frobenius norm of X - code @ dictionary', direction: 'lower' },
    change: { description: 'relative change of the error' },
  },
  tensors: { dictionary: { role: 'param', description: 'atoms as rows' } },
})

function reconstructionError(X: DenseMatrix, dictionary: Float64Array, code: Float64Array, k: number): number {
  let s = 0
  for (let i = 0; i < X.rows; i++)
    for (let f = 0; f < X.cols; f++) {
      let v = X.data[i * X.cols + f] ?? 0
      for (let j = 0; j < k; j++) v -= (code[i * k + j] ?? 0) * (dictionary[j * X.cols + f] ?? 0)
      s += v * v
    }
  return Math.sqrt(s)
}

function normaliseRows(d: Float64Array, k: number, m: number): Float64Array {
  const out = Float64Array.from(d)
  for (let j = 0; j < k; j++) {
    const n = dnrm2(out.subarray(j * m, (j + 1) * m))
    if (n > 0) for (let f = 0; f < m; f++) out[j * m + f] = (out[j * m + f] ?? 0) / n
  }
  return out
}

/** Initial atoms: the given ones, or distinct random samples (normalised). */
function initialAtoms(X: DenseMatrix, k: number, options: DictionaryLearningOptions, key: Key): Float64Array {
  if (options.init !== undefined) {
    const init = matrix(options.init, 'dictionary learning')
    if (init.rows !== k || init.cols !== X.cols) throw new OmaiError('shape', 'init must be [nComponents, nFeatures]')
    return normaliseRows(init.data, k, X.cols)
  }
  if (k > X.rows) throw new OmaiError('value', 'dictionary learning: more atoms than samples needs an init')
  const order = permutation(key, X.rows).data
  const d = new Float64Array(k * X.cols)
  for (let j = 0; j < k; j++) d.set(row(X, order[j] ?? 0), j * X.cols)
  return normaliseRows(d, k, X.cols)
}

function encodeOmp(X: DenseMatrix, dictionary: Float64Array, k: number, sparsity: number): Float64Array {
  return Float64Array.from(
    toFlat(
      sparseEncode(asTensorOf(X), fromStorage(dictionary, [k, X.cols], 'float64'), {
        algorithm: 'omp',
        nNonzeroCoefs: sparsity,
      }),
    ),
  )
}

/** The sample with the largest residual, normalised: the replacement for an unused atom. */
function worstSample(X: DenseMatrix, dictionary: Float64Array, code: Float64Array, k: number): Float64Array {
  let best = 0
  let worst = -1
  for (let i = 0; i < X.rows; i++) {
    let s = 0
    for (let f = 0; f < X.cols; f++) {
      let v = X.data[i * X.cols + f] ?? 0
      for (let j = 0; j < k; j++) v -= (code[i * k + j] ?? 0) * (dictionary[j * X.cols + f] ?? 0)
      s += v * v
    }
    if (s > worst) {
      worst = s
      best = i
    }
  }
  const r = row(X, best)
  const n = dnrm2(r)
  return Float64Array.from(r, (v) => v / Math.max(n, 1e-300))
}

function step(previous: number, error: number): number {
  return Math.abs(previous - error) / Math.max(previous, 1e-300)
}

/**
 * K-SVD (Aharon, Elad and Bruckstein 2006) as a traced algorithm: sparse coding of every sample by OMP, then each atom
 * and its coefficients replaced by the rank-one SVD approximation of the residual restricted to the samples that use
 * it (unused atoms are re-seeded with the worst-represented sample).
 *
 * @param x - Samples `[nSamples, nFeatures]`.
 * @param nComponents - The number of atoms.
 * @param sparsity - Atoms per sample in the coding step.
 * @param options - Key, initial atoms and tolerance.
 * @returns The algorithm.
 * @concept dictionary-learning
 * @example
 * ```ts
 * import { ksvdAlgorithm } from 'omai-engine/optim'
 * ksvdAlgorithm([[1, 0], [0, 1], [1, 1]], 2, 1).name // => 'ksvd'
 * ```
 */
export function ksvdAlgorithm(
  x: TensorLike,
  nComponents: number,
  sparsity: number,
  options: DictionaryLearningOptions = {},
): Algorithm<DictionaryState> {
  const X = matrix(x, 'ksvd')
  const k = nComponents
  const m = X.cols
  const tol = options.tol ?? 1e-8
  return {
    name: 'ksvd',
    config: { nComponents, sparsity, tol } as Json,
    schema: DICT_SCHEMA,
    init: (rk) => {
      const dictionary = initialAtoms(X, k, options, options.key ?? rk)
      const code = encodeOmp(X, dictionary, k, sparsity)
      return { dictionary, code, error: reconstructionError(X, dictionary, code, k), change: Infinity }
    },
    step: (s) => {
      const code = encodeOmp(X, s.dictionary, k, sparsity)
      const dictionary = Float64Array.from(s.dictionary)
      for (let j = 0; j < k; j++) {
        const users: number[] = []
        for (let i = 0; i < X.rows; i++) if ((code[i * k + j] ?? 0) !== 0) users.push(i)
        if (users.length === 0) {
          dictionary.set(worstSample(X, dictionary, code, k), j * m)
          continue
        }
        // E = residual on the users with atom j's contribution added back, as nFeatures x |users|.
        const e = new Float64Array(m * users.length)
        for (const [c, i] of users.entries())
          for (let f = 0; f < m; f++) {
            let v = X.data[i * m + f] ?? 0
            for (let q = 0; q < k; q++) if (q !== j) v -= (code[i * k + q] ?? 0) * (dictionary[q * m + f] ?? 0)
            e[f * users.length + c] = v
          }
        const { u, s: sv, vh } = svd(fromStorage(e, [m, users.length], 'float64'))
        const uu = toFlat(primalOf(u))
        const ss = toFlat(primalOf(sv))
        const vv = toFlat(primalOf(vh))
        const r = Math.min(m, users.length)
        for (let f = 0; f < m; f++) dictionary[j * m + f] = uu[f * r] ?? 0
        for (const [c, i] of users.entries()) code[i * k + j] = (ss[0] ?? 0) * (vv[c] ?? 0)
      }
      const error = reconstructionError(X, dictionary, code, k)
      return { dictionary, code, error, change: step(s.error, error) }
    },
    done: (s) => (s.change <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { error: s.error, change: s.change },
      tensors: { dictionary: { role: 'param', value: s.dictionary, shape: [k, m] } },
    }),
  }
}

function finish(
  algorithm: Algorithm<DictionaryState>,
  options: DictionaryLearningOptions,
  k: number,
  m: number,
  n: number,
  defaultIterations: number,
): DictionaryResult {
  const { key: _k, init: _i, maxIterations, tol: _t, ...run } = options
  const history: number[] = []
  const tracked: Algorithm<DictionaryState> = {
    ...algorithm,
    init: (rk) => {
      const s = algorithm.init(rk)
      history.push(s.error)
      return s
    },
    step: (s, ctx) => {
      const next = algorithm.step(s, ctx)
      history.push(next.error)
      return next
    },
  }
  const result = runAlgorithm(tracked, {
    ...run,
    ...(options.key === undefined ? {} : { key: options.key }),
    maxSteps: maxIterations ?? defaultIterations,
  })
  const s = result.state
  return {
    dictionary: fromStorage(Float64Array.from(s.dictionary), [k, m], 'float64'),
    code: fromStorage(Float64Array.from(s.code), [n, k], 'float64'),
    error: s.error,
    iterations: result.steps,
    status: result.status,
    history,
  }
}

/**
 * Learn a dictionary by K-SVD.
 *
 * @param x - Samples `[nSamples, nFeatures]`.
 * @param nComponents - The number of atoms.
 * @param sparsity - Atoms per sample.
 * @param options - Key, initial atoms, tolerance, iteration cap (default 50) and run options.
 * @returns The atoms (rows, unit norm), the codes and the error history.
 * @concept dictionary-learning
 * @example
 * ```ts
 * import { ksvd } from 'omai-engine/optim'
 * ksvd([[2, 0], [0, 3], [1, 0]], 2, 1, { init: [[1, 0], [0, 1]] }).error // => 0
 * ```
 */
export function ksvd(
  x: TensorLike,
  nComponents: number,
  sparsity: number,
  options: DictionaryLearningOptions = {},
): DictionaryResult {
  const X = matrix(x, 'ksvd')
  return finish(ksvdAlgorithm(x, nComponents, sparsity, options), options, nComponents, X.cols, X.rows, 50)
}

/**
 * The method of optimal directions (Engan, Aase and Husoy 1999) as a traced algorithm: OMP codes, then the dictionary
 * that minimises the reconstruction error for those codes (a least-squares solve), with atoms normalised.
 *
 * @param x - Samples `[nSamples, nFeatures]`.
 * @param nComponents - The number of atoms.
 * @param sparsity - Atoms per sample.
 * @param options - Key, initial atoms and tolerance.
 * @returns The algorithm.
 * @concept dictionary-learning
 * @example
 * ```ts
 * import { modAlgorithm } from 'omai-engine/optim'
 * modAlgorithm([[1, 0], [0, 1], [1, 1]], 2, 1).name // => 'mod'
 * ```
 */
export function modAlgorithm(
  x: TensorLike,
  nComponents: number,
  sparsity: number,
  options: DictionaryLearningOptions = {},
): Algorithm<DictionaryState> {
  const X = matrix(x, 'mod')
  const k = nComponents
  const m = X.cols
  const tol = options.tol ?? 1e-8
  return {
    name: 'mod',
    config: { nComponents, sparsity, tol } as Json,
    schema: DICT_SCHEMA,
    init: (rk) => {
      const dictionary = initialAtoms(X, k, options, options.key ?? rk)
      const code = encodeOmp(X, dictionary, k, sparsity)
      return { dictionary, code, error: reconstructionError(X, dictionary, code, k), change: Infinity }
    },
    step: (s) => {
      const code = encodeOmp(X, s.dictionary, k, sparsity)
      // dictionary = argmin |X - code D|_F = lstsq(code, X), then unit rows (codes absorb the scale next time).
      const solved = Float64Array.from(
        toFlat(primalOf(lstsq(fromStorage(code, [X.rows, k], 'float64'), asTensorOf(X)))),
      )
      for (let j = 0; j < k; j++) {
        if (dnrm2(solved.subarray(j * m, (j + 1) * m)) === 0) solved.set(worstSample(X, s.dictionary, code, k), j * m)
      }
      const dictionary = normaliseRows(solved, k, m)
      const recoded = encodeOmp(X, dictionary, k, sparsity)
      const error = reconstructionError(X, dictionary, recoded, k)
      return { dictionary, code: recoded, error, change: step(s.error, error) }
    },
    done: (s) => (s.change <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { error: s.error, change: s.change },
      tensors: { dictionary: { role: 'param', value: s.dictionary, shape: [k, m] } },
    }),
  }
}

/**
 * Learn a dictionary by the method of optimal directions.
 *
 * @param x - Samples `[nSamples, nFeatures]`.
 * @param nComponents - The number of atoms.
 * @param sparsity - Atoms per sample.
 * @param options - Key, initial atoms, tolerance, iteration cap (default 50) and run options.
 * @returns The atoms, codes and error history.
 * @concept dictionary-learning
 * @example
 * ```ts
 * import { mod } from 'omai-engine/optim'
 * mod([[2, 0], [0, 3], [1, 0]], 2, 1, { init: [[1, 0], [0, 1]] }).error // => 0
 * ```
 */
export function mod(
  x: TensorLike,
  nComponents: number,
  sparsity: number,
  options: DictionaryLearningOptions = {},
): DictionaryResult {
  const X = matrix(x, 'mod')
  return finish(modAlgorithm(x, nComponents, sparsity, options), options, nComponents, X.cols, X.rows, 50)
}

/** Options of {@link onlineDictionaryLearning}. */
export interface OnlineDictionaryOptions extends DictionaryLearningOptions {
  /** The lasso weight of the coding step (default 1). */
  readonly alpha?: number
  /** Samples per minibatch (default 3, sklearn's). */
  readonly batchSize?: number
  /** Forgetting factor for the statistics A and B in (0, 1] (default 1: no forgetting). */
  readonly forget?: number
}

/**
 * Online dictionary learning (Mairal, Bach, Ponce and Sapiro 2009) as a traced algorithm: each step draws a minibatch
 * (from the step key), codes it by the lasso, accumulates A += sum c c^T and B += sum x c^T, and updates each atom by
 * one pass of block coordinate descent, d_j = proj_{|d| <= 1}((b_j - D a_j) / A_jj + d_j).
 *
 * @param x - Samples `[nSamples, nFeatures]`.
 * @param nComponents - The number of atoms.
 * @param options - Lasso weight, batch size, forgetting factor, key and initial atoms.
 * @returns The algorithm (the error is over all samples, coded at the current dictionary).
 * @concept dictionary-learning
 * @example
 * ```ts
 * import { onlineDictionaryLearningAlgorithm } from 'omai-engine/optim'
 * onlineDictionaryLearningAlgorithm([[1, 0], [0, 1], [1, 1]], 2).name // => 'onlineDictionaryLearning'
 * ```
 */
export function onlineDictionaryLearningAlgorithm(
  x: TensorLike,
  nComponents: number,
  options: OnlineDictionaryOptions = {},
): Algorithm<DictionaryState> {
  const X = matrix(x, 'onlineDictionaryLearning')
  const k = nComponents
  const m = X.cols
  const alpha = options.alpha ?? 1
  const batch = Math.min(options.batchSize ?? 3, X.rows)
  const forget = options.forget ?? 1
  const code = (dictionary: Float64Array, rows: DenseMatrix): Float64Array =>
    Float64Array.from(
      toFlat(
        sparseEncode(asTensorOf(rows), fromStorage(dictionary, [k, m], 'float64'), { algorithm: 'lassoCd', alpha }),
      ),
    )
  return {
    name: 'onlineDictionaryLearning',
    config: { nComponents, alpha, batchSize: batch, forget } as Json,
    schema: DICT_SCHEMA,
    init: (rk) => {
      const dictionary = initialAtoms(X, k, options, options.key ?? rk)
      const c = code(dictionary, X)
      return {
        dictionary,
        code: c,
        error: reconstructionError(X, dictionary, c, k),
        change: Infinity,
        a: new Float64Array(k * k),
        b: new Float64Array(m * k),
      }
    },
    step: (s, ctx) => {
      const order = permutation(foldIn(ctx.key, 7), X.rows).data
      const rows = new Float64Array(batch * m)
      for (let i = 0; i < batch; i++) rows.set(row(X, order[i] ?? 0), i * m)
      const mb: DenseMatrix = { data: rows, rows: batch, cols: m }
      const c = code(s.dictionary, mb)
      const a = Float64Array.from(s.a ?? new Float64Array(k * k), (v) => forget * v)
      const b = Float64Array.from(s.b ?? new Float64Array(m * k), (v) => forget * v)
      for (let i = 0; i < batch; i++)
        for (let p = 0; p < k; p++) {
          const cp = c[i * k + p] ?? 0
          if (cp === 0) continue
          for (let q = 0; q < k; q++) a[p * k + q] = (a[p * k + q] ?? 0) + cp * (c[i * k + q] ?? 0)
          for (let f = 0; f < m; f++) b[f * k + p] = (b[f * k + p] ?? 0) + (rows[i * m + f] ?? 0) * cp
        }
      const dictionary = Float64Array.from(s.dictionary)
      for (let j = 0; j < k; j++) {
        const ajj = a[j * k + j] ?? 0
        if (ajj <= 1e-12) continue
        const u = new Float64Array(m)
        for (let f = 0; f < m; f++) {
          let da = 0
          for (let q = 0; q < k; q++) da += (dictionary[q * m + f] ?? 0) * (a[q * k + j] ?? 0)
          u[f] = ((b[f * k + j] ?? 0) - da) / ajj + (dictionary[j * m + f] ?? 0)
        }
        const n = Math.max(dnrm2(u), 1)
        for (let f = 0; f < m; f++) dictionary[j * m + f] = (u[f] ?? 0) / n
      }
      const full = code(dictionary, X)
      const error = reconstructionError(X, dictionary, full, k)
      return { dictionary, code: full, error, change: step(s.error, error), a, b }
    },
    done: (s) => (s.change <= (options.tol ?? 0) ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { error: s.error, change: s.change },
      tensors: { dictionary: { role: 'param', value: s.dictionary, shape: [k, m] } },
    }),
  }
}

/**
 * Learn a dictionary online from minibatches (Mairal et al. 2009; sklearn's MiniBatchDictionaryLearning family).
 *
 * @param x - Samples `[nSamples, nFeatures]`.
 * @param nComponents - The number of atoms.
 * @param options - Lasso weight, batch size, forgetting, key, initial atoms, iteration cap (default 100) and run options.
 * @returns The atoms (norm at most 1), the lasso codes and the error history.
 * @concept dictionary-learning
 * @example
 * ```ts
 * import { onlineDictionaryLearning } from 'omai-engine/optim'
 * import { key } from 'omai-engine/random'
 * onlineDictionaryLearning([[1, 0], [0, 1], [1, 1]], 2, { key: key(1), maxIterations: 5, alpha: 0.1 }).dictionary.shape // => [2, 2]
 * ```
 */
export function onlineDictionaryLearning(
  x: TensorLike,
  nComponents: number,
  options: OnlineDictionaryOptions = {},
): DictionaryResult {
  const X = matrix(x, 'onlineDictionaryLearning')
  const { alpha: _a, batchSize: _b, forget: _f, ...rest } = options
  return finish(onlineDictionaryLearningAlgorithm(x, nComponents, options), rest, nComponents, X.cols, X.rows, 100)
}

/** The normalised Gram matrix |<d_i, d_j>| / (|d_i| |d_j|) of a dictionary's columns. */
function normalisedGram(d: DenseMatrix): Float64Array {
  const cols = Array.from({ length: d.cols }, (_, j) =>
    Float64Array.from({ length: d.rows }, (_, i) => d.data[i * d.cols + j] ?? 0),
  )
  const norms = cols.map((c) => dnrm2(c))
  const g = new Float64Array(d.cols * d.cols)
  for (let i = 0; i < d.cols; i++)
    for (let j = 0; j < d.cols; j++)
      g[i * d.cols + j] =
        Math.abs(ddot(cols[i] as Float64Array, cols[j] as Float64Array)) /
        Math.max((norms[i] ?? 0) * (norms[j] ?? 0), 1e-300)
  return g
}

/**
 * The mutual coherence of a dictionary: the largest |cosine| between two distinct atoms (columns). Recovery of every
 * s-sparse vector by OMP and basis pursuit is guaranteed when s < (1 + 1 / mu) / 2 ({@link coherenceSparsityBound}).
 *
 * @param dictionary - The dictionary `[m, k]`, atoms as columns.
 * @returns mu in [0, 1].
 * @concept compressed-sensing
 * @example
 * ```ts
 * import { mutualCoherence } from 'omai-engine/optim'
 * Math.round(mutualCoherence([[1, 0, 1], [0, 1, 1]]) * 1e12) / 1e12 // => 0.707106781187
 * ```
 */
export function mutualCoherence(dictionary: TensorLike): number {
  const d = matrix(dictionary, 'mutualCoherence')
  const g = normalisedGram(d)
  let mu = 0
  for (let i = 0; i < d.cols; i++)
    for (let j = 0; j < d.cols; j++) if (i !== j) mu = Math.max(mu, g[i * d.cols + j] ?? 0)
  return mu
}

/**
 * The Babel function mu_1(s) (Tropp 2004): the largest total |cosine| between one atom and s others. OMP recovers every
 * s-sparse vector when mu_1(s - 1) + mu_1(s) < 1.
 *
 * @param dictionary - The dictionary `[m, k]`.
 * @param s - The number of other atoms.
 * @returns mu_1(s).
 * @concept compressed-sensing
 * @example
 * ```ts
 * import { babelFunction } from 'omai-engine/optim'
 * babelFunction([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 2) // => 0
 * ```
 */
export function babelFunction(dictionary: TensorLike, s: number): number {
  const d = matrix(dictionary, 'babelFunction')
  const g = normalisedGram(d)
  let best = 0
  for (let i = 0; i < d.cols; i++) {
    const others = Array.from({ length: d.cols }, (_, j) => (j === i ? -1 : (g[i * d.cols + j] ?? 0)))
      .filter((v) => v >= 0)
      .sort((a, b) => b - a)
    best = Math.max(
      best,
      others.slice(0, s).reduce((a, b) => a + b, 0),
    )
  }
  return best
}

/**
 * The sparsity below which coherence guarantees exact recovery by OMP and basis pursuit (Donoho and Elad 2003;
 * Tropp 2004): (1 + 1 / mu) / 2. Also 1 + 1 / mu bounds the spark from below.
 *
 * @param mu - The mutual coherence.
 * @returns The bound (any s strictly below it is recoverable).
 * @example
 * ```ts
 * import { coherenceSparsityBound } from 'omai-engine/optim'
 * coherenceSparsityBound(0.25) // => 2.5
 * ```
 */
export function coherenceSparsityBound(mu: number): number {
  return mu === 0 ? Infinity : (1 + 1 / mu) / 2
}

/** Result of {@link ripEstimate}. */
export interface RipEstimate {
  /** A lower bound on the restricted isometry constant delta_s: max(1 - min sigma^2, max sigma^2 - 1). */
  readonly delta: number
  /** The smallest and largest squared singular values seen over the sampled supports. */
  readonly lower: number
  readonly upper: number
  readonly samples: number
}

/**
 * Estimate the restricted isometry constant delta_s of a matrix by sampling supports: the extreme squared singular
 * values of column submatrices of size s (computing delta_s exactly is NP-hard, so this is a lower bound).
 *
 * @param dictionary - The matrix `[m, k]` (use unit-norm columns, for example scaled by 1 / sqrt(m) for Gaussians).
 * @param s - The support size.
 * @param options - The key and the number of sampled supports (default 200).
 * @returns The estimate.
 * @concept compressed-sensing
 * @example
 * ```ts
 * import { ripEstimate } from 'omai-engine/optim'
 * import { key } from 'omai-engine/random'
 * ripEstimate([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 2, { key: key(0), samples: 5 }).delta // => 0
 * ```
 */
export function ripEstimate(
  dictionary: TensorLike,
  s: number,
  options: { readonly key?: Key; readonly samples?: number } = {},
): RipEstimate {
  const d = matrix(dictionary, 'ripEstimate')
  if (!(s >= 1 && s <= d.cols)) throw new OmaiError('value', `ripEstimate: s must be in 1..${d.cols}`)
  const samples = options.samples ?? 200
  const base = options.key ?? makeKey(0)
  let lower = Infinity
  let upper = 0
  for (let t = 0; t < samples; t++) {
    const support = Array.from(permutation(foldIn(base, t), d.cols).data.slice(0, s)).sort((a, b) => a - b)
    const sub = new Float64Array(d.rows * s)
    for (let i = 0; i < d.rows; i++)
      for (const [c, j] of support.entries()) sub[i * s + c] = d.data[i * d.cols + j] ?? 0
    const sv = toFlat(primalOf(svd(fromStorage(sub, [d.rows, s], 'float64')).s))
    const smax = sv[0] ?? 0
    const smin = s > d.rows ? 0 : (sv.at(-1) ?? 0)
    upper = Math.max(upper, smax * smax)
    lower = Math.min(lower, smin * smin)
  }
  return { delta: Math.max(1 - lower, upper - 1), lower, upper, samples }
}
