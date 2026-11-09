/**
 * Sparse coding over a fixed dictionary (scikit-learn's `sparse_encode` and `SparseCoder`): each signal written as a
 * combination of few atoms by orthogonal matching pursuit, least angle regression (plain, or the lasso path stopped at
 * a penalty), coordinate descent on the lasso, or soft thresholding of the correlations. The greedy and
 * coordinate-descent coders are `optim`'s; the least-angle ones run `linear`'s path, one signal at a time. The
 * dictionary learners of this folder alternate this coding with an update of the dictionary.
 */

import { OmaiError } from '../base/index.ts'
import { defineEstimator, type Estimator, type Fitted, matrixView } from '../learn/index.ts'
import { coefficientsAt, larsPath } from '../linear/index.ts'
import { sparseEncode } from '../optim/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import { tr } from './_linear.ts'

/**
 * How a signal is coded over the atoms. scikit-learn's names are `omp`, `lars`, `lasso_lars`, `lasso_cd` and
 * `threshold` (its dictionary learners' `fit_algorithm` `'lars'` and `'cd'` are `lassoLars` and `lassoCd` here).
 */
export type SparseCodeAlgorithm = 'omp' | 'lars' | 'lassoLars' | 'lassoCd' | 'threshold'

/** Options of {@link sparseCode}. */
export interface SparseCodeOptions {
  /** The coder (default `lassoLars`, scikit-learn's `lasso_lars`). */
  readonly algorithm?: SparseCodeAlgorithm
  /**
   * Atoms per signal for `omp` and `lars` (default min(max(nFeatures / 10, 1), nComponents), truncated to an
   * integer, as scikit-learn).
   */
  readonly nNonzeroCoefs?: number
  /** The penalty of `lassoLars` and `lassoCd` on the objective |x - c D|^2 / 2 + alpha |c|_1, and the threshold of `threshold` (default 1). */
  readonly alpha?: number
  /** Constrain the codes to be non-negative: `lassoCd` and `threshold` (the least-angle path has no such mode here). */
  readonly positive?: boolean
  /** The duality-gap tolerance of `lassoCd` relative to |x|^2 (default 1e-8, scikit-learn's in `sparse_encode`). */
  readonly tol?: number
  /** The sweep cap of `lassoCd` (default 1000). */
  readonly maxIter?: number
}

/** The number of atoms an `omp` or `lars` coder uses when none is given (scikit-learn's default, truncated). */
function defaultNonzeroCoefs(nFeatures: number, nComponents: number): number {
  return Math.max(1, Math.min(Math.floor(Math.max(nFeatures / 10, 1)), nComponents))
}

/**
 * Sparse codes of signals over a dictionary: row `i` of the result is a sparse vector `c` with `c D ~ x_i`.
 *
 * - `omp`: orthogonal matching pursuit with `nNonzeroCoefs` atoms.
 * - `lars`: least angle regression, stopped when `nNonzeroCoefs` atoms are active.
 * - `lassoLars`: the lasso path of least angle regression, at `alpha` (exact: the penalty is `alpha` on the unscaled
 *   squared error, scikit-learn's `alpha / nFeatures` on the mean).
 * - `lassoCd`: the lasso by coordinate descent at `alpha`.
 * - `threshold`: `sign(d.x) max(|d.x| - alpha, 0)`.
 *
 * @param x - Signals `[n, nFeatures]`.
 * @param dictionary - Atoms as rows `[nComponents, nFeatures]`.
 * @param options - Coder, sparsity or penalty, positivity and coordinate descent tolerances.
 * @returns The codes `[n, nComponents]`.
 * @throws OmaiError `value` when `positive` is asked of `omp`, `lars` or `lassoLars`; `shape` when signals and atoms
 *   differ in length.
 * @concept sparse-coding
 * @index function Sparse codes of signals over a dictionary by OMP, LARS, lasso or thresholding.
 * @example
 * ```ts
 * import { sparseCode } from 'omai-engine/decomposition'
 * import { toArray } from 'omai-engine/tensor'
 * const d = [[1, 0], [0, 1], [0.6, 0.8]]
 * toArray(sparseCode([[3, 4]], d, { algorithm: 'omp', nNonzeroCoefs: 1 })) // => [[0, 0, 5]]
 * toArray(sparseCode([[3, 0.5]], [[1, 0], [0, 1]], { algorithm: 'lassoLars', alpha: 1 })) // => [[2, 0]]
 * ```
 */
export function sparseCode(x: TensorLike, dictionary: TensorLike, options: SparseCodeOptions = {}): Tensor {
  const signals = matrixView(x, 'sparseCode')
  const atoms = matrixView(dictionary, 'sparseCode', { nFeatures: signals.d })
  const algorithm = options.algorithm ?? 'lassoLars'
  const alpha = options.alpha ?? 1
  const k = atoms.n
  const f = signals.d
  const positive = options.positive ?? false
  if (positive && (algorithm === 'omp' || algorithm === 'lars' || algorithm === 'lassoLars'))
    throw new OmaiError('value', `sparseCode: ${algorithm} has no positive option; use lassoCd or threshold`)
  const nnz = options.nNonzeroCoefs ?? defaultNonzeroCoefs(f, k)
  if (algorithm === 'lars' || algorithm === 'lassoLars') {
    const out = new Float64Array(signals.n * k)
    const design = tr(Array.from({ length: k }, (_, j) => Array.from(atoms.data.subarray(j * f, (j + 1) * f))))
    for (let i = 0; i < signals.n; i++) {
      const y = Array.from(signals.data.subarray(i * f, (i + 1) * f))
      let coef: number[]
      if (algorithm === 'lars') {
        const path = larsPath({ x: design, y }, { method: 'lar', fitIntercept: false, maxFeatures: nnz })
        coef = path.coefs[path.coefs.length - 1] ?? new Array<number>(k).fill(0)
      } else {
        const path = larsPath({ x: design, y }, { method: 'lasso', fitIntercept: false, alphaMin: alpha / f })
        coef = coefficientsAt(path, alpha / f)
      }
      out.set(coef, i * k)
    }
    return fromStorage(out, [signals.n, k], 'float64')
  }
  return sparseEncode(
    fromStorage(signals.data, [signals.n, f], 'float64'),
    fromStorage(atoms.data, [k, f], 'float64'),
    {
      algorithm: algorithm === 'lassoCd' ? 'lassoCd' : algorithm,
      nNonzeroCoefs: nnz,
      alpha,
      positive,
      tol: options.tol ?? 1e-8,
      maxIterations: options.maxIter ?? 1000,
    },
  )
}

/**
 * The codes with their signs split (scikit-learn's `split_sign`): `[max(c, 0), max(-c, 0)]` side by side, so that a
 * downstream learner sees non-negative features.
 *
 * @param code - Codes `[n, k]`.
 * @returns The split codes `[n, 2 k]`.
 * @example
 * ```ts
 * import { splitSign } from 'omai-engine/decomposition'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(splitSign([[2, -3]])) // => [[2, 0, 0, 3]]
 * ```
 */
export function splitSign(code: TensorLike): Tensor {
  const m = matrixView(code, 'splitSign')
  const out = new Float64Array(m.n * 2 * m.d)
  for (let i = 0; i < m.n; i++)
    for (let j = 0; j < m.d; j++) {
      const v = m.data[i * m.d + j] ?? 0
      out[i * 2 * m.d + j] = Math.max(v, 0)
      out[i * 2 * m.d + m.d + j] = Math.max(-v, 0)
    }
  return fromStorage(out, [m.n, 2 * m.d], 'float64')
}

/** Hyperparameters of {@link sparseCoder}. */
export interface SparseCoderHyper {
  /** The fixed dictionary, atoms as rows `[nComponents, nFeatures]`. */
  readonly dictionary: number[][]
  /** The coder (default `omp`, as `SparseCoder`). */
  readonly transformAlgorithm: SparseCodeAlgorithm
  /** Atoms per signal for `omp` and `lars` (`null`: scikit-learn's default). */
  readonly transformNonzeroCoefs: number | null
  /** The penalty or threshold (`null`: 1). */
  readonly transformAlpha: number | null
  /** Split the code into its positive and negative parts, `[n, 2 nComponents]`. */
  readonly splitSign: boolean
  readonly positiveCode: boolean
  /** The sweep cap of the coordinate descent coder. */
  readonly transformMaxIter: number
}

/** The parameters of a sparse coder: its dictionary and how it codes. */
export type SparseCoderParams = {
  readonly components: number[][]
  readonly transformAlgorithm: SparseCodeAlgorithm
}

/** The verbs of a sparse coder: the codes of new signals and the signals the codes write. */
export interface SparseCoderVerbs {
  readonly transform: (x: TensorLike) => Tensor
  readonly inverseTransform: (code: TensorLike) => Tensor
  readonly nFeaturesIn: number
}

/**
 * The coder of a set of hyperparameters (`transformAlgorithm`, `transformNonzeroCoefs`, `transformAlpha`,
 * `positiveCode`, `transformMaxIter`, the names the coder and the learners share) as options of {@link sparseCode}.
 *
 * @param h - The hyperparameters.
 * @returns The options, with the unset (`null`) ones left to the defaults.
 * @example
 * ```ts
 * import { coderOptions } from 'omai-engine/decomposition'
 * const h = { transformAlgorithm: 'lars', transformNonzeroCoefs: 3, transformAlpha: null, positiveCode: false, transformMaxIter: 100 } as const
 * coderOptions(h).nNonzeroCoefs // => 3
 * ```
 */
export function coderOptions(h: {
  readonly transformAlgorithm: SparseCodeAlgorithm
  readonly transformNonzeroCoefs: number | null
  readonly transformAlpha: number | null
  readonly positiveCode: boolean
  readonly transformMaxIter: number
}): SparseCodeOptions {
  return {
    algorithm: h.transformAlgorithm,
    ...(h.transformNonzeroCoefs === null ? {} : { nNonzeroCoefs: h.transformNonzeroCoefs }),
    ...(h.transformAlpha === null ? {} : { alpha: h.transformAlpha }),
    positive: h.positiveCode,
    maxIter: h.transformMaxIter,
  }
}

/**
 * The signals that codes write over a dictionary, `c D`, taking split codes back to their signed form first.
 *
 * @param code - Codes `[n, k]`, or `[n, 2 k]` when `split` (as {@link splitSign} makes).
 * @param dictionary - Atoms as rows `[k, nFeatures]`.
 * @param split - Whether the codes are sign-split.
 * @returns The signals `[n, nFeatures]`.
 * @example
 * ```ts
 * import { synthesise } from 'omai-engine/decomposition'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(synthesise([[2, 0, 0, 3]], [[1, 0], [0, 1]], true)) // => [[2, -3]]
 * ```
 */
export function synthesise(code: TensorLike, dictionary: readonly (readonly number[])[], split = false): Tensor {
  const k = dictionary.length
  const d = dictionary[0]?.length ?? 0
  const m = matrixView(code, 'synthesise', { nFeatures: split ? 2 * k : k })
  const out = new Float64Array(m.n * d)
  for (let i = 0; i < m.n; i++)
    for (let j = 0; j < k; j++) {
      const c = split ? (m.data[i * 2 * k + j] ?? 0) - (m.data[i * 2 * k + k + j] ?? 0) : (m.data[i * k + j] ?? 0)
      if (c === 0) continue
      for (let q = 0; q < d; q++) out[i * d + q] = (out[i * d + q] ?? 0) + c * (dictionary[j]?.[q] ?? 0)
    }
  return fromStorage(out, [m.n, d], 'float64')
}

/**
 * A sparse coder (scikit-learn's `SparseCoder`): a transformer over a fixed dictionary, with nothing to fit. Its
 * `transform` is {@link sparseCode} with the chosen coder (and `splitSign` when asked), and `inverseTransform` writes
 * the signals back.
 *
 * @param hyper - The dictionary, and optionally the coder and its sparsity or penalty.
 * @returns The estimator.
 * @concept sparse-coding
 * @index estimator Transformer coding signals sparsely over a fixed dictionary.
 * @example
 * ```ts
 * import { sparseCoder } from 'omai-engine/decomposition'
 * import { toArray } from 'omai-engine/tensor'
 * const coder = sparseCoder({ dictionary: [[1, 0], [0, 1]], transformNonzeroCoefs: 1 }).fit({ x: [[2, 0.1]] })
 * toArray(coder.transform([[2, 0.1]])) // => [[2, 0]]
 * ```
 */
export function sparseCoder(
  hyper: Pick<SparseCoderHyper, 'dictionary'> & Partial<SparseCoderHyper>,
): Estimator<SparseCoderHyper, Fitted<SparseCoderParams> & SparseCoderVerbs> {
  return defineEstimator<SparseCoderHyper, SparseCoderParams, SparseCoderVerbs>(
    {
      name: 'decomposition/sparse-coder',
      kind: 'transformer',
      hyperSpace: (h) =>
        dict({
          transformAlgorithm: choice(['omp', 'lars', 'lassoLars', 'lassoCd', 'threshold'] as const, {
            default: h.transformAlgorithm,
            label: 'coder',
          }),
          transformAlpha: real(0.001, 10, { log: true, default: h.transformAlpha ?? 1, label: 'penalty' }),
          transformNonzeroCoefs: int(1, 50, { default: h.transformNonzeroCoefs ?? 1, label: 'atoms per signal' }),
        }),
      fit: (_data, h) => ({ components: h.dictionary, transformAlgorithm: h.transformAlgorithm }),
      methods: (p, h) => ({
        transform: (x) => {
          const code = sparseCode(x, p.components, coderOptions(h))
          return h.splitSign ? splitSign(code) : code
        },
        inverseTransform: (code) => synthesise(code, p.components, h.splitSign),
        nFeaturesIn: p.components[0]?.length ?? 0,
      }),
    },
    {
      transformAlgorithm: 'omp',
      transformNonzeroCoefs: null,
      transformAlpha: null,
      splitSign: false,
      positiveCode: false,
      transformMaxIter: 1000,
      ...hyper,
    },
  )
}
