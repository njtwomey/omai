/**
 * Convolutional sparse coding and convolutional dictionary learning: a signal y of N samples written as a sum of K
 * short filters d_k of length L, each convolved with a sparse activation map z_k of length M = N - L + 1,
 * y ~ sum_k z_k * d_k (the full convolution, so every placed copy of a filter lies inside the signal; Grosse et al.
 * 2007, Zeiler et al. 2010). It is sparse coding over the N x KM dictionary of every shift of every filter, a banded
 * Toeplitz block per filter, applied by convolution and never formed.
 *
 * Coding (`convolutionalSparseCodeAlgorithm`) is the convolutional lasso 1/2 |y - sum_k z_k * d_k|^2 + lambda sum_k
 * |z_k|_1 by the library's proximal gradient (`proximalGradientAlgorithm` with `proxL1`, FISTA or ISTA): the gradient
 * with respect to z_k is minus the residual correlated with d_k over the valid lags, the adjoint of the convolution,
 * and the step is 1 / (1.1 L_hat) for L_hat a power-iteration estimate of the Lipschitz constant. Learning
 * (`convolutionalDictionaryLearningAlgorithm`) alternates a filter update for fixed maps, the least-squares filters
 * with |d_k| <= 1 by projected FISTA on their normal equations (Garcia-Cardona and Wohlberg 2018; the constraint fixes
 * the scale ambiguity (d, z) -> (c d, z / c) that would let the penalty shrink without limit), with coding warm-started
 * from the previous maps. The learner is a traced algorithm with a contract (`convolutionalDictionaryContract`): the
 * objective, its parts and the sparsity on the tape, and the filters, maps and reconstruction as probes at any step.
 * Fast solvers work in the Fourier domain with ADMM (Bristow et al. 2013, Wohlberg 2016); this module keeps the direct
 * convolutions, which suit short filters and signals of a few thousand samples.
 *
 * Filters are the rows of a K x L matrix and the maps the rows of a K x M matrix (unlike `dictionary.ts`, whose atoms
 * are also rows, a filter here is a short pattern slid along the signal). A signal is a vector, or a column `[N, 1]`
 * such as the `values` of a `TimeSeriesData`.
 *
 * @index structure Convolutional sparse coding and dictionary learning: filters, sparse maps, a signal.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { curveSet } from '../exposition/index.ts'
import { convolve, correlate } from '../numerics/convolution/index.ts'
import { daxpy, ddot, dgemv, dnrm2, dscal } from '../numerics/linalg/index.ts'
import { foldIn, type Key, permutation } from '../random/index.ts'
import {
  abs,
  asTensor,
  fromStorage,
  mul,
  sum,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  derived,
  type RunOptions,
  type RunStatus,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'
import { flatOf, matrixOf, vectorOf } from './_vector.ts'
import { analyticObjective } from './analytic.ts'
import { projectL2Ball, proxL1 } from './proximal.ts'
import { type ProximalGradientState, proximalGradientAlgorithm } from './splitting.ts'

/** A signal as plain numbers: a vector, or a column `[N, 1]` (the `values` of a one-channel `TimeSeriesData`). */
function seriesOf(y: TensorLike, name: string): Float64Array {
  const t = asTensor(y)
  const column = t.shape.length === 2 && t.shape[1] === 1
  if (t.shape.length !== 1 && !column)
    throw new OmaiError('shape', `${name}: the signal must be a vector or a column, got shape [${t.shape.join(', ')}]`)
  return Float64Array.from(toFlat(t))
}

/** Filters as a `K x L` matrix read once into plain numbers. */
function filtersOf(d: TensorLike, name: string): { data: Float64Array; K: number; L: number } {
  const { data, rows, cols } = matrixOf(d, name)
  if (rows === 0 || cols === 0) throw new OmaiError('shape', `${name}: there are no filters`)
  return { data, K: rows, L: cols }
}

/** The synthesis sum_k z_k * d_k of flat filters (K rows of L) and maps (K rows of M): N = M + L - 1 samples. */
function synthesis(d: Float64Array, z: Float64Array, K: number, L: number, M: number): Float64Array {
  let out: Float64Array = new Float64Array(M + L - 1)
  for (let k = 0; k < K; k++)
    out = daxpy(1, flatOf(convolve(z.subarray(k * M, (k + 1) * M), d.subarray(k * L, (k + 1) * L))), out)
  return out
}

/** The adjoint of the synthesis: r correlated with every filter over the valid lags, K maps of M joined row by row. */
function analysis(d: Float64Array, r: Float64Array, K: number, L: number, M: number): Float64Array {
  const out = new Float64Array(K * M)
  for (let k = 0; k < K; k++) out.set(flatOf(correlate(r, d.subarray(k * L, (k + 1) * L), 'valid')), k * M)
  return out
}

/**
 * The largest eigenvalue of the Gram operator of the synthesis, by power iteration from a fixed start (so it is
 * deterministic). Power iteration approaches it from below; callers add a margin.
 */
function synthesisLipschitz(d: Float64Array, K: number, L: number, M: number, iterations = 10): number {
  let v: Float64Array = Float64Array.from({ length: K * M }, (_, i) => 1 + 0.5 * Math.sin(i))
  let value = 0
  for (let it = 0; it < iterations; it++) {
    v = dscal(1 / (dnrm2(v) || 1), v)
    v = analysis(d, synthesis(d, v, K, L, M), K, L, M)
    value = dnrm2(v)
  }
  return value
}

/** The objective's parts at filters and maps: the squared error 1/2 |y - sum z * d|^2, the l1 norm and the non-zeros. */
function measure(
  signal: Float64Array,
  d: Float64Array,
  z: Float64Array,
  K: number,
  L: number,
  M: number,
): { fit: number; l1: number; nonzeros: number } {
  const r = daxpy(-1, synthesis(d, z, K, L, M), signal)
  let l1 = 0
  let nonzeros = 0
  for (const v of z) {
    l1 += Math.abs(v)
    if (v !== 0) nonzeros++
  }
  return { fit: 0.5 * ddot(r, r), l1, nonzeros }
}

/**
 * The synthesis of filters and activation maps: each filter placed at every position its map is non-zero, scaled by
 * the map's value, and the copies added (the full convolution, summed over filters).
 *
 * @param filters - The filters, `[K, L]`, one per row.
 * @param codes - The activation maps, `[K, M]`, one per row.
 * @returns The signal, `[M + L - 1]`.
 * @throws OmaiError with code `shape` unless there are as many maps as filters.
 * @concept convolutional-sparse-coding
 * @index function Sum of filters convolved with sparse activation maps: the signal they make.
 * @example
 * ```ts
 * import { convolutionalSynthesis } from 'omai-engine/optim'
 * import { toArray } from 'omai-engine/tensor'
 * // Two copies of one filter, the second negated and halved.
 * toArray(convolutionalSynthesis([[1, 2, 1]], [[0, 1, 0, 0, -0.5]])) // => [0, 1, 2, 1, -0.5, -1, -0.5]
 * ```
 */
export function convolutionalSynthesis(filters: TensorLike, codes: TensorLike): Tensor<'float64', [number]> {
  const d = filtersOf(filters, 'convolutionalSynthesis')
  const z = matrixOf(codes, 'convolutionalSynthesis')
  if (z.rows !== d.K)
    throw new OmaiError('shape', `convolutionalSynthesis: ${d.K} filters but ${z.rows} activation maps`)
  return vectorOf(synthesis(d.data, z.data, d.K, d.L, z.cols))
}

/**
 * The adjoint of the synthesis, `convolutionalSynthesis`'s transpose: a signal correlated with every filter over the
 * valid lags. It is the negative gradient of the squared error with respect to the maps, and the first thing a sparse
 * coder looks at (where does each filter match the signal?).
 *
 * @param filters - The filters, `[K, L]`.
 * @param signal - A signal, `[N]` with `N >= L` (a vector or a column).
 * @returns `K` maps of `M = N - L + 1` values, `[K, M]`.
 * @throws OmaiError with code `shape` when the filters are longer than the signal.
 * @concept convolutional-sparse-coding
 * @example
 * ```ts
 * import { convolutionalAnalysis } from 'omai-engine/optim'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(convolutionalAnalysis([[1, 2, 1]], [0, 1, 2, 1, 0])) // => [[4, 6, 4]]
 * ```
 */
export function convolutionalAnalysis(filters: TensorLike, signal: TensorLike): Tensor<'float64', [number, number]> {
  const d = filtersOf(filters, 'convolutionalAnalysis')
  const y = seriesOf(signal, 'convolutionalAnalysis')
  if (d.L > y.length)
    throw new OmaiError('shape', `convolutionalAnalysis: filters of length ${d.L} are longer than the signal`)
  const M = y.length - d.L + 1
  return fromStorage(analysis(d.data, y, d.K, d.L, M), [d.K, M], 'float64')
}

/** Options of {@link convolutionalSparseCodeAlgorithm} and {@link convolutionalSparseCode}. */
export interface ConvolutionalCodeOptions {
  /** The weight `lambda > 0` of the l1 penalty: larger gives fewer, larger activations and a looser fit. */
  readonly lambda: number
  /** The starting maps, `[K, M]` (a warm start); default zero. */
  readonly init?: TensorLike
  /** FISTA's extrapolation (default true); false runs ISTA. */
  readonly accelerate?: boolean
  /** Stop once the infinity norm of the gradient mapping is at most this (default 1e-7). */
  readonly gradTol?: number
}

/**
 * Convolutional sparse coding as a traced algorithm: proximal gradient (FISTA, or ISTA) on
 * `1/2 |y - sum_k z_k * d_k|^2 + lambda sum_k |z_k|_1` over the maps, with step `1 / (1.1 L_hat)`, `L_hat` the
 * power-iteration estimate of the gradient's Lipschitz constant. The state and the tape are those of
 * `proximalGradientAlgorithm`, the maps' rows joined in `x` (`K M` values, map `k` at `[kM, (k + 1)M)`) and the
 * objective, penalty included, in `value`.
 *
 * @param filters - The filters, `[K, L]`, one per row, `L <= N`.
 * @param signal - The signal, `[N]` (a vector or a column).
 * @param options - The penalty, the warm start, the acceleration and the tolerance.
 * @returns The algorithm.
 * @concept convolutional-sparse-coding
 * @index algorithm Convolutional lasso over filters by proximal gradient, FISTA or ISTA.
 * @example
 * ```ts
 * import { convolutionalSparseCodeAlgorithm } from 'omai-engine/optim'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const y = [0, 0, 1, 2, 1, 0, 0, 0, -1, -2, -1, 0]
 * const alg = convolutionalSparseCodeAlgorithm([[1, 2, 1]], y, { lambda: 0.1 })
 * const falls = [1, 10, 100].map((t) => runAlgorithm(alg, { maxSteps: t }).state.value)
 * falls[1] < falls[0] && falls[2] <= falls[1] // => true
 * ```
 */
export function convolutionalSparseCodeAlgorithm(
  filters: TensorLike,
  signal: TensorLike,
  options: ConvolutionalCodeOptions,
): Algorithm<ProximalGradientState> {
  const where = 'convolutionalSparseCode'
  const { lambda, accelerate = true, gradTol = 1e-7 } = options
  if (!(lambda > 0)) throw new OmaiError('value', `${where}: lambda must be positive, got ${lambda}`)
  const { data: d, K, L } = filtersOf(filters, where)
  const y = seriesOf(signal, where)
  if (L > y.length)
    throw new OmaiError('shape', `${where}: filters of length ${L} are longer than the signal (${y.length})`)
  const M = y.length - L + 1
  let x0: Float64Array = new Float64Array(K * M)
  if (options.init !== undefined) {
    const z0 = matrixOf(options.init, where)
    if (z0.rows !== K || z0.cols !== M)
      throw new OmaiError('shape', `${where}: init must be ${K} x ${M}, got ${z0.rows} x ${z0.cols}`)
    x0 = z0.data
  }
  const residual = (x: Float64Array): Float64Array => daxpy(-1, synthesis(d, x, K, L, M), y)
  const smooth = analyticObjective({
    value: (x) => {
      const r = residual(x)
      return 0.5 * ddot(r, r)
    },
    gradient: (x) => dscal(-1, analysis(d, residual(x), K, L, M)),
  })
  // With every filter zero the gradient is zero and any step will do.
  const lipschitz = synthesisLipschitz(d, K, L, M)
  const inner = proximalGradientAlgorithm(smooth, (x, t) => proxL1(x, t * lambda), x0, {
    stepSize: lipschitz > 1e-12 ? 1 / (1.1 * lipschitz) : 1,
    accelerate,
    gradTol,
    g: (x: Value) => mul(sum(abs(x)), lambda),
  })
  return {
    ...inner,
    name: where,
    config: { filters: K, length: L, signal: y.length, lambda, accelerate, gradTol },
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ProximalGradientState,
  }
}

/** The result of {@link convolutionalSparseCode}. */
export interface ConvolutionalCode {
  /** The activation maps, `[K, M]`. */
  readonly codes: Tensor<'float64', [number, number]>
  /** `y - sum_k z_k * d_k`, `[N]`. */
  readonly residual: Tensor<'float64', [number]>
  /** The residual's Euclidean norm. */
  readonly residualNorm: number
  /** The objective at the maps. */
  readonly objective: number
  /** Non-zero activations in all maps. */
  readonly nonzeros: number
  /** Proximal gradient steps taken. */
  readonly steps: number
  /** Whether the tolerance was met. */
  readonly converged: boolean
}

/**
 * Convolutional sparse coding run to the tolerance or for at most `maxSteps` steps: the one-call form of
 * `convolutionalSparseCodeAlgorithm`.
 *
 * @param filters - The filters, `[K, L]`, one per row.
 * @param signal - The signal, `[N]`.
 * @param options - The options of `convolutionalSparseCodeAlgorithm` and the step budget (`maxSteps`, default 200).
 * @returns The maps, the residual and its norm, the objective, the number of non-zeros, the steps and whether the
 *   tolerance was met.
 * @concept convolutional-sparse-coding
 * @index function Sparse activations of known filters in a signal: where does each occur?
 * @example
 * ```ts
 * import { convolutionalSparseCode } from 'omai-engine/optim'
 * import { toArray } from 'omai-engine/tensor'
 * // Find where a pulse (1, 2, 1) occurs, upright at 2 and upside down at 8.
 * const y = [0, 0, 1, 2, 1, 0, 0, 0, -1, -2, -1, 0]
 * const { codes, nonzeros } = convolutionalSparseCode([[1, 2, 1]], y, { lambda: 0.1, maxSteps: 2000, gradTol: 1e-12 })
 * nonzeros // => 2
 * toArray(codes)[0]?.map((v) => Math.round(v * 1e6) / 1e6) // => [0, 0, 0.983333, 0, 0, 0, 0, 0, -0.983333, 0]
 * ```
 */
export function convolutionalSparseCode(
  filters: TensorLike,
  signal: TensorLike,
  options: ConvolutionalCodeOptions & { readonly maxSteps?: number },
): ConvolutionalCode {
  const { data: d, K, L } = filtersOf(filters, 'convolutionalSparseCode')
  const y = seriesOf(signal, 'convolutionalSparseCode')
  const run = runAlgorithm(convolutionalSparseCodeAlgorithm(filters, signal, options), {
    maxSteps: options.maxSteps ?? 200,
  })
  const M = y.length - L + 1
  const z = Float64Array.from(run.state.x)
  const r = daxpy(-1, synthesis(d, z, K, L, M), y)
  return {
    codes: fromStorage(z, [K, M], 'float64'),
    residual: vectorOf(r),
    residualNorm: dnrm2(r),
    objective: run.state.value,
    nonzeros: z.reduce((n, v) => n + (v !== 0 ? 1 : 0), 0),
    steps: run.steps,
    converged: run.status === 'converged',
  }
}

/**
 * The least-squares filters for fixed maps subject to `|d_k| <= 1`, by projected FISTA from `d0`. The Gram entry for
 * taps `(a, l)` and `(b, l')` is the correlation of `z_a` and `z_b` at lag `l - l'`, and the right-hand side is `y`
 * correlated with each map, so the design matrix (every map shifted by every tap) is never formed.
 */
function updateFilters(
  y: Float64Array,
  z: Float64Array,
  d0: Float64Array,
  K: number,
  L: number,
  M: number,
  steps: number,
): Float64Array {
  const P = K * L
  const G = new Float64Array(P * P)
  for (let a = 0; a < K; a++)
    for (let b = 0; b < K; b++) {
      // full[c] is the correlation of z_a with z_b at lag c - (M - 1).
      const full = flatOf(correlate(z.subarray(a * M, (a + 1) * M), z.subarray(b * M, (b + 1) * M), 'full'))
      for (let l = 0; l < L; l++)
        for (let lp = 0; lp < L; lp++) G[(a * L + l) * P + b * L + lp] = full[M - 1 + lp - l] ?? 0
    }
  const rhs = new Float64Array(P)
  for (let k = 0; k < K; k++) rhs.set(flatOf(correlate(y, z.subarray(k * M, (k + 1) * M), 'valid')), k * L)
  let bound = Number.MIN_VALUE
  for (let i = 0; i < P; i++) {
    let s = 0
    for (let j = 0; j < P; j++) s += Math.abs(G[i * P + j] ?? 0)
    bound = Math.max(bound, s)
  }
  const normal = analyticObjective({
    value: (v) => 0.5 * ddot(v, dgemv(G, P, P, v)) - ddot(v, rhs),
    gradient: (v) => daxpy(-1, rhs, dgemv(G, P, P, v)),
  })
  // The projection onto the product of unit balls: a filter that is too long is scaled back to unit norm.
  const balls = (v: Value): Value => {
    const flat = flatOf(v)
    const out = new Float64Array(P)
    for (let k = 0; k < K; k++) out.set(flatOf(projectL2Ball(flat.subarray(k * L, (k + 1) * L), 1)), k * L)
    return vectorOf(out)
  }
  // With every map zero the normal equations are empty and the filters stay where they are.
  if (bound < 1e-200) return d0
  const inner = proximalGradientAlgorithm(normal, balls, d0, { stepSize: 1 / bound, accelerate: true, gradTol: 1e-9 })
  return Float64Array.from(runAlgorithm(inner, { maxSteps: steps }).state.x)
}

/** Options of {@link convolutionalDictionaryLearningAlgorithm}. */
export interface ConvolutionalDictionaryOptions {
  /** The number of filters `K`. */
  readonly filters: number
  /** The filter length `L`, at most the signal's length. */
  readonly length: number
  /** The penalty `lambda > 0` of the coding. */
  readonly lambda: number
  /** Proximal gradient steps of coding per round, warm-started from the previous maps (default 25). */
  readonly codeSteps?: number
  /** Projected FISTA steps of the filter update per round (default 60). */
  readonly filterSteps?: number
  /** The initial filters, `[K, L]` (scaled to unit norm); without them, the stretches `initialisation` picks. */
  readonly init?: TensorLike
  /**
   * Where the initial filters come from when `init` is not given: `random` (default; `K` stretches of the signal at
   * random positions, less their means, as in the papers) or `energy` (the `K` non-overlapping stretches of greatest
   * energy, as they are, so a filter starts on a whole pattern rather than on a piece of one, and no key is needed;
   * take a baseline out of the signal first).
   */
  readonly initialisation?: 'random' | 'energy'
  /** Stop when the objective changes by less than this fraction of itself in one round (default 1e-6). */
  readonly tol?: number
}

/** The state of {@link convolutionalDictionaryLearningAlgorithm}. */
export interface ConvolutionalDictionaryState {
  /** Rounds taken: a filter update then a coding. */
  readonly t: number
  /** The filters, `K L` values (row `k` at `[kL, (k + 1)L)`), each of norm at most 1. */
  readonly filters: Float64Array
  /** The activation maps, `K M` values (row `k` at `[kM, (k + 1)M)`). */
  readonly codes: Float64Array
  /** `1/2 |y - sum_k z_k * d_k|^2 + lambda sum_k |z_k|_1`. */
  readonly objective: number
  /** The squared-error part, `1/2 |y - sum_k z_k * d_k|^2`. */
  readonly fit: number
  /** Non-zero activations in all maps. */
  readonly nonzeros: number
  /** The objective's relative change in the last round (infinite before the first). */
  readonly change: number
}

/** The paths of {@link convolutionalDictionaryContract}. */
export type ConvolutionalPath =
  | 'cdl/objective'
  | 'cdl/fit'
  | 'cdl/nonzeros'
  | 'cdl/change'
  | 'cdl/filters'
  | 'cdl/codes'
  | 'cdl/reconstruction'

const CDL_SCHEMA = channels({
  metrics: {
    objective: { description: 'squared error plus lambda times the l1 norm of the maps', direction: 'lower' },
    fit: { description: 'half the squared error of the reconstruction', direction: 'lower' },
    nonzeros: { description: 'non-zero activations in all maps' },
    change: { description: 'relative change of the objective in the round' },
  },
  tensors: {
    filters: { role: 'param', description: 'the filters, one per row' },
    codes: { role: 'state', description: 'the activation maps, one per row' },
  },
})

/** The checked shape of a learner's options and signal. */
function learnerShape(
  signal: Float64Array,
  options: ConvolutionalDictionaryOptions,
): { K: number; L: number; M: number } {
  const where = 'convolutionalDictionaryLearning'
  const { filters: K, length: L, lambda } = options
  if (!(Number.isInteger(K) && K >= 1))
    throw new OmaiError('value', `${where}: filters must be a positive integer, got ${K}`)
  if (!(Number.isInteger(L) && L >= 1 && L <= signal.length))
    throw new OmaiError('value', `${where}: length must be an integer from 1 to ${signal.length}, got ${L}`)
  if (!(lambda > 0)) throw new OmaiError('value', `${where}: lambda must be positive, got ${lambda}`)
  return { K, L, M: signal.length - L + 1 }
}

/**
 * The contract of the learner: its objective, the squared error, the number of activations and the objective's
 * change as scalars logged each round, and three probes at any step, each an exposition record of curves: the
 * filters against the tap (`cdl/filters`), the activation maps against the position (`cdl/codes`) and the signal with
 * its reconstruction (`cdl/reconstruction`).
 *
 * @param signal - The signal the learner runs on (for the reconstruction probe).
 * @param options - The learner's options (the filter length and count).
 * @returns The contract.
 * @concept log-contract
 * @example
 * ```ts
 * import { convolutionalDictionaryContract } from 'omai-engine/optim'
 * Object.keys(convolutionalDictionaryContract([0, 1, 2, 1, 0, 0, 1, 2, 1, 0], { filters: 1, length: 3, lambda: 0.1 })).length // => 7
 * ```
 */
export function convolutionalDictionaryContract(
  signal: TensorLike,
  options: ConvolutionalDictionaryOptions,
): Contract<ConvolutionalDictionaryState, ConvolutionalPath> {
  const y = seriesOf(signal, 'convolutionalDictionaryContract')
  const { K, L, M } = learnerShape(y, options)
  return defineContract({
    'cdl/objective': scalar({ description: 'the squared error plus lambda times the l1 norm of the maps' }),
    'cdl/fit': scalar({ description: 'half the squared error of the reconstruction' }),
    'cdl/nonzeros': scalar({ description: 'the number of non-zero activations in all maps' }),
    'cdl/change': scalar({ description: 'the objective’s relative change in the round' }),
    'cdl/filters': derived<ConvolutionalDictionaryState>({
      description: 'the filters against the tap, one curve each',
      compute: (s) =>
        curveSet({
          x: Array.from({ length: L }, (_, l) => l),
          xLabel: 'tap',
          series: Array.from({ length: K }, (_, k) => ({
            name: `filter ${k + 1}`,
            y: Array.from(s.filters.subarray(k * L, (k + 1) * L)),
          })),
        }),
    }),
    'cdl/codes': derived<ConvolutionalDictionaryState>({
      description: 'the activation maps against the position, one curve each',
      compute: (s) =>
        curveSet({
          x: Array.from({ length: M }, (_, m) => m),
          xLabel: 'position',
          series: Array.from({ length: K }, (_, k) => ({
            name: `map ${k + 1}`,
            y: Array.from(s.codes.subarray(k * M, (k + 1) * M)),
          })),
        }),
    }),
    'cdl/reconstruction': derived<ConvolutionalDictionaryState>({
      description: 'the signal and the sum of the filters placed where the maps are non-zero',
      compute: (s) =>
        curveSet({
          x: Array.from({ length: y.length }, (_, n) => n),
          xLabel: 'sample',
          series: [
            { name: 'signal', y: Array.from(y) },
            { name: 'reconstruction', y: Array.from(synthesis(s.filters, s.codes, K, L, M)) },
          ],
        }),
    }),
  })
}

/**
 * Convolutional dictionary learning as a traced algorithm. The initial state codes the signal over the initial
 * filters; each step updates the filters for the current maps (least squares with `|d_k| <= 1`) and then recodes,
 * warm-started. Neither stage increases the objective, up to the coding's finite step budget. The objective is not
 * convex in filters and maps together, so the result depends on the start: a filter that starts as a misaligned
 * stretch of a pattern can settle on a piece of it, coded by two neighbouring activations. A filter a few samples
 * longer than the patterns avoids this, and holds each pattern at some shift within its window (a shifted filter
 * with a shifted map fits equally well). The state serialises, so a run resumes and seeks; `init` takes its random
 * filters from the run's key.
 *
 * The signal is a plain series, so a window of a recording works as it is: the `values` of a `TimeSeriesData`, sliced
 * with `toFlat(values).slice(from, to)`. For an electrocardiogram (`electrocardiogram` of `omai-engine/datasets`, 360
 * Hz, a remote asset, `await load(electrocardiogram, { cache })`), a couple of seconds and filters about a beat
 * long learn heartbeat-shaped pulses:
 *
 * ```ts
 * const ecg = await load(electrocardiogram, { cache })
 * const y = toFlat(ecg.values).slice(0, 1080) // three seconds
 * const { filters, codes } = convolutionalDictionaryLearning(y, { filters: 2, length: 90, lambda: 0.3, maxSteps: 40 })
 * ```
 *
 * The penalty scales with the signal: the maps are all zero once `lambda` reaches the largest correlation of a
 * filter with the signal (`convolutionalAnalysis`), so for millivolts start well below it.
 *
 * Remove the baseline first (subtract a moving median, or the mean of the window) when it wanders: a filter is
 * constrained to unit norm and a drift comparable with the beats takes filters for itself. `initialisation:
 * 'energy'` starts the filters on the most energetic stretches, the beats, instead of at random.
 *
 * @param signal - The signal, `[N]` (a vector or a column).
 * @param options - The number and length of the filters, the penalty, the step budgets, the start and the tolerance.
 * @returns The algorithm, typed by the paths of its contract.
 * @concept convolutional-sparse-coding
 * @concept dictionary-learning
 * @index algorithm Learn shift-invariant filters and their sparse activations from one signal.
 * @example
 * ```ts
 * import { convolutionalDictionaryLearningAlgorithm } from 'omai-engine/optim'
 * import { runAlgorithm } from 'omai-engine/trace'
 * // Six copies of the pulse (1, 3, 1) at irregular places and scales.
 * const y = new Array(60).fill(0)
 * for (const [t, a] of [[3, 1], [11, -0.5], [19, 2], [30, 1.5], [41, -1], [50, 0.8]] as const)
 *   [1, 3, 1].forEach((v, l) => { y[t + l] += a * v })
 * const alg = convolutionalDictionaryLearningAlgorithm(y, { filters: 1, length: 6, lambda: 0.1 })
 * const objective = (rounds: number): number => runAlgorithm(alg, { maxSteps: rounds }).state.objective
 * objective(20) < objective(0) // => true
 * ```
 *
 * @example A window of a time series
 * ```ts
 * import { generate } from 'omai-engine/data'
 * import { pulseTrain } from 'omai-engine/datasets'
 * import { convolutionalDictionaryLearning } from 'omai-engine/optim'
 * import { toFlat } from 'omai-engine/tensor'
 * // `values` of a TimeSeriesData is a column; a window of it is a plain series, as the window of a recording would be.
 * const series = generate(pulseTrain, { knobs: { n: 400, pulses: 1, length: 10, occurrences: 8 } }).data
 * const window = toFlat(series.values).slice(0, 200)
 * const { filters } = convolutionalDictionaryLearning(window, { filters: 1, length: 14, lambda: 0.1, initialisation: 'energy' })
 * filters.shape // => [1, 14]
 * ```
 */
export function convolutionalDictionaryLearningAlgorithm(
  signal: TensorLike,
  options: ConvolutionalDictionaryOptions,
): Algorithm<ConvolutionalDictionaryState, ConvolutionalPath> {
  const where = 'convolutionalDictionaryLearning'
  const y = seriesOf(signal, where)
  const { K, L, M } = learnerShape(y, options)
  const { lambda, codeSteps = 25, filterSteps = 60, tol = 1e-6 } = options
  const code = (d: Float64Array, z: Float64Array): Float64Array =>
    Float64Array.from(
      runAlgorithm(
        convolutionalSparseCodeAlgorithm(fromStorage(d, [K, L], 'float64'), y, {
          lambda,
          init: fromStorage(z, [K, M], 'float64'),
        }),
        { maxSteps: codeSteps },
      ).state.x,
    )
  const state = (t: number, d: Float64Array, z: Float64Array, previous: number): ConvolutionalDictionaryState => {
    const { fit, l1, nonzeros } = measure(y, d, z, K, L, M)
    const objective = fit + lambda * l1
    return {
      t,
      filters: d,
      codes: z,
      objective,
      fit,
      nonzeros,
      change: Math.abs(previous - objective) / Math.max(previous, Number.MIN_VALUE),
    }
  }
  const unit = (d: Float64Array): Float64Array => {
    const out = Float64Array.from(d)
    for (let k = 0; k < K; k++) {
      const row = out.subarray(k * L, (k + 1) * L)
      const n = dnrm2(row)
      if (n > 0) row.set(dscal(1 / n, row))
    }
    return out
  }
  const initialFilters = (key: Key): Float64Array => {
    if (options.init !== undefined) {
      const r = matrixOf(options.init, `${where}: init`)
      if (r.rows !== K || r.cols !== L)
        throw new OmaiError('shape', `${where}: init must be ${K} x ${L}, got ${r.rows} x ${r.cols}`)
      return r.data
    }
    // K stretches of the signal, less their means, skipping flat ones: at random positions, or the most energetic.
    const energetic = options.initialisation === 'energy'
    const stretch = (at: number): Float64Array => {
      const w = y.slice(at, at + L)
      const mean = energetic ? 0 : w.reduce((a, b) => a + b, 0) / L
      for (let l = 0; l < L; l++) w[l] = (w[l] ?? 0) - mean
      return w
    }
    const d = new Float64Array(K * L)
    let found = 0
    if (energetic) {
      const energy = Float64Array.from({ length: M }, (_, at) => ddot(stretch(at), stretch(at)))
      const taken: number[] = []
      while (found < K) {
        let best = -1
        for (let at = 0; at < M; at++)
          if ((best < 0 || (energy[at] ?? 0) > (energy[best] ?? 0)) && taken.every((u) => Math.abs(u - at) >= L))
            best = at
        if (best < 0 || (energy[best] ?? 0) <= 1e-12) break
        taken.push(best)
        d.set(stretch(best), found++ * L)
      }
    } else {
      const order = permutation(foldIn(key, 0), M).data
      for (let i = 0; i < M && found < K; i++) {
        const w = stretch(order[i] ?? 0)
        if (ddot(w, w) > 1e-12) d.set(w, found++ * L)
      }
    }
    for (; found < K; found++) d[found * L + (found % L)] = 1
    return d
  }
  const config: Json = { filters: K, length: L, lambda, codeSteps, filterSteps, tol, signal: y.length }
  return {
    name: 'convolutionalDictionaryLearning',
    config,
    schema: CDL_SCHEMA,
    init: (key) => {
      const d = unit(initialFilters(key))
      const z = code(d, new Float64Array(K * M))
      return { ...state(0, d, z, Number.POSITIVE_INFINITY), change: Number.POSITIVE_INFINITY }
    },
    step: (s, ctx) => {
      const d = updateFilters(y, s.codes, s.filters, K, L, M, filterSteps)
      const next = state(s.t + 1, d, code(d, s.codes), s.objective)
      ctx.log('cdl/objective', next.objective)
      ctx.log('cdl/fit', next.fit)
      ctx.log('cdl/nonzeros', next.nonzeros)
      ctx.log('cdl/change', next.change)
      return next
    },
    done: (s) => (!Number.isFinite(s.objective) ? 'diverged' : s.change < tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        objective: s.objective,
        fit: s.fit,
        nonzeros: s.nonzeros,
        change: Number.isFinite(s.change) ? s.change : 0,
      },
      tensors: {
        filters: { role: 'param', value: s.filters, shape: [K, L] },
        codes: { role: 'state', value: s.codes, shape: [K, M] },
      },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as ConvolutionalDictionaryState,
  }
}

/** Options of {@link convolutionalDictionaryLearning}: those of the algorithm, the round budget and the run's key. */
export interface ConvolutionalDictionaryRunOptions
  extends ConvolutionalDictionaryOptions,
    Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** The most rounds (default 20). */
  readonly maxSteps?: number
}

/** The result of {@link convolutionalDictionaryLearning}. */
export interface ConvolutionalDictionaryResult {
  /** The filters, `[K, L]`, each of norm at most 1. */
  readonly filters: Tensor<'float64', [number, number]>
  /** The activation maps, `[K, M]`. */
  readonly codes: Tensor<'float64', [number, number]>
  readonly objective: number
  /** Rounds taken. */
  readonly steps: number
  readonly status: RunStatus
  /** The objective at the start and after each round. */
  readonly history: readonly number[]
}

/**
 * Convolutional dictionary learning run to the tolerance or for at most `maxSteps` rounds: the one-call form of
 * `convolutionalDictionaryLearningAlgorithm`.
 *
 * @param signal - The signal, `[N]` (a vector or a column; see the algorithm for an electrocardiogram).
 * @param options - The options of the algorithm, the round budget (`maxSteps`, default 20) and the run's `key`.
 * @returns The filters, the maps, the objective, the rounds taken, the status and the objective's history.
 * @concept convolutional-sparse-coding
 * @concept dictionary-learning
 * @index function Learn filters and sparse activation maps from a signal.
 * @example
 * ```ts
 * import { convolutionalDictionaryLearning } from 'omai-engine/optim'
 * import { toArray } from 'omai-engine/tensor'
 * // Six copies of the pulse (1, 3, 1) at irregular places and scales. The learnt filter is the pulse at unit norm,
 * // (0.30, 0.90, 0.30), at some shift within its six samples, and its map is non-zero once per copy.
 * const y = new Array(60).fill(0)
 * for (const [t, a] of [[3, 1], [11, -0.5], [19, 2], [30, 1.5], [41, -1], [50, 0.8]] as const)
 *   [1, 3, 1].forEach((v, l) => { y[t + l] += a * v })
 * const { filters, codes } = convolutionalDictionaryLearning(y, {
 *   filters: 1, length: 6, lambda: 0.1, maxSteps: 60, initialisation: 'energy',
 * })
 * toArray(filters)[0]?.map((v) => Math.round(v * 100) / 100) // => [0, 0, 0, 0.3, 0.9, 0.3]
 * toArray(codes)[0]?.filter((v) => v !== 0).length // => 6
 * ```
 */
export function convolutionalDictionaryLearning(
  signal: TensorLike,
  options: ConvolutionalDictionaryRunOptions,
): ConvolutionalDictionaryResult {
  const { maxSteps, filters: K, length: L, lambda, codeSteps, filterSteps, init, initialisation, tol, ...run } = options
  const algorithm = convolutionalDictionaryLearningAlgorithm(signal, {
    filters: K,
    length: L,
    lambda,
    ...(codeSteps === undefined ? {} : { codeSteps }),
    ...(filterSteps === undefined ? {} : { filterSteps }),
    ...(init === undefined ? {} : { init }),
    ...(initialisation === undefined ? {} : { initialisation }),
    ...(tol === undefined ? {} : { tol }),
  })
  const history: number[] = []
  const tracked: Algorithm<ConvolutionalDictionaryState, ConvolutionalPath> = {
    ...algorithm,
    init: (key) => {
      const s = algorithm.init(key)
      history.push(s.objective)
      return s
    },
    step: (s, ctx) => {
      const next = algorithm.step(s, ctx)
      history.push(next.objective)
      return next
    },
  }
  const result = runAlgorithm(tracked, { ...run, maxSteps: maxSteps ?? 20 })
  const s = result.state
  const M = s.codes.length / K
  return {
    filters: fromStorage(Float64Array.from(s.filters), [K, L], 'float64'),
    codes: fromStorage(Float64Array.from(s.codes), [K, M], 'float64'),
    objective: s.objective,
    steps: result.steps,
    status: result.status,
    history,
  }
}

/** How well one planted filter is found among the learnt ones: see {@link matchFilters}. */
export interface FilterMatch {
  /** The index of the planted filter. */
  readonly planted: number
  /** The index of the learnt filter that matches it best. */
  readonly learnt: number
  /** The largest absolute normalised cross-correlation over every shift (1 is a perfect match up to shift and sign). */
  readonly score: number
  /** The shift of the planted filter within the learnt one at that score (negative: it starts before the window). */
  readonly shift: number
  /** The sign of the match: -1 when the learnt filter is the planted one negated. */
  readonly sign: 1 | -1
}

/**
 * Compare learnt filters with known ones up to shift and sign, the two ambiguities of convolutional dictionary
 * learning (a filter shifted by `s` with its map shifted by `-s`, or negated with its map negated, fits as well). For
 * every planted filter, the learnt filter with the largest absolute normalised cross-correlation over all shifts,
 * the shifts that hang over an end included (the overlap is what is compared, and the score is at most 1).
 *
 * @param learnt - The learnt filters, `[K, L]`.
 * @param planted - The known filters, `[P, L']` (their length may differ).
 * @returns One match per planted filter.
 * @concept convolutional-sparse-coding
 * @example
 * ```ts
 * import { matchFilters } from 'omai-engine/optim'
 * // The pulse (1, 3, 1), negated and shifted by one inside a window of five.
 * const [m] = matchFilters([[0, -1, -3, -1, 0]], [[1, 3, 1]])
 * [m?.sign, m?.shift, Math.round((m?.score ?? 0) * 1e6) / 1e6] // => [-1, 1, 1]
 * ```
 */
export function matchFilters(learnt: TensorLike, planted: TensorLike): FilterMatch[] {
  const a = filtersOf(learnt, 'matchFilters')
  const b = filtersOf(planted, 'matchFilters')
  return Array.from({ length: b.K }, (_, p) => {
    const target = b.data.subarray(p * b.L, (p + 1) * b.L)
    let best: FilterMatch = { planted: p, learnt: 0, score: -1, shift: 0, sign: 1 }
    for (let k = 0; k < a.K; k++) {
      const row = a.data.subarray(k * a.L, (k + 1) * a.L)
      const scale = dnrm2(row) * dnrm2(target) || 1
      // full[c] pairs the target's tap j with the learnt filter's tap j + c - (b.L - 1).
      const full = flatOf(correlate(row, target, 'full'))
      for (let c = 0; c < full.length; c++) {
        const score = Math.abs(full[c] ?? 0) / scale
        if (score > best.score + 1e-12)
          best = { planted: p, learnt: k, score, shift: c - (b.L - 1), sign: (full[c] ?? 0) < 0 ? -1 : 1 }
      }
    }
    return best
  })
}
