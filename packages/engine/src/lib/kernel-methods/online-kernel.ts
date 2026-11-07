/**
 * Online kernel regression, one observation per step: kernel recursive least squares (Engel, Mannor and Meir) with the
 * approximate-linear-dependence test or a coherence test deciding which observations enter the dictionary, its
 * sliding-window form (Van Vaerenbergh et al.) that keeps the exact ridge solution on the last observations through
 * rank-one updates of the inverse, and NORMA (Kivinen, Smola and Williamson), stochastic gradient descent in the
 * reproducing kernel Hilbert space with weight decay and a truncated expansion. Each is an `Algorithm` over the stream
 * with serialisable state and an estimator on the iterative base, so `fittedAt(t)` is the predictor after `t` observations.
 */

import { MdmlError } from '../base/index.ts'
import { type Distribution, normal } from '../dists/index.ts'
import {
  type Kernel,
  type KernelChoice,
  kernelMatrix,
  kernelSpace,
  type PairwiseKernelMetric,
  type ResolvedKernel,
  resolveKernel,
} from '../kernels/index.ts'
import { type IterativeEstimator, iterative, type LearnData, matrixView, targetVector } from '../learn/index.ts'
import { r2Score } from '../metrics/index.ts'
import { ddot } from '../numerics/linalg/index.ts'
import { choice, dict, int, real } from '../space/index.ts'
import { fromStorage, type Tensor, type TensorLike } from '../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  derived,
  Log,
  stateFromJson,
  stateToJson,
} from '../trace/index.ts'

/** How KRLS decides which observations enter the dictionary. */
export type KrlsCriterion = 'ald' | 'coherence' | 'sliding-window'

/** The hyperparameters of {@link krls}. */
export interface OnlineKernelHyper {
  /**
   * `ald`: add an observation when its squared distance in feature space to the span of the dictionary exceeds
   * `aldThreshold`; `coherence`: add it when its largest normalised kernel value with the dictionary is below
   * `coherence`; `sliding-window`: keep the last `budget` observations, all of them.
   */
  readonly criterion: KrlsCriterion
  /** The most dictionary centres (the window length for `sliding-window`; default 64). */
  readonly budget: number
  /** The ALD threshold nu: the smallest squared residual of the projection that makes an observation novel (default 1e-2). */
  readonly aldThreshold: number
  /** The coherence threshold mu in (0, 1]: novel when the largest normalised kernel value is below it (default 0.9). */
  readonly coherence: number
  /** The ridge regularisation of the `sliding-window` solution, `(K + lambda I)^(-1) y` (default 1e-3). */
  readonly regularisation: number
  readonly kernel: PairwiseKernelMetric | Kernel
  readonly gamma: number | 'scale' | 'auto'
  readonly degree: number
  readonly coef0: number
}

/**
 * The path {@link krls} adds to the base's: the dictionary so far (the algorithm logs through the solver's channels
 * alone).
 *
 * @example
 * ```ts
 * import type { KrlsPath } from 'mdml-engine/kernel-methods'
 * const path: KrlsPath = 'model/dictionary'
 * path // => 'model/dictionary'
 * ```
 */
export type KrlsPath = 'model/dictionary'

/** The state of KRLS after `step` observations: plain JSON. */
export type KrlsState = {
  /** Observations absorbed. */
  readonly step: number
  readonly nFeatures: number
  /** The dictionary (or window) `[m][d]`. */
  readonly dictionary: number[][]
  /** The kernel of each centre with itself. */
  readonly dictionaryDiagonal: number[]
  /** The targets of the window (sliding-window only). */
  readonly targets: number[]
  /** The expansion coefficients over the dictionary. */
  readonly alpha: number[]
  /** `K^(-1)` of the dictionary (of `K + lambda I` for the sliding window), `[m][m]`. */
  readonly kInv: number[][]
  /** The covariance-like matrix `P` of the reduced-order recursive least squares, `[m][m]`. */
  readonly p: number[][]
  /** The novelty score of the last observation (ALD residual, or one minus the coherence). */
  readonly novelty: number
  /** The one-step-ahead error `y - f(x)` of the last observation, before it was absorbed. */
  readonly error: number
  /** Whether the last observation entered the dictionary. */
  readonly added: boolean
  /** The sum of squared one-step-ahead errors. */
  readonly sse: number
}

/** The fitted parameters of {@link krls}: plain JSON. */
export type OnlineKernelParams = {
  readonly kernel: ResolvedKernel
  /** The dictionary centres `[m][d]`. */
  readonly dictionary: number[][]
  /** Their coefficients `[m]`. */
  readonly dualCoef: number[]
  readonly nFeatures: number
  readonly nObservations: number
  /** The mean squared one-step-ahead (prequential) error. */
  readonly prequentialMse: number
}

/** What a fitted KRLS offers. */
export interface OnlineKernelVerbs {
  readonly predict: (x: TensorLike) => Tensor
  /** A normal distribution around the prediction with the prequential error as its spread. */
  readonly predictDistribution: (x: TensorLike) => Distribution
  readonly score: (data: LearnData) => number
  readonly dictionarySize: number
  readonly nFeaturesIn: number
}

const WHO = 'krls'

const KRLS_DEFAULTS: OnlineKernelHyper = {
  criterion: 'ald',
  budget: 64,
  aldThreshold: 1e-2,
  coherence: 0.9,
  regularisation: 1e-3,
  kernel: 'rbf',
  gamma: 'auto',
  degree: 3,
  coef0: 1,
}

const TINY = 1e-12
/** `M v` for a square matrix of rows. */
const matVec = (m: readonly (readonly number[])[], v: readonly number[]): number[] => m.map((row) => ddot(row, v))

function validate(h: OnlineKernelHyper, who: string): void {
  if (!Number.isInteger(h.budget) || h.budget < 1)
    throw new MdmlError('value', `${who}: budget must be a positive integer`)
  if (!(h.regularisation > 0)) throw new MdmlError('value', `${who}: regularisation must be positive`)
  if (!(h.aldThreshold >= 0)) throw new MdmlError('value', `${who}: aldThreshold must not be negative`)
  if (!(h.coherence > 0 && h.coherence <= 1)) throw new MdmlError('value', `${who}: coherence must be in (0, 1]`)
}

/**
 * The empty state of {@link krls} for inputs with `nFeatures` features.
 *
 * @param nFeatures - The number of features `d >= 1`.
 * @returns A state with an empty dictionary.
 * @throws MdmlError `value` for a non-positive `nFeatures`.
 * @concept online-kernel
 * @example
 * ```ts
 * import { createKrlsState } from 'mdml-engine/kernel-methods'
 * createKrlsState(2).dictionary.length // => 0
 * ```
 */
export function createKrlsState(nFeatures: number): KrlsState {
  if (!Number.isInteger(nFeatures) || nFeatures < 1) {
    throw new MdmlError('value', `createKrlsState: nFeatures must be a positive integer, got ${nFeatures}`)
  }
  return {
    step: 0,
    nFeatures,
    dictionary: [],
    dictionaryDiagonal: [],
    targets: [],
    alpha: [],
    kInv: [],
    p: [],
    novelty: 0,
    error: 0,
    added: false,
    sse: 0,
  }
}

/**
 * Absorb one observation `(x, y)` into a KRLS state.
 *
 * With `ald` (Engel et al. 2004) the residual `delta = k(x, x) - k_t' K^(-1) k_t` of projecting `phi(x)` on the span of
 * the dictionary decides: above `aldThreshold` (and with room under `budget`) the dictionary grows by `x` through a
 * block inversion, otherwise the coefficients take one reduced-order recursive least squares step on the projection.
 * With `coherence` the test is the largest normalised kernel value of `x` with the centres. With `sliding-window` the
 * window of the last `budget` observations is kept, and `K + lambda I` is inverted by a bordering update when an
 * observation enters and a downdate when the oldest leaves, so the coefficients are always the exact ridge solution
 * `(K + lambda I)^(-1) y` on the window.
 *
 * @param state - The current state.
 * @param x - The observation's features `[d]`.
 * @param y - Its target.
 * @param hyper - The hyperparameters.
 * @param kernel - The resolved kernel (default: resolved from the hyperparameters on this one row).
 * @returns The next state.
 * @throws MdmlError `shape` for a wrong feature count.
 * @concept online-kernel
 * @reference Y. Engel, S. Mannor and R. Meir, The kernel recursive least-squares algorithm (IEEE Trans. Signal
 *   Processing 52, 2004); S. Van Vaerenbergh, J. Via and I. Santamaria, A sliding-window kernel RLS algorithm and its
 *   application to nonlinear channel identification (ICASSP 2006).
 * @example
 * ```ts
 * import { createKrlsState, krlsStep } from 'mdml-engine/kernel-methods'
 * const hyper = { criterion: 'ald', budget: 10, aldThreshold: 0.05, coherence: 0.9, regularisation: 1e-3,
 *   kernel: 'linear', gamma: 1, degree: 3, coef0: 0 } as const
 * const s = krlsStep(createKrlsState(1), [2], 4, hyper)
 * s.alpha // => [1]
 * ```
 */
export function krlsStep(
  state: KrlsState,
  x: readonly number[],
  y: number,
  hyper: OnlineKernelHyper,
  kernel: ResolvedKernel = resolveKernel(hyper as KernelChoice, [Array.from(x)]),
): KrlsState {
  if (x.length !== state.nFeatures) {
    throw new MdmlError('shape', `krlsStep: expected ${state.nFeatures} features, got ${x.length}`)
  }
  validate(hyper, 'krlsStep')
  const point = Array.from(x)
  const m = state.dictionary.length
  const kxx = kernelMatrix(hyper, kernel, [point])[0] ?? 0
  const kv = m === 0 ? [] : Array.from(kernelMatrix(hyper, kernel, state.dictionary, [point]))
  const prior = ddot(kv, state.alpha)
  const error = y - prior
  const sse = state.sse + error * error
  const next = (patch: Partial<KrlsState>): KrlsState => ({
    ...state,
    step: state.step + 1,
    error,
    sse,
    added: false,
    ...patch,
  })

  if (hyper.criterion === 'sliding-window') {
    const lam = hyper.regularisation
    let dictionary = state.dictionary
    let targets = state.targets
    let diag = state.dictionaryDiagonal
    let inv = state.kInv
    let b = kv
    if (m >= hyper.budget) {
      // Drop the oldest: with the inverse [[e, f'], [f, G]], the inverse of the trailing block is G - f f' / e.
      const e = inv[0]?.[0] ?? 1
      const f = inv.slice(1).map((row) => row[0] ?? 0)
      inv = inv.slice(1).map((row, i) => row.slice(1).map((g, j) => g - ((f[i] ?? 0) * (f[j] ?? 0)) / e))
      dictionary = dictionary.slice(1)
      targets = targets.slice(1)
      diag = diag.slice(1)
      b = b.slice(1)
    }
    const w = dictionary.length
    // Border with the new point: g = 1 / (d - b' A^(-1) b), the standard block inverse.
    const ab = matVec(inv, b)
    const schur = kxx + lam - ddot(b, ab)
    const g = 1 / schur
    const grown: number[][] = Array.from({ length: w + 1 }, (_, i) =>
      Array.from({ length: w + 1 }, (_, j) => {
        if (i < w && j < w) return (inv[i]?.[j] ?? 0) + g * (ab[i] ?? 0) * (ab[j] ?? 0)
        if (i < w) return -g * (ab[i] ?? 0)
        if (j < w) return -g * (ab[j] ?? 0)
        return g
      }),
    )
    const newTargets = [...targets, y]
    return next({
      dictionary: [...dictionary, point],
      dictionaryDiagonal: [...diag, kxx],
      targets: newTargets,
      kInv: grown,
      alpha: matVec(grown, newTargets),
      p: [],
      novelty: schur,
      added: true,
    })
  }

  if (m === 0) {
    if (kxx <= TINY) return next({ novelty: 0 })
    return next({
      dictionary: [point],
      dictionaryDiagonal: [kxx],
      targets: [y],
      alpha: [y / kxx],
      kInv: [[1 / kxx]],
      p: [[1]],
      novelty: kxx,
      added: true,
    })
  }

  // The projection of phi(x) on the span of the dictionary, and the squared residual of the projection.
  const a = matVec(state.kInv, kv)
  const delta = Math.max(0, kxx - ddot(kv, a))
  let novelty = delta
  let novel = false
  if (m < hyper.budget && delta > TINY * Math.max(1, kxx)) {
    if (hyper.criterion === 'ald') novel = delta > hyper.aldThreshold
    else {
      let coh = 0
      for (let j = 0; j < m; j++) {
        const denom = Math.sqrt(Math.max(kxx, TINY) * Math.max(state.dictionaryDiagonal[j] ?? 0, TINY))
        coh = Math.max(coh, Math.abs(kv[j] ?? 0) / denom)
      }
      novelty = 1 - coh
      novel = coh < hyper.coherence
    }
  }

  if (novel) {
    const size = m + 1
    const kInv: number[][] = Array.from({ length: size }, (_, i) =>
      Array.from({ length: size }, (_, j) => {
        if (i < m && j < m) return (state.kInv[i]?.[j] ?? 0) + ((a[i] ?? 0) * (a[j] ?? 0)) / delta
        if (i < m) return -(a[i] ?? 0) / delta
        if (j < m) return -(a[j] ?? 0) / delta
        return 1 / delta
      }),
    )
    const p: number[][] = Array.from({ length: size }, (_, i) =>
      Array.from({ length: size }, (_, j) => (i < m && j < m ? (state.p[i]?.[j] ?? 0) : i === j ? 1 : 0)),
    )
    return next({
      dictionary: [...state.dictionary, point],
      dictionaryDiagonal: [...state.dictionaryDiagonal, kxx],
      targets: [...state.targets, y],
      alpha: [...state.alpha.map((c, i) => c - ((a[i] ?? 0) * error) / delta), error / delta],
      kInv,
      p,
      novelty,
      added: true,
    })
  }

  // Reduced-order recursive least squares on the projection a: q = P a / (1 + a' P a).
  const pa = matVec(state.p, a)
  const denom = 1 + ddot(a, pa)
  const q = pa.map((v) => v / denom)
  const p = state.p.map((row, i) => row.map((v, j) => v - (q[i] ?? 0) * (pa[j] ?? 0)))
  const kq = matVec(state.kInv, q)
  return next({ alpha: state.alpha.map((c, i) => c + (kq[i] ?? 0) * error), p, novelty })
}

/**
 * The predictions of a KRLS state, `sum_j alpha_j k(c_j, x)`.
 *
 * @param state - The state.
 * @param x - Query rows `[n, d]`.
 * @param hyper - The hyperparameters (the kernel).
 * @param kernel - The resolved kernel (default: resolved from the hyperparameters on the dictionary).
 * @returns The predictions `[n]` (zero for an empty dictionary).
 * @concept online-kernel
 * @example
 * ```ts
 * import { createKrlsState, krlsPredict, krlsStep } from 'mdml-engine/kernel-methods'
 * import { toFlat } from 'mdml-engine/tensor'
 * const hyper = { criterion: 'ald', budget: 10, aldThreshold: 0.05, coherence: 0.9, regularisation: 1e-3,
 *   kernel: 'linear', gamma: 1, degree: 3, coef0: 0 } as const
 * const s = krlsStep(createKrlsState(1), [2], 4, hyper)
 * toFlat(krlsPredict(s, [[3]], hyper)) // => [6]
 * ```
 */
export function krlsPredict(
  state: KrlsState,
  x: TensorLike,
  hyper: OnlineKernelHyper,
  kernel: ResolvedKernel = resolveKernel(hyper as KernelChoice, state.dictionary.length > 0 ? state.dictionary : x),
): Tensor {
  const q = matrixView(x, 'krlsPredict', { nFeatures: state.nFeatures })
  const m = state.dictionary.length
  if (m === 0) return fromStorage(new Float64Array(q.n), [q.n], 'float64')
  const k = kernelMatrix(hyper, kernel, x, state.dictionary)
  return fromStorage(
    Float64Array.from({ length: q.n }, (_, i) => {
      let s = 0
      for (let j = 0; j < m; j++) s += (k[i * m + j] ?? 0) * (state.alpha[j] ?? 0)
      return s
    }),
    [q.n],
    'float64',
  )
}

/**
 * Kernel recursive least squares as an algorithm over the rows of `data` in order, one per step. It logs the
 * dictionary size, the novelty score, the one-step-ahead squared error and its running mean, and the coefficients,
 * with a message `dictionaryAdd` when an observation enters the dictionary.
 *
 * @param data - Features and targets.
 * @param hyper - The hyperparameters.
 * @returns The algorithm.
 * @concept online-kernel
 * @example
 * ```ts
 * import { krlsAlgorithm, krlsDefaults } from 'mdml-engine/kernel-methods'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const hyper = { ...krlsDefaults, kernel: 'linear', gamma: 1, aldThreshold: 1e-6 } as const
 * const alg = krlsAlgorithm({ x: [[1], [2], [3]], y: [2, 4, 6] }, hyper)
 * runAlgorithm(alg, { maxSteps: 10 }).state.dictionary.length // => 1
 * ```
 */
export function krlsAlgorithm(data: LearnData, hyper: OnlineKernelHyper): Algorithm<KrlsState, never> {
  validate(hyper, WHO)
  const m = matrixView(data.x, WHO)
  const y = targetVector(data.y ?? [], WHO, m.n)
  const kernel = resolveKernel(hyper as KernelChoice, data.x)
  const row = (i: number): number[] => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))
  return {
    name: 'krls',
    config: { criterion: hyper.criterion, budget: hyper.budget, kernel: kernel.kernel, gamma: kernel.gamma },
    schema: channels({
      metrics: {
        dictionarySize: { description: 'the number of centres in the dictionary' },
        novelty: { description: 'the novelty score of the observation (ALD residual, or one minus the coherence)' },
        loss: { description: 'the squared one-step-ahead error of the observation', direction: 'lower' },
        runningMse: { description: 'the mean squared one-step-ahead error so far', direction: 'lower' },
      },
      tensors: { alpha: { role: 'param', description: 'the coefficients over the dictionary' } },
      messages: { dictionaryAdd: { description: 'an observation entered the dictionary: [row, size]' } },
    }),
    init: () => createKrlsState(m.d),
    step: (s) => krlsStep(s, row(s.step), y[s.step] ?? 0, hyper, kernel),
    done: (s) => (s.step >= m.n ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        dictionarySize: s.dictionary.length,
        novelty: s.novelty,
        loss: s.error * s.error,
        runningMse: s.step === 0 ? 0 : s.sse / s.step,
      },
      tensors: { alpha: { role: 'param', value: s.alpha } },
      ...(s.added ? { messages: [{ kind: 'dictionaryAdd', payload: [s.step - 1, s.dictionary.length] }] } : {}),
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as KrlsState,
  }
}

/** The default hyperparameters of {@link krls}. */
export const krlsDefaults: OnlineKernelHyper = KRLS_DEFAULTS

function expansionVerbs(
  who: string,
  centres: number[][],
  coef: number[],
  d: number,
  kernel: ResolvedKernel,
  h: Pick<KernelChoice, 'kernel'>,
  spread: number,
): OnlineKernelVerbs {
  const m = centres.length
  const predict = (x: TensorLike): Tensor => {
    const q = matrixView(x, who, { nFeatures: d })
    if (m === 0) return fromStorage(new Float64Array(q.n), [q.n], 'float64')
    const k = kernelMatrix(h, kernel, x, centres)
    return fromStorage(
      Float64Array.from({ length: q.n }, (_, i) => {
        let s = 0
        for (let j = 0; j < m; j++) s += (k[i * m + j] ?? 0) * (coef[j] ?? 0)
        return s
      }),
      [q.n],
      'float64',
    )
  }
  return {
    predict,
    predictDistribution: (x) => {
      const mean = predict(x)
      return normal(
        mean,
        fromStorage(
          new Float64Array(mean.shape.reduce((a, b) => a * b, 1)).fill(Math.max(spread, 1e-12)),
          [...mean.shape],
          'float64',
        ),
      )
    },
    score: (data) => r2Score(data.y ?? [], predict(data.x)).estimate.value,
    dictionarySize: m,
    nFeaturesIn: d,
  }
}

/**
 * Kernel recursive least squares (Engel et al. 2004), its coherence-test variant and the sliding-window form, as an
 * estimator on the iterative base. A session steps one observation at a time, so `fittedAt(t)` is the predictor after
 * `t` observations, and the tape shows the dictionary growing and the one-step-ahead error falling.
 *
 * @concept online-kernel
 * @reference Y. Engel, S. Mannor and R. Meir, The kernel recursive least-squares algorithm (IEEE Trans. Signal
 *   Processing 52, 2004).
 * @example
 * ```ts
 * import { krls } from 'mdml-engine/kernel-methods'
 * const fitted = krls.with({ kernel: 'rbf', gamma: 0.5, aldThreshold: 0.01 }).fit({ x: [[0], [1], [2], [3]], y: [0, 1, 0, -1] })
 * fitted.dictionarySize <= 4 // => true
 * ```
 */
export const krls: IterativeEstimator<OnlineKernelHyper, KrlsState, OnlineKernelParams, OnlineKernelVerbs, KrlsPath> =
  iterative<OnlineKernelHyper, KrlsState, OnlineKernelParams, OnlineKernelVerbs, KrlsPath>({
    name: 'kernel-methods/krls',
    kind: 'regressor',
    defaults: KRLS_DEFAULTS,
    hyperSpace: (h) =>
      dict({
        criterion: choice(['ald', 'coherence', 'sliding-window'], { default: h.criterion, label: 'dictionary rule' }),
        budget: int(Math.min(2, h.budget), Math.max(256, h.budget), { default: h.budget, label: 'dictionary size' }),
        aldThreshold: real(h.aldThreshold > 0 ? Math.min(1e-5, h.aldThreshold) : 0, Math.max(1, h.aldThreshold), {
          log: h.aldThreshold > 0,
          default: h.aldThreshold,
          label: 'ALD threshold',
          when: { key: 'criterion', in: ['ald'] },
        }),
        regularisation: real(Math.min(1e-6, h.regularisation), Math.max(1, h.regularisation), {
          log: true,
          default: h.regularisation,
          label: 'regularisation',
          when: { key: 'criterion', in: ['sliding-window'] },
        }),
        ...kernelSpace(h),
      }),
    algorithm: (data, h): Algorithm<KrlsState, KrlsPath> => krlsAlgorithm(data, h),
    maxSteps: (_h, data) => matrixView(data.x, WHO).n,
    params: (s, data, h) => ({
      kernel: resolveKernel(h as KernelChoice, data.x),
      dictionary: s.dictionary,
      dualCoef: s.alpha,
      nFeatures: s.nFeatures,
      nObservations: s.step,
      prequentialMse: s.step === 0 ? 0 : s.sse / s.step,
    }),
    verbs: (p, h) =>
      expansionVerbs(WHO, p.dictionary, p.dualCoef, p.nFeatures, p.kernel, h, Math.sqrt(p.prequentialMse)),
    contract: (): Contract<KrlsState, KrlsPath> =>
      defineContract({
        'model/dictionary': derived<KrlsState>({
          description: 'the dictionary centres and their coefficients so far',
          compute: (s) => ({ centres: s.dictionary, coef: s.alpha }),
        }),
      }),
    logs: { 'solver/dictionarySize': Log(), 'solver/runningMse': Log() },
  })

// ---------------------------------------------------------------------------------------------------------------------
// NORMA
// ---------------------------------------------------------------------------------------------------------------------

/** The losses of {@link norma}. */
export type NormaLoss = 'squared' | 'epsilon-insensitive'

/** The hyperparameters of {@link norma}. */
export interface NormaHyper {
  readonly loss: NormaLoss
  /** The half-width of the insensitive tube (`epsilon-insensitive` only). */
  readonly epsilon: number
  /** The step size scale: `eta / sqrt(t)` at step `t` (default 0.5). */
  readonly eta: number
  /** The weight decay `lambda` of the regulariser `lambda/2 ||f||^2` (default 1e-3). */
  readonly lambda: number
  /** The most centres kept: the oldest are dropped (default 100). */
  readonly budget: number
  /** Passes over the data (default 1). */
  readonly epochs: number
  readonly kernel: PairwiseKernelMetric | Kernel
  readonly gamma: number | 'scale' | 'auto'
  readonly degree: number
  readonly coef0: number
}

/** The state of NORMA: the truncated expansion. */
export type NormaState = {
  readonly step: number
  readonly centres: number[][]
  readonly coef: number[]
  /** The prediction error `f(x) - y` of the last step, before the update. */
  readonly error: number
  /** The sum of the losses of the steps so far (online loss). */
  readonly cumulativeLoss: number
}

/** The fitted parameters of {@link norma}: plain JSON. */
export type NormaParams = {
  readonly kernel: ResolvedKernel
  readonly centres: number[][]
  readonly coef: number[]
  readonly nFeatures: number
  readonly nSteps: number
  readonly meanLoss: number
}

const NORMA_DEFAULTS: NormaHyper = {
  loss: 'squared',
  epsilon: 0.1,
  eta: 0.5,
  lambda: 1e-3,
  budget: 100,
  epochs: 1,
  kernel: 'rbf',
  gamma: 'auto',
  degree: 3,
  coef0: 1,
}

/**
 * NORMA (Naive Online R_reg Minimisation Algorithm) as an algorithm: stochastic gradient descent on
 * `loss(f(x_t), y_t) + lambda/2 ||f||^2` in the reproducing kernel Hilbert space, where the update
 * `f <- (1 - eta_t lambda) f - eta_t loss'(f(x_t), y_t) k(x_t, .)` shrinks every coefficient and adds one centre, and
 * the expansion is truncated to the last `budget` centres (the shrinkage makes old terms decay geometrically, which is
 * why truncation costs little). The step size is `eta_t = eta / sqrt(t)`.
 *
 * @param data - Features and targets, visited in order, `epochs` times.
 * @param hyper - The hyperparameters.
 * @returns The algorithm, logging the loss, its running mean and the expansion size.
 * @concept online-kernel
 * @reference J. Kivinen, A. J. Smola and R. C. Williamson, Online learning with kernels (IEEE Trans. Signal Processing
 *   52, 2004).
 * @example
 * ```ts
 * import { normaAlgorithm, normaDefaults } from 'mdml-engine/kernel-methods'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = normaAlgorithm({ x: [[0], [1], [2]], y: [0, 1, 2] }, { ...normaDefaults, kernel: 'linear', gamma: 1 })
 * runAlgorithm(alg, { maxSteps: 10 }).state.step // => 3
 * ```
 */
export function normaAlgorithm(data: LearnData, hyper: NormaHyper): Algorithm<NormaState, never> {
  const who = 'norma'
  if (!(hyper.eta > 0)) throw new MdmlError('value', `${who}: eta must be positive`)
  if (!(hyper.lambda >= 0)) throw new MdmlError('value', `${who}: lambda must not be negative`)
  if (!Number.isInteger(hyper.budget) || hyper.budget < 1)
    throw new MdmlError('value', `${who}: budget must be a positive integer`)
  const m = matrixView(data.x, who)
  const y = targetVector(data.y ?? [], who, m.n)
  const kernel = resolveKernel(hyper as KernelChoice, data.x)
  const row = (i: number): number[] => Array.from(m.data.subarray(i * m.d, (i + 1) * m.d))
  const total = m.n * hyper.epochs
  return {
    name: 'norma',
    config: { loss: hyper.loss, eta: hyper.eta, lambda: hyper.lambda, budget: hyper.budget, kernel: kernel.kernel },
    schema: channels({
      metrics: {
        loss: { description: 'the loss of the observation before the update', direction: 'lower' },
        meanLoss: { description: 'the mean online loss so far', direction: 'lower' },
        centres: { description: 'the number of centres in the expansion' },
      },
      tensors: { coef: { role: 'param', description: 'the coefficients of the expansion' } },
    }),
    init: () => ({ step: 0, centres: [], coef: [], error: 0, cumulativeLoss: 0 }),
    step: (s) => {
      const i = s.step % m.n
      const point = row(i)
      const t = s.step + 1
      const eta = hyper.eta / Math.sqrt(t)
      const f = s.centres.length === 0 ? 0 : ddot(Array.from(kernelMatrix(hyper, kernel, s.centres, [point])), s.coef)
      const error = f - (y[i] ?? 0)
      const loss = hyper.loss === 'squared' ? 0.5 * error * error : Math.max(0, Math.abs(error) - hyper.epsilon)
      const grad = hyper.loss === 'squared' ? error : Math.abs(error) > hyper.epsilon ? Math.sign(error) : 0
      const shrink = 1 - eta * hyper.lambda
      const coef = s.coef.map((c) => c * shrink)
      const centres = [...s.centres]
      if (grad !== 0) {
        centres.push(point)
        coef.push(-eta * grad)
      }
      const over = Math.max(0, centres.length - hyper.budget)
      return {
        step: t,
        centres: centres.slice(over),
        coef: coef.slice(over),
        error,
        cumulativeLoss: s.cumulativeLoss + loss,
      }
    },
    done: (s) => (s.step >= total ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        loss: s.step === 0 ? 0 : lossOf(hyper, s.error),
        meanLoss: s.step === 0 ? 0 : s.cumulativeLoss / s.step,
        centres: s.centres.length,
      },
      tensors: { coef: { role: 'param', value: s.coef } },
    }),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as NormaState,
  }
}

function lossOf(h: NormaHyper, error: number): number {
  return h.loss === 'squared' ? 0.5 * error * error : Math.max(0, Math.abs(error) - h.epsilon)
}

/** The default hyperparameters of {@link norma}. */
export const normaDefaults: NormaHyper = NORMA_DEFAULTS

/**
 * NORMA as an estimator on the iterative base: online kernel regression by stochastic gradient descent with weight
 * decay and a truncated expansion, one observation per step. There is no intercept term.
 *
 * @concept online-kernel
 * @reference J. Kivinen, A. J. Smola and R. C. Williamson, Online learning with kernels (IEEE Trans. Signal Processing
 *   52, 2004).
 * @example
 * ```ts
 * import { norma } from 'mdml-engine/kernel-methods'
 * const f = norma.with({ kernel: 'linear', gamma: 1, lambda: 0, epochs: 50 }).fit({ x: [[1], [2], [3]], y: [2, 4, 6] })
 * f.dictionarySize <= 100 // => true
 * ```
 */
export const norma: IterativeEstimator<NormaHyper, NormaState, NormaParams, OnlineKernelVerbs, never> = iterative<
  NormaHyper,
  NormaState,
  NormaParams,
  OnlineKernelVerbs,
  never
>({
  name: 'kernel-methods/norma',
  kind: 'regressor',
  defaults: NORMA_DEFAULTS,
  hyperSpace: (h) =>
    dict({
      eta: real(Math.min(1e-3, h.eta), Math.max(5, h.eta), { log: true, default: h.eta, label: 'step size' }),
      lambda: real(h.lambda > 0 ? Math.min(1e-6, h.lambda) : 0, Math.max(1, h.lambda), {
        log: h.lambda > 0,
        default: h.lambda,
        label: 'weight decay',
      }),
      budget: int(Math.min(2, h.budget), Math.max(500, h.budget), { default: h.budget, label: 'expansion size' }),
      ...kernelSpace(h),
    }),
  algorithm: (data, h): Algorithm<NormaState, never> => normaAlgorithm(data, h),
  maxSteps: (h, data) => matrixView(data.x, 'norma').n * h.epochs,
  params: (s, data, h) => ({
    kernel: resolveKernel(h as KernelChoice, data.x),
    centres: s.centres,
    coef: s.coef,
    nFeatures: matrixView(data.x, 'norma').d,
    nSteps: s.step,
    meanLoss: s.step === 0 ? 0 : s.cumulativeLoss / s.step,
  }),
  verbs: (p, h) => expansionVerbs('norma', p.centres, p.coef, p.nFeatures, p.kernel, h, Math.sqrt(2 * p.meanLoss)),
  logs: { 'solver/loss': Log(), 'solver/meanLoss': Log() },
})
