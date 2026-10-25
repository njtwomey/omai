/**
 * Warm-up tuning for Hamiltonian samplers, following Stan and BlackJAX: Nesterov's dual averaging of the step size
 * towards a target acceptance rate, the heuristic for a reasonable first step size, Welford estimation of the mass
 * matrix with Stan's regularisation, Stan's warm-up schedule of fast and doubling slow windows, and `windowAdaptation`,
 * which combines them.
 */

import { type Json, MdmlError } from '../../base/index.ts'
import { foldIn, type Key } from '../../random/index.ts'
import type { TensorLike } from '../../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../../trace/index.ts'
import { squareMatrix, vectorOf } from '../_linear.ts'
import { welfordCovariance, welfordInit, welfordUpdate } from '../_mcmc.ts'
import type { Adaptation, MCMCKernel, MCMCState, TransitionInfo, Warmup } from '../chains.ts'

/** The state of Nesterov's dual averaging of log step sizes (BlackJAX `DualAveragingState`). */
export interface DualAveragingState {
  readonly logStepSize: number
  readonly logStepSizeAvg: number
  /** The iteration, starting at 1. */
  readonly step: number
  /** The running average of target - acceptance. */
  readonly avgError: number
  /** The shrinkage point log(10 eps_0). */
  readonly mu: number
}

/** Options of {@link dualAveragingUpdate}. */
export interface DualAveragingOptions {
  /** The acceptance rate to reach (default 0.8, as Stan and BlackJAX). */
  readonly targetAcceptance?: number
  /** Iteration offset t0 that damps early iterations (default 10). */
  readonly t0?: number
  /** The shrinkage gamma (default 0.05). */
  readonly gamma?: number
  /** The averaging exponent kappa in (0.5, 1] (default 0.75). */
  readonly kappa?: number
}

/**
 * Start dual averaging of the step size at eps_0 (BlackJAX `dual_averaging_adaptation` init): log eps = log eps_0,
 * mu = log(10 eps_0), averages zero, step 1.
 *
 * @param stepSize - The starting step size eps_0 (positive).
 * @returns The state.
 * @throws MdmlError with code `value` for a step size that is not positive and finite.
 * @concept dual-averaging
 * @example
 * ```ts
 * import { dualAveragingInit } from 'mdml-engine/sampling'
 * dualAveragingInit(1).mu === Math.log(10) // => true
 * ```
 */
export function dualAveragingInit(stepSize: number): DualAveragingState {
  if (!(stepSize > 0 && Number.isFinite(stepSize))) {
    throw new MdmlError('value', 'dualAveragingInit: the step size must be positive and finite')
  }
  return { logStepSize: Math.log(stepSize), logStepSizeAvg: 0, step: 1, avgError: 0, mu: Math.log(10 * stepSize) }
}

/**
 * One update of Nesterov's primal-dual averaging of log step sizes towards a target acceptance rate (Hoffman and
 * Gelman 2014, section 3.2.1, in BlackJAX's form): with H_t = delta - alpha_t,
 * Hbar_t = (1 - 1 / (t + t0)) Hbar_{t-1} + H_t / (t + t0), log eps_t = mu - sqrt(t) Hbar_t / gamma, and the average
 * log epsbar_t = t^-kappa log eps_{t-1} + (1 - t^-kappa) log epsbar_{t-1}. A NaN acceptance rate counts as 0.
 *
 * @param state - The state.
 * @param acceptanceRate - The acceptance rate of the last transition.
 * @param options - Target acceptance, t0, gamma and kappa.
 * @returns The next state.
 * @concept dual-averaging
 * @reference M. D. Hoffman and A. Gelman, The No-U-Turn Sampler (JMLR 2014); Y. Nesterov, Primal-dual subgradient
 *   methods for convex problems (Mathematical Programming 2009).
 * @example
 * ```ts
 * import { dualAveragingInit, dualAveragingUpdate } from 'mdml-engine/sampling'
 * // Acceptance above the target makes the step size grow.
 * dualAveragingUpdate(dualAveragingInit(1), 1).logStepSize > 0 // => true
 * ```
 */
export function dualAveragingUpdate(
  state: DualAveragingState,
  acceptanceRate: number,
  options: DualAveragingOptions = {},
): DualAveragingState {
  const target = options.targetAcceptance ?? 0.8
  const t0 = options.t0 ?? 10
  const gamma = options.gamma ?? 0.05
  const kappa = options.kappa ?? 0.75
  const alpha = Number.isNaN(acceptanceRate) ? 0 : acceptanceRate
  const gradient = target - alpha
  const t = state.step
  const reg = t + t0
  const eta = t ** -kappa
  const avgError = (1 - 1 / reg) * state.avgError + gradient / reg
  const logStepSize = state.mu - (Math.sqrt(t) / gamma) * avgError
  const logStepSizeAvg = eta * state.logStepSize + (1 - eta) * state.logStepSizeAvg
  return { logStepSize, logStepSizeAvg, step: t + 1, avgError, mu: state.mu }
}

/**
 * The step size dual averaging settles on: exp of the averaged log step size.
 *
 * @param state - The state.
 * @returns The step size.
 * @concept dual-averaging
 * @example
 * ```ts
 * import { dualAveragingFinal, dualAveragingInit } from 'mdml-engine/sampling'
 * dualAveragingFinal(dualAveragingInit(0.5)) // => 1
 * ```
 */
export function dualAveragingFinal(state: DualAveragingState): number {
  return Math.exp(state.logStepSizeAvg)
}

/** Options of {@link findReasonableStepSize}. */
export interface ReasonableStepSizeOptions {
  readonly key: Key
  /** The first step size tried (default 1). */
  readonly initialStepSize?: number
  /** The acceptance rate that separates too large from too small (default 0.65). */
  readonly targetAcceptance?: number
  /** The most doublings or halvings (default 100). */
  readonly maxIter?: number
}

/**
 * Hoffman and Gelman's heuristic for a first step size (BlackJAX `find_reasonable_step_size`): from the reference state,
 * double the step size while one transition accepts with probability above the target, or halve it while it accepts
 * below, and stop at the step size where the direction flips. The chain never moves, the search stops at the
 * floating-point limits, and it gives up after `maxIter` tries.
 *
 * @param kernelFor - The kernel at a given step size.
 * @param state - The reference state.
 * @param options - Key, starting step size, target acceptance and iteration cap.
 * @returns The step size.
 * @concept dual-averaging
 * @example
 * ```ts
 * import { findReasonableStepSize, hmcKernel, numericTarget } from 'mdml-engine/sampling'
 * import { key } from 'mdml-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2 / 0.01, 1, { grad: (x) => [-(x[0] ?? 0) / 0.01] })
 * const kernelFor = (stepSize: number) => hmcKernel(t, { stepSize, numIntegrationSteps: 1 })
 * findReasonableStepSize(kernelFor, kernelFor(1).init([0.1]), { key: key(0) }) < 1 // => true
 * ```
 */
export function findReasonableStepSize<S extends MCMCState>(
  kernelFor: (stepSize: number) => MCMCKernel<S>,
  state: S,
  options: ReasonableStepSizeOptions,
): number {
  const target = options.targetAcceptance ?? 0.65
  const maxIter = options.maxIter ?? 100
  let stepSize = options.initialStepSize ?? 1
  let direction = 0
  let previous = 0
  for (let i = 0; i < maxIter; i++) {
    const notTooLarge = stepSize < Number.MAX_VALUE || direction <= 0
    const notTooSmall = stepSize > Number.MIN_VALUE * 2 ** 52 || direction >= 0
    if (!(notTooLarge && notTooSmall && (previous === 0 || direction === previous))) break
    stepSize *= 2 ** direction
    const info = kernelFor(stepSize).step(foldIn(options.key, i), state).info
    previous = direction
    direction = target < info.acceptanceRate ? 1 : -1
  }
  return stepSize
}

/** The state of mass-matrix adaptation: the inverse mass matrix in force and Welford's running moments. */
export interface MassMatrixState {
  readonly dense: boolean
  /** The diagonal of M^-1 (length dim) or M^-1 row-major (dim * dim). */
  readonly inverseMassMatrix: Float64Array
  readonly count: number
  readonly mean: Float64Array
  /** Sums of squared (diagonal) or outer-product (dense) deviations. */
  readonly m2: Float64Array
}

/**
 * Start mass-matrix adaptation (BlackJAX `mass_matrix_adaptation` init): the identity (or a given inverse mass
 * matrix) and empty moments.
 *
 * @param dim - The dimension.
 * @param options - `dense` (default false: a diagonal matrix) and an initial inverse mass matrix.
 * @returns The state.
 * @throws MdmlError with code `shape` for an initial matrix of the wrong shape.
 * @concept mass-matrix-adaptation
 * @example
 * ```ts
 * import { massMatrixInit } from 'mdml-engine/sampling'
 * Array.from(massMatrixInit(2, { dense: true }).inverseMassMatrix) // => [1, 0, 0, 1]
 * ```
 */
export function massMatrixInit(
  dim: number,
  options: { readonly dense?: boolean; readonly initial?: TensorLike } = {},
): MassMatrixState {
  const dense = options.dense ?? false
  const identity = dense
    ? Float64Array.from({ length: dim * dim }, (_, k) => (k % (dim + 1) === 0 ? 1 : 0))
    : new Float64Array(dim).fill(1)
  const inverseMassMatrix =
    options.initial === undefined
      ? identity
      : dense
        ? squareMatrix(options.initial, dim, 'massMatrixInit')
        : vectorOf(options.initial, dim, 'massMatrixInit')
  const w = welfordInit(dim, !dense)
  return { dense, inverseMassMatrix, count: w.count, mean: w.mean, m2: w.m2 }
}

/**
 * Add a draw to the mass-matrix moments (Welford's update; the matrix in force is unchanged until the window ends).
 *
 * @param state - The state.
 * @param position - The draw.
 * @returns The next state.
 * @concept mass-matrix-adaptation
 * @example
 * ```ts
 * import { massMatrixInit, massMatrixUpdate } from 'mdml-engine/sampling'
 * massMatrixUpdate(massMatrixInit(1), [2]).count // => 1
 * ```
 */
export function massMatrixUpdate(state: MassMatrixState, position: ArrayLike<number>): MassMatrixState {
  return { ...state, ...welfordUpdate(state, position) }
}

/**
 * End a slow window (BlackJAX `mass_matrix_adaptation` final, Stan's regularisation): the new inverse mass matrix is
 * (n / (n + 5)) Cov + 1e-3 (5 / (n + 5)) I from the window's n draws, shrinking a small window's covariance towards
 * a small multiple of the identity, and the moments restart for the next window.
 *
 * @param state - The state at the end of the window.
 * @returns The state with the new matrix and empty moments.
 * @concept mass-matrix-adaptation
 * @example
 * ```ts
 * import { massMatrixFinal, massMatrixInit, massMatrixUpdate } from 'mdml-engine/sampling'
 * let s = massMatrixInit(1)
 * for (const x of [0, 2, 4]) s = massMatrixUpdate(s, [x])
 * // Variance 4 from three draws: (3 / 8) 4 + 1e-3 (5 / 8).
 * massMatrixFinal(s).inverseMassMatrix[0] // => 1.500625
 * ```
 */
export function massMatrixFinal(state: MassMatrixState): MassMatrixState {
  const dim = state.mean.length
  const cov = welfordCovariance(state)
  const n = state.count
  const denom = n + 5
  const inverseMassMatrix = cov.map(
    (v, k) => (n / denom) * v + (!state.dense || k % (dim + 1) === 0 ? (5 / denom) * 1e-3 : 0),
  )
  const w = welfordInit(dim, !state.dense)
  return { dense: state.dense, inverseMassMatrix, count: w.count, mean: w.mean, m2: w.m2 }
}

/** One iteration of the warm-up schedule. */
export interface WindowStage {
  /** Whether the mass matrix is being estimated (a slow window) or only the step size (a fast buffer). */
  readonly slow: boolean
  /** Whether this iteration ends a slow window. */
  readonly windowEnd: boolean
}

/** Options of {@link windowSchedule}. */
export interface WindowScheduleOptions {
  /** The first fast buffer (default 75). */
  readonly initialBuffer?: number
  /** The last fast buffer (default 50). */
  readonly finalBuffer?: number
  /** The first slow window (default 25); later windows double. */
  readonly firstWindow?: number
}

/**
 * Stan's warm-up schedule (BlackJAX `build_schedule`): a fast buffer adapting only the step size, slow windows that
 * double in size and estimate the mass matrix (the last one stretched to fill), and a final fast buffer. Fewer than 20
 * iterations are all fast; when the buffers do not fit they shrink to 15%, 75% and 10% of the warm-up.
 *
 * @param numSteps - The number of warm-up iterations.
 * @param options - Buffer and first window sizes.
 * @returns One stage per iteration.
 * @throws MdmlError with code `value` for a negative or fractional number of steps.
 * @concept window-adaptation
 * @reference Stan Reference Manual, HMC algorithm parameters: automatic parameter tuning.
 * @example
 * ```ts
 * import { windowSchedule } from 'mdml-engine/sampling'
 * const s = windowSchedule(1000)
 * // Windows of 25, 50, 100, 200 and the stretched last one, between buffers of 75 and 50.
 * s.flatMap((x, i) => (x.windowEnd ? [i] : [])) // => [99, 149, 249, 449, 949]
 * ```
 */
export function windowSchedule(numSteps: number, options: WindowScheduleOptions = {}): WindowStage[] {
  if (!(Number.isInteger(numSteps) && numSteps >= 0)) {
    throw new MdmlError('value', 'windowSchedule: numSteps must be a non-negative integer')
  }
  const fast: WindowStage = { slow: false, windowEnd: false }
  if (numSteps < 20) return Array.from({ length: numSteps }, () => fast)
  let initial = options.initialBuffer ?? 75
  let final = options.finalBuffer ?? 50
  let first = options.firstWindow ?? 25
  if (initial + first + final > numSteps) {
    initial = Math.trunc(0.15 * numSteps)
    final = Math.trunc(0.1 * numSteps)
    first = numSteps - initial - final
  }
  const out: WindowStage[] = Array.from({ length: initial }, () => fast)
  const finalStart = numSteps - final
  let size = first
  let start = initial
  while (start < finalStart) {
    let current = size
    if (3 * current <= finalStart - start) size = 2 * current
    else current = finalStart - start
    const end = start + current
    for (let i = start; i < end - 1; i++) out.push({ slow: true, windowEnd: false })
    out.push({ slow: true, windowEnd: true })
    start = end
  }
  for (let i = finalStart; i < numSteps; i++) out.push(fast)
  return out
}

/** The tuned parameters of a Hamiltonian kernel. */
export interface HamiltonianParameters {
  readonly stepSize: number
  /** The diagonal of M^-1 or M^-1 row-major. */
  readonly inverseMassMatrix: Float64Array
}

/** Options of {@link windowAdaptation}. */
export interface WindowAdaptationOptions extends WindowScheduleOptions, Omit<DualAveragingOptions, 'targetAcceptance'> {
  /** Adapt a dense mass matrix (default false: diagonal). */
  readonly dense?: boolean
  /** The acceptance rate dual averaging aims for (default 0.8). */
  readonly targetAcceptance?: number
  /** The starting step size (default 1). */
  readonly initialStepSize?: number
  /** Search for a reasonable first step size from the starting point, as Stan (default true). */
  readonly findStepSize?: boolean
  /** The starting inverse mass matrix (default the identity). */
  readonly initialInverseMassMatrix?: TensorLike
}

interface WarmupState<S extends MCMCState> {
  readonly chain: S
  readonly info?: TransitionInfo
  readonly da: DualAveragingState
  readonly mm: MassMatrixState
  readonly stepSize: number
  readonly windowEnded: boolean
}

/** Keep log step sizes inside the range where exp is finite and positive. */
function clampLog(x: number): number {
  return Math.min(300, Math.max(-300, x))
}

/**
 * Stan's windowed adaptation of the step size and mass matrix for Hamiltonian kernels (BlackJAX `window_adaptation`):
 * dual averaging tunes the step size in every iteration; during the slow windows of {@link windowSchedule} Welford's
 * algorithm estimates the posterior covariance, which becomes the inverse mass matrix (regularised) at each window's
 * end, where dual averaging restarts from the window's averaged step size. The final kernel uses the averaged step
 * size and the last window's mass matrix. Warm-up is traced: step size, acceptance, divergences and tree depth per
 * iteration, and a `windowEnd` message per window.
 *
 * @param dim - The target's dimension.
 * @param kernelFor - The kernel with given parameters (HMC, NUTS, MALA with a preconditioner, ...).
 * @param options - Mass-matrix form, target acceptance, initial step size and matrix, schedule and dual-averaging
 *   options.
 * @returns The adaptation (pass to `sampleChains`); its parameters are `{ stepSize, inverseMassMatrix }`.
 * @concept window-adaptation
 * @example
 * ```ts
 * import { hmcKernel, numericTarget, sampleChains, windowAdaptation } from 'mdml-engine/sampling'
 * import { key } from 'mdml-engine/random'
 * const t = numericTarget((x) => -0.5 * (x[0] ?? 0) ** 2 / 4, 1, { grad: (x) => [-(x[0] ?? 0) / 4] })
 * const adapt = windowAdaptation(1, (p) => hmcKernel(t, { ...p, numIntegrationSteps: 5 }))
 * const c = sampleChains(adapt, { key: key(1), numChains: 1, numWarmup: 150, numDraws: 10 })
 * (c.parameters[0] as { stepSize: number }).stepSize > 0 // => true
 * ```
 */
export function windowAdaptation<S extends MCMCState>(
  dim: number,
  kernelFor: (parameters: HamiltonianParameters) => MCMCKernel<S>,
  options: WindowAdaptationOptions = {},
): Adaptation<S> {
  const dense = options.dense ?? false
  const daOptions: DualAveragingOptions = {
    targetAcceptance: options.targetAcceptance ?? 0.8,
    ...(options.t0 === undefined ? {} : { t0: options.t0 }),
    ...(options.gamma === undefined ? {} : { gamma: options.gamma }),
    ...(options.kappa === undefined ? {} : { kappa: options.kappa }),
  }
  const initialStepSize = options.initialStepSize ?? 1
  const findStepSize = options.findStepSize ?? true
  const mmInit = (): MassMatrixState =>
    massMatrixInit(dim, {
      dense,
      ...(options.initialInverseMassMatrix === undefined ? {} : { initial: options.initialInverseMassMatrix }),
    })
  return {
    name: 'windowAdaptation',
    dim,
    run: (position, run) => {
      let mm = mmInit()
      const first = kernelFor({ stepSize: initialStepSize, inverseMassMatrix: mm.inverseMassMatrix })
      const chain0 = first.init(position)
      if (run.numWarmup === 0) {
        const parameters: Json = { stepSize: initialStepSize, inverseMassMatrix: Array.from(mm.inverseMassMatrix) }
        return { kernel: first, state: chain0, parameters }
      }
      const eps0 = findStepSize
        ? findReasonableStepSize(
            (stepSize) => kernelFor({ stepSize, inverseMassMatrix: mm.inverseMassMatrix }),
            chain0,
            { key: foldIn(run.key, 0), initialStepSize },
          )
        : initialStepSize
      const schedule = windowSchedule(run.numWarmup, options)
      const algorithm: Algorithm<WarmupState<S>> = {
        name: 'windowAdaptation',
        config: { dim, dense, numWarmup: run.numWarmup, targetAcceptance: daOptions.targetAcceptance ?? 0.8 },
        schema: channels({
          metrics: {
            stepSize: { description: 'step size used in the transition' },
            acceptanceRate: { description: 'acceptance rate of the transition', direction: 'higher' },
            logDensity: { description: 'log density at the current position' },
            divergent: { description: '1 when the transition diverged', direction: 'lower' },
            treeDepth: { description: 'NUTS tree depth' },
          },
          messages: {
            windowEnd: {
              description: 'a slow window ended: new mass matrix and restarted dual averaging',
              payload: { stepSize: 'number', inverseMassMatrix: 'number[]' },
            },
          },
        }),
        init: () => ({ chain: chain0, da: dualAveragingInit(eps0), mm, stepSize: eps0, windowEnded: false }),
        step: (s, ctx) => {
          const kernel = kernelFor({ stepSize: s.stepSize, inverseMassMatrix: s.mm.inverseMassMatrix })
          const tr = kernel.step(ctx.key, s.chain)
          run.onTransition?.(tr.state, tr.info)
          const stage = schedule[ctx.t] ?? { slow: false, windowEnd: false }
          let da = dualAveragingUpdate(s.da, tr.info.acceptanceRate, daOptions)
          da = { ...da, logStepSize: clampLog(da.logStepSize) }
          let next = stage.slow ? massMatrixUpdate(s.mm, tr.state.position) : s.mm
          if (stage.windowEnd) {
            next = massMatrixFinal(next)
            da = dualAveragingInit(Math.exp(clampLog(da.logStepSizeAvg)))
          }
          return {
            chain: tr.state,
            info: tr.info,
            da,
            mm: next,
            stepSize: Math.exp(da.logStepSize),
            windowEnded: stage.windowEnd,
          }
        },
        observe: (s) => ({
          metrics: {
            stepSize: s.info?.stepSize ?? s.stepSize,
            logDensity: s.chain.logDensity,
            ...(s.info === undefined ? {} : { acceptanceRate: s.info.acceptanceRate }),
            ...(s.info?.divergent === undefined ? {} : { divergent: s.info.divergent ? 1 : 0 }),
            ...(s.info?.treeDepth === undefined ? {} : { treeDepth: s.info.treeDepth }),
          },
          ...(s.windowEnded
            ? {
                messages: [
                  {
                    kind: 'windowEnd',
                    payload: { stepSize: s.stepSize, inverseMassMatrix: Array.from(s.mm.inverseMassMatrix) },
                  },
                ],
              }
            : {}),
        }),
      }
      const result = runAlgorithm(algorithm, {
        key: foldIn(run.key, 1),
        maxSteps: run.numWarmup,
        ...(run.parent === undefined ? {} : { parent: run.parent }),
        ...(run.tracer === undefined ? {} : { tracer: run.tracer }),
        ...(run.policy === undefined ? {} : { policy: run.policy }),
      })
      const s = result.state
      mm = s.mm
      const stepSize = Math.exp(clampLog(s.da.logStepSizeAvg))
      const warmup: Warmup<S> = {
        kernel: kernelFor({ stepSize, inverseMassMatrix: mm.inverseMassMatrix }),
        state: s.chain,
        parameters: { stepSize, inverseMassMatrix: Array.from(mm.inverseMassMatrix) },
      }
      return warmup
    },
  }
}
