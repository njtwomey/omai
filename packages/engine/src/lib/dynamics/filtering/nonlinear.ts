/**
 * Kalman filtering for nonlinear state-space models: the extended Kalman filter, which linearises the dynamics and
 * observation by automatic differentiation; the unscented Kalman filter with Van der Merwe or Julier sigma points and
 * the unscented transform; and the ensemble Kalman filter with perturbed observations, all following filterpy.
 */

import { jacfwd } from '../../autodiff/index.ts'
import { OmaiError } from '../../base/index.ts'
import { cholesky, eigh, solve, symmetricPart } from '../../numerics/linalg/index.ts'
import { foldIn, type Key, key as makeKey, normal } from '../../random/index.ts'
import {
  add,
  asTensor,
  diagonal,
  log,
  matmul,
  maximum,
  mean,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  sqrt,
  stack,
  sub,
  sum,
  type TensorLike,
  transpose,
  type Value,
} from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  channels,
  defineContract,
  type Observation,
  type RunOptions,
  runAlgorithm,
  type StepContext,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { fromNumbers, numbersOf } from '../ode/_vec.ts'
import { kalmanUpdate } from './kalman.ts'

/**
 * A nonlinear state-space model x_t = f(x_{t-1}) + w_t, z_t = h(x_t) + v_t with Gaussian noise covariances Q and R
 * and prior N(x0, P0); f and h act on single states `[n]` and are written with operations.
 */
export interface NonlinearModel {
  readonly f: (x: Value) => Value
  readonly h: (x: Value) => Value
  readonly Q: TensorLike
  readonly R: TensorLike
  readonly x0: TensorLike
  readonly P0: TensorLike
}

/** Filtered moments of a nonlinear filter. */
export interface NonlinearFilterResult {
  /** `[T, n]` and `[T, n, n]`. */
  readonly means: Value
  readonly covariances: Value
  /** Sum of the innovation log-densities (Gaussian approximation). */
  readonly logLikelihood: Value
}

const vecOf = (v: TensorLike): Value => reshape(asTensor(v), [-1])
const mv = (a: Value, x: Value): Value => reshape(matmul(a, reshape(x, [-1, 1])), [-1])

/** The state of the nonlinear Kalman filters after `t` observations: the filtered moments and what the last step saw. */
export interface NonlinearFilterState {
  /** Observations consumed. */
  readonly t: number
  /** The filtered mean `[n]` and covariance `[n, n]` after the last observation. */
  readonly x: Value
  readonly P: Value
  /** The prediction `[n]` the last observation corrected (the prior mean before it). */
  readonly predicted: Value
  /** The innovation `[p]` of the last step (the observation minus its prediction). */
  readonly innovation: Value
  /** Every step's filtered mean and covariance. */
  readonly means: readonly Value[]
  readonly covs: readonly Value[]
  /** The log-likelihood of the observations so far (Gaussian approximation). */
  readonly logLik: Value
}

/** The state of {@link unscentedKalmanAlgorithm}: the filtered moments and the sigma points of the last step. */
export interface UnscentedFilterState extends NonlinearFilterState {
  /** The sigma points `[count, n]` pushed through f at the last step (empty before the first). */
  readonly sigmaPoints: Value
}

/** The state of {@link ensembleKalmanAlgorithm}: the filtered moments and the ensemble itself. */
export interface EnsembleFilterState extends NonlinearFilterState {
  /** The analysis ensemble `[N, n]` (the initial draw before the first observation). */
  readonly ensemble: Value
}

/** The paths every nonlinear Kalman filter logs, as a type. */
export type NonlinearFilterPath =
  | 'filter/mean'
  | 'filter/cov'
  | 'filter/predicted-mean'
  | 'filter/innovation'
  | 'filter/log-likelihood'

/** The paths of the unscented filter: the common ones and the sigma points. */
export type UnscentedFilterPath = NonlinearFilterPath | 'filter/sigma-points'

/** The paths of the ensemble filter: the common ones and the ensemble. */
export type EnsembleFilterPath = NonlinearFilterPath | 'filter/ensemble'

/**
 * What the nonlinear Kalman filters log, by path: the filtered mean and covariance after each observation, the mean
 * it predicted before it, the innovation, and the running log-likelihood.
 */
export const nonlinearFilterContract: Contract<NonlinearFilterState, NonlinearFilterPath> = defineContract({
  'filter/mean': tensor({ role: 'state', shape: ['state'], description: 'the filtered mean after this observation' }),
  'filter/cov': tensor({
    role: 'state',
    shape: ['state', 'state'],
    description: 'the filtered covariance after this observation',
  }),
  'filter/predicted-mean': tensor({
    role: 'state',
    shape: ['state'],
    description: 'the mean predicted before this observation',
  }),
  'filter/innovation': tensor({
    role: 'output',
    shape: ['observation'],
    description: 'the observation minus its prediction',
  }),
  'filter/log-likelihood': scalar({ description: 'log p(z_1..t), Gaussian approximation', direction: 'higher' }),
})

/** The contract of {@link unscentedKalmanAlgorithm}: the common paths and the sigma points. */
export const unscentedFilterContract: Contract<UnscentedFilterState, UnscentedFilterPath> = defineContract({
  ...nonlinearFilterContract,
  'filter/sigma-points': tensor({
    role: 'state',
    shape: ['point', 'state'],
    description: 'the sigma points of the posterior, before they are pushed through the dynamics',
  }),
})

/** The contract of {@link ensembleKalmanAlgorithm}: the common paths and the ensemble. */
export const ensembleFilterContract: Contract<EnsembleFilterState, EnsembleFilterPath> = defineContract({
  ...nonlinearFilterContract,
  'filter/ensemble': tensor({ role: 'state', shape: ['member', 'state'], description: 'the analysis ensemble' }),
})

/** The starting state: the prior moments, nothing seen. */
function startState(x0: Value, P0: Value, p: number): NonlinearFilterState {
  return {
    t: 0,
    x: x0,
    P: P0,
    predicted: x0,
    innovation: fromNumbers(new Float64Array(p), [p]),
    means: [],
    covs: [],
    logLik: mul(sum(x0), 0),
  }
}

/** The result of one correction: the new moments, the prediction they corrected, the innovation, the log-likelihood. */
interface Correction {
  readonly x: Value
  readonly P: Value
  readonly predicted: Value
  readonly innovation: Value
  readonly logLikelihood: Value
}

function advance<S extends NonlinearFilterState>(s: S, c: Correction): S {
  return {
    ...s,
    t: s.t + 1,
    x: c.x,
    P: c.P,
    predicted: c.predicted,
    innovation: c.innovation,
    means: [...s.means, c.x],
    covs: [...s.covs, c.P],
    logLik: add(s.logLik, c.logLikelihood),
  }
}

function logFilter<K extends string>(ctx: StepContext<K>, s: NonlinearFilterState): void {
  const c = ctx as unknown as StepContext<NonlinearFilterPath>
  if (c.wants('filter/mean')) c.log('filter/mean', primalOf(s.x))
  if (c.wants('filter/cov')) c.log('filter/cov', primalOf(s.P))
  if (c.wants('filter/predicted-mean')) c.log('filter/predicted-mean', primalOf(s.predicted))
  if (c.wants('filter/innovation')) c.log('filter/innovation', primalOf(s.innovation))
  if (c.wants('filter/log-likelihood')) c.log('filter/log-likelihood', numbersOf(s.logLik)[0] ?? 0)
}

const filterSchema = channels({
  metrics: { logLikelihood: { description: 'log p(z_1..t) (Gaussian approximation)', direction: 'higher' } },
  tensors: { x: { role: 'state', description: 'the filtered mean' } },
})

function observeFilter(s: NonlinearFilterState): Observation {
  return {
    metrics: { logLikelihood: numbersOf(s.logLik)[0] ?? 0 },
    tensors: { x: { role: 'state', value: numbersOf(s.x) } },
  }
}

function observationsOf(name: string, observations: TensorLike): { readonly Z: Value; readonly T: number } {
  const Z = asTensor(observations)
  const T = Z.shape[0] ?? 0
  if (T === 0) throw new OmaiError('value', `${name}: no observations`)
  return { Z, T }
}

const rowOf = (Z: Value, t: number): Value => reshape(sliceAxis(Z, 0, t, t + 1), [-1])

function resultOf(s: NonlinearFilterState): NonlinearFilterResult {
  return { means: stack([...s.means]), covariances: stack([...s.covs]), logLikelihood: s.logLik }
}

/**
 * The extended Kalman filter as a steppable algorithm: each step consumes one observation, propagates the mean through
 * f and the covariance through its Jacobian F = df/dx (forward-mode autodiff), then updates with the Jacobian of h at
 * the prediction (Joseph form, as filterpy's ExtendedKalmanFilter). The state is tensors and serialises, so a run
 * resumes from a checkpoint; every step logs the filtered mean and covariance, the prediction, the innovation and the
 * log-likelihood (see {@link nonlinearFilterContract}).
 *
 * @param model - Transition, observation, noise and prior.
 * @param observations - `[T, p]`.
 * @returns The algorithm; it is done after the last observation.
 * @throws OmaiError with code `value` for no observations.
 * @concept extended-kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = { f: (x) => x, h: (x) => x, Q: [[0]], R: [[1]], x0: [0], P0: [[1]] }
 * runAlgorithm(filtering.extendedKalmanAlgorithm(model, [[2], [2]]), { maxSteps: 2 }).state.t // => 2
 * ```
 */
export function extendedKalmanAlgorithm(
  model: NonlinearModel,
  observations: TensorLike,
): Algorithm<NonlinearFilterState, NonlinearFilterPath> {
  const { Z, T } = observationsOf('extended-kalman-filter', observations)
  const Q = asTensor(model.Q)
  const R = asTensor(model.R)
  const x0 = vecOf(model.x0)
  return {
    name: 'extended-kalman-filter',
    config: { steps: T },
    schema: filterSchema,
    init: () => startState(x0, asTensor(model.P0), R.shape[0] ?? 0),
    step: (s, ctx) => {
      const F = jacfwd(model.f)(s.x)
      const xp = model.f(s.x)
      const Pp = add(matmul(F, matmul(s.P, transpose(F))), Q)
      const H = jacfwd(model.h)(xp)
      const zp = model.h(xp)
      const z = rowOf(Z, ctx.t)
      // The update linearised at the prediction: z - h(x_p) plays the innovation.
      const up = kalmanUpdate(xp, Pp, add(sub(z, zp), mv(H, xp)), H, R)
      const next = advance(s, {
        x: up.x,
        P: up.P,
        predicted: xp,
        innovation: sub(z, zp),
        logLikelihood: up.logLikelihood,
      })
      logFilter(ctx, next)
      return next
    },
    done: (s) => (s.t >= T ? 'converged' : undefined),
    observe: observeFilter,
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as NonlinearFilterState,
  }
}

/**
 * The extended Kalman filter: propagate the mean through f and the covariance through its Jacobian
 * F = df/dx (forward-mode autodiff), then update with the Jacobian of h at the prediction (Joseph form, as filterpy's
 * ExtendedKalmanFilter). {@link extendedKalmanAlgorithm} run to the end, with the running log-likelihood.
 *
 * @param model - Transition, observation, noise and prior.
 * @param observations - `[T, p]`.
 * @param options - Tracing.
 * @returns Filtered moments and the log-likelihood.
 * @concept extended-kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { mul, toArray } from 'omai-engine/tensor'
 * // With linear f and h the EKF is the Kalman filter.
 * toArray(filtering.extendedKalmanFilter({ f: (x) => x, h: (x) => x, Q: [[0]], R: [[1]], x0: [0], P0: [[1]] }, [[2]]).means) // => [[1]]
 * ```
 */
export function extendedKalmanFilter(
  model: NonlinearModel,
  observations: TensorLike,
  options: Partial<Omit<RunOptions, 'maxSteps' | 'start'>> = {},
): NonlinearFilterResult {
  const algorithm = extendedKalmanAlgorithm(model, observations)
  const T = asTensor(observations).shape[0] ?? 0
  return resultOf(runAlgorithm(algorithm, { maxSteps: T, ...options }).state)
}

/** Sigma points and their weights for the unscented transform. */
export interface SigmaPoints {
  readonly n: number
  readonly count: number
  /** Mean and covariance weights. */
  readonly wm: readonly number[]
  readonly wc: readonly number[]
  /** The points `[count, n]` for a mean and covariance. */
  points(x: Value, P: Value): Value
}

/**
 * Van der Merwe's scaled sigma points (filterpy's MerweScaledSigmaPoints): x, x +- the rows of the upper Cholesky
 * factor of (n + lambda) P, with lambda = alpha^2 (n + kappa) - n and the matching weights.
 *
 * @param n - State dimension.
 * @param alpha - Spread (e.g. 0.1 to 1e-3).
 * @param beta - Prior knowledge of the distribution (2 for Gaussians).
 * @param kappa - Secondary scaling (often 0 or 3 - n).
 * @returns The sigma-point generator.
 * @concept unscented-transform
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * filtering.merweSigmaPoints(2, 1, 2, 1).wm.length // => 5
 * ```
 */
export function merweSigmaPoints(n: number, alpha: number, beta: number, kappa: number): SigmaPoints {
  const lambda = alpha * alpha * (n + kappa) - n
  const c = 0.5 / (n + lambda)
  const wm = Array.from({ length: 2 * n + 1 }, (_, i) => (i === 0 ? lambda / (n + lambda) : c))
  const wc = wm.map((w, i) => (i === 0 ? w + (1 - alpha * alpha + beta) : w))
  return {
    n,
    count: 2 * n + 1,
    wm,
    wc,
    points: (x, P) => {
      // Rows of the upper factor U (U^T U = (n + lambda) P) are the columns of the lower factor.
      const L = cholesky(mul(P, n + lambda))
      const U = transpose(L)
      const rows: Value[] = [x]
      for (let k = 0; k < n; k++) rows.push(add(x, reshape(sliceAxis(U, 0, k, k + 1), [-1])))
      for (let k = 0; k < n; k++) rows.push(sub(x, reshape(sliceAxis(U, 0, k, k + 1), [-1])))
      return stack(rows)
    },
  }
}

/**
 * Julier's sigma points with a single parameter kappa: x, x +- the rows of the Cholesky factor of (n + kappa) P,
 * weights kappa / (n + kappa) and 1 / (2 (n + kappa)).
 *
 * @param n - State dimension.
 * @param kappa - The spread (default 3 - n).
 * @returns The sigma-point generator.
 * @concept unscented-transform
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * filtering.julierSigmaPoints(1, 2).wm // => [0.6666666666666666, 0.16666666666666666, 0.16666666666666666]
 * ```
 */
export function julierSigmaPoints(n: number, kappa: number = 3 - n): SigmaPoints {
  const wm = Array.from({ length: 2 * n + 1 }, (_, i) => (i === 0 ? kappa / (n + kappa) : 0.5 / (n + kappa)))
  return {
    n,
    count: 2 * n + 1,
    wm,
    wc: wm,
    points: (x, P) => {
      const U = transpose(cholesky(mul(P, n + kappa)))
      const rows: Value[] = [x]
      for (let k = 0; k < n; k++) rows.push(add(x, reshape(sliceAxis(U, 0, k, k + 1), [-1])))
      for (let k = 0; k < n; k++) rows.push(sub(x, reshape(sliceAxis(U, 0, k, k + 1), [-1])))
      return stack(rows)
    },
  }
}

/**
 * The unscented transform (filterpy's unscented_transform): the weighted mean and covariance of propagated sigma
 * points, plus a noise covariance.
 *
 * @param sigmas - Points `[count, d]`.
 * @param wm - Mean weights.
 * @param wc - Covariance weights.
 * @param noise - A covariance to add (optional).
 * @returns The mean `[d]` and covariance `[d, d]`.
 * @concept unscented-transform
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(filtering.unscentedTransform([[0], [1], [-1]], [0, 0.5, 0.5], [0, 0.5, 0.5]).P) // => [[1]]
 * ```
 */
export function unscentedTransform(
  sigmas: TensorLike,
  wm: readonly number[],
  wc: readonly number[],
  noise?: TensorLike,
): { readonly x: Value; readonly P: Value } {
  const S = asTensor(sigmas)
  const k = S.shape[0] ?? 0
  const Wm = fromNumbers(wm, [1, k])
  const x = reshape(matmul(Wm, S), [-1])
  const Y = sub(S, reshape(x, [1, -1]))
  const P = matmul(transpose(Y), mul(Y, fromNumbers(wc, [k, 1])))
  return { x, P: noise === undefined ? P : add(P, asTensor(noise)) }
}

/** Options of {@link unscentedKalmanAlgorithm}. */
export interface UnscentedOptions {
  /** The sigma points (default Merwe with alpha 0.1, beta 2, kappa 3 - n). */
  readonly points?: SigmaPoints
  /** Draw fresh sigma points of the prediction for the observation step (default false, as filterpy). */
  readonly redraw?: boolean
}

/**
 * The unscented Kalman filter (Julier and Uhlmann; filterpy's UnscentedKalmanFilter with any sigma points) as a
 * steppable algorithm: sigma points of the posterior are pushed through f, their unscented transform plus Q is the
 * prediction, the same points pushed through h give the predicted observation and the cross covariance, and the update
 * uses the gain Pxz S^-1 with P = P - K S K^T. With `redraw` the observation step uses fresh sigma points of the
 * prediction (the additive-noise UKF of Wan and van der Merwe, exact for linear models); by default the propagated
 * points are reused as filterpy does. Each step logs the common paths and the sigma points it used (see
 * {@link unscentedFilterContract}); the state serialises.
 *
 * @param model - Transition, observation, noise and prior.
 * @param observations - `[T, p]`.
 * @param options - Sigma points and redrawing.
 * @returns The algorithm; it is done after the last observation.
 * @throws OmaiError with code `value` for no observations.
 * @concept unscented-kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = { f: (x) => x, h: (x) => x, Q: [[0]], R: [[1]], x0: [0], P0: [[1]] }
 * runAlgorithm(filtering.unscentedKalmanAlgorithm(model, [[2]]), { maxSteps: 1 }).state.sigmaPoints.shape // => [3, 1]
 * ```
 */
export function unscentedKalmanAlgorithm(
  model: NonlinearModel,
  observations: TensorLike,
  options: UnscentedOptions = {},
): Algorithm<UnscentedFilterState, UnscentedFilterPath> {
  const { Z, T } = observationsOf('unscented-kalman-filter', observations)
  const Q = asTensor(model.Q)
  const R = asTensor(model.R)
  const x0 = vecOf(model.x0)
  const n = x0.shape[0] ?? 0
  const pts = options.points ?? merweSigmaPoints(n, 0.1, 2, 3 - n)
  return {
    name: 'unscented-kalman-filter',
    config: { steps: T, points: pts.count, redraw: options.redraw === true },
    schema: filterSchema,
    init: () => ({
      ...startState(x0, asTensor(model.P0), R.shape[0] ?? 0),
      sigmaPoints: fromNumbers(new Float64Array(0), [0, n]),
    }),
    step: (s, ctx) => {
      const z = rowOf(Z, ctx.t)
      const sig = pts.points(s.x, s.P)
      const sf = stack(Array.from({ length: pts.count }, (_, i) => model.f(reshape(sliceAxis(sig, 0, i, i + 1), [-1]))))
      const pred = unscentedTransform(sf, pts.wm, pts.wc, Q)
      // filterpy reuses the propagated points; redrawing them from the prediction accounts for Q in the update.
      const sx = options.redraw === true ? pts.points(pred.x, symmetricPart(pred.P)) : sf
      const sh = stack(Array.from({ length: pts.count }, (_, i) => model.h(reshape(sliceAxis(sx, 0, i, i + 1), [-1]))))
      const obs = unscentedTransform(sh, pts.wm, pts.wc, R)
      const dx = sub(sx, reshape(pred.x, [1, -1]))
      const dz = sub(sh, reshape(obs.x, [1, -1]))
      const Pxz = matmul(transpose(dx), mul(dz, fromNumbers(pts.wc, [pts.count, 1])))
      const S = symmetricPart(obs.P)
      const K = transpose(solve(S, transpose(Pxz)))
      const y = sub(z, obs.x)
      const x = add(pred.x, mv(K, y))
      const P = symmetricPart(sub(pred.P, matmul(K, matmul(S, transpose(K)))))
      const L = cholesky(S)
      const p = y.shape[0] ?? 0
      const w = solve(S, reshape(y, [-1, 1]))
      const quad = sum(mul(y, reshape(w, [-1])))
      const logdet = mul(sum(logDiag(L)), 2)
      const ll = mul(add(add(quad, logdet), p * Math.log(2 * Math.PI)), -0.5)
      const next: UnscentedFilterState = {
        ...advance(s, { x, P, predicted: pred.x, innovation: y, logLikelihood: ll }),
        sigmaPoints: sig,
      }
      logFilter(ctx, next)
      if (ctx.wants('filter/sigma-points')) ctx.log('filter/sigma-points', primalOf(sig))
      return next
    },
    done: (s) => (s.t >= T ? 'converged' : undefined),
    observe: observeFilter,
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as UnscentedFilterState,
  }
}

/**
 * The unscented Kalman filter (Julier and Uhlmann; filterpy's UnscentedKalmanFilter with any sigma points): sigma
 * points of the posterior are pushed through f, their unscented transform plus Q is the prediction, the same points
 * pushed through h give the predicted observation and the cross covariance, and the update uses the gain
 * Pxz S^-1 with P = P - K S K^T. With `redraw` the observation step uses fresh sigma points of the prediction (the
 * additive-noise UKF of Wan and van der Merwe, exact for linear models); by default the propagated points are reused
 * as filterpy does. {@link unscentedKalmanAlgorithm} run to the end.
 *
 * @param model - Transition, observation, noise and prior.
 * @param observations - `[T, p]`.
 * @param options - Sigma points (default Merwe with alpha 0.1, beta 2, kappa 3 - n), redrawing and tracing.
 * @returns Filtered moments and the log-likelihood.
 * @concept unscented-kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { toArray } from 'omai-engine/tensor'
 * const r = filtering.unscentedKalmanFilter({ f: (x) => x, h: (x) => x, Q: [[0]], R: [[1]], x0: [0], P0: [[1]] }, [[2]])
 * toArray(r.means).map((v) => Math.round(v[0] * 1e10) / 1e10) // => [1]
 * ```
 */
export function unscentedKalmanFilter(
  model: NonlinearModel,
  observations: TensorLike,
  options: UnscentedOptions & Partial<Omit<RunOptions, 'maxSteps' | 'start'>> = {},
): NonlinearFilterResult {
  const { points, redraw, ...run } = options
  const algorithm = unscentedKalmanAlgorithm(model, observations, {
    ...(points === undefined ? {} : { points }),
    ...(redraw === undefined ? {} : { redraw }),
  })
  const T = asTensor(observations).shape[0] ?? 0
  return resultOf(runAlgorithm(algorithm, { maxSteps: T, ...run }).state)
}

function logDiag(L: Value): Value {
  return log(diagonal(L))
}

/** Options of {@link ensembleKalmanAlgorithm}. */
export interface EnsembleOptions {
  /** Number of members (default 100). */
  readonly size?: number
  /** The key the draws derive from (default key 0). */
  readonly key?: Key
  /**
   * Multiplicative covariance inflation: the analysis anomalies are scaled by this factor about the ensemble mean
   * (default 1, none; above 1 counteracts the underestimated spread of a small ensemble).
   */
  readonly inflation?: number
}

/**
 * The ensemble Kalman filter with perturbed observations (Evensen; filterpy's EnsembleKalmanFilter) as a steppable
 * algorithm: N members are propagated through f with process noise, the gain comes from the ensemble covariances, and
 * each member is updated with its own perturbed observation; optional multiplicative inflation widens the analysis
 * spread. Randomness from a key (step t uses the key folded with t + 1), so a resumed run draws the same numbers. The
 * state holds the ensemble itself and each step logs it (see {@link ensembleFilterContract}).
 *
 * @param model - Transition, observation, noise and prior.
 * @param observations - `[T, p]`.
 * @param options - Ensemble size, key and inflation.
 * @returns The algorithm; it is done after the last observation.
 * @throws OmaiError with code `value` for no observations or an ensemble of fewer than two members.
 * @concept ensemble-kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { key } from 'omai-engine/random'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const model = { f: (x) => x, h: (x) => x, Q: [[0]], R: [[1]], x0: [0], P0: [[1]] }
 * runAlgorithm(filtering.ensembleKalmanAlgorithm(model, [[2]], { size: 8, key: key(0) }), { maxSteps: 1 }).state.ensemble.shape // => [8, 1]
 * ```
 */
export function ensembleKalmanAlgorithm(
  model: NonlinearModel,
  observations: TensorLike,
  options: EnsembleOptions = {},
): Algorithm<EnsembleFilterState, EnsembleFilterPath> {
  const { Z, T } = observationsOf('ensemble-kalman-filter', observations)
  const N = options.size ?? 100
  if (N < 2) throw new OmaiError('value', 'ensemble-kalman-filter: need at least two members')
  const root = options.key ?? makeKey(0)
  const inflation = options.inflation ?? 1
  const Q = asTensor(model.Q)
  const R = asTensor(model.R)
  const x0 = vecOf(model.x0)
  const n = x0.shape[0] ?? 0
  const LQ = psdRoot(Q)
  const LR = psdRoot(R)
  const p = R.shape[0] ?? 0
  const LP0 = psdRoot(asTensor(model.P0))
  return {
    name: 'ensemble-kalman-filter',
    config: { steps: T, size: N, inflation },
    schema: filterSchema,
    init: () => ({
      ...startState(x0, asTensor(model.P0), p),
      ensemble: add(reshape(x0, [1, n]), matmul(normal(foldIn(root, 0), [N, n]), transpose(LP0))),
    }),
    step: (s, ctx) => {
      const z = rowOf(Z, ctx.t)
      const kt = foldIn(root, ctx.t + 1)
      const prop = stack(
        Array.from({ length: N }, (_, i) => model.f(reshape(sliceAxis(s.ensemble, 0, i, i + 1), [-1]))),
      )
      const Ef = add(prop, matmul(normal(foldIn(kt, 0), [N, n]), transpose(LQ)))
      const Hz = stack(Array.from({ length: N }, (_, i) => model.h(reshape(sliceAxis(Ef, 0, i, i + 1), [-1]))))
      const xm = mean(Ef, 0)
      const zm = mean(Hz, 0)
      const dX = sub(Ef, reshape(xm, [1, -1]))
      const dZ = sub(Hz, reshape(zm, [1, -1]))
      const Pxz = mul(matmul(transpose(dX), dZ), 1 / (N - 1))
      const Pzz = add(mul(matmul(transpose(dZ), dZ), 1 / (N - 1)), R)
      const K = transpose(solve(Pzz, transpose(Pxz)))
      const perturbed = add(reshape(z, [1, -1]), matmul(normal(foldIn(kt, 1), [N, p]), transpose(LR)))
      const analysis = add(Ef, matmul(sub(perturbed, Hz), transpose(K)))
      const xa = mean(analysis, 0)
      const Ea =
        inflation === 1 ? analysis : add(reshape(xa, [1, -1]), mul(sub(analysis, reshape(xa, [1, -1])), inflation))
      const dA = sub(Ea, reshape(xa, [1, -1]))
      const Pa = mul(matmul(transpose(dA), dA), 1 / (N - 1))
      const y = sub(z, zm)
      const w = solve(Pzz, reshape(y, [-1, 1]))
      const L = cholesky(symmetricPart(Pzz))
      const ll = mul(add(add(sum(mul(y, reshape(w, [-1]))), mul(sum(logDiag(L)), 2)), p * Math.log(2 * Math.PI)), -0.5)
      const next: EnsembleFilterState = {
        ...advance(s, { x: xa, P: Pa, predicted: xm, innovation: y, logLikelihood: ll }),
        ensemble: Ea,
      }
      logFilter(ctx, next)
      if (ctx.wants('filter/ensemble')) ctx.log('filter/ensemble', primalOf(Ea))
      return next
    },
    done: (s) => (s.t >= T ? 'converged' : undefined),
    observe: observeFilter,
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as EnsembleFilterState,
  }
}

/**
 * The ensemble Kalman filter with perturbed observations (Evensen; filterpy's EnsembleKalmanFilter): N members are
 * propagated through f with process noise, the gain comes from the ensemble covariances, and each member is updated
 * with its own perturbed observation. Randomness from a key; the mean and covariance are the ensemble's.
 * {@link ensembleKalmanAlgorithm} run to the end.
 *
 * @param model - Transition, observation, noise and prior.
 * @param observations - `[T, p]`.
 * @param options - Ensemble size (default 100), key, inflation and tracing.
 * @returns Ensemble moments and the Gaussian log-likelihood of the innovations.
 * @concept ensemble-kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'omai-engine/dynamics'
 * import { key } from 'omai-engine/random'
 * const r = filtering.ensembleKalmanFilter({ f: (x) => x, h: (x) => x, Q: [[0]], R: [[1]], x0: [0], P0: [[1]] }, [[2]], { size: 2000, key: key(0) })
 * Math.abs((r.means.data[0] ?? 0) - 1) < 0.1 // => true
 * ```
 */
export function ensembleKalmanFilter(
  model: NonlinearModel,
  observations: TensorLike,
  options: EnsembleOptions & Partial<Omit<RunOptions, 'maxSteps' | 'start'>> = {},
): NonlinearFilterResult & { readonly ensemble: Value } {
  const { size, key, inflation, ...run } = options
  const algorithm = ensembleKalmanAlgorithm(model, observations, {
    ...(size === undefined ? {} : { size }),
    ...(key === undefined ? {} : { key }),
    ...(inflation === undefined ? {} : { inflation }),
  })
  const T = asTensor(observations).shape[0] ?? 0
  const s = runAlgorithm(algorithm, { maxSteps: T, ...run }).state
  return { ...resultOf(s), ensemble: s.ensemble }
}

/** A square root S with S S^T = M for a positive semidefinite M (Cholesky, or the symmetric eigendecomposition). */
function psdRoot(m: Value): Value {
  try {
    return cholesky(m)
  } catch {
    const { eigenvalues: values, eigenvectors: vectors } = eigh(m)
    return mul(vectors, reshape(sqrt(maximum(values, 0)), [1, -1]))
  }
}
