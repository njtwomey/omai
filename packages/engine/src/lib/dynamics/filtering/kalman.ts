/**
 * The Kalman filter for linear-Gaussian state-space models, filterpy style: the predict and Joseph-form update steps, a
 * traced filter over a sequence with missing observations and the exact log-likelihood, the Rauch-Tung-Striebel
 * smoother, and the steady-state gain. Also the consistency checks (normalised innovation and estimation error squared)
 * used to test whether a filter's covariances are honest.
 */

import { MdmlError } from '../../base/index.ts'
import { chi2 } from '../../dists/index.ts'
import { cholesky, choleskyLogdet, choSolve, solve, solveDare, symmetricPart } from '../../numerics/linalg/index.ts'
import {
  add,
  asTensor,
  cast,
  eye,
  matmul,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  stack,
  sub,
  sum,
  type TensorLike,
  take,
  transpose,
  type Value,
} from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { fromNumbers, numbersOf } from '../ode/_vec.ts'

/**
 * A linear-Gaussian state-space model x_t = F x_{t-1} + B u_t + w_t, z_t = H x_t + v_t with w ~ N(0, Q),
 * v ~ N(0, R) and x_0 ~ N(x0, P0) (filterpy's KalmanFilter attributes). Matrices may be traced values, so filters and
 * likelihoods are differentiable in them.
 */
export interface LinearGaussianModel {
  readonly F: TensorLike
  readonly H: TensorLike
  readonly Q: TensorLike
  readonly R: TensorLike
  readonly x0: TensorLike
  readonly P0: TensorLike
  readonly B?: TensorLike
}

interface Resolved {
  readonly F: Value
  readonly H: Value
  readonly Q: Value
  readonly R: Value
  readonly x0: Value
  readonly P0: Value
  readonly B: Value | undefined
  readonly n: number
  readonly p: number
}

function resolve(model: LinearGaussianModel): Resolved {
  const F = asTensor(model.F)
  const H = asTensor(model.H)
  return {
    F,
    H,
    Q: asTensor(model.Q),
    R: asTensor(model.R),
    x0: reshape(asTensor(model.x0), [-1]),
    P0: asTensor(model.P0),
    B: model.B === undefined ? undefined : asTensor(model.B),
    n: F.shape[0] ?? 0,
    p: H.shape[0] ?? 0,
  }
}

const mv = (a: Value, x: Value): Value => reshape(matmul(a, reshape(x, [-1, 1])), [-1])

/**
 * The Kalman prediction step: x = F x + B u, P = F P F^T + Q.
 *
 * @param x - Mean `[n]`.
 * @param P - Covariance `[n, n]`.
 * @param F - Transition `[n, n]`.
 * @param Q - Process noise `[n, n]`.
 * @param B - Control matrix and input (optional).
 * @param u - The input.
 * @returns The predicted mean and covariance.
 * @concept kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(filtering.kalmanPredict([1], [[1]], [[2]], [[0.5]]).P) // => [[4.5]]
 * ```
 */
export function kalmanPredict(
  x: TensorLike,
  P: TensorLike,
  F: TensorLike,
  Q: TensorLike,
  B?: TensorLike,
  u?: TensorLike,
): { readonly x: Value; readonly P: Value } {
  const Fv = asTensor(F)
  let xp = mv(Fv, asTensor(x))
  if (B !== undefined && u !== undefined) xp = add(xp, mv(asTensor(B), asTensor(u)))
  return { x: xp, P: add(matmul(Fv, matmul(asTensor(P), transpose(Fv))), asTensor(Q)) }
}

/** The result of one Kalman update. */
export interface KalmanUpdate {
  readonly x: Value
  readonly P: Value
  /** The innovation z - H x. */
  readonly y: Value
  /** The innovation covariance H P H^T + R. */
  readonly S: Value
  readonly K: Value
  /** log N(y; 0, S). */
  readonly logLikelihood: Value
}

/**
 * The Kalman update step in Joseph form (as filterpy): K = P H^T S^-1, x = x + K y,
 * P = (I - K H) P (I - K H)^T + K R K^T, which stays symmetric positive definite under rounding.
 *
 * @param x - Prior mean.
 * @param P - Prior covariance.
 * @param z - The observation `[p]`.
 * @param H - Observation matrix `[p, n]`.
 * @param R - Observation noise `[p, p]`.
 * @param options - `joseph` (default true) or the short form P = (I - K H) P.
 * @returns The posterior, innovation, its covariance, the gain and the log-likelihood of z.
 * @concept kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(filtering.kalmanUpdate([0], [[1]], [2], [[1]], [[1]]).x) // => [1]
 * ```
 */
export function kalmanUpdate(
  x: TensorLike,
  P: TensorLike,
  z: TensorLike,
  H: TensorLike,
  R: TensorLike,
  options: { readonly joseph?: boolean } = {},
): KalmanUpdate {
  const xv = reshape(asTensor(x), [-1])
  const Pv = asTensor(P)
  const Hv = asTensor(H)
  const Rv = asTensor(R)
  const zv = reshape(asTensor(z), [-1])
  const y = sub(zv, mv(Hv, xv))
  const S = symmetricPart(add(matmul(Hv, matmul(Pv, transpose(Hv))), Rv))
  const K = transpose(solve(S, matmul(Hv, Pv)))
  const xn = add(xv, mv(K, y))
  const n = xv.shape[0] ?? 0
  const IKH = sub(eye(n), matmul(K, Hv))
  const Pn =
    options.joseph === false
      ? matmul(IKH, Pv)
      : add(matmul(IKH, matmul(Pv, transpose(IKH))), matmul(K, matmul(Rv, transpose(K))))
  const L = cholesky(S)
  const p = zv.shape[0] ?? 0
  const quad = sum(mul(y, reshape(choSolve(L, reshape(y, [-1, 1])), [-1])))
  const logLikelihood = mul(add(add(quad, choleskyLogdet(L)), p * Math.log(2 * Math.PI)), -0.5)
  return { x: xn, P: symmetricPart(Pn), y, S, K, logLikelihood }
}

/** The output of {@link kalmanFilter}. */
export interface KalmanResult {
  /** Filtered means `[T, n]` and covariances `[T, n, n]`. */
  readonly means: Value
  readonly covariances: Value
  /** One-step predictions `[T, n]`, `[T, n, n]`. */
  readonly predictedMeans: Value
  readonly predictedCovariances: Value
  /** The total log-likelihood log p(z_1..T) (differentiable in the model). */
  readonly logLikelihood: Value
  /** Normalised innovation squared y^T S^-1 y per step (NaN where all of z_t is missing). */
  readonly nis: readonly number[]
  /** Observed components per step. */
  readonly observed: readonly number[]
}

/** The state of {@link kalmanFilterAlgorithm}. */
export interface KalmanState {
  readonly x: Value
  readonly P: Value
  readonly means: Value[]
  readonly covs: Value[]
  readonly pmeans: Value[]
  readonly pcovs: Value[]
  readonly logLik: Value
  readonly nis: number[]
  readonly observed: number[]
}

/**
 * The Kalman filter as a traced algorithm (one observation per step, the log-likelihood and NIS as metrics); missing
 * observation components (NaN) are skipped by updating with the observed rows of H and R only.
 *
 * @param model - The model.
 * @param observations - `[T, p]` (NaN marks missing values).
 * @param options - Inputs `[T, m]` and the Joseph form switch.
 * @returns The algorithm and the number of steps.
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * filtering.kalmanFilterAlgorithm({ F: [[1]], H: [[1]], Q: [[1]], R: [[1]], x0: [0], P0: [[1]] }, [[1], [2]]).steps // => 2
 * ```
 */
export function kalmanFilterAlgorithm(
  model: LinearGaussianModel,
  observations: TensorLike,
  options: { readonly inputs?: TensorLike; readonly joseph?: boolean } = {},
): { readonly algorithm: Algorithm<KalmanState>; readonly steps: number } {
  const m = resolve(model)
  const Z = asTensor(observations)
  const T = Z.shape[0] ?? 0
  const zNumbers = numbersOf(Z)
  const U = options.inputs === undefined ? undefined : asTensor(options.inputs)
  const joseph = options.joseph ?? true
  const algorithm: Algorithm<KalmanState> = {
    name: 'kalman-filter',
    config: { steps: T, joseph },
    schema: channels({
      metrics: {
        logLikelihood: { description: 'log p(z_1..t)', direction: 'higher' },
        nis: { description: 'normalised innovation squared of this step' },
      },
      tensors: { x: { role: 'state', description: 'the filtered mean' } },
    }),
    init: () => ({
      x: m.x0,
      P: m.P0,
      means: [],
      covs: [],
      pmeans: [],
      pcovs: [],
      logLik: mul(sum(m.x0), 0),
      nis: [],
      observed: [],
    }),
    step: (s, ctx) => {
      const t = ctx.t
      const ut = U === undefined ? undefined : reshape(sliceAxis(U, 0, t, t + 1), [-1])
      const pred = kalmanPredict(s.x, s.P, m.F, m.Q, m.B, ut)
      const obs: number[] = []
      for (let i = 0; i < m.p; i++) if (Number.isFinite(zNumbers[t * m.p + i] ?? Number.NaN)) obs.push(i)
      s.pmeans.push(pred.x)
      s.pcovs.push(pred.P)
      if (obs.length === 0) {
        s.means.push(pred.x)
        s.covs.push(pred.P)
        s.nis.push(Number.NaN)
        s.observed.push(0)
        return { ...s, x: pred.x, P: pred.P }
      }
      const idx = fromNumbers(obs, [obs.length])
      const pick = (a: Value, axis: number): Value => take(a, castInt(idx), axis)
      const zt = pick(reshape(sliceAxis(Z, 0, t, t + 1), [-1]), 0)
      const Ht = obs.length === m.p ? m.H : pick(m.H, 0)
      const Rt = obs.length === m.p ? m.R : pick(pick(m.R, 0), 1)
      const up = kalmanUpdate(pred.x, pred.P, zt, Ht, Rt, { joseph })
      const yv = numbersOf(up.y)
      const Sv = primalOf(up.S)
      const w = numbersOf(primalOf(solve(Sv, fromNumbers(yv, [yv.length]))))
      let nis = 0
      for (let i = 0; i < yv.length; i++) nis += (yv[i] ?? 0) * (w[i] ?? 0)
      s.means.push(up.x)
      s.covs.push(up.P)
      s.nis.push(nis)
      s.observed.push(obs.length)
      return { ...s, x: up.x, P: up.P, logLik: add(s.logLik, up.logLikelihood) }
    },
    observe: (s) => ({
      metrics: { logLikelihood: numbersOf(s.logLik)[0] ?? 0, nis: s.nis.at(-1) ?? 0 },
      tensors: { x: { role: 'state', value: numbersOf(s.x) } },
    }),
  }
  return { algorithm, steps: T }
}

function castInt(v: Value): Value {
  return cast(v, 'int32')
}

/**
 * The Kalman filter over a sequence of observations (filterpy's `batch_filter`: predict then update at every step),
 * with the Joseph-form update, missing data and the exact log-likelihood. Built from operations, so the means and the
 * likelihood are differentiable in the model's matrices (maximum likelihood by gradients, or EM).
 *
 * @param model - The model.
 * @param observations - `[T, p]` (NaN marks missing values).
 * @param options - Inputs, Joseph form and tracing.
 * @returns Filtered and predicted moments, the log-likelihood and the NIS per step.
 * @concept kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * import { toArray } from 'mdml-engine/tensor'
 * // A random walk observed with equal noise: the first filtered mean is halfway to z.
 * toArray(filtering.kalmanFilter({ F: [[1]], H: [[1]], Q: [[0]], R: [[1]], x0: [0], P0: [[1]] }, [[2]]).means) // => [[1]]
 * ```
 */
export function kalmanFilter(
  model: LinearGaussianModel,
  observations: TensorLike,
  options: { readonly inputs?: TensorLike; readonly joseph?: boolean } & Partial<
    Omit<RunOptions, 'maxSteps' | 'start'>
  > = {},
): KalmanResult {
  const { algorithm, steps } = kalmanFilterAlgorithm(model, observations, options)
  const s = runAlgorithm(algorithm, { maxSteps: steps, ...options }).state
  if (steps === 0) throw new MdmlError('value', 'kalmanFilter: no observations')
  return {
    means: stack(s.means),
    covariances: stack(s.covs),
    predictedMeans: stack(s.pmeans),
    predictedCovariances: stack(s.pcovs),
    logLikelihood: s.logLik,
    nis: s.nis,
    observed: s.observed,
  }
}

/** The output of {@link rtsSmoother}. */
export interface SmootherResult {
  /** Smoothed means `[T, n]` and covariances `[T, n, n]`. */
  readonly means: Value
  readonly covariances: Value
  /** Smoother gains J_t `[T - 1, n, n]` (for lag-one covariances and EM). */
  readonly gains: Value | undefined
}

/**
 * The Rauch-Tung-Striebel smoother (filterpy's `rts_smoother`): backwards from the last filtered estimate,
 * J = P_t F^T (P_{t+1|t})^-1, x_t|T = x_t + J (x_{t+1|T} - x_{t+1|t}), P_t|T = P_t + J (P_{t+1|T} - P_{t+1|t}) J^T.
 *
 * @param model - The model (F and Q are needed).
 * @param filtered - The output of {@link kalmanFilter}.
 * @returns Smoothed moments and gains.
 * @concept rts-smoother
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * import { toArray } from 'mdml-engine/tensor'
 * const m = { F: [[1]], H: [[1]], Q: [[0]], R: [[1]], x0: [0], P0: [[100]] }
 * // A constant observed twice: the smoothed estimates agree.
 * const s = filtering.rtsSmoother(m, filtering.kalmanFilter(m, [[1], [3]]))
 * toArray(s.means).map((r) => Math.round(r[0] * 1e6) / 1e6) // => [1.99005, 1.99005]
 * ```
 */
export function rtsSmoother(model: LinearGaussianModel, filtered: KalmanResult): SmootherResult {
  const m = resolve(model)
  const T = filtered.means.shape[0] ?? 0
  const at = (v: Value, t: number): Value => reshape(sliceAxis(v, 0, t, t + 1), v.shape.slice(1))
  const xs: Value[] = new Array(T)
  const Ps: Value[] = new Array(T)
  const Js: Value[] = []
  xs[T - 1] = at(filtered.means, T - 1)
  Ps[T - 1] = at(filtered.covariances, T - 1)
  for (let t = T - 2; t >= 0; t--) {
    const P = at(filtered.covariances, t)
    const Pp = add(matmul(m.F, matmul(P, transpose(m.F))), m.Q)
    const xp = mv(m.F, at(filtered.means, t))
    const step = rtsStep(
      { x: at(filtered.means, t), P },
      { x: xp, P: Pp },
      { x: xs[t + 1] as Value, P: Ps[t + 1] as Value },
      m.F,
    )
    xs[t] = step.x
    Ps[t] = step.P
    Js.unshift(step.J)
  }
  return { means: stack(xs), covariances: stack(Ps), gains: Js.length === 0 ? undefined : stack(Js) }
}

/**
 * One backward step of the Rauch-Tung-Striebel smoother, for a transition `F` that may differ from step to step: from
 * the filtered moments at t, the one-step prediction of t + 1 from them and the smoothed moments at t + 1,
 * `J = P_t F^T (P_{t+1|t})^-1`, `x_t|T = x_t + J (x_{t+1|T} - x_{t+1|t})`,
 * `P_t|T = P_t + J (P_{t+1|T} - P_{t+1|t}) J^T`. {@link rtsSmoother} is this over a whole run of a fixed model.
 *
 * @param filtered - The filtered mean `[n]` and covariance `[n, n]` at t.
 * @param predicted - The prediction of t + 1 made from them: mean and covariance.
 * @param smoothed - The smoothed mean and covariance at t + 1.
 * @param F - The transition from t to t + 1.
 * @returns The smoothed mean and covariance at t and the gain J.
 * @concept rts-smoother
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * import { item } from 'mdml-engine/tensor'
 * // A random walk observed once: the smoothed past follows the smoothed future by the gain.
 * const s = filtering.rtsStep({ x: [0], P: [[1]] }, { x: [0], P: [[2]] }, { x: [1], P: [[1]] }, [[1]])
 * item(s.x) // => 0.5
 * ```
 */
export function rtsStep(
  filtered: { readonly x: TensorLike; readonly P: TensorLike },
  predicted: { readonly x: TensorLike; readonly P: TensorLike },
  smoothed: { readonly x: TensorLike; readonly P: TensorLike },
  F: TensorLike,
): { readonly x: Value; readonly P: Value; readonly J: Value } {
  const Fv = asTensor(F)
  const P = asTensor(filtered.P)
  const Pp = asTensor(predicted.P)
  const J = transpose(solve(Pp, matmul(Fv, P)))
  const x = add(asTensor(filtered.x), mv(J, sub(asTensor(smoothed.x), asTensor(predicted.x))))
  return {
    x,
    P: symmetricPart(add(P, matmul(J, matmul(sub(asTensor(smoothed.P), Pp), transpose(J))))),
    J,
  }
}

/**
 * The steady-state Kalman filter: the limiting prior covariance P from the discrete Riccati equation and the constant
 * gain K = P H^T (H P H^T + R)^-1.
 *
 * @param F - Transition.
 * @param H - Observation matrix.
 * @param Q - Process noise.
 * @param R - Observation noise.
 * @returns The gain and the prior and posterior steady-state covariances.
 * @concept kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * import { item } from 'mdml-engine/tensor'
 * // Random walk with q = r = 1: P = (1 + sqrt 5) / 2.
 * Math.abs(item(filtering.steadyStateKalman([[1]], [[1]], [[1]], [[1]]).prior) - (1 + Math.sqrt(5)) / 2) < 1e-10 // => true
 * ```
 */
export function steadyStateKalman(
  F: TensorLike,
  H: TensorLike,
  Q: TensorLike,
  R: TensorLike,
): { readonly K: Value; readonly prior: Value; readonly posterior: Value } {
  const Fv = asTensor(F)
  const Hv = asTensor(H)
  const prior = solveDare(transpose(Fv), transpose(Hv), Q, R)
  const S = add(matmul(Hv, matmul(prior, transpose(Hv))), asTensor(R))
  const K = transpose(solve(S, matmul(Hv, prior)))
  const n = Fv.shape[0] ?? 0
  return { K, prior, posterior: matmul(sub(eye(n), matmul(K, Hv)), prior) }
}

/**
 * The log-likelihood log p(z_1..T) of a linear-Gaussian model by the prediction-error decomposition (the Kalman
 * filter's sum of innovation log-densities), differentiable in the model.
 *
 * @param model - The model.
 * @param observations - `[T, p]`.
 * @returns The log-likelihood.
 * @concept kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * import { item } from 'mdml-engine/tensor'
 * Math.abs(item(filtering.kalmanLogLikelihood({ F: [[1]], H: [[1]], Q: [[0]], R: [[1]], x0: [0], P0: [[1]] }, [[0]])) + 0.5 * Math.log(4 * Math.PI)) < 1e-12 // => true
 * ```
 */
export function kalmanLogLikelihood(model: LinearGaussianModel, observations: TensorLike): Value {
  return kalmanFilter(model, observations).logLikelihood
}

/** The outcome of {@link nisTest}. */
export interface ConsistencyReport {
  /** The mean NIS over the observed steps. */
  readonly meanNis: number
  /** The chi-square acceptance interval of the sum of NIS at the given level. */
  readonly interval: readonly [number, number]
  readonly consistent: boolean
  readonly degreesOfFreedom: number
}

/**
 * The normalised-innovation-squared consistency test: for a correct model the sum of NIS over the run is chi-square
 * with sum(p_t) degrees of freedom; the filter is consistent when it falls in the central (1 - alpha) interval.
 *
 * @param filtered - The output of {@link kalmanFilter}.
 * @param alpha - The level (default 0.05).
 * @returns The mean NIS, the interval and the verdict.
 * @concept filter-consistency
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * const f = filtering.kalmanFilter({ F: [[1]], H: [[1]], Q: [[0]], R: [[1]], x0: [0], P0: [[1]] }, [[0.3], [-0.5], [1.1], [-0.2]])
 * filtering.nisTest(f).degreesOfFreedom // => 4
 * ```
 */
export function nisTest(filtered: KalmanResult, alpha = 0.05): ConsistencyReport {
  let total = 0
  let dof = 0
  let count = 0
  filtered.nis.forEach((v, t) => {
    if (!Number.isFinite(v)) return
    total += v
    dof += filtered.observed[t] ?? 0
    count += 1
  })
  const d = chi2(dof)
  const lo = numbersOf(d.ppf(alpha / 2))[0] ?? 0
  const hi = numbersOf(d.ppf(1 - alpha / 2))[0] ?? Number.POSITIVE_INFINITY
  return {
    meanNis: count === 0 ? Number.NaN : total / count,
    interval: [lo, hi],
    consistent: total >= lo && total <= hi,
    degreesOfFreedom: dof,
  }
}

/**
 * The normalised estimation error squared (NEES) of filtered estimates against known true states: e^T P^-1 e per
 * step, chi-square with n degrees of freedom for a consistent filter.
 *
 * @param filtered - The output of {@link kalmanFilter} (or smoothed moments).
 * @param truth - True states `[T, n]`.
 * @returns NEES per step.
 * @concept filter-consistency
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * const f = filtering.kalmanFilter({ F: [[1]], H: [[1]], Q: [[0]], R: [[1]], x0: [0], P0: [[1]] }, [[2]])
 * filtering.nees(f, [[0]]) // => [2]
 * ```
 */
export function nees(filtered: Pick<KalmanResult, 'means' | 'covariances'>, truth: TensorLike): number[] {
  const X = primalOf(filtered.means)
  const P = primalOf(filtered.covariances)
  const tr = asTensor(truth)
  const T = X.shape[0] ?? 0
  const n = X.shape[1] ?? 0
  const tv = numbersOf(tr)
  return Array.from({ length: T }, (_, t) => {
    const e = Float64Array.from({ length: n }, (_, i) => (X.data[t * n + i] ?? 0) - (tv[t * n + i] ?? 0))
    const Pt = fromNumbers(P.data.slice(t * n * n, (t + 1) * n * n), [n, n])
    const w = numbersOf(primalOf(solve(Pt, fromNumbers(e, [n]))))
    let s = 0
    for (let i = 0; i < n; i++) s += (e[i] ?? 0) * (w[i] ?? 0)
    return s
  })
}
