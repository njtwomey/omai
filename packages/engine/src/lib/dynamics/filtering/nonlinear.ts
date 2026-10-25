/**
 * Kalman filtering for nonlinear state-space models: the extended Kalman filter, which linearises the dynamics and
 * observation by automatic differentiation; the unscented Kalman filter with Van der Merwe or Julier sigma points and
 * the unscented transform; and the ensemble Kalman filter with perturbed observations, all following filterpy.
 */

import { jacfwd } from '../../autodiff/index.ts'
import { MdmlError } from '../../base/index.ts'
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
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
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

function runFilter<S extends { x: Value; P: Value; means: Value[]; covs: Value[]; logLik: Value }>(
  name: string,
  init: () => S,
  step: (s: S, z: Value, t: number) => S,
  observations: TensorLike,
  options: Partial<Omit<RunOptions, 'maxSteps' | 'start'>>,
): NonlinearFilterResult {
  const Z = asTensor(observations)
  const T = Z.shape[0] ?? 0
  if (T === 0) throw new MdmlError('value', `${name}: no observations`)
  const algorithm: Algorithm<S> = {
    name,
    config: { steps: T },
    schema: channels({
      metrics: { logLikelihood: { description: 'log p(z_1..t) (Gaussian approximation)', direction: 'higher' } },
      tensors: { x: { role: 'state', description: 'the filtered mean' } },
    }),
    init,
    step: (s, ctx) => step(s, reshape(sliceAxis(Z, 0, ctx.t, ctx.t + 1), [-1]), ctx.t),
    observe: (s) => ({
      metrics: { logLikelihood: numbersOf(s.logLik)[0] ?? 0 },
      tensors: { x: { role: 'state', value: numbersOf(s.x) } },
    }),
  }
  const s = runAlgorithm(algorithm, { maxSteps: T, ...options }).state
  return { means: stack(s.means), covariances: stack(s.covs), logLikelihood: s.logLik }
}

interface FilterState {
  x: Value
  P: Value
  means: Value[]
  covs: Value[]
  logLik: Value
}

/**
 * The extended Kalman filter: propagate the mean through f and the covariance through its Jacobian
 * F = df/dx (forward-mode autodiff), then update with the Jacobian of h at the prediction (Joseph form, as filterpy's
 * ExtendedKalmanFilter). A traced algorithm with the running log-likelihood.
 *
 * @param model - Transition, observation, noise and prior.
 * @param observations - `[T, p]`.
 * @param options - Tracing.
 * @returns Filtered moments and the log-likelihood.
 * @concept extended-kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * import { mul, toArray } from 'mdml-engine/tensor'
 * // With linear f and h the EKF is the Kalman filter.
 * toArray(filtering.extendedKalmanFilter({ f: (x) => x, h: (x) => x, Q: [[0]], R: [[1]], x0: [0], P0: [[1]] }, [[2]]).means) // => [[1]]
 * ```
 */
export function extendedKalmanFilter(
  model: NonlinearModel,
  observations: TensorLike,
  options: Partial<Omit<RunOptions, 'maxSteps' | 'start'>> = {},
): NonlinearFilterResult {
  const Q = asTensor(model.Q)
  const R = asTensor(model.R)
  const x0 = vecOf(model.x0)
  return runFilter<FilterState>(
    'extended-kalman-filter',
    () => ({ x: x0, P: asTensor(model.P0), means: [], covs: [], logLik: mul(sum(x0), 0) }),
    (s, z) => {
      const F = jacfwd(model.f)(s.x)
      const xp = model.f(s.x)
      const Pp = add(matmul(F, matmul(s.P, transpose(F))), Q)
      const H = jacfwd(model.h)(xp)
      const zp = model.h(xp)
      // The update linearised at the prediction: z - h(x_p) plays the innovation.
      const up = kalmanUpdate(xp, Pp, add(sub(z, zp), mv(H, xp)), H, R)
      s.means.push(up.x)
      s.covs.push(up.P)
      return { ...s, x: up.x, P: up.P, logLik: add(s.logLik, up.logLikelihood) }
    },
    observations,
    options,
  )
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
 * import { filtering } from 'mdml-engine/dynamics'
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
 * import { filtering } from 'mdml-engine/dynamics'
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
 * import { filtering } from 'mdml-engine/dynamics'
 * import { toArray } from 'mdml-engine/tensor'
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

/**
 * The unscented Kalman filter (Julier and Uhlmann; filterpy's UnscentedKalmanFilter with any sigma points): sigma
 * points of the posterior are pushed through f, their unscented transform plus Q is the prediction, the same points
 * pushed through h give the predicted observation and the cross covariance, and the update uses the gain
 * Pxz S^-1 with P = P - K S K^T. With `redraw` the observation step uses fresh sigma points of the prediction (the
 * additive-noise UKF of Wan and van der Merwe, exact for linear models); by default the propagated points are reused
 * as filterpy does.
 *
 * @param model - Transition, observation, noise and prior.
 * @param observations - `[T, p]`.
 * @param options - Sigma points (default Merwe with alpha 0.1, beta 2, kappa 3 - n), redrawing and tracing.
 * @returns Filtered moments and the log-likelihood.
 * @concept unscented-kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * import { toArray } from 'mdml-engine/tensor'
 * const r = filtering.unscentedKalmanFilter({ f: (x) => x, h: (x) => x, Q: [[0]], R: [[1]], x0: [0], P0: [[1]] }, [[2]])
 * toArray(r.means).map((v) => Math.round(v[0] * 1e10) / 1e10) // => [1]
 * ```
 */
export function unscentedKalmanFilter(
  model: NonlinearModel,
  observations: TensorLike,
  options: { readonly points?: SigmaPoints; readonly redraw?: boolean } & Partial<
    Omit<RunOptions, 'maxSteps' | 'start'>
  > = {},
): NonlinearFilterResult {
  const Q = asTensor(model.Q)
  const R = asTensor(model.R)
  const x0 = vecOf(model.x0)
  const n = x0.shape[0] ?? 0
  const pts = options.points ?? merweSigmaPoints(n, 0.1, 2, 3 - n)
  return runFilter<FilterState>(
    'unscented-kalman-filter',
    () => ({ x: x0, P: asTensor(model.P0), means: [], covs: [], logLik: mul(sum(x0), 0) }),
    (s, z) => {
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
      s.means.push(x)
      s.covs.push(P)
      return { ...s, x, P, logLik: add(s.logLik, ll) }
    },
    observations,
    options,
  )
}

function logDiag(L: Value): Value {
  return log(diagonal(L))
}

/**
 * The ensemble Kalman filter with perturbed observations (Evensen; filterpy's EnsembleKalmanFilter): N members are
 * propagated through f with process noise, the gain comes from the ensemble covariances, and each member is updated
 * with its own perturbed observation. Randomness from a key; the mean and covariance are the ensemble's.
 *
 * @param model - Transition, observation, noise and prior.
 * @param observations - `[T, p]`.
 * @param options - Ensemble size (default 100), key and tracing.
 * @returns Ensemble moments and the Gaussian log-likelihood of the innovations.
 * @concept ensemble-kalman-filter
 * @example
 * ```ts
 * import { filtering } from 'mdml-engine/dynamics'
 * import { key } from 'mdml-engine/random'
 * const r = filtering.ensembleKalmanFilter({ f: (x) => x, h: (x) => x, Q: [[0]], R: [[1]], x0: [0], P0: [[1]] }, [[2]], { size: 2000, key: key(0) })
 * Math.abs((r.means.data[0] ?? 0) - 1) < 0.1 // => true
 * ```
 */
export function ensembleKalmanFilter(
  model: NonlinearModel,
  observations: TensorLike,
  options: { readonly size?: number; readonly key?: Key } & Partial<Omit<RunOptions, 'maxSteps' | 'start'>> = {},
): NonlinearFilterResult & { readonly ensemble: Value } {
  const N = options.size ?? 100
  const root = options.key ?? makeKey(0)
  const Q = asTensor(model.Q)
  const R = asTensor(model.R)
  const x0 = vecOf(model.x0)
  const n = x0.shape[0] ?? 0
  const LQ = psdRoot(Q)
  const LR = psdRoot(R)
  const p = R.shape[0] ?? 0
  const LP0 = psdRoot(asTensor(model.P0))
  let ensemble = add(reshape(x0, [1, n]), matmul(normal(foldIn(root, 0), [N, n]), transpose(LP0)))
  const result = runFilter<FilterState & { E: Value }>(
    'ensemble-kalman-filter',
    () => ({ x: x0, P: asTensor(model.P0), means: [], covs: [], logLik: mul(sum(x0), 0), E: ensemble }),
    (s, z, t) => {
      const kt = foldIn(root, t + 1)
      const prop = stack(Array.from({ length: N }, (_, i) => model.f(reshape(sliceAxis(s.E, 0, i, i + 1), [-1]))))
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
      const Ea = add(Ef, matmul(sub(perturbed, Hz), transpose(K)))
      const xa = mean(Ea, 0)
      const dA = sub(Ea, reshape(xa, [1, -1]))
      const Pa = mul(matmul(transpose(dA), dA), 1 / (N - 1))
      const y = sub(z, zm)
      const w = solve(Pzz, reshape(y, [-1, 1]))
      const L = cholesky(symmetricPart(Pzz))
      const ll = mul(add(add(sum(mul(y, reshape(w, [-1]))), mul(sum(logDiag(L)), 2)), p * Math.log(2 * Math.PI)), -0.5)
      s.means.push(xa)
      s.covs.push(Pa)
      ensemble = Ea
      return { ...s, x: xa, P: Pa, E: Ea, logLik: add(s.logLik, ll) }
    },
    observations,
    options,
  )
  return { ...result, ensemble }
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
