import { WhiteboxError } from '../../base/index.ts'
import { solve, symmetricPart } from '../../numerics/linalg/index.ts'
import {
  add,
  asTensor,
  matmul,
  mul,
  primalOf,
  reshape,
  sliceAxis,
  sub,
  type Tensor,
  type TensorLike,
  transpose,
  type Value,
} from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { numbersOf } from '../ode/_vec.ts'
import { kalmanFilter, type LinearGaussianModel, rtsSmoother } from './kalman.ts'

/** Which parameters EM updates (default all). */
export interface EmLearn {
  readonly F?: boolean
  readonly H?: boolean
  readonly Q?: boolean
  readonly R?: boolean
  readonly x0?: boolean
  readonly P0?: boolean
}

/** The state of {@link emAlgorithm}. */
export interface EmState {
  readonly model: { F: Tensor; H: Tensor; Q: Tensor; R: Tensor; x0: Tensor; P0: Tensor }
  readonly logLikelihood: number
  readonly history: number[]
  readonly converged: boolean
}

const outer = (a: Value, b: Value): Value => matmul(reshape(a, [-1, 1]), reshape(b, [1, -1]))
const row = (v: Value, t: number): Value => reshape(sliceAxis(v, 0, t, t + 1), v.shape.slice(1))
const concrete = (v: Value): Tensor => primalOf(v)

function eStep(
  model: LinearGaussianModel,
  Z: Value,
): {
  ll: number
  S11: Value
  S10: Value
  S00: Value
  Szx: Value
  Szz: Value
  x0: Value
  P0: Value
  T: number
} {
  const F = asTensor(model.F)
  const P0 = asTensor(model.P0)
  const x0 = reshape(asTensor(model.x0), [-1])
  const f = kalmanFilter(model, Z)
  const s = rtsSmoother(model, f)
  const T = Z.shape[0] ?? 0
  // Smooth the initial state too: J0 = P0 F^T (P_1|0)^-1.
  const P10 = row(f.predictedCovariances, 0)
  const J0 = transpose(solve(P10, matmul(F, P0)))
  const xs1 = row(s.means, 0)
  const Ps1 = row(s.covariances, 0)
  const x0s = add(x0, reshape(matmul(J0, reshape(sub(xs1, row(f.predictedMeans, 0)), [-1, 1])), [-1]))
  const P0s = symmetricPart(add(P0, matmul(J0, matmul(sub(Ps1, P10), transpose(J0)))))
  let S11: Value | undefined
  let S10: Value | undefined
  let S00: Value | undefined
  let Szx: Value | undefined
  let Szz: Value | undefined
  const acc = (a: Value | undefined, b: Value): Value => (a === undefined ? b : add(a, b))
  for (let t = 0; t < T; t++) {
    const xt = row(s.means, t)
    const Pt = row(s.covariances, t)
    const xp = t === 0 ? x0s : row(s.means, t - 1)
    const Pp = t === 0 ? P0s : row(s.covariances, t - 1)
    // Lag-one covariance P_{t,t-1|T} = P_t|T J_{t-1}^T (de Jong 1989).
    const Jprev = t === 0 ? J0 : row(s.gains as Value, t - 1)
    const Plag = matmul(Pt, transpose(Jprev))
    const zt = row(Z, t)
    S11 = acc(S11, add(outer(xt, xt), Pt))
    S10 = acc(S10, add(outer(xt, xp), Plag))
    S00 = acc(S00, add(outer(xp, xp), Pp))
    Szx = acc(Szx, outer(zt, xt))
    Szz = acc(Szz, outer(zt, zt))
  }
  return {
    ll: numbersOf(f.logLikelihood)[0] ?? Number.NaN,
    S11: S11 as Value,
    S10: S10 as Value,
    S00: S00 as Value,
    Szx: Szx as Value,
    Szz: Szz as Value,
    x0: x0s,
    P0: P0s,
    T,
  }
}

/**
 * The EM algorithm for the parameters of a linear-Gaussian state-space model (Shumway and Stoffer 1982) as a traced
 * algorithm: the E-step runs the Kalman filter and RTS smoother (with lag-one covariances), the M-step updates
 * F = S10 S00^-1, Q = (S11 - F S10^T) / T, H = Szx S11^-1, R = (Szz - H Szx^T) / T and the initial state from the
 * smoothed x_0. The log-likelihood never decreases.
 *
 * @param initial - The starting model.
 * @param observations - `[T, p]` (fully observed).
 * @param options - Which parameters to learn and the convergence tolerance on the log-likelihood (default 1e-8,
 * relative).
 * @returns The algorithm.
 * @throws WhiteboxError with code `value` for missing observations.
 * @concept expectation-maximisation
 * @example
 * ```ts
 * import { filtering } from 'whitebox-engine/dynamics'
 * filtering.emAlgorithm({ F: [[1]], H: [[1]], Q: [[1]], R: [[1]], x0: [0], P0: [[1]] }, [[1], [2], [1.5]]).name // => 'kalman-em'
 * ```
 */
export function emAlgorithm(
  initial: LinearGaussianModel,
  observations: TensorLike,
  options: { readonly learn?: EmLearn; readonly tol?: number } = {},
): Algorithm<EmState> {
  const Z = asTensor(observations)
  if (numbersOf(Z).some((v) => !Number.isFinite(v)))
    throw new WhiteboxError('value', 'kalmanEm: missing observations are not supported')
  const learn = { F: true, H: true, Q: true, R: true, x0: true, P0: true, ...options.learn }
  const tol = options.tol ?? 1e-8
  const start = {
    F: concrete(asTensor(initial.F)),
    H: concrete(asTensor(initial.H)),
    Q: concrete(asTensor(initial.Q)),
    R: concrete(asTensor(initial.R)),
    x0: concrete(reshape(asTensor(initial.x0), [-1])),
    P0: concrete(asTensor(initial.P0)),
  }
  return {
    name: 'kalman-em',
    config: { learn: { ...learn }, tol },
    schema: channels({
      metrics: { logLikelihood: { description: 'log p(z_1..T) before the update', direction: 'higher' } },
    }),
    init: () => ({ model: start, logLikelihood: Number.NEGATIVE_INFINITY, history: [], converged: false }),
    step: (s) => {
      const e = eStep(s.model, Z)
      const m = { ...s.model }
      const T = e.T
      if (learn.F) m.F = concrete(transpose(solve(e.S00, transpose(e.S10))))
      // The full forms (S11 - F S10^T - S10 F^T + F S00 F^T) / T and likewise for R: they reduce to the familiar
      // (S11 - F S10^T) / T only when F (or H) is learned in the same step, so they stay right for fixed F or H.
      const FS10 = matmul(m.F, transpose(e.S10))
      if (learn.Q)
        m.Q = concrete(
          symmetricPart(
            mul(add(sub(sub(e.S11, FS10), transpose(FS10)), matmul(m.F, matmul(e.S00, transpose(m.F)))), 1 / T),
          ),
        )
      if (learn.H) m.H = concrete(transpose(solve(e.S11, transpose(e.Szx))))
      const HSzx = matmul(m.H, transpose(e.Szx))
      if (learn.R)
        m.R = concrete(
          symmetricPart(
            mul(add(sub(sub(e.Szz, HSzx), transpose(HSzx)), matmul(m.H, matmul(e.S11, transpose(m.H)))), 1 / T),
          ),
        )
      if (learn.x0) m.x0 = concrete(e.x0)
      if (learn.P0) m.P0 = concrete(e.P0)
      const converged =
        Number.isFinite(s.logLikelihood) && Math.abs(e.ll - s.logLikelihood) <= tol * Math.max(1, Math.abs(e.ll))
      s.history.push(e.ll)
      return { model: m, logLikelihood: e.ll, history: s.history, converged }
    },
    done: (s) => (s.converged ? 'converged' : undefined),
    observe: (s) => ({ metrics: { logLikelihood: s.logLikelihood } }),
  }
}

/**
 * Fit a linear-Gaussian state-space model by EM (see {@link emAlgorithm}).
 *
 * @param initial - The starting model.
 * @param observations - `[T, p]`.
 * @param options - Which parameters to learn, tolerance, iteration cap (default 200) and tracing.
 * @returns The fitted model, its log-likelihood history and the convergence flag.
 * @concept expectation-maximisation
 * @example
 * ```ts
 * import { filtering } from 'whitebox-engine/dynamics'
 * const r = filtering.kalmanEm({ F: [[1]], H: [[1]], Q: [[1]], R: [[1]], x0: [0], P0: [[1]] }, [[1], [2], [1.5], [1.7]], { maxIterations: 5, learn: { F: false, H: false } })
 * r.history.every((v, i) => i === 0 || v >= (r.history[i - 1] ?? 0) - 1e-9) // => true
 * ```
 */
export function kalmanEm(
  initial: LinearGaussianModel,
  observations: TensorLike,
  options: { readonly learn?: EmLearn; readonly tol?: number; readonly maxIterations?: number } & Partial<
    Omit<RunOptions, 'maxSteps' | 'start'>
  > = {},
): { readonly model: EmState['model']; readonly history: readonly number[]; readonly converged: boolean } {
  const run = runAlgorithm(emAlgorithm(initial, observations, options), {
    maxSteps: options.maxIterations ?? 200,
    ...options,
  })
  return { model: run.state.model, history: run.state.history, converged: run.state.converged }
}
