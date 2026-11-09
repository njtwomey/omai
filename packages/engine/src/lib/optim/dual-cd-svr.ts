/**
 * Dual coordinate descent for the linear support vector regression (Ho and Lin 2012, the SVR solver of liblinear), for the
 * L1-loss (epsilon-insensitive) and L2-loss (squared epsilon-insensitive) primal min 1/2 |w|^2 + C sum_i loss(|w.x_i - y_i| -
 * epsilon). Each step is one sweep over the coordinates in a random order; a coordinate of the dual is minimised exactly (a
 * soft-threshold Newton step clipped to its box) and the primal vector w = sum_i beta_i x_i is updated in O(d). The primal
 * objective, the dual objective and their gap are on the tape. Like liblinear (and `sklearn.svm.LinearSVR`) the intercept is
 * a constant feature, regularised with the weights. It is the regression companion of `dual-cd.ts`.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { type Key, key as makeKey, permutation } from '../random/index.ts'
import { asTensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, type RunStatus, runAlgorithm } from '../trace/index.ts'

/** The loss on the residual outside the tube: the epsilon-insensitive (L1) or its square (L2). */
export type DualCdSvrLoss = 'l1' | 'l2'

/** Options of {@link linearSvrDualCdAlgorithm}. */
export interface DualCdSvrOptions {
  /** `'l1'` (epsilon-insensitive, default) or `'l2'` (squared epsilon-insensitive). */
  readonly loss?: DualCdSvrLoss
  /** The half-width of the tube inside which the residual costs nothing (default 0). */
  readonly epsilon?: number
  /** Stop when the largest projected-gradient violation of a sweep falls below this (default 1e-4). */
  readonly tol?: number
  /** Append a constant feature for the intercept (default true). */
  readonly fitIntercept?: boolean
  /** The value of the constant feature (liblinear's bias, sklearn's intercept_scaling; default 1). */
  readonly bias?: number
  /** A multiplier of `C` for each row `[n]` (sample weights; default 1). */
  readonly weights?: TensorLike
}

/** State of {@link linearSvrDualCdAlgorithm}. */
export interface DualCdSvrState {
  /** The dual variables, one per row, in `[-C_i, C_i]` (L1 loss) or unbounded (L2 loss). */
  readonly beta: Float64Array
  /** The primal vector sum beta_i x_i (the weights, then the intercept times `bias` when fitted). */
  readonly w: Float64Array
  readonly sweep: number
  /** The largest projected gradient of the last sweep. */
  readonly violation: number
  readonly primal: number
  readonly dual: number
  readonly converged: boolean
}

interface SerialisedState {
  readonly beta: number[]
  readonly w: number[]
  readonly sweep: number
  readonly violation: number | null
  readonly primal: number
  readonly dual: number
  readonly converged: boolean
}

/**
 * Dual coordinate descent for the linear SVR as a stepped algorithm, one sweep per step. With `lambda_i = 0`, `U_i = C_i`
 * (L1 loss) or `lambda_i = 1 / (2 C_i)`, `U_i = infinity` (L2 loss), a coordinate has gradient `G = w.x_i - y_i + lambda_i
 * beta_i` and curvature `H = |x_i|^2 + lambda_i`; the Newton step is taken on the side of the kink of `epsilon |beta_i|` that
 * the gradient points to (`-(G + epsilon) / H` or `-(G - epsilon) / H`, or to zero between them) and clipped to `[-U_i, U_i]`.
 * Logs `svr/primal`, `svr/dual`, `svr/gap` and the violation.
 *
 * @param x - The features `[n, d]`.
 * @param y - The targets `[n]`.
 * @param C - The penalty.
 * @param options - Loss, tube, tolerance, intercept, row weights.
 * @returns The algorithm over {@link DualCdSvrState}.
 * @throws OmaiError `value` for a non-positive `C`, a negative `epsilon` or bad weights, `shape` for mismatched inputs.
 * @concept support-vector-machines
 * @reference C.-H. Ho and C.-J. Lin, Large-scale linear support vector regression (2012).
 * @example
 * ```ts
 * import { linearSvrDualCdAlgorithm } from 'omai-engine/optim'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const alg = linearSvrDualCdAlgorithm([[1], [2], [3], [4]], [1, 2, 3, 4], 100, { fitIntercept: false, tol: 1e-9 })
 * Math.round((runAlgorithm(alg, { maxSteps: 500 }).state.w[0] ?? 0) * 100) / 100 // => 1
 * ```
 */
export function linearSvrDualCdAlgorithm(
  x: TensorLike,
  y: TensorLike,
  C: number,
  options: DualCdSvrOptions = {},
): Algorithm<DualCdSvrState> {
  if (!(C > 0)) throw new OmaiError('value', 'linearSvrDualCd: C must be positive')
  const epsilon = options.epsilon ?? 0
  if (!(epsilon >= 0)) throw new OmaiError('value', 'linearSvrDualCd: epsilon must be non-negative')
  const t = asTensor(x)
  if (t.shape.length !== 2) throw new OmaiError('shape', 'linearSvrDualCd: x must be a matrix')
  const n = t.shape[0] ?? 0
  const features = t.shape[1] ?? 0
  const targets = toFlat(asTensor(y))
  if (targets.length !== n) throw new OmaiError('shape', 'linearSvrDualCd: x and y differ in length')
  const loss = options.loss ?? 'l1'
  const tol = options.tol ?? 1e-4
  const intercept = options.fitIntercept ?? true
  const bias = options.bias ?? 1
  const d = features + (intercept ? 1 : 0)
  const raw = toFlat(t)
  const data = new Float64Array(n * d)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < features; j++) data[i * d + j] = raw[i * features + j] ?? 0
    if (intercept) data[i * d + features] = bias
  }
  const weights = options.weights === undefined ? undefined : toFlat(asTensor(options.weights))
  if (weights !== undefined && (weights.length !== n || weights.some((v) => !(v > 0))))
    throw new OmaiError('value', 'linearSvrDualCd: weights must be positive, one per row')
  const cs = Float64Array.from({ length: n }, (_, i) => C * (weights?.[i] ?? 1))
  const upper = (i: number): number => (loss === 'l1' ? (cs[i] ?? C) : Number.POSITIVE_INFINITY)
  const lambda = (i: number): number => (loss === 'l1' ? 0 : 1 / (2 * (cs[i] ?? C)))
  const qii = Float64Array.from({ length: n }, (_, i) => {
    let s = 0
    for (let j = 0; j < d; j++) s += (data[i * d + j] ?? 0) ** 2
    return s
  })
  const dot = (w: Float64Array, i: number): number => {
    let z = 0
    for (let j = 0; j < d; j++) z += (w[j] ?? 0) * (data[i * d + j] ?? 0)
    return z
  }
  const objectives = (beta: Float64Array, w: Float64Array): { primal: number; dual: number } => {
    let ww = 0
    for (const v of w) ww += v * v
    let primal = 0.5 * ww
    let dual = 0.5 * ww
    for (let i = 0; i < n; i++) {
      const excess = Math.max(0, Math.abs(dot(w, i) - (targets[i] ?? 0)) - epsilon)
      primal += (cs[i] ?? 0) * (loss === 'l1' ? excess : excess * excess)
      const b = beta[i] ?? 0
      dual += 0.5 * lambda(i) * b * b - (targets[i] ?? 0) * b + epsilon * Math.abs(b)
    }
    return { primal, dual: -dual }
  }
  return {
    name: 'linear-svr-dual-cd',
    config: { n, d, C, loss, epsilon, tol, weighted: weights !== undefined } as Json,
    schema: channels({
      metrics: {
        primal: { description: 'primal objective at the current weights', direction: 'lower' },
        dual: { description: 'dual objective', direction: 'higher' },
        gap: { description: 'primal minus dual, at least zero', direction: 'lower' },
        violation: { description: 'largest projected gradient of the sweep', direction: 'lower' },
      },
      tensors: { w: { role: 'param', description: 'the primal weights (then the scaled intercept)' } },
    }),
    init: () => {
      const beta = new Float64Array(n)
      const w = new Float64Array(d)
      return { beta, w, sweep: 0, violation: Number.POSITIVE_INFINITY, ...objectives(beta, w), converged: false }
    },
    step: (s, ctx) => {
      const beta = Float64Array.from(s.beta)
      const w = Float64Array.from(s.w)
      let worst = 0
      for (const i of toFlat(permutation(ctx.key, n))) {
        const b = beta[i] ?? 0
        const u = upper(i)
        const g = dot(w, i) - (targets[i] ?? 0) + lambda(i) * b
        const h = (qii[i] ?? 0) + lambda(i)
        const gp = g + epsilon
        const gn = g - epsilon
        let violation = 0
        if (b === 0) violation = Math.max(-gp, gn, 0)
        else if (b >= u) violation = Math.max(gp, 0)
        else if (b <= -u) violation = Math.max(-gn, 0)
        else violation = b > 0 ? Math.abs(gp) : Math.abs(gn)
        worst = Math.max(worst, violation)
        const step = gp < h * b ? -gp / h : gn > h * b ? -gn / h : -b
        if (Math.abs(step) < 1e-12) continue
        const next = Math.min(Math.max(b + step, -u), u)
        beta[i] = next
        const delta = next - b
        if (delta !== 0) for (let j = 0; j < d; j++) w[j] = (w[j] ?? 0) + delta * (data[i * d + j] ?? 0)
      }
      return { beta, w, sweep: s.sweep + 1, violation: worst, ...objectives(beta, w), converged: worst <= tol }
    },
    done: (s) => (s.converged ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        primal: s.primal,
        dual: s.dual,
        gap: s.primal - s.dual,
        violation: Number.isFinite(s.violation) ? s.violation : 0,
      },
      tensors: { w: { role: 'param', value: s.w } },
    }),
    serialise: (s) =>
      ({
        beta: Array.from(s.beta),
        w: Array.from(s.w),
        sweep: s.sweep,
        violation: Number.isFinite(s.violation) ? s.violation : null,
        primal: s.primal,
        dual: s.dual,
        converged: s.converged,
      }) satisfies SerialisedState as unknown as Json,
    deserialise: (j) => {
      const o = j as unknown as SerialisedState
      return {
        beta: Float64Array.from(o.beta),
        w: Float64Array.from(o.w),
        sweep: o.sweep,
        violation: o.violation ?? Number.POSITIVE_INFINITY,
        primal: o.primal,
        dual: o.dual,
        converged: o.converged,
      }
    },
  }
}

/** The result of {@link linearSvrDualCd}. */
export interface DualCdSvrResult {
  /** The weights `[d]`. */
  readonly coef: Float64Array
  /** The intercept (0 when not fitted). */
  readonly intercept: number
  /** The dual variables `[n]`. */
  readonly beta: Float64Array
  readonly primal: number
  readonly dual: number
  /** primal - dual, at least zero. */
  readonly gap: number
  readonly sweeps: number
  readonly converged: boolean
  readonly status: RunStatus
}

/**
 * Train a linear SVR by dual coordinate descent ({@link linearSvrDualCdAlgorithm} run to convergence).
 *
 * @param x - The features `[n, d]`.
 * @param y - The targets `[n]`.
 * @param C - The penalty.
 * @param options - Loss, tube, tolerance, intercept, weights, sweeps cap (`maxSweeps`, default 1000) and the key.
 * @returns The weights, intercept, dual variables, objectives and gap.
 * @concept support-vector-machines
 * @example
 * ```ts
 * import { linearSvrDualCd } from 'omai-engine/optim'
 * const r = linearSvrDualCd([[1], [2], [3], [4]], [2, 4, 6, 8], 1000, { fitIntercept: false, tol: 1e-10 })
 * Math.round(r.coef[0] * 1000) / 1000 // => 2
 * ```
 */
export function linearSvrDualCd(
  x: TensorLike,
  y: TensorLike,
  C: number,
  options: DualCdSvrOptions & { readonly maxSweeps?: number; readonly key?: Key } = {},
): DualCdSvrResult {
  const { maxSweeps, key, ...rest } = options
  const r = runAlgorithm(linearSvrDualCdAlgorithm(x, y, C, rest), {
    maxSteps: maxSweeps ?? 1000,
    key: key ?? makeKey(0),
  })
  const s = r.state
  const intercept = options.fitIntercept ?? true
  const bias = options.bias ?? 1
  const d = s.w.length - (intercept ? 1 : 0)
  return {
    coef: s.w.slice(0, d),
    intercept: intercept ? (s.w[d] ?? 0) * bias : 0,
    beta: s.beta,
    primal: s.primal,
    dual: s.dual,
    gap: s.primal - s.dual,
    sweeps: s.sweep,
    converged: s.converged,
    status: r.status,
  }
}
