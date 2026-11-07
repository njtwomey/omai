/**
 * Dual coordinate descent for the linear support vector machine (Hsieh, Chang, Lin, Keerthi and Sundararajan 2008, the
 * algorithm of liblinear), for the L1-loss (hinge) and L2-loss (squared hinge) primal min 1/2 |w|^2 + C sum_i loss(1 - y_i w.x_i).
 * Each step is one sweep over the active coordinates in a random order: a coordinate of the dual is minimised exactly, in
 * closed form, and the primal vector w = sum_i alpha_i y_i x_i is updated in O(d). Shrinking drops coordinates stuck at
 * their bounds. The primal objective, the dual objective and their gap are on the tape, so the gap is the stopping
 * certificate. Like liblinear (and `sklearn.svm.LinearSVC`) the intercept is a constant feature and is regularised with
 * the weights. It sits beside SMO (`smo.ts`), the kernel solver for the same dual.
 */

import { type Json, MdmlError } from '../base/index.ts'
import { type Key, key as makeKey, permutation } from '../random/index.ts'
import { asTensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, type RunStatus, runAlgorithm } from '../trace/index.ts'

/** The loss on the margin violation: the hinge (L1) or its square (L2). */
export type DualCdLoss = 'l1' | 'l2'

/** Options of {@link linearSvmDualCdAlgorithm}. */
export interface DualCdOptions {
  /** `'l1'` (hinge, default) or `'l2'` (squared hinge). */
  readonly loss?: DualCdLoss
  /** Stop when the largest projected-gradient violation over the coordinates falls below this (default 1e-4, liblinear's eps). */
  readonly tol?: number
  /** Drop coordinates stuck at a bound from the sweeps (default true). */
  readonly shrinking?: boolean
  /** Append a constant feature for the intercept (default true). */
  readonly fitIntercept?: boolean
  /** The value of the constant feature (liblinear's bias, sklearn's intercept_scaling; default 1). */
  readonly bias?: number
}

/** State of {@link linearSvmDualCdAlgorithm}. */
export interface DualCdState {
  readonly alpha: Float64Array
  /** The primal vector sum alpha_i y_i x_i (the weights, then the intercept times `bias` when fitted). */
  readonly w: Float64Array
  readonly sweep: number
  /** The coordinates still in the sweeps. */
  readonly active: readonly number[]
  /** The largest and smallest projected gradient of the last sweep (the stopping quantity is their difference). */
  readonly maxViolation: number
  readonly minViolation: number
  /** The thresholds the next sweep shrinks against. */
  readonly shrinkMax: number
  readonly shrinkMin: number
  readonly primal: number
  readonly dual: number
  readonly converged: boolean
}

interface SerialisedDualCd {
  readonly alpha: number[]
  readonly w: number[]
  readonly sweep: number
  readonly active: number[]
  readonly maxViolation: number | null
  readonly minViolation: number | null
  readonly shrinkMax: number | null
  readonly shrinkMin: number | null
  readonly primal: number
  readonly dual: number
  readonly converged: boolean
}

interface Problem {
  readonly n: number
  readonly d: number
  /** Row-major `[n, d]` with the constant feature appended when the intercept is fitted. */
  readonly x: Float64Array
  readonly y: Float64Array
  readonly features: number
}

function problemOf(x: TensorLike, y: TensorLike, options: DualCdOptions): Problem {
  const t = asTensor(x)
  if (t.shape.length !== 2) throw new MdmlError('shape', 'linearSvmDualCd: x must be a matrix')
  const n = t.shape[0] ?? 0
  const features = t.shape[1] ?? 0
  const raw = toFlat(t)
  const labels = toFlat(asTensor(y))
  if (labels.length !== n) throw new MdmlError('shape', 'linearSvmDualCd: x and y differ in length')
  const classes = [...new Set(labels)].sort((a, b) => a - b)
  if (classes.length !== 2) throw new MdmlError('value', 'linearSvmDualCd: y needs exactly two distinct values')
  const intercept = options.fitIntercept ?? true
  const bias = options.bias ?? 1
  const d = features + (intercept ? 1 : 0)
  const data = new Float64Array(n * d)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < features; j++) data[i * d + j] = raw[i * features + j] ?? 0
    if (intercept) data[i * d + features] = bias
  }
  return { n, d, x: data, y: Float64Array.from(labels, (v) => (v === classes[1] ? 1 : -1)), features }
}

/** The primal and dual objectives at (alpha, w). */
function objectives(
  p: Problem,
  alpha: Float64Array,
  w: Float64Array,
  C: number,
  loss: DualCdLoss,
): { primal: number; dual: number } {
  let ww = 0
  for (const v of w) ww += v * v
  let hinge = 0
  let a = 0
  let aa = 0
  for (let i = 0; i < p.n; i++) {
    let z = 0
    for (let j = 0; j < p.d; j++) z += (w[j] ?? 0) * (p.x[i * p.d + j] ?? 0)
    const slack = Math.max(0, 1 - (p.y[i] ?? 0) * z)
    hinge += loss === 'l1' ? slack : slack * slack
    a += alpha[i] ?? 0
    aa += (alpha[i] ?? 0) ** 2
  }
  return { primal: 0.5 * ww + C * hinge, dual: a - 0.5 * ww - (loss === 'l2' ? aa / (4 * C) : 0) }
}

/**
 * Dual coordinate descent for the linear SVM as a stepped algorithm, one sweep over the active coordinates per step. The
 * dual variables live in [0, C] (L1 loss) or [0, infinity) with a diagonal term 1 / (2C) (L2 loss); a coordinate is updated by
 * alpha_i <- clip(alpha_i - G_i / Q_ii) with G_i = y_i w.x_i - 1 + D_ii alpha_i and Q_ii = |x_i|^2 + D_ii, and w follows. Coordinates
 * at a bound whose gradient points out by more than the last sweep's extreme violation are shrunk; when the violation is
 * under `tol` on a shrunk set all coordinates return for a final check. Logs `svm/primal`, `svm/dual`, `svm/gap`, the
 * violation and the active-set size.
 *
 * @param x - The features `[n, d]`.
 * @param y - Two class labels `[n]` (the larger value is the positive class).
 * @param C - The penalty.
 * @param options - Loss, tolerance, shrinking, intercept.
 * @returns The algorithm over {@link DualCdState}.
 * @concept support-vector-machines
 * @reference C.-J. Hsieh, K.-W. Chang, C.-J. Lin, S. S. Keerthi and S. Sundararajan, A dual coordinate descent method for
 * large-scale linear SVM (2008).
 * @example
 * ```ts
 * import { linearSvmDualCdAlgorithm } from 'mdml-engine/optim'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = linearSvmDualCdAlgorithm([[2], [3], [-2], [-3]], [1, 1, -1, -1], 1, { fitIntercept: false })
 * runAlgorithm(alg, { maxSteps: 50 }).state.converged // => true
 * ```
 */
export function linearSvmDualCdAlgorithm(
  x: TensorLike,
  y: TensorLike,
  C: number,
  options: DualCdOptions = {},
): Algorithm<DualCdState> {
  if (!(C > 0)) throw new MdmlError('value', 'linearSvmDualCd: C must be positive')
  const p = problemOf(x, y, options)
  const loss = options.loss ?? 'l1'
  const tol = options.tol ?? 1e-4
  const shrinking = options.shrinking ?? true
  const upper = loss === 'l1' ? C : Number.POSITIVE_INFINITY
  const diag = loss === 'l1' ? 0 : 1 / (2 * C)
  const qii = Float64Array.from({ length: p.n }, (_, i) => {
    let s = diag
    for (let j = 0; j < p.d; j++) s += (p.x[i * p.d + j] ?? 0) ** 2
    return s
  })
  return {
    name: 'linear-svm-dual-cd',
    config: { n: p.n, d: p.d, C, loss, tol, shrinking } as Json,
    schema: channels({
      metrics: {
        primal: { description: 'primal objective at the current weights', direction: 'lower' },
        dual: { description: 'dual objective', direction: 'higher' },
        gap: { description: 'primal minus dual, at least zero', direction: 'lower' },
        violation: { description: 'largest minus smallest projected gradient of the sweep', direction: 'lower' },
        active: { description: 'coordinates still in the sweep' },
      },
      tensors: { w: { role: 'param', description: 'the primal weights (then the scaled intercept)' } },
    }),
    init: () => ({
      alpha: new Float64Array(p.n),
      w: new Float64Array(p.d),
      sweep: 0,
      active: Array.from({ length: p.n }, (_, i) => i),
      maxViolation: Number.POSITIVE_INFINITY,
      minViolation: Number.NEGATIVE_INFINITY,
      shrinkMax: Number.POSITIVE_INFINITY,
      shrinkMin: Number.NEGATIVE_INFINITY,
      primal: C * p.n,
      dual: 0,
      converged: false,
    }),
    step: (s, ctx) => {
      const alpha = Float64Array.from(s.alpha)
      const w = Float64Array.from(s.w)
      const order = toFlat(permutation(ctx.key, s.active.length))
      let active = [...s.active]
      const kept: number[] = []
      let pgMax = Number.NEGATIVE_INFINITY
      let pgMin = Number.POSITIVE_INFINITY
      for (const slot of order) {
        const i = active[slot] as number
        let z = 0
        for (let j = 0; j < p.d; j++) z += (w[j] ?? 0) * (p.x[i * p.d + j] ?? 0)
        const yi = p.y[i] ?? 1
        const a = alpha[i] ?? 0
        const g = yi * z - 1 + diag * a
        let pg = 0
        let drop = false
        if (a === 0) {
          if (g > s.shrinkMax && shrinking) drop = true
          else if (g < 0) pg = g
        } else if (a === upper) {
          if (g < s.shrinkMin && shrinking) drop = true
          else if (g > 0) pg = g
        } else pg = g
        if (drop) continue
        kept.push(i)
        pgMax = Math.max(pgMax, pg)
        pgMin = Math.min(pgMin, pg)
        if (Math.abs(pg) > 1e-12) {
          const next = Math.min(Math.max(a - g / (qii[i] ?? 1), 0), upper)
          alpha[i] = next
          const delta = (next - a) * yi
          for (let j = 0; j < p.d; j++) w[j] = (w[j] ?? 0) + delta * (p.x[i * p.d + j] ?? 0)
        }
      }
      active = kept
      let shrinkMax = pgMax <= 0 ? Number.POSITIVE_INFINITY : pgMax
      let shrinkMin = pgMin >= 0 ? Number.NEGATIVE_INFINITY : pgMin
      let converged = false
      if (pgMax - pgMin <= tol) {
        if (active.length === p.n) converged = true
        else {
          active = Array.from({ length: p.n }, (_, i) => i)
          shrinkMax = Number.POSITIVE_INFINITY
          shrinkMin = Number.NEGATIVE_INFINITY
        }
      }
      const { primal, dual } = objectives(p, alpha, w, C, loss)
      return {
        alpha,
        w,
        sweep: s.sweep + 1,
        active,
        maxViolation: pgMax,
        minViolation: pgMin,
        shrinkMax,
        shrinkMin,
        primal,
        dual,
        converged,
      }
    },
    done: (s) => (s.converged ? 'converged' : undefined),
    observe: (s) => ({
      metrics: {
        primal: s.primal,
        dual: s.dual,
        gap: s.primal - s.dual,
        violation: Number.isFinite(s.maxViolation - s.minViolation) ? s.maxViolation - s.minViolation : 0,
        active: s.active.length,
      },
      tensors: { w: { role: 'param', value: s.w } },
    }),
    serialise: (s) => ({
      alpha: Array.from(s.alpha),
      w: Array.from(s.w),
      sweep: s.sweep,
      active: [...s.active],
      maxViolation: Number.isFinite(s.maxViolation) ? s.maxViolation : null,
      minViolation: Number.isFinite(s.minViolation) ? s.minViolation : null,
      shrinkMax: Number.isFinite(s.shrinkMax) ? s.shrinkMax : null,
      shrinkMin: Number.isFinite(s.shrinkMin) ? s.shrinkMin : null,
      primal: s.primal,
      dual: s.dual,
      converged: s.converged,
    }),
    deserialise: (j) => {
      const o = j as unknown as SerialisedDualCd
      return {
        alpha: Float64Array.from(o.alpha),
        w: Float64Array.from(o.w),
        sweep: o.sweep,
        active: o.active,
        maxViolation: o.maxViolation ?? Number.POSITIVE_INFINITY,
        minViolation: o.minViolation ?? Number.NEGATIVE_INFINITY,
        shrinkMax: o.shrinkMax ?? Number.POSITIVE_INFINITY,
        shrinkMin: o.shrinkMin ?? Number.NEGATIVE_INFINITY,
        primal: o.primal,
        dual: o.dual,
        converged: o.converged,
      }
    },
  }
}

/** The result of {@link linearSvmDualCd}. */
export interface DualCdResult {
  /** The weights `[d]`. */
  readonly coef: Float64Array
  /** The intercept (0 when not fitted). */
  readonly intercept: number
  /** The dual variables `[n]`. */
  readonly alpha: Float64Array
  readonly primal: number
  readonly dual: number
  /** primal - dual, at least zero. */
  readonly gap: number
  readonly sweeps: number
  readonly converged: boolean
  /** How the run ended in the one vocabulary: `converged`, or `completed` when the iteration cap was reached. */
  readonly status: RunStatus
}

/**
 * Train a linear SVM by dual coordinate descent ({@link linearSvmDualCdAlgorithm} run to convergence).
 *
 * @param x - The features `[n, d]`.
 * @param y - Two class labels.
 * @param C - The penalty.
 * @param options - Loss, tolerance, shrinking, intercept, sweeps cap (`maxSweeps`, or its alias `maxIterations`; default 1000) and the key.
 * @returns The weights, intercept, dual variables, objectives and gap.
 * @concept support-vector-machines
 * @example
 * ```ts
 * import { linearSvmDualCd } from 'mdml-engine/optim'
 * const r = linearSvmDualCd([[2], [3], [-2], [-3]], [1, 1, -1, -1], 1, { fitIntercept: false, tol: 1e-10 })
 * Math.round(r.coef[0] * 1e6) / 1e6 // => 0.5
 * ```
 */
export function linearSvmDualCd(
  x: TensorLike,
  y: TensorLike,
  C: number,
  options: DualCdOptions & { readonly maxSweeps?: number; readonly maxIterations?: number; readonly key?: Key } = {},
): DualCdResult {
  const { maxSweeps, maxIterations, key, ...rest } = options
  const r = runAlgorithm(linearSvmDualCdAlgorithm(x, y, C, rest), {
    maxSteps: maxSweeps ?? maxIterations ?? 1000,
    key: key ?? makeKey(0),
  })
  const s = r.state
  const intercept = options.fitIntercept ?? true
  const bias = options.bias ?? 1
  const d = s.w.length - (intercept ? 1 : 0)
  return {
    coef: s.w.slice(0, d),
    intercept: intercept ? (s.w[d] ?? 0) * bias : 0,
    alpha: s.alpha,
    primal: s.primal,
    dual: s.dual,
    gap: s.primal - s.dual,
    sweeps: s.sweep,
    converged: s.converged,
    status: r.status,
  }
}
