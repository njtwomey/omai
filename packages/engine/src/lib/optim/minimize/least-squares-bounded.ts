/**
 * The dogbox method for bounded nonlinear least squares (Voglis and Lagaris), a port of scipy's
 * `least_squares(method='dogbox')`: a dogleg trust-region step in a rectangular region, with variables at an active
 * bound held fixed, as a traced algorithm.
 */

import { MdmlError } from '../../base/index.ts'
import { daxpy, ddot, dnrm2, dscal } from '../../numerics/linalg/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { jtv, jv, lstsqSvd, robustCost, runLeastSquares, scaleForLoss } from './_least-squares.ts'
import { configOf, observeVector, vectorDone, vectorSchema } from './_shared.ts'
import type { LeastSquaresOptions, LeastSquaresResult, LeastSquaresState, RobustLoss } from './least-squares.ts'
import {
  type Bounds,
  type Box,
  boxOf,
  clipToBox,
  type FlatResiduals,
  flatResiduals,
  type Parameters,
  type ResidualFunction,
} from './objective.ts'

/** Options of {@link dogbox}. */
export interface DogboxOptions extends LeastSquaresOptions {
  /** Box bounds on the flattened parameters (the start must lie inside). */
  readonly bounds?: Bounds
  /** A robust loss of the residuals (default linear) and its scale (default 1). */
  readonly loss?: RobustLoss
  readonly fScale?: number
  /** Variable scales: a number or one per variable (default 1), or 'jac' for Jacobian column norms. */
  readonly xScale?: 'jac' | ArrayLike<number> | number
}

/** The state of a {@link dogbox} run. */
export interface DogboxState extends LeastSquaresState {
  /** The rectangular trust radius (infinity norm of the scaled step). */
  readonly delta: number
  readonly scaleInv: Float64Array
  readonly onBound: Int8Array
  readonly ratio: number
  /** The infinity norm of the gradient with the active variables removed (the result's optimality). */
  readonly projectedGradNorm: number
}

/** max_i of the largest steps along s from x that stay in [lb, ub], with which bounds are hit (scipy's). */
function stepToBound(
  x: ArrayLike<number>,
  s: ArrayLike<number>,
  lb: ArrayLike<number>,
  ub: ArrayLike<number>,
): { step: number; hits: Int8Array } {
  const n = s.length
  const steps = new Float64Array(n).fill(Infinity)
  for (let i = 0; i < n; i++) {
    const si = s[i] ?? 0
    if (si === 0) continue
    steps[i] = Math.max(((lb[i] ?? 0) - (x[i] ?? 0)) / si, ((ub[i] ?? 0) - (x[i] ?? 0)) / si)
  }
  let min = Infinity
  for (const v of steps) if (v < min) min = v
  const hits = new Int8Array(n)
  for (let i = 0; i < n; i++) if (steps[i] === min) hits[i] = Math.sign(s[i] ?? 0)
  return { step: min, hits }
}

/** The minimiser of t (a t + b) on [lb, ub] (scipy's minimize_quadratic_1d). */
function minimizeQuadratic1d(a: number, b: number, lb: number, ub: number): number {
  const ts = [lb, ub]
  if (a !== 0) {
    const ext = (-0.5 * b) / a
    if (lb < ext && ext < ub) ts.push(ext)
  }
  let best = ts[0] ?? 0
  let bestY = Infinity
  for (const t of ts) {
    const y = t * (a * t + b)
    if (y < bestY) {
      bestY = y
      best = t
    }
  }
  return best
}

/** The dogleg step inside the intersection of the bounds and the rectangular trust region (scipy's dogleg_step). */
function doglegBoxStep(
  x: Float64Array,
  newton: Float64Array,
  g: Float64Array,
  a: number,
  b: number,
  tr: Float64Array,
  lb: Float64Array,
  ub: Float64Array,
): { step: Float64Array; hits: Int8Array; trHit: boolean } {
  const n = x.length
  const lo = new Float64Array(n)
  const hi = new Float64Array(n)
  const origL: boolean[] = []
  const origU: boolean[] = []
  const trL: boolean[] = []
  const trU: boolean[] = []
  for (let i = 0; i < n; i++) {
    const lc = (lb[i] ?? 0) - (x[i] ?? 0)
    const uc = (ub[i] ?? 0) - (x[i] ?? 0)
    const t = tr[i] ?? 0
    lo[i] = Math.max(lc, -t)
    hi[i] = Math.min(uc, t)
    origL.push(lo[i] === lc)
    origU.push(hi[i] === uc)
    trL.push(lo[i] === -t)
    trU.push(hi[i] === t)
  }
  const hitsOut = new Int8Array(n)
  let inside = true
  for (let i = 0; i < n; i++)
    if (!((newton[i] ?? 0) >= (lo[i] ?? 0) && (newton[i] ?? 0) <= (hi[i] ?? 0))) inside = false
  if (inside) return { step: newton, hits: hitsOut, trHit: false }
  const zero = new Float64Array(n)
  const toBounds = stepToBound(zero, dscal(-1, g), lo, hi).step
  const cauchy = dscal(-minimizeQuadratic1d(a, b, 0, toBounds), g)
  const diff = daxpy(-1, cauchy, newton)
  const { step, hits } = stepToBound(cauchy, diff, lo, hi)
  let trHit = false
  for (let i = 0; i < n; i++) {
    const h = hits[i] ?? 0
    if (h < 0 && origL[i] === true) hitsOut[i] = -1
    if (h > 0 && origU[i] === true) hitsOut[i] = 1
    if ((h < 0 && trL[i] === true) || (h > 0 && trU[i] === true)) trHit = true
  }
  return { step: daxpy(step, diff, cauchy), hits: hitsOut, trHit }
}

const columnScaleInv = (J: Float64Array, m: number, n: number, old?: Float64Array): Float64Array => {
  const out = new Float64Array(n)
  for (let j = 0; j < n; j++) {
    let s = 0
    for (let i = 0; i < m; i++) s += (J[i * n + j] ?? 0) ** 2
    const c = Math.sqrt(s)
    out[j] = old === undefined ? (c === 0 ? 1 : c) : Math.max(c, old[j] ?? 0)
  }
  return out
}

/**
 * The dogbox method for bounded nonlinear least squares as a traced algorithm, a port of scipy's least_squares with
 * method 'dogbox' (Voglis and Lagaris): variables at a bound whose gradient points out of the box are frozen, and a
 * dogleg step on the free variables runs inside the intersection of the bounds and a rectangular trust region. Robust
 * losses scale the residuals and Jacobian as scipy does.
 *
 * @param r - The residual function.
 * @param x0 - The start (inside the bounds).
 * @param options - Bounds, robust loss, variable scaling and tolerances (ftol, xtol, gtol: default 1e-8).
 * @returns The algorithm.
 * @throws MdmlError with code `value` when the start lies outside the bounds.
 * @concept least-squares
 * @reference C. Voglis and I. E. Lagaris, A rectangular trust region dogleg approach for unconstrained and bound
 * constrained nonlinear optimization, WSEAS (2004).
 * @example
 * ```ts
 * import { dogboxAlgorithm } from 'mdml-engine/optim'
 * import { sub, type Value } from 'mdml-engine/tensor'
 * dogboxAlgorithm((p: Value) => sub(p, 3), [0], { bounds: [[null, 1]] }).name // => 'dogbox'
 * ```
 */
export function dogboxAlgorithm(
  r: ResidualFunction,
  x0: Parameters,
  options: DogboxOptions = {},
): Algorithm<DogboxState> {
  return dogboxOn(flatResiduals(r, x0), options)
}

function dogboxOn(res: FlatResiduals, options: DogboxOptions): Algorithm<DogboxState> {
  const { m, n } = res
  const box: Box = boxOf(options.bounds, n)
  const ftol = options.ftol ?? 1e-8
  const xtol = options.xtol ?? 1e-8
  const gtol = options.gradTol ?? 1e-8
  const loss = options.loss ?? 'linear'
  const fScale = options.fScale ?? 1
  const maxfev = options.maxEvaluations ?? 100 * n
  const xs = options.xScale ?? 1
  const jacScale = xs === 'jac'
  for (let i = 0; i < n; i++) {
    const xi = res.x0[i] ?? 0
    if (!(xi >= (box.lower[i] ?? -Infinity) && xi <= (box.upper[i] ?? Infinity)))
      throw new MdmlError('value', `dogbox: the start is outside the bounds (variable ${i})`)
  }
  const fixedScaleInv = (): Float64Array =>
    typeof xs === 'number' ? new Float64Array(n).fill(1 / xs) : Float64Array.from(xs as ArrayLike<number>, (v) => 1 / v)
  const model = (f: Float64Array, J: Float64Array) => {
    const scaled = scaleForLoss(J, f, m, n, loss, fScale)
    return { ...scaled, g: jtv(scaled.J, m, n, scaled.f) }
  }
  const projected = (g: Float64Array, onBound: Int8Array): number => {
    let mx = 0
    for (let i = 0; i < n; i++) if (!((onBound[i] ?? 0) * (g[i] ?? 0) < 0)) mx = Math.max(mx, Math.abs(g[i] ?? 0))
    return mx
  }
  return {
    name: 'dogbox',
    config: configOf(options),
    schema: vectorSchema({
      metrics: {
        cost: { description: 'the (robust) cost' },
        delta: { description: 'rectangular trust radius' },
        ratio: { description: 'actual over predicted reduction' },
        projectedGradNorm: { description: 'infinity norm of the gradient on the free variables' },
      },
    }),
    init: () => {
      const x = Float64Array.from(res.x0)
      const f = res.residuals(x)
      const J = res.jacobian(x)
      const md = model(f, J)
      const scaleInv = jacScale ? columnScaleInv(md.J, m, n) : fixedScaleInv()
      let delta = 0
      for (let i = 0; i < n; i++) delta = Math.max(delta, Math.abs((x[i] ?? 0) * (scaleInv[i] ?? 1)))
      if (delta === 0) delta = 1
      const onBound = new Int8Array(n)
      for (let i = 0; i < n; i++) {
        if (x[i] === box.lower[i]) onBound[i] = -1
        else if (x[i] === box.upper[i]) onBound[i] = 1
      }
      const pg = projected(md.g, onBound)
      return {
        x,
        value: robustCost(f, loss, fScale),
        grad: md.g,
        nfev: 1,
        ngev: 1,
        nhev: 0,
        njev: 1,
        step: 0,
        residuals: f,
        jacobian: J,
        delta,
        scaleInv,
        onBound,
        activeMask: onBound,
        ratio: Number.NaN,
        projectedGradNorm: pg,
        ...(pg < gtol ? { stop: { status: 'converged' as const, message: 'gradient norm below gtol' } } : {}),
      }
    },
    step: (s) => {
      const md = model(s.residuals, s.jacobian)
      const g = md.g
      const free: number[] = []
      for (let i = 0; i < n; i++) if (!((s.onBound[i] ?? 0) * (g[i] ?? 0) < 0)) free.push(i)
      const nf = free.length
      const Jf = new Float64Array(m * nf)
      for (let r0 = 0; r0 < m; r0++) for (const [c, j] of free.entries()) Jf[r0 * nf + c] = md.J[r0 * n + j] ?? 0
      const gf = Float64Array.from(free, (j) => g[j] ?? 0)
      const xf = Float64Array.from(free, (j) => s.x[j] ?? 0)
      const lbf = Float64Array.from(free, (j) => box.lower[j] ?? -Infinity)
      const ubf = Float64Array.from(free, (j) => box.upper[j] ?? Infinity)
      const scaleF = Float64Array.from(free, (j) => 1 / (s.scaleInv[j] ?? 1))
      const newton = lstsqSvd(Jf, m, nf, dscal(-1, md.f), Number.EPSILON)
      const v = jv(Jf, m, nf, dscal(-1, gf))
      const a = 0.5 * ddot(v, v)
      const b = -ddot(gf, gf)
      let delta = s.delta
      let nfev = s.nfev
      let actual = -1
      let ratio = 0
      let stop: DogboxState['stop']
      let accepted: { x: Float64Array; f: Float64Array; cost: number; hits: Int8Array } | undefined
      let stepNorm = 0
      while (actual <= 0 && nfev < maxfev) {
        const tr = scaleF.map((sc) => delta * sc)
        const { step: sf, hits, trHit } = doglegBoxStep(xf, newton, gf, a, b, tr, lbf, ubf)
        const step = new Float64Array(n)
        for (const [c, j] of free.entries()) step[j] = sf[c] ?? 0
        const Js = jv(Jf, m, nf, sf)
        const predicted = -(0.5 * ddot(Js, Js) + ddot(gf, sf))
        const xNew = clipToBox(daxpy(1, step, s.x), box)
        const fNew = res.residuals(xNew)
        nfev += 1
        let stepH = 0
        for (let i = 0; i < n; i++) stepH = Math.max(stepH, Math.abs((step[i] ?? 0) * (s.scaleInv[i] ?? 1)))
        if (!fNew.every(Number.isFinite)) {
          delta = 0.25 * stepH
          continue
        }
        const costNew = robustCost(fNew, loss, fScale)
        actual = s.value - costNew
        ratio = predicted > 0 ? actual / predicted : predicted === 0 && actual === 0 ? 1 : 0
        if (ratio < 0.25) delta = 0.25 * stepH
        else if (ratio > 0.75 && trHit) delta *= 2
        stepNorm = dnrm2(step)
        const ftolMet = actual < ftol * s.value && ratio > 0.25
        const xtolMet = stepNorm < xtol * (xtol + dnrm2(s.x))
        if (ftolMet || xtolMet) {
          stop = {
            status: 'converged',
            message: ftolMet && xtolMet ? 'ftol and xtol satisfied' : ftolMet ? 'ftol satisfied' : 'xtol satisfied',
          }
        }
        if (actual > 0) accepted = { x: xNew, f: fNew, cost: costNew, hits }
        if (stop !== undefined) break
      }
      if (accepted === undefined) {
        const pg = projected(g, s.onBound)
        return {
          ...s,
          nfev,
          delta,
          ratio,
          step: 0,
          ...(stop !== undefined
            ? { stop }
            : nfev >= maxfev
              ? { stop: { status: 'failed' as const, message: 'maximum number of residual evaluations reached' } }
              : {}),
          projectedGradNorm: pg,
        }
      }
      const onBound = Int8Array.from(s.onBound)
      for (const [c, j] of free.entries()) onBound[j] = accepted.hits[c] ?? 0
      const x = accepted.x
      for (let i = 0; i < n; i++) {
        if (onBound[i] === -1) x[i] = box.lower[i] ?? 0
        else if (onBound[i] === 1) x[i] = box.upper[i] ?? 0
      }
      const J = res.jacobian(x)
      const mdNew = model(accepted.f, J)
      const scaleInv = jacScale ? columnScaleInv(mdNew.J, m, n, s.scaleInv) : s.scaleInv
      const pg = projected(mdNew.g, onBound)
      const final =
        pg < gtol
          ? { status: 'converged' as const, message: 'gradient norm below gtol' }
          : (stop ??
            (nfev >= maxfev
              ? { status: 'failed' as const, message: 'maximum number of residual evaluations reached' }
              : undefined))
      return {
        x,
        value: accepted.cost,
        grad: mdNew.g,
        nfev,
        ngev: s.ngev + 1,
        nhev: 0,
        njev: s.njev + 1,
        step: stepNorm,
        residuals: accepted.f,
        jacobian: J,
        delta,
        scaleInv,
        onBound,
        activeMask: onBound,
        ratio,
        projectedGradNorm: pg,
        ...(final === undefined ? {} : { stop: final }),
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { cost: s.value, delta: s.delta, ratio: s.ratio, projectedGradNorm: s.projectedGradNorm },
      }),
  }
}

/**
 * Solve a bounded nonlinear least-squares problem by the dogbox method (scipy.optimize.least_squares with method
 * 'dogbox').
 *
 * @param r - The residual function.
 * @param x0 - The start (inside the bounds).
 * @param options - Bounds, robust loss, scaling, tolerances, the evaluation cap (default 100 n), iteration cap and run
 * options.
 * @returns The result, with the active mask of the bounds.
 * @concept least-squares
 * @example
 * ```ts
 * import { dogbox } from 'mdml-engine/optim'
 * import { sub, tensor, toFlat, type Value } from 'mdml-engine/tensor'
 * const r = dogbox((p: Value) => sub(p, tensor([3, -3])), [0, 0], { bounds: [[-1, 1], [-1, 1]] })
 * [toFlat(r.x as Value), r.activeMask] // => [[1, -1], [1, -1]]
 * ```
 */
export function dogbox(r: ResidualFunction, x0: Parameters, options: DogboxOptions = {}): LeastSquaresResult {
  const res = flatResiduals(r, x0)
  return runLeastSquares(dogboxOn(res, options), res, options, 100 * res.n)
}
