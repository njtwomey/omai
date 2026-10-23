import { WhiteboxError } from '../../base/index.ts'
import { daxpy, ddot, dgesvd, dnrm2, dscal } from '../../numerics/linalg/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { jtv, jv, robustCost, runLeastSquares, scaleForLoss } from './_least-squares.ts'
import { configOf, observeVector, vectorDone, vectorSchema } from './_shared.ts'
import type { LeastSquaresResult, LeastSquaresState } from './least-squares.ts'
import type { DogboxOptions } from './least-squares-bounded.ts'
import {
  type Box,
  boxOf,
  type FlatResiduals,
  flatResiduals,
  type Parameters,
  type ResidualFunction,
} from './objective.ts'

/** Options of {@link trf} (the same as dogbox's: bounds, robust loss, scaling and tolerances). */
export type TrfOptions = DogboxOptions

/** The state of a {@link trf} run. */
export interface TrfState extends LeastSquaresState {
  /** The trust radius (in the scaled variables). */
  readonly delta: number
  /** The Levenberg-Marquardt parameter of the last subproblem. */
  readonly lmParameter: number
  readonly scaleInv: Float64Array
  readonly ratio: number
  /** The infinity norm of the Coleman-Li scaled gradient (the result's optimality). */
  readonly projectedGradNorm: number
}

/** The Coleman-Li scaling vector v and its derivative dv (scipy's CL_scaling_vector). */
function clScaling(x: Float64Array, g: Float64Array, box: Box): { v: Float64Array; dv: Float64Array } {
  const n = x.length
  const v = new Float64Array(n).fill(1)
  const dv = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const gi = g[i] ?? 0
    const ub = box.upper[i] ?? Infinity
    const lb = box.lower[i] ?? -Infinity
    if (gi < 0 && Number.isFinite(ub)) {
      v[i] = ub - (x[i] ?? 0)
      dv[i] = -1
    }
    if (gi > 0 && Number.isFinite(lb)) {
      v[i] = (x[i] ?? 0) - lb
      dv[i] = 1
    }
  }
  return { v, dv }
}

/** Active constraints within rtol (scipy's find_active_constraints). */
function activeConstraints(x: Float64Array, box: Box, rtol: number): Int8Array {
  const n = x.length
  const out = new Int8Array(n)
  for (let i = 0; i < n; i++) {
    const lb = box.lower[i] ?? -Infinity
    const ub = box.upper[i] ?? Infinity
    const xi = x[i] ?? 0
    if (rtol === 0) {
      if (xi <= lb) out[i] = -1
      else if (xi >= ub) out[i] = 1
      continue
    }
    const ld = xi - lb
    const ud = ub - xi
    if (Number.isFinite(lb) && ld <= Math.min(ud, rtol * Math.max(1, Math.abs(lb)))) out[i] = -1
    else if (Number.isFinite(ub) && ud <= Math.min(ld, rtol * Math.max(1, Math.abs(ub)))) out[i] = 1
  }
  return out
}

/** Move points on or near a bound strictly inside (scipy's make_strictly_feasible). */
function strictlyFeasible(x: Float64Array, box: Box, rstep: number): Float64Array {
  const active = activeConstraints(x, box, rstep)
  const out = Float64Array.from(x)
  for (let i = 0; i < x.length; i++) {
    const lb = box.lower[i] ?? -Infinity
    const ub = box.upper[i] ?? Infinity
    if (active[i] === -1) out[i] = rstep === 0 ? nextAfter(lb, ub) : lb + rstep * Math.max(1, Math.abs(lb))
    else if (active[i] === 1) out[i] = rstep === 0 ? nextAfter(ub, lb) : ub - rstep * Math.max(1, Math.abs(ub))
    if ((out[i] ?? 0) < lb || (out[i] ?? 0) > ub) out[i] = 0.5 * (lb + ub)
  }
  return out
}

/** The next double after x towards y. */
function nextAfter(x: number, y: number): number {
  if (Number.isNaN(x) || Number.isNaN(y)) return Number.NaN
  if (x === y) return y
  if (x === 0) return y > 0 ? Number.MIN_VALUE : -Number.MIN_VALUE
  const buf = new DataView(new ArrayBuffer(8))
  buf.setFloat64(0, x)
  let bits = buf.getBigInt64(0)
  bits += y > x === x > 0 ? 1n : -1n
  buf.setBigInt64(0, bits)
  return buf.getFloat64(0)
}

/** The largest step along s from x within the bounds and which bounds it hits (scipy's step_size_to_bound). */
function stepToBound(x: ArrayLike<number>, s: ArrayLike<number>, box: Box): { step: number; hits: Int8Array } {
  const n = s.length
  let min = Infinity
  const steps = new Float64Array(n).fill(Infinity)
  for (let i = 0; i < n; i++) {
    const si = s[i] ?? 0
    if (si === 0) continue
    steps[i] = Math.max(
      ((box.lower[i] ?? -Infinity) - (x[i] ?? 0)) / si,
      ((box.upper[i] ?? Infinity) - (x[i] ?? 0)) / si,
    )
    min = Math.min(min, steps[i] ?? Infinity)
  }
  const hits = new Int8Array(n)
  for (let i = 0; i < n; i++) if (steps[i] === min) hits[i] = Math.sign(s[i] ?? 0)
  return { step: min, hits }
}

/** The two values of t with |x + t s| = delta (scipy's intersect_trust_region). */
function intersectTrustRegion(x: Float64Array, s: Float64Array, delta: number): [number, number] {
  const a = ddot(s, s)
  if (a === 0) throw new WhiteboxError('value', 'trf: zero direction')
  const b = ddot(x, s)
  const c = ddot(x, x) - delta * delta
  const d = Math.sqrt(Math.max(0, b * b - a * c))
  const q = -(b + (b >= 0 ? d : -d))
  const t1 = q / a
  const t2 = c / q
  return t1 < t2 ? [t1, t2] : [t2, t1]
}

/** a t^2 + b t + c on [lb, ub]: its minimiser and value (scipy's minimize_quadratic_1d). */
function minimize1d(a: number, b: number, lb: number, ub: number, c = 0): [number, number] {
  const ts = [lb, ub]
  if (a !== 0) {
    const ext = (-0.5 * b) / a
    if (lb < ext && ext < ub) ts.push(ext)
  }
  let best = ts[0] ?? 0
  let bestY = Infinity
  for (const t of ts) {
    const y = t * (a * t + b) + c
    if (y < bestY) {
      bestY = y
      best = t
    }
  }
  return [best, bestY]
}

/** The model of the scaled problem: J_h = J diag(d), diagonal term diag_h and gradient g_h. */
interface ScaledModel {
  readonly J: Float64Array
  readonly m: number
  readonly n: number
  readonly diag?: Float64Array
  readonly g: Float64Array
}

const quadValue = (q: ScaledModel, s: Float64Array): number => {
  const Js = jv(q.J, q.m, q.n, s)
  let v = ddot(Js, Js)
  if (q.diag !== undefined) for (let i = 0; i < q.n; i++) v += (q.diag[i] ?? 0) * (s[i] ?? 0) ** 2
  return 0.5 * v + ddot(s, q.g)
}

/** a, b (and c) of the model along s (from s0) (scipy's build_quadratic_1d). */
function quadratic1d(q: ScaledModel, s: Float64Array, s0?: Float64Array): [number, number, number] {
  const v = jv(q.J, q.m, q.n, s)
  let a = ddot(v, v)
  if (q.diag !== undefined) for (let i = 0; i < q.n; i++) a += (q.diag[i] ?? 0) * (s[i] ?? 0) ** 2
  a *= 0.5
  let b = ddot(q.g, s)
  let c = 0
  if (s0 !== undefined) {
    const u = jv(q.J, q.m, q.n, s0)
    b += ddot(u, v)
    c = 0.5 * ddot(u, u) + ddot(q.g, s0)
    if (q.diag !== undefined)
      for (let i = 0; i < q.n; i++) {
        b += (q.diag[i] ?? 0) * (s0[i] ?? 0) * (s[i] ?? 0)
        c += 0.5 * (q.diag[i] ?? 0) * (s0[i] ?? 0) ** 2
      }
  }
  return [a, b, c]
}

/** The trust-region least-squares subproblem by the SVD (scipy's solve_lsq_trust_region). */
function lsqTrustRegion(
  n: number,
  m: number,
  uf: Float64Array,
  s: Float64Array,
  V: Float64Array,
  delta: number,
  initialAlpha: number,
): { p: Float64Array; alpha: number } {
  const k = s.length
  const suf = s.map((si, i) => si * (uf[i] ?? 0))
  const fullRank = m >= n && (s[k - 1] ?? 0) > Number.EPSILON * m * (s[0] ?? 0)
  const pOf = (w: (i: number) => number): Float64Array => {
    const p = new Float64Array(n)
    for (let c = 0; c < k; c++) {
      const wc = w(c)
      for (let j = 0; j < n; j++) p[j] = (p[j] ?? 0) - (V[j * k + c] ?? 0) * wc
    }
    return p
  }
  if (fullRank) {
    const p = pOf((c) => (uf[c] ?? 0) / (s[c] ?? 1))
    if (dnrm2(p) <= delta) return { p, alpha: 0 }
  }
  const phi = (alpha: number): [number, number] => {
    let pn = 0
    let d = 0
    for (let c = 0; c < k; c++) {
      const den = (s[c] ?? 0) ** 2 + alpha
      pn += ((suf[c] ?? 0) / den) ** 2
      d += (suf[c] ?? 0) ** 2 / den ** 3
    }
    pn = Math.sqrt(pn)
    return [pn - delta, -d / pn]
  }
  let upper = dnrm2(suf) / delta
  let lower = 0
  if (fullRank) {
    const [p0, d0] = phi(0)
    lower = -p0 / d0
  }
  let alpha = !fullRank && initialAlpha === 0 ? Math.max(0.001 * upper, Math.sqrt(lower * upper)) : initialAlpha
  for (let it = 0; it < 10; it++) {
    if (alpha < lower || alpha > upper) alpha = Math.max(0.001 * upper, Math.sqrt(lower * upper))
    const [ph, dph] = phi(alpha)
    if (ph < 0) upper = alpha
    const ratio = ph / dph
    lower = Math.max(lower, alpha - ratio)
    alpha -= ((ph + delta) * ratio) / delta
    if (Math.abs(ph) < 0.01 * delta) break
  }
  const p = pOf((c) => (suf[c] ?? 0) / ((s[c] ?? 0) ** 2 + alpha))
  return { p: dscal(delta / dnrm2(p), p), alpha }
}

/** The best of the reflected, truncated and gradient steps (scipy's select_step). */
function selectStep(
  x: Float64Array,
  q: ScaledModel,
  p0: Float64Array,
  ph0: Float64Array,
  d: Float64Array,
  delta: number,
  box: Box,
  theta: number,
): { step: Float64Array; stepH: Float64Array; predicted: number } {
  const n = x.length
  const xp = daxpy(1, p0, x)
  let inside = true
  for (let i = 0; i < n; i++)
    if (!((xp[i] ?? 0) >= (box.lower[i] ?? -Infinity) && (xp[i] ?? 0) <= (box.upper[i] ?? Infinity))) inside = false
  if (inside) return { step: p0, stepH: ph0, predicted: -quadValue(q, ph0) }
  const { step: pStride, hits } = stepToBound(x, p0, box)
  const rh = Float64Array.from(ph0)
  for (let i = 0; i < n; i++) if (hits[i] !== 0) rh[i] = -(rh[i] ?? 0)
  let r: Float64Array = rh.map((v, i) => v * (d[i] ?? 1))
  let p = dscal(pStride, p0)
  let ph = dscal(pStride, ph0)
  const onBound = daxpy(1, p, x)
  const [, toTr] = intersectTrustRegion(ph, rh, delta)
  const toBound = stepToBound(onBound, r, box).step
  let rStride = Math.min(toBound, toTr)
  let rl: number
  let ru: number
  if (rStride > 0) {
    rl = ((1 - theta) * pStride) / rStride
    ru = rStride === toBound ? theta * toBound : toTr
  } else {
    rl = 0
    ru = -1
  }
  let rValue = Infinity
  let rH: Float64Array = rh
  if (rl <= ru) {
    const [a, b, c] = quadratic1d(q, rh, ph)
    const [t, v] = minimize1d(a, b, rl, ru, c)
    rStride = t
    rValue = v
    rH = daxpy(1, ph, dscal(rStride, rh))
    r = rH.map((vv, i) => vv * (d[i] ?? 1))
  }
  p = dscal(theta, p)
  ph = dscal(theta, ph)
  const pValue = quadValue(q, ph)
  const agH = dscal(-1, q.g)
  const ag = agH.map((v, i) => v * (d[i] ?? 1))
  const toTrG = delta / dnrm2(agH)
  const toBoundG = stepToBound(x, ag, box).step
  const agLimit = toBoundG < toTrG ? theta * toBoundG : toTrG
  const [ga, gb] = quadratic1d(q, agH)
  const [agStride, agValue] = minimize1d(ga, gb, 0, agLimit)
  const agHs = dscal(agStride, agH)
  const ags = dscal(agStride, ag)
  if (pValue < rValue && pValue < agValue) return { step: p, stepH: ph, predicted: -pValue }
  if (rValue < pValue && rValue < agValue) return { step: r, stepH: rH, predicted: -rValue }
  return { step: ags, stepH: agHs, predicted: -agValue }
}

/**
 * The trust region reflective method for (bounded) nonlinear least squares as a traced algorithm, a port of
 * scipy's least_squares with method 'trf' (its default) and the exact (SVD) subproblem solver: Coleman-Li scaling of
 * the variables near active bounds, the trust-region subproblem on the scaled, augmented Jacobian, and the best of
 * the reflected, truncated and scaled-gradient steps; without bounds it is a trust-region Levenberg-Marquardt method.
 * Robust losses scale the residuals as scipy does.
 *
 * @param r - The residual function.
 * @param x0 - The start (made strictly feasible).
 * @param options - Bounds, robust loss, variable scaling and tolerances (ftol, xtol, gtol: default 1e-8).
 * @returns The algorithm.
 * @concept least-squares
 * @reference M. A. Branch, T. F. Coleman and Y. Li, A subspace, interior, and conjugate gradient method for large-scale
 * bound-constrained minimization problems, SIAM J. Sci. Comput. 21 (1999).
 * @example
 * ```ts
 * import { trfAlgorithm } from 'whitebox-engine/optim'
 * import { sub, type Value } from 'whitebox-engine/tensor'
 * trfAlgorithm((p: Value) => sub(p, 3), [0], { bounds: [[null, 1]] }).name // => 'trf'
 * ```
 */
export function trfAlgorithm(r: ResidualFunction, x0: Parameters, options: TrfOptions = {}): Algorithm<TrfState> {
  return trfOn(flatResiduals(r, x0), options)
}

function trfOn(res: FlatResiduals, options: TrfOptions): Algorithm<TrfState> {
  const { m, n } = res
  const box = boxOf(options.bounds, n)
  const bounded = box.lower.some(Number.isFinite) || box.upper.some(Number.isFinite)
  const ftol = options.ftol ?? 1e-8
  const xtol = options.xtol ?? 1e-8
  const gtol = options.gradTol ?? 1e-8
  const loss = options.loss ?? 'linear'
  const fScale = options.fScale ?? 1
  const maxfev = options.maxEvaluations ?? 100 * n
  const xs = options.xScale ?? 1
  const jacScale = xs === 'jac'
  const fixedScaleInv = (): Float64Array =>
    typeof xs === 'number' ? new Float64Array(n).fill(1 / xs) : Float64Array.from(xs as ArrayLike<number>, (v) => 1 / v)
  const columnScaleInv = (J: Float64Array, old?: Float64Array): Float64Array => {
    const out = new Float64Array(n)
    for (let j = 0; j < n; j++) {
      let s = 0
      for (let i = 0; i < m; i++) s += (J[i * n + j] ?? 0) ** 2
      const c = Math.sqrt(s)
      out[j] = old === undefined ? (c === 0 ? 1 : c) : Math.max(c, old[j] ?? 0)
    }
    return out
  }
  const model = (f: Float64Array, J: Float64Array) => {
    const scaled = scaleForLoss(J, f, m, n, loss, fScale)
    return { ...scaled, g: jtv(scaled.J, m, n, scaled.f) }
  }
  const gNorm = (x: Float64Array, g: Float64Array): number => {
    if (!bounded) return g.reduce((mx, v) => Math.max(mx, Math.abs(v)), 0)
    const { v } = clScaling(x, g, box)
    let mx = 0
    for (let i = 0; i < n; i++) mx = Math.max(mx, Math.abs((g[i] ?? 0) * (v[i] ?? 1)))
    return mx
  }
  return {
    name: 'trf',
    config: configOf(options),
    schema: vectorSchema({
      metrics: {
        cost: { description: 'the (robust) cost' },
        delta: { description: 'trust radius in the scaled variables' },
        ratio: { description: 'actual over predicted reduction' },
        projectedGradNorm: { description: 'infinity norm of the Coleman-Li scaled gradient' },
      },
    }),
    init: () => {
      for (let i = 0; i < n; i++) {
        const xi = res.x0[i] ?? 0
        if (!(xi >= (box.lower[i] ?? -Infinity) && xi <= (box.upper[i] ?? Infinity)))
          throw new WhiteboxError('value', `trf: the start is outside the bounds (variable ${i})`)
      }
      const x = bounded ? strictlyFeasible(Float64Array.from(res.x0), box, 1e-10) : Float64Array.from(res.x0)
      const f = res.residuals(x)
      const J = res.jacobian(x)
      const md = model(f, J)
      const scaleInv = jacScale ? columnScaleInv(md.J) : fixedScaleInv()
      let delta = 0
      if (bounded) {
        const { v, dv } = clScaling(x, md.g, box)
        for (let i = 0; i < n; i++) if ((dv[i] ?? 0) !== 0) v[i] = (v[i] ?? 1) * (scaleInv[i] ?? 1)
        delta = dnrm2(x.map((xi, i) => (xi * (scaleInv[i] ?? 1)) / Math.sqrt(v[i] ?? 1)))
      } else delta = dnrm2(x.map((xi, i) => xi * (scaleInv[i] ?? 1)))
      if (delta === 0) delta = 1
      const pg = gNorm(x, md.g)
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
        lmParameter: 0,
        scaleInv,
        ratio: Number.NaN,
        projectedGradNorm: pg,
        activeMask: bounded ? activeConstraints(x, box, xtol) : new Int8Array(n),
        ...(pg < gtol ? { stop: { status: 'converged' as const, message: 'gradient norm below gtol' } } : {}),
      }
    },
    step: (s) => {
      const md = model(s.residuals, s.jacobian)
      const g = md.g
      const scale = s.scaleInv.map((v) => 1 / v)
      let d: Float64Array
      let diagH: Float64Array | undefined
      let theta = 1
      if (bounded) {
        const { v, dv } = clScaling(s.x, g, box)
        for (let i = 0; i < n; i++) if ((dv[i] ?? 0) !== 0) v[i] = (v[i] ?? 1) * (s.scaleInv[i] ?? 1)
        d = v.map((vi, i) => Math.sqrt(vi) * (scale[i] ?? 1))
        diagH = g.map((gi, i) => gi * (dv[i] ?? 0) * (scale[i] ?? 1))
        theta = Math.max(0.995, 1 - s.projectedGradNorm)
      } else d = scale
      const gH = g.map((gi, i) => (d[i] ?? 1) * gi)
      const Jh = new Float64Array(m * n)
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) Jh[i * n + j] = (md.J[i * n + j] ?? 0) * (d[j] ?? 1)
      // The SVD of the (augmented) scaled Jacobian.
      const rows = bounded ? m + n : m
      const Ja = new Float64Array(rows * n)
      Ja.set(Jh, 0)
      const fa = new Float64Array(rows)
      fa.set(md.f, 0)
      if (bounded && diagH !== undefined) for (let j = 0; j < n; j++) Ja[(m + j) * n + j] = Math.sqrt(diagH[j] ?? 0)
      const { u, s: sv, v: V } = dgesvd(Ja, rows, n)
      const k = sv.length
      const uf = new Float64Array(k)
      for (let c = 0; c < k; c++) {
        let t = 0
        for (let i = 0; i < rows; i++) t += (u[i * k + c] ?? 0) * (fa[i] ?? 0)
        uf[c] = t
      }
      const q: ScaledModel = { J: Jh, m, n, ...(diagH === undefined ? {} : { diag: diagH }), g: gH }
      let delta = s.delta
      let alpha = s.lmParameter
      let nfev = s.nfev
      let actual = -1
      let ratio = 0
      let stepNorm = 0
      let stop: TrfState['stop']
      let accepted: { x: Float64Array; f: Float64Array; cost: number } | undefined
      while (actual <= 0 && nfev < maxfev) {
        const sol = lsqTrustRegion(n, m, uf, sv, V, delta, alpha)
        alpha = sol.alpha
        let step: Float64Array
        let stepH: Float64Array
        let predicted: number
        if (bounded) {
          const p = sol.p.map((ph, i) => ph * (d[i] ?? 1))
          const sel = selectStep(s.x, q, p, sol.p, d, delta, box, theta)
          step = sel.step
          stepH = sel.stepH
          predicted = sel.predicted
        } else {
          stepH = sol.p
          step = stepH.map((ph, i) => ph * (d[i] ?? 1))
          predicted = -quadValue(q, stepH)
        }
        const xNew = bounded ? strictlyFeasible(daxpy(1, step, s.x), box, 0) : daxpy(1, step, s.x)
        const fNew = res.residuals(xNew)
        nfev += 1
        const stepHNorm = dnrm2(stepH)
        if (!fNew.every(Number.isFinite)) {
          delta = 0.25 * stepHNorm
          continue
        }
        const costNew = robustCost(fNew, loss, fScale)
        actual = s.value - costNew
        ratio = predicted > 0 ? actual / predicted : predicted === 0 && actual === 0 ? 1 : 0
        let deltaNew = delta
        if (ratio < 0.25) deltaNew = 0.25 * stepHNorm
        else if (ratio > 0.75 && stepHNorm > 0.95 * delta) deltaNew = delta * 2
        stepNorm = dnrm2(step)
        const ftolMet = actual < ftol * s.value && ratio > 0.25
        const xtolMet = stepNorm < xtol * (xtol + dnrm2(s.x))
        if (actual > 0) accepted = { x: xNew, f: fNew, cost: costNew }
        if (ftolMet || xtolMet) {
          stop = {
            status: 'converged',
            message: ftolMet && xtolMet ? 'ftol and xtol satisfied' : ftolMet ? 'ftol satisfied' : 'xtol satisfied',
          }
          delta = deltaNew
          break
        }
        alpha *= delta / deltaNew
        delta = deltaNew
      }
      if (accepted === undefined) {
        return {
          ...s,
          nfev,
          delta,
          lmParameter: alpha,
          ratio,
          step: 0,
          ...(stop !== undefined
            ? { stop }
            : nfev >= maxfev
              ? { stop: { status: 'failed' as const, message: 'maximum number of residual evaluations reached' } }
              : {}),
        }
      }
      const J = res.jacobian(accepted.x)
      const mdNew = model(accepted.f, J)
      const scaleInv = jacScale ? columnScaleInv(mdNew.J, s.scaleInv) : s.scaleInv
      const pg = gNorm(accepted.x, mdNew.g)
      const final =
        pg < gtol
          ? { status: 'converged' as const, message: 'gradient norm below gtol' }
          : (stop ??
            (nfev >= maxfev
              ? { status: 'failed' as const, message: 'maximum number of residual evaluations reached' }
              : undefined))
      return {
        x: accepted.x,
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
        lmParameter: alpha,
        scaleInv,
        ratio,
        projectedGradNorm: pg,
        activeMask: bounded ? activeConstraints(accepted.x, box, xtol) : new Int8Array(n),
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
 * Solve a (bounded) nonlinear least-squares problem by the trust region reflective method (scipy.optimize.least_squares
 * with method 'trf', its default).
 *
 * @param r - The residual function.
 * @param x0 - The start (inside the bounds).
 * @param options - Bounds, robust loss, scaling, tolerances, the evaluation cap (default 100 n), iteration cap and run
 * options.
 * @returns The result, with the active mask of the bounds.
 * @concept least-squares
 * @example
 * ```ts
 * import { trf } from 'whitebox-engine/optim'
 * import { sub, tensor, toFlat, type Value } from 'whitebox-engine/tensor'
 * const r = trf((p: Value) => sub(p, tensor([3, -3])), [0, 0], { bounds: [[-1, 1], [-1, 1]] })
 * [toFlat(r.x as Value).map((v) => Math.round(v * 1e6) / 1e6), r.activeMask] // => [[1, -1], [1, -1]]
 * ```
 */
export function trf(r: ResidualFunction, x0: Parameters, options: TrfOptions = {}): LeastSquaresResult {
  const res = flatResiduals(r, x0)
  return runLeastSquares(trfOn(res, options), res, options, 100 * res.n)
}
