import { daxpy, ddot, dgemv, dgesv, dnrm2 } from '../../numerics/linalg/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { type LineSearchResult, lineFunction } from '../line-search.ts'
import { moreThuente } from '../line-search-interval.ts'
import {
  configOf,
  LINE_SEARCH_CHANNEL,
  lineSearchMessage,
  observeVector,
  runVector,
  type VectorState,
  vectorDone,
  vectorSchema,
} from './_shared.ts'
import {
  type Bounds,
  type Box,
  boxOf,
  clipToBox,
  type FlatObjective,
  flatObjective,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'

/** Options of {@link lbfgsb}. */
export interface LbfgsbOptions extends VectorMethodOptions {
  /** Box bounds on the flattened parameters. */
  readonly bounds?: Bounds
  /** Curvature pairs kept (default 10, scipy's maxcor). */
  readonly memory?: number
  /** Stop when (f_k - f_k+1) / max(|f_k|, |f_k+1|, 1) <= ftol (default 2.22e-9, scipy's). */
  readonly ftol?: number
  /** Line-search evaluations per iteration (default 20, scipy's maxls). */
  readonly maxLineSearch?: number
  /** Objective evaluations in total (default 15000, scipy's maxfun). */
  readonly maxEvaluations?: number
}

/** The state of an {@link lbfgsb} run. */
export interface LbfgsbState extends VectorState {
  /** Curvature pairs, oldest first. */
  readonly s: readonly Float64Array[]
  readonly y: readonly Float64Array[]
  readonly theta: number
  /** Iterations so far (the first is special: its step is bounded by 1 / |d|). */
  readonly iterations: number
  /** The infinity norm of the projected gradient. */
  readonly projectedGradNorm: number
  /** The generalized Cauchy point of the last iteration. */
  readonly cauchy?: Float64Array
  /** Variables free (off their bounds) at the Cauchy point. */
  readonly free: number
  /** Breakpoints passed in the last Cauchy search. */
  readonly segments: number
  readonly lineSearch?: LineSearchResult
}

/**
 * The infinity norm of the projected gradient of a box-constrained problem (L-BFGS-B's projgr): components pointing
 * out of the box at an active bound are cut to the distance to it.
 *
 * @param x - The point (inside the box).
 * @param g - The gradient.
 * @param box - The bounds.
 * @returns max_i |proj g_i|.
 * @example
 * ```ts
 * import { boxOf, projectedGradientNorm } from 'whitebox-engine/optim'
 * // At the lower bound with a positive gradient, the bound blocks the descent direction.
 * projectedGradientNorm([0, 1], [3, -0.5], boxOf({ lower: 0 }, 2)) // => 0.5
 * ```
 */
export function projectedGradientNorm(x: ArrayLike<number>, g: ArrayLike<number>, box: Box): number {
  let m = 0
  for (let i = 0; i < x.length; i++) {
    let gi = g[i] ?? 0
    const xi = x[i] ?? 0
    if (gi < 0) gi = Math.max(xi - (box.upper[i] ?? Infinity), gi)
    else gi = Math.min(xi - (box.lower[i] ?? -Infinity), gi)
    m = Math.max(m, Math.abs(gi))
  }
  return m
}

/** The compact limited-memory representation B = theta I - W M W' with W = [Y, theta S] (n x 2m, row-major). */
interface Compact {
  readonly k: number
  readonly w: Float64Array
  readonly m: Float64Array
}

function compact(S: readonly Float64Array[], Y: readonly Float64Array[], theta: number, n: number): Compact {
  const col = S.length
  const k = 2 * col
  const w = new Float64Array(n * k)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < col; j++) {
      w[i * k + j] = (Y[j] as Float64Array)[i] ?? 0
      w[i * k + col + j] = theta * ((S[j] as Float64Array)[i] ?? 0)
    }
  }
  if (col === 0) return { k: 0, w, m: new Float64Array(0) }
  // K = [[-D, L'], [L, theta S'S]] with D = diag(s_i'y_i), L_ij = s_i'y_j (i > j); M = K^-1.
  const K = new Float64Array(k * k)
  for (let i = 0; i < col; i++) {
    for (let j = 0; j < col; j++) {
      const sy = ddot(S[i] as Float64Array, Y[j] as Float64Array)
      if (i === j) K[i * k + j] = -sy
      if (i > j) {
        K[(col + i) * k + j] = sy
        K[j * k + (col + i)] = sy
      }
      K[(col + i) * k + (col + j)] = theta * ddot(S[i] as Float64Array, S[j] as Float64Array)
    }
  }
  const m = new Float64Array(k * k)
  for (let c = 0; c < k; c++) {
    const e = new Float64Array(k)
    e[c] = 1
    const col_ = dgesv(K, e, k)
    for (let r = 0; r < k; r++) m[r * k + c] = col_[r] ?? 0
  }
  return { k, w, m }
}

const row = (c: Compact, i: number): Float64Array => c.w.subarray(i * c.k, (i + 1) * c.k)
const mv = (c: Compact, v: Float64Array): Float64Array => (c.k === 0 ? new Float64Array(0) : dgemv(c.m, c.k, c.k, v))

/** The generalized Cauchy point (L-BFGS-B's cauchy): the first local minimiser of the model along P(x - t g). */
function cauchyPoint(
  x: Float64Array,
  g: Float64Array,
  box: Box,
  theta: number,
  c: Compact,
): { xcp: Float64Array; cvec: Float64Array; free: boolean[]; segments: number } {
  const n = x.length
  const xcp = Float64Array.from(x)
  const d = new Float64Array(n)
  const atBound: boolean[] = new Array<boolean>(n).fill(false)
  const breaks: { t: number; i: number }[] = []
  let p: Float64Array = new Float64Array(c.k)
  let f1 = 0
  let bounded = true
  for (let i = 0; i < n; i++) {
    const lo = box.lower[i] ?? -Infinity
    const hi = box.upper[i] ?? Infinity
    const neg = -(g[i] ?? 0)
    const fixed = lo === hi
    const tl = (x[i] ?? 0) - lo
    const tu = hi - (x[i] ?? 0)
    let active = fixed
    if (!fixed) {
      if (Number.isFinite(lo) && tl <= 0) active = neg <= 0
      else if (Number.isFinite(hi) && tu <= 0) active = neg >= 0
      else active = Math.abs(neg) <= 0
    }
    if (active) {
      atBound[i] = Number.isFinite(lo) || Number.isFinite(hi)
      continue
    }
    d[i] = neg
    f1 -= neg * neg
    if (c.k > 0) p = daxpy(neg, row(c, i), p)
    if (Number.isFinite(lo) && neg < 0) breaks.push({ t: tl / -neg, i })
    else if (Number.isFinite(hi) && neg > 0) breaks.push({ t: tu / neg, i })
    else if (Math.abs(neg) > 0) bounded = false
  }
  let cvec: Float64Array = new Float64Array(c.k)
  if (breaks.length === 0 && f1 === 0) return { xcp, cvec, free: atBound.map((b) => !b), segments: 0 }
  let f2 = -theta * f1 - ddot(p, mv(c, p))
  const f2org = f2
  let dtm = -f1 / f2
  let tsum = 0
  let segments = 1
  breaks.sort((a, b) => a.t - b.t)
  let tj = 0
  let consumed = 0
  let allFixed = false
  for (const brk of breaks) {
    const tj0 = tj
    tj = brk.t
    const dt = tj - tj0
    if (dtm < dt) break
    tsum += dt
    consumed += 1
    const ib = brk.i
    const dib = d[ib] ?? 0
    d[ib] = 0
    let zib: number
    if (dib > 0) {
      zib = (box.upper[ib] ?? 0) - (x[ib] ?? 0)
      xcp[ib] = box.upper[ib] ?? 0
    } else {
      zib = (box.lower[ib] ?? 0) - (x[ib] ?? 0)
      xcp[ib] = box.lower[ib] ?? 0
    }
    atBound[ib] = true
    if (consumed === breaks.length && breaks.length === n) {
      dtm = dt
      allFixed = true
      break
    }
    segments += 1
    const dib2 = dib * dib
    f1 = f1 + dt * f2 + dib2 - theta * dib * zib
    f2 -= theta * dib2
    if (c.k > 0) {
      cvec = daxpy(dt, p, cvec)
      const wb = row(c, ib)
      const v = mv(c, wb)
      const wmc = ddot(cvec, v)
      const wmp = ddot(p, v)
      const wmw = ddot(wb, v)
      p = daxpy(-dib, wb, p)
      f1 += dib * wmc
      f2 += 2 * dib * wmp - dib2 * wmw
    }
    f2 = Math.max(Number.EPSILON * f2org, f2)
    if (consumed < breaks.length) dtm = -f1 / f2
    else if (bounded) {
      f1 = 0
      f2 = 0
      dtm = 0
    } else dtm = -f1 / f2
  }
  if (!allFixed) {
    if (dtm <= 0) dtm = 0
    tsum += dtm
    for (let i = 0; i < n; i++) if ((d[i] ?? 0) !== 0) xcp[i] = (x[i] ?? 0) + tsum * (d[i] ?? 0)
  }
  if (c.k > 0) cvec = daxpy(dtm, p, cvec)
  return { xcp, cvec, free: atBound.map((b) => !b), segments }
}

/** Subspace minimisation over the free variables at the Cauchy point (direct primal method, Morales-Nocedal 2011). */
function subspaceMin(
  x: Float64Array,
  g: Float64Array,
  box: Box,
  theta: number,
  c: Compact,
  xcp: Float64Array,
  cvec: Float64Array,
  free: readonly boolean[],
  unconstrained: boolean,
): Float64Array {
  const n = x.length
  const ind: number[] = []
  for (let i = 0; i < n; i++) if (free[i] === true) ind.push(i)
  if (ind.length === 0 || c.k === 0) return xcp
  // r = -(g + theta (xcp - x) - W M c) on the free variables (cmprlb).
  const mc = mv(c, cvec)
  const r = new Float64Array(ind.length)
  for (const [j, i] of ind.entries()) {
    r[j] = unconstrained ? -(g[i] ?? 0) : -theta * ((xcp[i] ?? 0) - (x[i] ?? 0)) - (g[i] ?? 0) + ddot(row(c, i), mc)
  }
  // v = M W'Z r; N = I - M W'ZZ'W / theta; v = N^-1 v; du = r / theta + Z'W v / theta^2.
  let wzr: Float64Array = new Float64Array(c.k)
  const A = new Float64Array(c.k * c.k)
  for (const [j, i] of ind.entries()) {
    const wi = row(c, i)
    wzr = daxpy(r[j] ?? 0, wi, wzr)
    for (let a = 0; a < c.k; a++)
      for (let b = 0; b < c.k; b++) A[a * c.k + b] = (A[a * c.k + b] ?? 0) + (wi[a] ?? 0) * (wi[b] ?? 0)
  }
  const v0 = mv(c, wzr)
  const N = new Float64Array(c.k * c.k)
  for (let a = 0; a < c.k; a++) {
    for (let b = 0; b < c.k; b++) {
      let s = 0
      for (let q = 0; q < c.k; q++) s += (c.m[a * c.k + q] ?? 0) * (A[q * c.k + b] ?? 0)
      N[a * c.k + b] = (a === b ? 1 : 0) - s / theta
    }
  }
  let v: Float64Array
  try {
    v = dgesv(N, v0, c.k)
  } catch {
    return xcp
  }
  const du = new Float64Array(ind.length)
  for (const [j, i] of ind.entries()) du[j] = (r[j] ?? 0) / theta + ddot(row(c, i), v) / (theta * theta)
  // Project the Newton point onto the box; keep it if it is still a descent direction from x.
  const xp = Float64Array.from(xcp)
  let hit = false
  for (const [j, i] of ind.entries()) {
    const lo = box.lower[i] ?? -Infinity
    const hi = box.upper[i] ?? Infinity
    const t = Math.min(Math.max((xcp[i] ?? 0) + (du[j] ?? 0), lo), hi)
    if ((Number.isFinite(lo) && t === lo) || (Number.isFinite(hi) && t === hi)) hit = true
    xp[i] = t
  }
  if (!hit) return xp
  let dd = 0
  for (let i = 0; i < n; i++) dd += ((xp[i] ?? 0) - (x[i] ?? 0)) * (g[i] ?? 0)
  if (dd <= 0) return xp
  // Otherwise backtrack along du from the Cauchy point to the first bound (the 1995 method).
  let alpha = 1
  let ibd = -1
  for (const [j, i] of ind.entries()) {
    const dk = du[j] ?? 0
    const lo = box.lower[i] ?? -Infinity
    const hi = box.upper[i] ?? Infinity
    let t1 = alpha
    if (dk < 0 && Number.isFinite(lo)) {
      const t2 = lo - (xcp[i] ?? 0)
      if (t2 >= 0) t1 = 0
      else if (dk * alpha < t2) t1 = t2 / dk
    } else if (dk > 0 && Number.isFinite(hi)) {
      const t2 = hi - (xcp[i] ?? 0)
      if (t2 <= 0) t1 = 0
      else if (dk * alpha > t2) t1 = t2 / dk
    }
    if (t1 < alpha) {
      alpha = t1
      ibd = j
    }
  }
  const out = Float64Array.from(xcp)
  if (alpha < 1 && ibd >= 0) {
    const i = ind[ibd] as number
    const dk = du[ibd] ?? 0
    if (dk > 0) out[i] = box.upper[i] ?? 0
    else if (dk < 0) out[i] = box.lower[i] ?? 0
    du[ibd] = 0
  }
  for (const [j, i] of ind.entries()) out[i] = (out[i] ?? 0) + alpha * (du[j] ?? 0)
  return out
}

/**
 * L-BFGS-B (Byrd, Lu, Nocedal and Zhu 1995, with the subspace projection of Morales and Nocedal 2011) as a traced
 * algorithm: the generalized Cauchy point along the projected steepest-descent path of the limited-memory model, a
 * subspace minimisation over the free variables, then a More-Thuente search (ftol 1e-3, gtol 0.9, xtol 0.1) kept inside
 * the box. Stops on the projected gradient's infinity norm or a relative reduction below `ftol`, as scipy's.
 *
 * @param f - The objective.
 * @param x0 - The start (projected into the box).
 * @param options - Bounds, memory, tolerances and caps.
 * @returns The algorithm.
 * @concept quasi-newton
 * @reference R. H. Byrd, P. Lu, J. Nocedal and C. Zhu, A limited memory algorithm for bound constrained optimization,
 * SIAM J. Sci. Comput. 16 (1995); J. L. Morales and J. Nocedal, Remark on Algorithm 778, ACM TOMS 38 (2011).
 * @example
 * ```ts
 * import { lbfgsbAlgorithm } from 'whitebox-engine/optim'
 * import { square, sum, type Value } from 'whitebox-engine/tensor'
 * lbfgsbAlgorithm((x: Value) => sum(square(x)), [1, 2], { bounds: { lower: 0.5 } }).name // => 'lbfgsb'
 * ```
 */
export function lbfgsbAlgorithm(f: Objective, x0: Parameters, options: LbfgsbOptions = {}): Algorithm<LbfgsbState> {
  return lbfgsbOn(flatObjective(f, x0), options)
}

function lbfgsbOn(obj: FlatObjective, options: LbfgsbOptions): Algorithm<LbfgsbState> {
  const n = obj.n
  const box = boxOf(options.bounds, n)
  const memory = options.memory ?? 10
  const pgtol = options.gradTol ?? 1e-5
  const ftol = options.ftol ?? 2.220446049250313e-9
  const maxls = options.maxLineSearch ?? 20
  const maxfun = options.maxEvaluations ?? 15000
  let constrained = false
  let boxed = true
  for (let i = 0; i < n; i++) {
    const lo = Number.isFinite(box.lower[i] ?? -Infinity)
    const hi = Number.isFinite(box.upper[i] ?? Infinity)
    if (lo || hi) constrained = true
    if (!(lo && hi)) boxed = false
  }
  const converged = (pg: number) =>
    pg <= pgtol ? { status: 'converged' as const, message: 'norm of projected gradient below gtol' } : undefined
  return {
    name: 'lbfgsb',
    config: configOf(options),
    schema: vectorSchema({
      metrics: {
        projectedGradNorm: { description: 'infinity norm of the projected gradient' },
        free: { description: 'variables free at the Cauchy point' },
        segments: { description: 'segments of the Cauchy search' },
        pairs: { description: 'curvature pairs held' },
      },
      tensors: { cauchy: { role: 'state', description: 'the generalized Cauchy point' } },
      messages: LINE_SEARCH_CHANNEL,
    }),
    init: () => {
      const x = clipToBox(obj.x0, box)
      const { value, grad } = obj.valueAndGrad(x)
      const pg = projectedGradientNorm(x, grad, box)
      const stop = converged(pg)
      return {
        x,
        value,
        grad,
        nfev: 1,
        ngev: 1,
        nhev: 0,
        step: 0,
        s: [],
        y: [],
        theta: 1,
        iterations: 0,
        projectedGradNorm: pg,
        free: n,
        segments: 0,
        ...(stop === undefined ? {} : { stop }),
      }
    },
    step: (st) => {
      const c = compact(st.s, st.y, st.theta, n)
      const col = st.s.length
      let z: Float64Array
      let cvec: Float64Array = new Float64Array(c.k)
      let free: boolean[]
      let segments = 0
      if (!constrained && col > 0) {
        z = Float64Array.from(st.x)
        free = new Array<boolean>(n).fill(true)
      } else {
        const cp = cauchyPoint(st.x, st.grad, box, st.theta, c)
        z = cp.xcp
        cvec = cp.cvec
        free = cp.free
        segments = cp.segments
      }
      const cauchy = Float64Array.from(z)
      z = subspaceMin(st.x, st.grad, box, st.theta, c, z, cvec, free, !constrained && col > 0)
      const d = daxpy(-1, st.x, z)
      // The largest step that stays in the box (one on the very first iteration of a constrained problem).
      let stpmx = 1e10
      if (constrained) {
        if (st.iterations === 0) stpmx = 1
        else {
          for (let i = 0; i < n; i++) {
            const a1 = d[i] ?? 0
            const xi = st.x[i] ?? 0
            const lo = box.lower[i] ?? -Infinity
            const hi = box.upper[i] ?? Infinity
            if (a1 < 0 && Number.isFinite(lo)) {
              const a2 = lo - xi
              if (a2 >= 0) stpmx = 0
              else if (a1 * stpmx < a2) stpmx = a2 / a1
            } else if (a1 > 0 && Number.isFinite(hi)) {
              const a2 = hi - xi
              if (a2 <= 0) stpmx = 0
              else if (a1 * stpmx > a2) stpmx = a2 / a1
            }
          }
        }
      }
      const dnorm = dnrm2(d)
      const alpha0 = st.iterations === 0 && !boxed ? Math.min(1 / dnorm, stpmx) : 1
      const slope = ddot(st.grad, d)
      const freeCount = free.filter(Boolean).length
      const fail = (ls: LineSearchResult | undefined, message: string): LbfgsbState => {
        const base = {
          ...st,
          nfev: st.nfev + (ls?.evaluations ?? 0),
          ngev: st.ngev + (ls?.gradEvaluations ?? 0),
          cauchy,
          free: freeCount,
          segments,
          ...(ls === undefined ? {} : { lineSearch: ls }),
        }
        // With corrections, restart from steepest descent; without, give up (L-BFGS-B's abnormal termination).
        if (col > 0) return { ...base, s: [], y: [], theta: 1 }
        return { ...base, stop: { status: 'failed', message } }
      }
      if (!(slope < 0) || !(stpmx > 0) || dnorm === 0) return fail(undefined, 'abnormal termination in line search')
      const ls = moreThuente({ c1: 1e-3, c2: 0.9, xtol: 0.1, alphaMin: 0, alphaMax: stpmx, maxEvaluations: maxls })(
        lineFunction(obj, st.x, d),
        { value: st.value, slope, alpha: Math.min(alpha0, stpmx) },
      )
      // dcsrch's warnings (xtol, step bounds, rounding) end the search at its last trial, which L-BFGS-B accepts; the
      // evaluation cap and errors are failures.
      const accepted = ls.success || (Number.isFinite(ls.alpha) && /xtol|alphaMax|alphaMin|rounding/.test(ls.message))
      if (!accepted) return fail(ls, `abnormal termination in line search: ${ls.message}`)
      const xRaw = daxpy(ls.alpha, d, st.x)
      const x = clipToBox(xRaw, box)
      const clipped = x.some((v, i) => v !== xRaw[i])
      const reuse = !clipped && ls.grad !== undefined
      const ev = reuse ? { value: ls.value, grad: ls.grad as Float64Array } : obj.valueAndGrad(x)
      const nfev = st.nfev + ls.evaluations + (reuse ? 0 : 1)
      const ngev = st.ngev + ls.gradEvaluations + (reuse ? 0 : 1)
      const pg = projectedGradientNorm(x, ev.grad, box)
      const reduction = (st.value - ev.value) / Math.max(Math.abs(st.value), Math.abs(ev.value), 1)
      const stop =
        converged(pg) ??
        (reduction <= ftol
          ? { status: 'converged' as const, message: 'relative reduction of f below ftol' }
          : undefined) ??
        (nfev >= maxfun ? { status: 'failed' as const, message: 'maximum number of evaluations reached' } : undefined)
      // Update the corrections unless the curvature is too small (dr <= eps ddum in L-BFGS-B).
      const s = daxpy(-1, st.x, x)
      const y = daxpy(-1, st.grad, ev.grad)
      const sy = ddot(s, y)
      const used = sy > Number.EPSILON * -slope * ls.alpha
      return {
        x,
        value: ev.value,
        grad: ev.grad,
        nfev,
        ngev,
        nhev: 0,
        step: dnrm2(s),
        iterations: st.iterations + 1,
        s: used ? [...st.s, s].slice(-memory) : st.s,
        y: used ? [...st.y, y].slice(-memory) : st.y,
        theta: used ? ddot(y, y) / sy : st.theta,
        projectedGradNorm: pg,
        cauchy,
        free: freeCount,
        segments,
        lineSearch: ls,
        ...(stop === undefined ? {} : { stop }),
      }
    },
    done: vectorDone,
    observe: (s) =>
      observeVector(s, {
        metrics: { projectedGradNorm: s.projectedGradNorm, free: s.free, segments: s.segments, pairs: s.s.length },
        ...(s.cauchy === undefined ? {} : { tensors: { cauchy: { role: 'state', value: s.cauchy } } }),
        ...(s.lineSearch === undefined ? {} : { messages: [lineSearchMessage(s.lineSearch)] }),
      }),
  }
}

/**
 * Minimise subject to box bounds by L-BFGS-B (scipy.optimize.minimize with method 'L-BFGS-B'; gradients by autodiff).
 * Without bounds it is limited-memory BFGS.
 *
 * @param f - The objective.
 * @param x0 - The start (any parameter tree; bounds apply to its flattened entries).
 * @param options - Bounds, memory (default 10), projected-gradient tolerance (default 1e-5), ftol (default 2.22e-9),
 * line-search and evaluation caps, iteration cap (default 15000) and run options.
 * @returns The result.
 * @concept quasi-newton
 * @example
 * ```ts
 * import { lbfgsb } from 'whitebox-engine/optim'
 * import { square, sub, sum, toFlat, type Value } from 'whitebox-engine/tensor'
 * const r = lbfgsb((x: Value) => sum(square(sub(x, 3))), [0, 0], { bounds: [[null, 1], [0, 5]] })
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e6) / 1e6) // => [1, 3]
 * ```
 */
export function lbfgsb(f: Objective, x0: Parameters, options: LbfgsbOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(lbfgsbOn(obj, options), obj, options, 15000)
}
