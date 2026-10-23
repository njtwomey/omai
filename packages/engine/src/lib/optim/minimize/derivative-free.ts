import { WhiteboxError } from '../../base/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { type BoundArrays, boundArrays, clipInto } from './_bounds.ts'
import { configOf, observeVector, runVector, type VectorState, vectorDone, vectorSchema } from './_shared.ts'
import {
  flatObjective,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'

/** Box bounds as `(lower, upper)` pairs, one per parameter (scipy's sequence form); infinite ends are allowed. */
export type Bounds = readonly (readonly [number, number])[]

// Thrown by a counted objective when the evaluation budget is spent (scipy's _MaxFuncCallError).
const BUDGET: unique symbol = Symbol('evaluation budget spent')

/** A counted objective: throws BUDGET once `max` evaluations (including `used` earlier ones) have been made. */
function counted(value: (x: ArrayLike<number>) => number, used: number, max: number) {
  let calls = 0
  return {
    f: (x: ArrayLike<number>): number => {
      if (used + calls >= max) throw BUDGET
      calls += 1
      return value(x)
    },
    calls: (): number => calls,
  }
}

const isBudget = (e: unknown): boolean => e === BUDGET

// ---------------------------------------------------------------------------------------------------------------
// Nelder-Mead

/** Options of {@link nelderMead}. */
export interface NelderMeadOptions extends VectorMethodOptions {
  /** Stop when every vertex is within this of the best in every coordinate (default 1e-4)... */
  readonly xatol?: number
  /** ...and every vertex's value is within this of the best (default 1e-4). */
  readonly fatol?: number
  /** Adapt the coefficients to the dimension (Gao and Han 2012), for high-dimensional problems. */
  readonly adaptive?: boolean
  /** The starting simplex, n + 1 points (default: x0 and x0 with each coordinate scaled by 1.05, or 0.00025 if zero). */
  readonly initialSimplex?: readonly (readonly number[])[]
  /** The evaluation budget (scipy's maxfev; default 200 n). */
  readonly maxEvaluations?: number
  /** Box bounds: trial points are clipped into the box. */
  readonly bounds?: Bounds
}

/** State of a Nelder-Mead run. */
export interface NelderMeadState extends VectorState {
  /** The simplex vertices, best first. */
  readonly simplex: readonly Float64Array[]
  /** Their objective values, ascending. */
  readonly fsim: Float64Array
  /** The move made by the last step. */
  readonly operation: 'initial' | 'reflect' | 'expand' | 'contract outside' | 'contract inside' | 'shrink'
}

const EMPTY = new Float64Array(0)

/** Sort a simplex by value (stable, as numpy's argsort on small arrays). */
function sortSimplex(sim: Float64Array[], fsim: Float64Array): { sim: Float64Array[]; fsim: Float64Array } {
  const order = Array.from(fsim, (_, i) => i).sort((a, b) => {
    const fa = fsim[a] ?? Number.NaN
    const fb = fsim[b] ?? Number.NaN
    if (Number.isNaN(fa)) return Number.isNaN(fb) ? a - b : 1
    if (Number.isNaN(fb)) return -1
    return fa < fb ? -1 : fa > fb ? 1 : a - b
  })
  return { sim: order.map((i) => sim[i] as Float64Array), fsim: Float64Array.from(order, (i) => fsim[i] ?? 0) }
}

function simplexSpread(sim: readonly Float64Array[], fsim: Float64Array): { dx: number; df: number } {
  const best = sim[0] as Float64Array
  let dx = 0
  let df = 0
  for (let k = 1; k < sim.length; k++) {
    const v = sim[k] as Float64Array
    for (let j = 0; j < best.length; j++) dx = Math.max(dx, Math.abs((v[j] ?? 0) - (best[j] ?? 0)))
    df = Math.max(df, Math.abs((fsim[0] ?? 0) - (fsim[k] ?? 0)))
  }
  return { dx, df }
}

/**
 * The Nelder-Mead downhill simplex method as a traced algorithm, step for step as scipy's `minimize(method='Nelder-Mead')`:
 * reflection, expansion, outside and inside contraction and shrinking, with optional dimension-adapted coefficients and
 * clipping into box bounds. Each step is one simplex move; the trace records the best value, the simplex size and the
 * move.
 *
 * @param f - The objective.
 * @param x0 - The starting point (any parameter tree).
 * @param options - Tolerances, adaptive coefficients, starting simplex, budget and bounds.
 * @returns The algorithm.
 * @concept derivative-free-optimisation
 * @example
 * ```ts
 * import { nelderMeadAlgorithm } from 'whitebox-engine/optim'
 * import { square, sum, type Value } from 'whitebox-engine/tensor'
 * nelderMeadAlgorithm((x) => sum(square(x as Value)), [1, 2]).name // => 'nelderMead'
 * ```
 */
export function nelderMeadAlgorithm(
  f: Objective,
  x0: Parameters,
  options: NelderMeadOptions = {},
): Algorithm<NelderMeadState> {
  const obj = flatObjective(f, x0)
  const n = obj.n
  if (n === 0) throw new WhiteboxError('value', 'nelderMead: no parameters')
  const box = options.bounds === undefined ? undefined : boundArrays(options.bounds, n, 'nelderMead')
  const clip = (x: Float64Array): Float64Array => (box === undefined ? x : clipInto(x, box))
  const adaptive = options.adaptive ?? false
  const rho = 1
  const chi = adaptive ? 1 + 2 / n : 2
  const psi = adaptive ? 0.75 - 1 / (2 * n) : 0.5
  const sigma = adaptive ? 1 - 1 / n : 0.5
  const xatol = options.xatol ?? 1e-4
  const fatol = options.fatol ?? 1e-4
  const maxfun = options.maxEvaluations ?? 200 * n
  const simplex0 = options.initialSimplex
  if (simplex0 !== undefined && (simplex0.length !== n + 1 || simplex0.some((v) => v.length !== n)))
    throw new WhiteboxError('shape', `nelderMead: initialSimplex must be ${n + 1} points of length ${n}`)

  const decide = (s: Omit<NelderMeadState, 'stop'>): NelderMeadState => {
    const { dx, df } = simplexSpread(s.simplex, s.fsim)
    if (dx <= xatol && df <= fatol)
      return { ...s, stop: { status: 'converged', message: 'simplex within xatol and fatol' } }
    if (s.nfev >= maxfun)
      return { ...s, stop: { status: 'failed', message: 'maximum number of function evaluations reached' } }
    return s
  }

  return {
    name: 'nelderMead',
    config: configOf({ xatol, fatol, adaptive, maxEvaluations: maxfun }),
    schema: vectorSchema({
      metrics: {
        simplexSize: { description: 'largest coordinate distance from a vertex to the best vertex' },
        valueSpread: { description: 'largest value difference from a vertex to the best vertex' },
      },
      tensors: { simplex: { role: 'state', description: 'the simplex vertices, best first' } },
      messages: { move: { description: 'the simplex move of the step' } },
    }),
    init: () => {
      let start: Float64Array[]
      if (options.initialSimplex !== undefined) {
        start = options.initialSimplex.map((v) => Float64Array.from(v))
      } else {
        const x = clip(Float64Array.from(obj.x0))
        start = [x]
        for (let k = 0; k < n; k++) {
          const y = Float64Array.from(x)
          y[k] = (y[k] ?? 0) !== 0 ? 1.05 * (y[k] ?? 0) : 0.00025
          start.push(y)
        }
      }
      if (box !== undefined) {
        // Reflect vertices beyond an upper bound into the interior, then clip (scipy).
        start = start.map((v) =>
          clip(Float64Array.from(v, (vj, j) => (vj > (box.upper[j] ?? 0) ? 2 * (box.upper[j] ?? 0) - vj : vj))),
        )
      }
      const fsim = new Float64Array(n + 1).fill(Number.POSITIVE_INFINITY)
      const ev = counted(obj.value, 0, maxfun)
      try {
        for (let k = 0; k <= n; k++) fsim[k] = ev.f(start[k] as Float64Array)
      } catch (e) {
        if (!isBudget(e)) throw e
      }
      const sorted = sortSimplex(start, fsim)
      return decide({
        x: sorted.sim[0] as Float64Array,
        value: sorted.fsim[0] ?? Number.NaN,
        grad: EMPTY,
        nfev: ev.calls(),
        ngev: 0,
        nhev: 0,
        step: simplexSpread(sorted.sim, sorted.fsim).dx,
        simplex: sorted.sim,
        fsim: sorted.fsim,
        operation: 'initial',
      })
    },
    step: (s) => {
      const sim = s.simplex.map((v) => v)
      const fsim = Float64Array.from(s.fsim)
      const ev = counted(obj.value, s.nfev, maxfun)
      let operation: NelderMeadState['operation'] = 'reflect'
      try {
        const xbar = new Float64Array(n)
        for (let k = 0; k < n; k++) {
          const v = sim[k] as Float64Array
          for (let j = 0; j < n; j++) xbar[j] = (xbar[j] ?? 0) + (v[j] ?? 0)
        }
        for (let j = 0; j < n; j++) xbar[j] = (xbar[j] ?? 0) / n
        const worst = sim[n] as Float64Array
        const along = (a: number, b: number): Float64Array =>
          clip(Float64Array.from(xbar, (xb, j) => a * xb - b * (worst[j] ?? 0)))
        const xr = along(1 + rho, rho)
        const fxr = ev.f(xr)
        let shrink = false
        if (fxr < (fsim[0] ?? 0)) {
          const xe = along(1 + rho * chi, rho * chi)
          const fxe = ev.f(xe)
          if (fxe < fxr) {
            sim[n] = xe
            fsim[n] = fxe
            operation = 'expand'
          } else {
            sim[n] = xr
            fsim[n] = fxr
          }
        } else if (fxr < (fsim[n - 1] ?? 0)) {
          sim[n] = xr
          fsim[n] = fxr
        } else if (fxr < (fsim[n] ?? 0)) {
          const xc = along(1 + psi * rho, psi * rho)
          const fxc = ev.f(xc)
          operation = 'contract outside'
          if (fxc <= fxr) {
            sim[n] = xc
            fsim[n] = fxc
          } else shrink = true
        } else {
          const xcc = clip(Float64Array.from(xbar, (xb, j) => (1 - psi) * xb + psi * (worst[j] ?? 0)))
          const fxcc = ev.f(xcc)
          operation = 'contract inside'
          if (fxcc < (fsim[n] ?? 0)) {
            sim[n] = xcc
            fsim[n] = fxcc
          } else shrink = true
        }
        if (shrink) {
          operation = 'shrink'
          const best = sim[0] as Float64Array
          for (let k = 1; k <= n; k++) {
            const v = sim[k] as Float64Array
            sim[k] = clip(Float64Array.from(best, (b, j) => b + sigma * ((v[j] ?? 0) - b)))
            fsim[k] = ev.f(sim[k] as Float64Array)
          }
        }
      } catch (e) {
        if (!isBudget(e)) throw e
      }
      const sorted = sortSimplex(sim, fsim)
      return decide({
        x: sorted.sim[0] as Float64Array,
        value: sorted.fsim[0] ?? Number.NaN,
        grad: EMPTY,
        nfev: s.nfev + ev.calls(),
        ngev: 0,
        nhev: 0,
        step: simplexSpread(sorted.sim, sorted.fsim).dx,
        simplex: sorted.sim,
        fsim: sorted.fsim,
        operation,
      })
    },
    done: vectorDone,
    observe: (s) => {
      const { dx, df } = simplexSpread(s.simplex, s.fsim)
      const flat = new Float64Array((n + 1) * n)
      for (const [k, v] of s.simplex.entries()) flat.set(v, k * n)
      return observeVector(s, {
        metrics: { simplexSize: dx, valueSpread: df },
        tensors: { simplex: { role: 'state', value: flat, shape: [n + 1, n] } },
        messages: [{ kind: 'move', payload: { operation: s.operation } }],
      })
    },
  }
}

/**
 * Minimise by the Nelder-Mead simplex method (scipy.optimize.minimize with method 'Nelder-Mead').
 *
 * @param f - The objective.
 * @param x0 - The starting point (any parameter tree).
 * @param options - Tolerances, adaptive coefficients, starting simplex, budgets (default 200 n iterations and
 *   evaluations), bounds and run options.
 * @returns The result.
 * @concept derivative-free-optimisation
 * @example
 * ```ts
 * import { nelderMead } from 'whitebox-engine/optim'
 * import { square, sub, sum, toFlat, type Value } from 'whitebox-engine/tensor'
 * const r = nelderMead((x) => sum(square(sub(x as Value, [1, -2]))), [0, 0], { xatol: 1e-8, fatol: 1e-12 })
 * r.converged // => true
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e6) / 1e6) // => [1, -2]
 * ```
 */
export function nelderMead(f: Objective, x0: Parameters, options: NelderMeadOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(nelderMeadAlgorithm(f, x0, options), obj, options, 200 * obj.n)
}

// ---------------------------------------------------------------------------------------------------------------
// One-dimensional minimisation for Powell's method, exactly as scipy's bracket, Brent and fminbound.

interface LineMin {
  readonly x: number
  readonly fun: number
}

/** scipy.optimize.bracket from (0, 1); on an invalid bracket, the best of its three points (scipy's recovery). */
function bracketBrent(g: (a: number) => number, tol: number): LineMin {
  const gold = 1.618034
  const tiny = 1e-21
  const growLimit = 110
  let xa = 0
  let xb = 1
  let fa = g(xa)
  let fb = g(xb)
  if (fa < fb) {
    ;[xa, xb] = [xb, xa]
    ;[fa, fb] = [fb, fa]
  }
  let xc = xb + gold * (xb - xa)
  let fc = g(xc)
  let iter = 0
  while (fc < fb) {
    const tmp1 = (xb - xa) * (fb - fc)
    const tmp2 = (xb - xc) * (fb - fa)
    const val = tmp2 - tmp1
    const denom = Math.abs(val) < tiny ? 2 * tiny : 2 * val
    let w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
    const wlim = xb + growLimit * (xc - xb)
    if (iter > 1000) throw new WhiteboxError('convergence', 'powell: no bracket found in 1000 iterations')
    iter += 1
    let fw: number
    if ((w - xc) * (xb - w) > 0) {
      fw = g(w)
      if (fw < fc) {
        xa = xb
        xb = w
        fa = fb
        fb = fw
        break
      }
      if (fw > fb) {
        xc = w
        fc = fw
        break
      }
      w = xc + gold * (xc - xb)
      fw = g(w)
    } else if ((w - wlim) * (wlim - xc) >= 0) {
      w = wlim
      fw = g(w)
    } else if ((w - wlim) * (xc - w) > 0) {
      fw = g(w)
      if (fw < fc) {
        xb = xc
        xc = w
        w = xc + gold * (xc - xb)
        fb = fc
        fc = fw
        fw = g(w)
      }
    } else {
      w = xc + gold * (xc - xb)
      fw = g(w)
    }
    xa = xb
    xb = xc
    xc = w
    fa = fb
    fb = fc
    fc = fw
  }
  const valid =
    ((fb < fc && fb <= fa) || (fb < fa && fb <= fc)) &&
    ((xa < xb && xb < xc) || (xc < xb && xb < xa)) &&
    Number.isFinite(xa) &&
    Number.isFinite(xb) &&
    Number.isFinite(xc)
  if (!valid) {
    if ([xa, xb, xc, fa, fb, fc].some(Number.isNaN)) return { x: Number.NaN, fun: Number.NaN }
    const fs = [fa, fb, fc]
    const xs = [xa, xb, xc]
    let i = 0
    for (let k = 1; k < 3; k++) if ((fs[k] ?? 0) < (fs[i] ?? 0)) i = k
    return { x: xs[i] ?? 0, fun: fs[i] ?? 0 }
  }
  // Brent's method on the bracket (scipy's Brent.optimize, maxiter 500).
  const mintol = 1e-11
  const cg = 0.381966
  let x = xb
  let w = xb
  let v = xb
  let fx = fb
  let fw = fb
  let fv = fb
  let a = Math.min(xa, xc)
  let b = Math.max(xa, xc)
  let deltax = 0
  let rat = 0
  for (let it = 0; it < 500; it++) {
    const tol1 = tol * Math.abs(x) + mintol
    const tol2 = 2 * tol1
    const xmid = 0.5 * (a + b)
    if (Math.abs(x - xmid) < tol2 - 0.5 * (b - a)) break
    if (Math.abs(deltax) <= tol1) {
      deltax = x >= xmid ? a - x : b - x
      rat = cg * deltax
    } else {
      const t1 = (x - w) * (fx - fv)
      let t2 = (x - v) * (fx - fw)
      let p = (x - v) * t2 - (x - w) * t1
      t2 = 2 * (t2 - t1)
      if (t2 > 0) p = -p
      t2 = Math.abs(t2)
      const dxTemp = deltax
      deltax = rat
      if (p > t2 * (a - x) && p < t2 * (b - x) && Math.abs(p) < Math.abs(0.5 * t2 * dxTemp)) {
        rat = p / t2
        const u = x + rat
        if (u - a < tol2 || b - u < tol2) rat = xmid - x >= 0 ? tol1 : -tol1
      } else {
        deltax = x >= xmid ? a - x : b - x
        rat = cg * deltax
      }
    }
    const u = Math.abs(rat) < tol1 ? (rat >= 0 ? x + tol1 : x - tol1) : x + rat
    const fu = g(u)
    if (fu > fx) {
      if (u < x) a = u
      else b = u
      if (fu <= fw || w === x) {
        v = w
        w = u
        fv = fw
        fw = fu
      } else if (fu <= fv || v === x || v === w) {
        v = u
        fv = fu
      }
    } else {
      if (u >= x) a = x
      else b = x
      v = w
      w = x
      x = u
      fv = fw
      fw = fx
      fx = fu
    }
  }
  return { x, fun: fx }
}

/** scipy's bounded scalar minimiser (fminbound: golden section with parabolic steps) on [x1, x2]. */
function fminbound(g: (a: number) => number, x1: number, x2: number, xatol: number, maxfun = 500): LineMin {
  const sqrtEps = Math.sqrt(2.2e-16)
  const goldenMean = 0.5 * (3 - Math.sqrt(5))
  let a = x1
  let b = x2
  let fulc = a + goldenMean * (b - a)
  let nfc = fulc
  let xf = fulc
  let rat = 0
  let e = 0
  let x = xf
  let fx = g(x)
  let num = 1
  let ffulc = fx
  let fnfc = fx
  let xm = 0.5 * (a + b)
  let tol1 = sqrtEps * Math.abs(xf) + xatol / 3
  let tol2 = 2 * tol1
  while (Math.abs(xf - xm) > tol2 - 0.5 * (b - a)) {
    let golden = true
    if (Math.abs(e) > tol1) {
      golden = false
      let r = (xf - nfc) * (fx - ffulc)
      let q = (xf - fulc) * (fx - fnfc)
      let p = (xf - fulc) * q - (xf - nfc) * r
      q = 2 * (q - r)
      if (q > 0) p = -p
      q = Math.abs(q)
      r = e
      e = rat
      if (Math.abs(p) < Math.abs(0.5 * q * r) && p > q * (a - xf) && p < q * (b - xf)) {
        rat = (p + 0) / q
        x = xf + rat
        if (x - a < tol2 || b - x < tol2) {
          const si = Math.sign(xm - xf) + (xm - xf === 0 ? 1 : 0)
          rat = tol1 * si
        }
      } else golden = true
    }
    if (golden) {
      e = xf >= xm ? a - xf : b - xf
      rat = goldenMean * e
    }
    const si = Math.sign(rat) + (rat === 0 ? 1 : 0)
    x = xf + si * Math.max(Math.abs(rat), tol1)
    const fu = g(x)
    num += 1
    if (fu <= fx) {
      if (x >= xf) a = xf
      else b = xf
      fulc = nfc
      ffulc = fnfc
      nfc = xf
      fnfc = fx
      xf = x
      fx = fu
    } else {
      if (x < xf) a = x
      else b = x
      if (fu <= fnfc || nfc === xf) {
        fulc = nfc
        ffulc = fnfc
        nfc = x
        fnfc = fu
      } else if (fu <= ffulc || fulc === xf || fulc === nfc) {
        fulc = x
        ffulc = fu
      }
    }
    xm = 0.5 * (a + b)
    tol1 = sqrtEps * Math.abs(xf) + xatol / 3
    tol2 = 2 * tol1
    if (num >= maxfun) break
  }
  return { x: xf, fun: fx }
}

/** The interval of t with lower <= x + t d <= upper over the coordinates where d is non-zero (scipy). */
function lineForSearch(x: Float64Array, d: Float64Array, box: BoundArrays): [number, number] {
  let lmin = Number.NEGATIVE_INFINITY
  let lmax = Number.POSITIVE_INFINITY
  for (let i = 0; i < x.length; i++) {
    const di = d[i] ?? 0
    if (di === 0) continue
    const low = ((box.lower[i] ?? 0) - (x[i] ?? 0)) / di
    const high = ((box.upper[i] ?? 0) - (x[i] ?? 0)) / di
    lmin = Math.max(lmin, di > 0 ? low : high)
    lmax = Math.min(lmax, di > 0 ? high : low)
  }
  return lmax >= lmin ? [lmin, lmax] : [0, 0]
}

/** Minimise along x + t d (scipy's _linesearch_powell): returns the value, the new point and the step taken. */
function powellLine(
  fun: (x: ArrayLike<number>) => number,
  x: Float64Array,
  d: Float64Array,
  tol: number,
  box: BoundArrays | undefined,
  fval: number,
): { value: number; x: Float64Array; step: Float64Array } {
  if (d.every((v) => v === 0)) return { value: fval, x, step: d }
  const along = (t: number): Float64Array => Float64Array.from(x, (xi, i) => xi + t * (d[i] ?? 0))
  const g = (t: number): number => fun(along(t))
  let res: LineMin
  let t: number
  const [lo, hi] = box === undefined ? [Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY] : lineForSearch(x, d, box)
  if (lo === Number.NEGATIVE_INFINITY && hi === Number.POSITIVE_INFINITY) {
    res = bracketBrent(g, tol)
    t = res.x
  } else if (Number.isFinite(lo) && Number.isFinite(hi)) {
    res = fminbound(g, lo, hi, tol / 100)
    t = res.x
  } else {
    // Bounded on one side only: search over arctan of the step.
    res = fminbound((u) => g(Math.tan(u)), Math.atan(lo), Math.atan(hi), tol / 100)
    t = Math.tan(res.x)
  }
  const step = Float64Array.from(d, (v) => t * v)
  return { value: res.fun, x: Float64Array.from(x, (xi, i) => xi + (step[i] ?? 0)), step }
}

// ---------------------------------------------------------------------------------------------------------------
// Powell

/** Options of {@link powell}. */
export interface PowellOptions extends VectorMethodOptions {
  /** Relative tolerance of the line minimisations (default 1e-4; scipy passes 100 xtol to Brent). */
  readonly xtol?: number
  /** Stop when an iteration decreases f by less than ftol (|f| + |f_new|) / 2 (default 1e-4). */
  readonly ftol?: number
  /** The initial direction set, n vectors (default the coordinate axes). */
  readonly directions?: readonly (readonly number[])[]
  /** The evaluation budget (scipy's maxfev; default 1000 n). */
  readonly maxEvaluations?: number
  /** Box bounds: line searches stay inside the box. */
  readonly bounds?: Bounds
}

/** State of a Powell run. */
export interface PowellState extends VectorState {
  /** The current direction set. */
  readonly directions: readonly Float64Array[]
  /** The point at the start of the last iteration (for the extrapolated direction). */
  readonly start: Float64Array
  /** The largest decrease along one direction in the last iteration, and its index. */
  readonly largestDecrease: number
  readonly largestIndex: number
  /** Whether the last iteration replaced a direction by the extrapolated one. */
  readonly replaced: boolean
}

/**
 * Powell's conjugate direction method as a traced algorithm, as scipy's `minimize(method='Powell')`: each step
 * minimises along every direction of the set by Brent's method (bounded minimisation inside box bounds), then tries
 * the extrapolated direction and replaces the direction of largest decrease when Powell's test allows.
 *
 * @param f - The objective.
 * @param x0 - The starting point (any parameter tree).
 * @param options - Tolerances, initial directions, budget, bounds and the iteration cap.
 * @returns The algorithm.
 * @concept derivative-free-optimisation
 * @example
 * ```ts
 * import { powellAlgorithm } from 'whitebox-engine/optim'
 * import { square, sum, type Value } from 'whitebox-engine/tensor'
 * powellAlgorithm((x) => sum(square(x as Value)), [1, 2]).name // => 'powell'
 * ```
 */
export function powellAlgorithm(f: Objective, x0: Parameters, options: PowellOptions = {}): Algorithm<PowellState> {
  const obj = flatObjective(f, x0)
  const n = obj.n
  if (n === 0) throw new WhiteboxError('value', 'powell: no parameters')
  const box = options.bounds === undefined ? undefined : boundArrays(options.bounds, n, 'powell')
  const xtol = options.xtol ?? 1e-4
  const ftol = options.ftol ?? 1e-4
  const maxfun = options.maxEvaluations ?? 1000 * n
  const maxiter = options.maxIterations ?? 1000 * n
  const directions0 = (
    options.directions ?? Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
  ).map((d) => {
    if (d.length !== n) throw new WhiteboxError('shape', `powell: directions must have length ${n}`)
    return Float64Array.from(d)
  })
  if (directions0.length !== n) throw new WhiteboxError('shape', `powell: ${n} directions are needed`)
  return {
    name: 'powell',
    config: configOf({ xtol, ftol, maxEvaluations: maxfun }),
    schema: vectorSchema({
      metrics: {
        largestDecrease: { description: 'largest decrease of f along one direction in the iteration' },
        largestIndex: { description: 'index of the direction with the largest decrease' },
      },
      tensors: { directions: { role: 'state', description: 'the direction set, one row per direction' } },
    }),
    init: () => {
      const x = Float64Array.from(obj.x0)
      return {
        x,
        value: obj.value(x),
        grad: EMPTY,
        nfev: 1,
        ngev: 0,
        nhev: 0,
        step: 0,
        directions: directions0,
        start: x,
        largestDecrease: 0,
        largestIndex: 0,
        replaced: false,
      }
    },
    step: (s, ctx) => {
      const ev = counted(obj.value, s.nfev, maxfun)
      const direc = s.directions.map((d) => d)
      let x = s.x
      let fval = s.value
      let x1 = s.start
      const fx = fval
      let bigind = 0
      let delta = 0
      let replaced = false
      let stop: PowellState['stop']
      const xStart = s.x
      try {
        for (let i = 0; i < n; i++) {
          const fx2 = fval
          const r = powellLine(ev.f, x, direc[i] as Float64Array, xtol * 100, box, fval)
          fval = r.value
          x = r.x
          if (fx2 - fval > delta) {
            delta = fx2 - fval
            bigind = i
          }
        }
        const bnd = ftol * (Math.abs(fx) + Math.abs(fval)) + 1e-20
        if (2 * (fx - fval) <= bnd) stop = { status: 'converged', message: 'relative decrease of f below ftol' }
        else if (s.nfev + ev.calls() >= maxfun)
          stop = { status: 'failed', message: 'maximum number of function evaluations reached' }
        else if (Number.isNaN(fx) && Number.isNaN(fval)) stop = { status: 'failed', message: 'NaN encountered' }
        else if (ctx.t + 1 < maxiter) {
          const direc1 = Float64Array.from(x, (xi, i) => xi - (x1[i] ?? 0))
          x1 = x
          const lmax = box === undefined ? 1 : lineForSearch(x, direc1, box)[1]
          const x2 = Float64Array.from(x, (xi, i) => xi + Math.min(lmax, 1) * (direc1[i] ?? 0))
          const fx2 = ev.f(x2)
          if (fx > fx2) {
            let t = 2 * (fx + fx2 - 2 * fval)
            let temp = fx - fval - delta
            t *= temp * temp
            temp = fx - fx2
            t -= delta * temp * temp
            if (t < 0) {
              const r = powellLine(ev.f, x, direc1, xtol * 100, box, fval)
              fval = r.value
              x = r.x
              if (r.step.some((v) => v !== 0)) {
                direc[bigind] = direc[n - 1] as Float64Array
                direc[n - 1] = r.step
                replaced = true
              }
            }
          }
        }
      } catch (e) {
        if (!isBudget(e)) throw e
        stop = { status: 'failed', message: 'maximum number of function evaluations reached' }
      }
      let moved = 0
      for (let i = 0; i < n; i++) moved += ((x[i] ?? 0) - (xStart[i] ?? 0)) ** 2
      return {
        x,
        value: fval,
        grad: EMPTY,
        nfev: s.nfev + ev.calls(),
        ngev: 0,
        nhev: 0,
        step: Math.sqrt(moved),
        directions: direc,
        start: x1,
        largestDecrease: delta,
        largestIndex: bigind,
        replaced,
        ...(stop === undefined ? {} : { stop }),
      }
    },
    done: vectorDone,
    observe: (s) => {
      const flat = new Float64Array(n * n)
      for (const [k, d] of s.directions.entries()) flat.set(d, k * n)
      return observeVector(s, {
        metrics: { largestDecrease: s.largestDecrease, largestIndex: s.largestIndex },
        tensors: { directions: { role: 'state', value: flat, shape: [n, n] } },
      })
    },
  }
}

/**
 * Minimise by Powell's conjugate direction method (scipy.optimize.minimize with method 'Powell').
 *
 * @param f - The objective.
 * @param x0 - The starting point (any parameter tree).
 * @param options - Tolerances, initial directions, budgets (default 1000 n iterations and evaluations), bounds and run
 *   options.
 * @returns The result.
 * @concept derivative-free-optimisation
 * @example
 * ```ts
 * import { powell } from 'whitebox-engine/optim'
 * import { square, sub, sum, toFlat, type Value } from 'whitebox-engine/tensor'
 * const r = powell((x) => sum(square(sub(x as Value, [1, -2]))), [0, 0])
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e6) / 1e6) // => [1, -2]
 * ```
 */
export function powell(f: Objective, x0: Parameters, options: PowellOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(powellAlgorithm(f, x0, options), obj, options, 1000 * obj.n)
}
