/**
 * Powell's conjugate direction method, a derivative-free minimiser that line-searches along a set of directions by
 * Brent's method and replaces a direction by the overall displacement, as a traced algorithm matching
 * `scipy.optimize.minimize(method='Powell')`.
 */

import { isWhiteboxError, WhiteboxError } from '../../base/index.ts'
import { metricFunction } from '../../numerics/geometry/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { brent, fminbound } from '../scalar.ts'
import { type BoundArrays, boundArrays } from './_bounds.ts'
import { counted, isBudget } from './_budget.ts'
import { configOf, observeVector, runVector, type VectorState, vectorDone, vectorSchema } from './_shared.ts'
import type { Bounds } from './objective.ts'
import {
  flatObjective,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'

const EMPTY = new Float64Array(0)

// ---------------------------------------------------------------------------------------------------------------
// One-dimensional minimisation for Powell's method, exactly as scipy's bracket, Brent and fminbound.

interface LineMin {
  readonly x: number
  readonly fun: number
}

/**
 * scipy's line minimisation of Powell's method: Brent's method on a bracket grown from (0, 1); on an invalid bracket,
 * the best of its three points (scipy's _recover_from_bracket_error).
 */
function bracketBrent(g: (a: number) => number, tol: number): LineMin {
  try {
    const r = brent(g, { tol })
    return { x: r.x, fun: r.fun }
  } catch (e) {
    if (!isWhiteboxError(e, 'convergence') || !('fb' in e.details)) throw e
    const d = e.details as Readonly<Record<'xa' | 'xb' | 'xc' | 'fa' | 'fb' | 'fc', number>>
    const xs = [d.xa, d.xb, d.xc]
    const fs = [d.fa, d.fb, d.fc]
    if ([...xs, ...fs].some(Number.isNaN)) return { x: Number.NaN, fun: Number.NaN }
    let i = 0
    for (let k = 1; k < 3; k++) if ((fs[k] ?? 0) < (fs[i] ?? 0)) i = k
    return { x: xs[i] ?? 0, fun: fs[i] ?? 0 }
  }
}

/** scipy's bounded scalar minimiser on [x1, x2] (fminbound), as Powell's method calls it. */
function boundedLine(g: (a: number) => number, x1: number, x2: number, xatol: number): LineMin {
  const r = fminbound(g, { bounds: [x1, x2], tol: xatol })
  return { x: r.x, fun: r.fun }
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
    res = boundedLine(g, lo, hi, tol / 100)
    t = res.x
  } else {
    // Bounded on one side only: search over arctan of the step.
    res = boundedLine((u) => g(Math.tan(u)), Math.atan(lo), Math.atan(hi), tol / 100)
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
  const euclid = metricFunction('euclidean', n)
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
      return {
        x,
        value: fval,
        grad: EMPTY,
        nfev: s.nfev + ev.calls(),
        ngev: 0,
        nhev: 0,
        step: euclid(x, 0, xStart, 0),
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
