import { WhiteboxError } from '../../base/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { boundArrays, clipInto } from './_bounds.ts'
import { counted, isBudget } from './_budget.ts'
import { configOf, observeVector, runVector, type VectorState, vectorDone, vectorSchema } from './_shared.ts'
import {
  type Bounds,
  flatObjective,
  type MinimizeResult,
  type Objective,
  type Parameters,
  type VectorMethodOptions,
} from './objective.ts'

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
