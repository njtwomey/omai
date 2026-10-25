/**
 * Pattern (direct) search methods, which need only function values: Hooke-Jeeves exploratory and pattern moves, and
 * compass search, the simplest generalised pattern search, which polls a mesh along a set of directions and shrinks it
 * when nothing improves. Both are traced algorithms.
 */

import { MdmlError } from '../../base/index.ts'
import { metricFunction } from '../../numerics/geometry/index.ts'
import type { Algorithm } from '../../trace/index.ts'
import { type BoundArrays, boundArrays } from './_bounds.ts'
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

const inBox = (x: ArrayLike<number>, box: BoundArrays | undefined): boolean => {
  if (box === undefined) return true
  for (let i = 0; i < x.length; i++)
    if ((x[i] ?? 0) < (box.lower[i] ?? 0) || (x[i] ?? 0) > (box.upper[i] ?? 0)) return false
  return true
}

/** Options of {@link hookeJeeves}. */
export interface HookeJeevesOptions extends VectorMethodOptions {
  /** The initial step along each coordinate (default 0.5, or 10% of |x0_i| when larger). */
  readonly step?: number
  /** Stop when the step falls below this (default 1e-6). */
  readonly stepTol?: number
  /** The factor by which the step shrinks after a failed exploration (default 0.5). */
  readonly shrink?: number
  /** The evaluation budget (default 1000 n). */
  readonly maxEvaluations?: number
  /** Box bounds: points outside the box are never evaluated. */
  readonly bounds?: Bounds
}

/** State of a Hooke-Jeeves run. */
export interface HookeJeevesState extends VectorState {
  /** The base point of the last pattern move. */
  readonly base: Float64Array
  /** The step size per coordinate. */
  readonly steps: Float64Array
  /** What the last step did. */
  readonly move: 'initial' | 'pattern' | 'explore' | 'shrink'
}

/**
 * The Hooke-Jeeves direct search (1961) as a traced algorithm: an exploratory move tries +-step along each coordinate,
 * keeping improvements; a pattern move then jumps from the new point along the direction of progress and explores
 * there. When exploration fails, the step shrinks. Each algorithm step is one exploration (with its pattern move).
 *
 * @param f - The objective.
 * @param x0 - The starting point (any parameter tree).
 * @param options - Initial step, step tolerance, shrink factor, budget and bounds.
 * @returns The algorithm.
 * @concept derivative-free-optimisation
 * @example
 * ```ts
 * import { hookeJeevesAlgorithm } from 'mdml-engine/optim'
 * import { square, sum, type Value } from 'mdml-engine/tensor'
 * hookeJeevesAlgorithm((x) => sum(square(x as Value)), [1, 2]).name // => 'hookeJeeves'
 * ```
 */
export function hookeJeevesAlgorithm(
  f: Objective,
  x0: Parameters,
  options: HookeJeevesOptions = {},
): Algorithm<HookeJeevesState> {
  const obj = flatObjective(f, x0)
  const n = obj.n
  if (n === 0) throw new MdmlError('value', 'hookeJeeves: no parameters')
  const euclid = metricFunction('euclidean', n)
  const dist = (a: Float64Array, b: Float64Array): number => euclid(a, 0, b, 0)
  const box = options.bounds === undefined ? undefined : boundArrays(options.bounds, n, 'hookeJeeves')
  if (!inBox(obj.x0, box)) throw new MdmlError('value', 'hookeJeeves: x0 is outside the bounds')
  const stepTol = options.stepTol ?? 1e-6
  const shrink = options.shrink ?? 0.5
  const maxfun = options.maxEvaluations ?? 1000 * n
  const step0 = options.step ?? 0.5

  /** Explore around x (value fx): returns the improved point, its value and the evaluations used. */
  const explore = (x: Float64Array, fx: number, steps: Float64Array, budget: number) => {
    let y = Float64Array.from(x)
    let fy = fx
    let used = 0
    for (let i = 0; i < n && used < budget; i++) {
      for (const sign of [1, -1]) {
        if (used >= budget) break
        const z = Float64Array.from(y)
        z[i] = (z[i] ?? 0) + sign * (steps[i] ?? 0)
        if (!inBox(z, box)) continue
        const fz = obj.value(z)
        used += 1
        if (fz < fy) {
          y = z
          fy = fz
          break
        }
      }
    }
    return { y, fy, used }
  }

  const decide = (s: Omit<HookeJeevesState, 'stop'>): HookeJeevesState => {
    let largest = 0
    for (const v of s.steps) largest = Math.max(largest, v)
    if (largest < stepTol) return { ...s, stop: { status: 'converged', message: 'step size below stepTol' } }
    if (s.nfev >= maxfun)
      return { ...s, stop: { status: 'failed', message: 'maximum number of function evaluations reached' } }
    return s
  }

  return {
    name: 'hookeJeeves',
    config: configOf({ step: step0, stepTol, shrink, maxEvaluations: maxfun }),
    schema: vectorSchema({
      metrics: { stepSize: { description: 'largest coordinate step' } },
      messages: { move: { description: 'pattern, explore or shrink' } },
    }),
    init: () => {
      const x = Float64Array.from(obj.x0)
      const steps = Float64Array.from(x, (v) => Math.max(step0, 0.1 * Math.abs(v)))
      return decide({
        x,
        value: obj.value(x),
        grad: EMPTY,
        nfev: 1,
        ngev: 0,
        nhev: 0,
        step: 0,
        base: x,
        steps,
        move: 'initial',
      })
    },
    step: (s) => {
      const budget = maxfun - s.nfev
      const e = explore(s.x, s.value, s.steps, budget)
      let nfev = s.nfev + e.used
      if (e.fy < s.value) {
        // Pattern move: x + (x_new - x), explored; keep it if it beats x_new.
        const pattern = Float64Array.from(e.y, (v, i) => 2 * v - (s.x[i] ?? 0))
        if (inBox(pattern, box) && nfev < maxfun) {
          const fp = obj.value(pattern)
          nfev += 1
          const ep = explore(pattern, fp, s.steps, maxfun - nfev)
          nfev += ep.used
          if (ep.fy < e.fy)
            return decide({ ...s, x: ep.y, value: ep.fy, nfev, step: dist(ep.y, s.x), base: e.y, move: 'pattern' })
        }
        return decide({ ...s, x: e.y, value: e.fy, nfev, step: dist(e.y, s.x), base: s.x, move: 'explore' })
      }
      return decide({ ...s, nfev, step: 0, steps: Float64Array.from(s.steps, (v) => v * shrink), move: 'shrink' })
    },
    done: vectorDone,
    observe: (s) => {
      let largest = 0
      for (const v of s.steps) largest = Math.max(largest, v)
      return observeVector(s, {
        metrics: { stepSize: largest },
        messages: [{ kind: 'move', payload: { move: s.move } }],
      })
    },
  }
}

/**
 * Minimise by the Hooke-Jeeves pattern search.
 *
 * @param f - The objective.
 * @param x0 - The starting point (any parameter tree).
 * @param options - Initial step, step tolerance, shrink factor, budgets, bounds and run options.
 * @returns The result.
 * @concept derivative-free-optimisation
 * @example
 * ```ts
 * import { hookeJeeves } from 'mdml-engine/optim'
 * import { square, sub, sum, toFlat, type Value } from 'mdml-engine/tensor'
 * const r = hookeJeeves((x) => sum(square(sub(x as Value, [1, -2]))), [0, 0], { stepTol: 1e-9 })
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e6) / 1e6) // => [1, -2]
 * ```
 */
export function hookeJeeves(f: Objective, x0: Parameters, options: HookeJeevesOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(hookeJeevesAlgorithm(f, x0, options), obj, options, 100_000)
}

/** Options of {@link compassSearch}. */
export interface CompassSearchOptions extends VectorMethodOptions {
  /** The initial mesh size (default 1). */
  readonly step?: number
  /** Stop when the mesh size falls below this (default 1e-6). */
  readonly stepTol?: number
  /** The mesh expansion after a successful poll (default 2; 1 keeps the mesh). */
  readonly expansion?: number
  /** The mesh contraction after an unsuccessful poll (default 0.5). */
  readonly contraction?: number
  /** Poll directions: `coordinate` (the 2n vectors +-e_i) or `minimal` (the n + 1 vectors e_i and -sum e_i). */
  readonly poll?: 'coordinate' | 'minimal'
  /** Move at the first improving poll point (default true) instead of polling them all and taking the best. */
  readonly opportunistic?: boolean
  /** The evaluation budget (default 1000 n). */
  readonly maxEvaluations?: number
  /** Box bounds: infeasible poll points are skipped (the coordinate directions conform to the box). */
  readonly bounds?: Bounds
}

/** State of a compass (generalised pattern) search. */
export interface CompassSearchState extends VectorState {
  /** The mesh size. */
  readonly mesh: number
  /** Whether the last poll found a better point. */
  readonly success: boolean
  /** The index of the direction polled first (the last successful one, so it is tried again first). */
  readonly first: number
}

/**
 * Compass search, the simplest generalised pattern search (Kolda, Lewis and Torczon 2003), as a traced algorithm:
 * each step polls the mesh points x + mesh d over a positive spanning set of directions, moves to an improving point
 * and expands the mesh, or contracts the mesh when none improves. Converges to stationary points of smooth functions
 * without using derivatives.
 *
 * @param f - The objective.
 * @param x0 - The starting point (any parameter tree).
 * @param options - Mesh size, tolerance, expansion and contraction, poll set, opportunism, budget and bounds.
 * @returns The algorithm.
 * @concept derivative-free-optimisation
 * @example
 * ```ts
 * import { compassSearchAlgorithm } from 'mdml-engine/optim'
 * import { square, sum, type Value } from 'mdml-engine/tensor'
 * compassSearchAlgorithm((x) => sum(square(x as Value)), [1, 2]).name // => 'compassSearch'
 * ```
 */
export function compassSearchAlgorithm(
  f: Objective,
  x0: Parameters,
  options: CompassSearchOptions = {},
): Algorithm<CompassSearchState> {
  const obj = flatObjective(f, x0)
  const n = obj.n
  if (n === 0) throw new MdmlError('value', 'compassSearch: no parameters')
  const box = options.bounds === undefined ? undefined : boundArrays(options.bounds, n, 'compassSearch')
  if (!inBox(obj.x0, box)) throw new MdmlError('value', 'compassSearch: x0 is outside the bounds')
  const stepTol = options.stepTol ?? 1e-6
  const expansion = options.expansion ?? 2
  const contraction = options.contraction ?? 0.5
  const opportunistic = options.opportunistic ?? true
  const maxfun = options.maxEvaluations ?? 1000 * n
  const poll = options.poll ?? 'coordinate'
  const directions: Float64Array[] = []
  for (let i = 0; i < n; i++) directions.push(Float64Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)))
  if (poll === 'coordinate')
    for (let i = 0; i < n; i++) directions.push(Float64Array.from({ length: n }, (_, j) => (i === j ? -1 : 0)))
  else directions.push(new Float64Array(n).fill(-1))

  const decide = (s: Omit<CompassSearchState, 'stop'>): CompassSearchState => {
    if (s.mesh < stepTol) return { ...s, stop: { status: 'converged', message: 'mesh size below stepTol' } }
    if (s.nfev >= maxfun)
      return { ...s, stop: { status: 'failed', message: 'maximum number of function evaluations reached' } }
    return s
  }

  return {
    name: 'compassSearch',
    config: configOf({
      step: options.step ?? 1,
      stepTol,
      expansion,
      contraction,
      poll,
      opportunistic,
      maxEvaluations: maxfun,
    }),
    schema: vectorSchema({ metrics: { mesh: { description: 'mesh size' } } }),
    init: () => {
      const x = Float64Array.from(obj.x0)
      return decide({
        x,
        value: obj.value(x),
        grad: EMPTY,
        nfev: 1,
        ngev: 0,
        nhev: 0,
        step: 0,
        mesh: options.step ?? 1,
        success: false,
        first: 0,
      })
    },
    step: (s) => {
      let best = s.x
      let fbest = s.value
      let bestDir = -1
      let nfev = s.nfev
      for (let k = 0; k < directions.length && nfev < maxfun; k++) {
        const j = (s.first + k) % directions.length
        const d = directions[j] as Float64Array
        const y = Float64Array.from(s.x, (v, i) => v + s.mesh * (d[i] ?? 0))
        if (!inBox(y, box)) continue
        const fy = obj.value(y)
        nfev += 1
        if (fy < fbest) {
          best = y
          fbest = fy
          bestDir = j
          if (opportunistic) break
        }
      }
      if (bestDir >= 0)
        return decide({
          ...s,
          x: best,
          value: fbest,
          nfev,
          step: s.mesh,
          mesh: s.mesh * expansion,
          success: true,
          first: bestDir,
        })
      return decide({ ...s, nfev, step: 0, mesh: s.mesh * contraction, success: false })
    },
    done: vectorDone,
    observe: (s) => observeVector(s, { metrics: { mesh: s.mesh } }),
  }
}

/**
 * Minimise by compass search (a generalised pattern search).
 *
 * @param f - The objective.
 * @param x0 - The starting point (any parameter tree).
 * @param options - Mesh size, tolerance, expansion and contraction, poll set, opportunism, budgets, bounds and run
 *   options.
 * @returns The result.
 * @concept derivative-free-optimisation
 * @example
 * ```ts
 * import { compassSearch } from 'mdml-engine/optim'
 * import { square, sub, sum, toFlat, type Value } from 'mdml-engine/tensor'
 * const r = compassSearch((x) => sum(square(sub(x as Value, [1, -2]))), [0, 0], { stepTol: 1e-9 })
 * toFlat(r.x as Value).map((v) => Math.round(v * 1e6) / 1e6) // => [1, -2]
 * ```
 */
export function compassSearch(f: Objective, x0: Parameters, options: CompassSearchOptions = {}): MinimizeResult {
  const obj = flatObjective(f, x0)
  return runVector(compassSearchAlgorithm(f, x0, options), obj, options, 100_000)
}
