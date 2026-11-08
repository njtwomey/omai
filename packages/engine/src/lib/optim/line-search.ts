/**
 * Line searches, which choose how far to step along a descent direction: the shared line-function and result types,
 * backtracking (Armijo) search, the strong Wolfe search of Nocedal and Wright (as `scipy.optimize.line_search`),
 * fallbacks between searches, and `lineSearch` on a vector objective. Trial points are recorded for figures.
 */

import { OmaiError } from '../base/index.ts'
import { daxpy, ddot } from '../numerics/linalg/index.ts'
import { type TensorLike, toVector } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { checkConstants, firstStep, probe } from './_line-search.ts'
import { backtrackingSearch } from './line-search-algorithms.ts'
import { type FlatObjective, flatObjective, type Objective, type Parameters } from './minimize/objective.ts'

/** The line function at one step: phi(alpha) = f(x + alpha p), its slope phi'(alpha) = grad f . p, and the gradient. */
export interface LinePoint {
  readonly value: number
  readonly slope: number
  /** The full gradient at x + alpha p, when the line function comes from a vector objective. */
  readonly grad?: Float64Array
}

/**
 * The one-dimensional function a line search explores. `at` returns the value and slope; `valueAt`, when given, is a
 * cheaper value-only evaluation that backtracking uses.
 */
export interface LineFunction {
  readonly at: (alpha: number) => LinePoint
  readonly valueAt?: (alpha: number) => number
}

/** Where a line search starts: phi(0), phi'(0) < 0, the first trial step and the previous objective value. */
export interface LineStart {
  readonly value: number
  readonly slope: number
  /** The first trial step (default 1, or the scipy heuristic from `previousValue`). */
  readonly alpha?: number
  /** The objective at the previous iterate (scipy's old_old_fval), for the first-step heuristic. */
  readonly previousValue?: number
}

/** One evaluation made by a line search (the trial points of figures). `slope` is NaN for value-only evaluations. */
export interface LineTrial {
  readonly alpha: number
  readonly value: number
  readonly slope: number
}

/** Why a search that did not meet its conditions still ended at a usable last trial step (More-Thuente's warnings). */
export type LineSearchWarning = 'xtol' | 'rounding' | 'alphaMax' | 'alphaMin'

/** The outcome of a line search. */
export interface LineSearchResult {
  /** The accepted step (NaN when the search failed). */
  readonly alpha: number
  readonly value: number
  /** phi'(alpha), NaN when it was not evaluated at the accepted step. */
  readonly slope: number
  /** The gradient at the accepted step, when the line function provides gradients. */
  readonly grad?: Float64Array
  readonly success: boolean
  /** Evaluations of the value (with or without the slope). */
  readonly evaluations: number
  /** Evaluations that included the slope (gradient evaluations). */
  readonly gradEvaluations: number
  readonly trials: readonly LineTrial[]
  readonly message: string
  /**
   * Set when `success` is false because the search stopped at a tolerance (`xtol`, `rounding`) or a step bound
   * (`alphaMax`, `alphaMin`) before the conditions held: the last trial step is finite and a solver may take it.
   */
  readonly warning?: LineSearchWarning
}

/** A line search: a function of the line function and the start, as every second-order method takes it. */
export type LineSearch = (phi: LineFunction, start: LineStart) => LineSearchResult

/** An optional acceptance test beyond the Wolfe conditions (scipy's extra_condition), given the step and its point. */
export type ExtraCondition = (alpha: number, point: LinePoint) => boolean

/** Options of {@link backtracking}. */
export interface BacktrackingOptions {
  /** Sufficient-decrease constant (default 1e-4). */
  readonly c1?: number
  /** Smallest step before giving up (default 0). */
  readonly alphaMin?: number
  /**
   * `cubic` (default): scipy's scalar_search_armijo, quadratic then cubic interpolation of the rejected values;
   * `halving`: multiply the step by `rho` until the Armijo condition holds.
   */
  readonly interpolation?: 'cubic' | 'halving'
  /** Contraction factor of the halving variant (default 0.5). */
  readonly rho?: number
  /** Most evaluations of the halving variant (default 50). */
  readonly maxEvaluations?: number
  /**
   * Clip each interpolated step to [lo, hi] times the previous one (Nocedal and Wright, section 3.5, typically
   * [0.1, 0.5]); without it the steps follow scipy exactly, whose first quadratic step can shrink by orders of
   * magnitude.
   */
  readonly safeguard?: readonly [number, number]
}

/**
 * Backtracking (Armijo) line search: shrink the step until phi(alpha) <= phi(0) + c1 alpha phi'(0). The default
 * interpolates the rejected values (a quadratic, then cubics) as scipy.optimize's scalar_search_armijo, optionally
 * safeguarded; `halving` multiplies by `rho`. Uses value-only evaluations when the line function has them.
 *
 * @param options - c1, smallest step, interpolation, contraction and evaluation cap.
 * @returns The line search.
 * @concept line-search
 * @example
 * ```ts
 * import { backtracking } from 'omai-engine/optim'
 * // phi(alpha) = (alpha - 0.3)^2 from phi(0) = 0.09, phi'(0) = -0.6
 * const phi = { at: (a: number) => ({ value: (a - 0.3) ** 2, slope: 2 * (a - 0.3) }) }
 * const r = backtracking()(phi, { value: 0.09, slope: -0.6 })
 * [r.success, Math.round(r.alpha * 1e12) / 1e12] // => [true, 0.3]
 * ```
 */
export function backtracking(options: BacktrackingOptions = {}): LineSearch {
  return (phi, start) => {
    const search = backtrackingSearch(phi, start, options)
    return search.result(runAlgorithm(search.algorithm, { maxSteps: Number.MAX_SAFE_INTEGER }).state)
  }
}

/** Options of {@link strongWolfe}. */
export interface StrongWolfeOptions {
  /** Sufficient-decrease constant (default 1e-4). */
  readonly c1?: number
  /** Curvature constant (default 0.9; 0.1 to 0.4 for conjugate gradients). */
  readonly c2?: number
  /** Largest step. */
  readonly alphaMax?: number
  /** Bracketing iterations (default 10, scipy's maxiter). */
  readonly maxIterations?: number
  /** Zoom iterations (default 10, as scipy's _zoom). */
  readonly zoomIterations?: number
  /** An extra acceptance test. */
  readonly extraCondition?: ExtraCondition
}

/** The minimiser of the cubic through (a, fa) with slope fpa, (b, fb) and (c, fc) (scipy's _cubicmin). */
function cubicmin(
  a: number,
  fa: number,
  fpa: number,
  b: number,
  fb: number,
  c: number,
  fc: number,
): number | undefined {
  const db = b - a
  const dc = c - a
  const denom = (db * dc) ** 2 * (db - dc)
  if (denom === 0) return undefined
  const r1 = fb - fa - fpa * db
  const r2 = fc - fa - fpa * dc
  const A = (dc * dc * r1 - db * db * r2) / denom
  const B = (-(dc ** 3) * r1 + db ** 3 * r2) / denom
  const radical = B * B - 3 * A * fpa
  if (radical < 0 || A === 0) return undefined
  const x = a + (-B + Math.sqrt(radical)) / (3 * A)
  return Number.isFinite(x) ? x : undefined
}

/** The minimiser of the quadratic through (a, fa) with slope fpa and (b, fb) (scipy's _quadmin). */
function quadmin(a: number, fa: number, fpa: number, b: number, fb: number): number | undefined {
  const db = b - a
  if (db === 0) return undefined
  const B = (fb - fa - fpa * db) / (db * db)
  if (B === 0) return undefined
  const x = a - fpa / (2 * B)
  return Number.isFinite(x) ? x : undefined
}

/**
 * The strong Wolfe line search of Nocedal and Wright (Algorithms 3.5 and 3.6), as scipy.optimize.line_search: bracket
 * a step by doubling, then zoom by cubic interpolation (falling back to quadratic interpolation and bisection) until
 * phi(alpha) <= phi(0) + c1 alpha phi'(0) and |phi'(alpha)| <= c2 |phi'(0)|.
 *
 * @param options - c1, c2, the largest step, iteration cap and an extra condition.
 * @returns The line search.
 * @concept line-search
 * @example
 * ```ts
 * import { strongWolfe } from 'omai-engine/optim'
 * // phi(alpha) = (alpha - 0.25)^2: the first trial (alpha = 1) overshoots, interpolation finds the minimum
 * const phi = { at: (a: number) => ({ value: (a - 0.25) ** 2, slope: 2 * (a - 0.25) }) }
 * const r = strongWolfe()(phi, { value: 0.0625, slope: -0.5 })
 * [r.success, r.alpha, r.trials.length] // => [true, 0.25, 2]
 * ```
 */
export function strongWolfe(options: StrongWolfeOptions = {}): LineSearch {
  const c1 = options.c1 ?? 1e-4
  const c2 = options.c2 ?? 0.9
  const amax = options.alphaMax
  const maxiter = options.maxIterations ?? 10
  const zoomIterations = options.zoomIterations ?? 10
  const extra = options.extraCondition ?? (() => true)
  checkConstants(c1, c2)
  return (phi, start) => {
    const p = probe(phi)
    const phi0 = start.value
    const d0 = start.slope
    // scipy's zoom search accepts a zero slope (a stationary start): the first trial then satisfies the conditions.
    if (d0 > 0 || Number.isNaN(d0)) return p.finish(undefined, phi0, Number.NaN, 'not a descent direction')
    const zoom = (
      aLo0: number,
      aHi0: number,
      phiLo0: number,
      phiHi0: number,
      dLo0: number,
    ): { a?: number; v: number; d: number } => {
      let aLo = aLo0
      let aHi = aHi0
      let phiLo = phiLo0
      let phiHi = phiHi0
      let dLo = dLo0
      let phiRec = phi0
      let aRec = 0
      for (let i = 0; i <= zoomIterations; i++) {
        const dalpha = aHi - aLo
        const [a, b] = dalpha < 0 ? [aHi, aLo] : [aLo, aHi]
        let aj: number | undefined
        if (i > 0) {
          const cchk = 0.2 * dalpha
          aj = cubicmin(aLo, phiLo, dLo, aHi, phiHi, aRec, phiRec)
          if (aj !== undefined && (aj > b - cchk || aj < a + cchk)) aj = undefined
        }
        if (aj === undefined) {
          const qchk = 0.1 * dalpha
          aj = quadmin(aLo, phiLo, dLo, aHi, phiHi)
          if (aj === undefined || aj > b - qchk || aj < a + qchk) aj = aLo + 0.5 * dalpha
        }
        const pj = p.at(aj)
        const phiJ = pj.value
        if (phiJ > phi0 + c1 * aj * d0 || phiJ >= phiLo) {
          phiRec = phiHi
          aRec = aHi
          aHi = aj
          phiHi = phiJ
        } else {
          const dj = pj.slope
          if (Math.abs(dj) <= -c2 * d0 && extra(aj, pj)) return { a: aj, v: phiJ, d: dj }
          if (dj * (aHi - aLo) >= 0) {
            phiRec = phiHi
            aRec = aHi
            aHi = aLo
            phiHi = phiLo
          } else {
            phiRec = phiLo
            aRec = aLo
          }
          aLo = aj
          phiLo = phiJ
          dLo = dj
        }
      }
      return { v: phiLo, d: Number.NaN }
    }
    let a0 = 0
    let a1 = firstStep(start)
    if (amax !== undefined) a1 = Math.min(a1, amax)
    let pt = p.at(a1)
    let phiA1 = pt.value
    let phiA0 = phi0
    let dA0 = d0
    for (let i = 0; i < maxiter; i++) {
      if (a1 === 0 || (amax !== undefined && a0 > amax)) {
        return p.finish(
          undefined,
          phi0,
          Number.NaN,
          a1 === 0 ? 'rounding errors prevent progress' : `no acceptable step up to alphaMax ${amax}`,
        )
      }
      if (phiA1 > phi0 + c1 * a1 * d0 || (phiA1 >= phiA0 && i > 0)) {
        const z = zoom(a0, a1, phiA0, phiA1, dA0)
        return p.finish(
          z.a,
          z.v,
          z.d,
          z.a === undefined ? 'zoom did not converge' : 'strong Wolfe conditions satisfied',
        )
      }
      const dA1 = pt.slope
      if (Math.abs(dA1) <= -c2 * d0 && extra(a1, pt)) {
        return p.finish(a1, phiA1, dA1, 'strong Wolfe conditions satisfied')
      }
      if (dA1 >= 0) {
        const z = zoom(a1, a0, phiA1, phiA0, dA1)
        return p.finish(
          z.a,
          z.v,
          z.d,
          z.a === undefined ? 'zoom did not converge' : 'strong Wolfe conditions satisfied',
        )
      }
      let a2 = 2 * a1
      if (amax !== undefined) a2 = Math.min(a2, amax)
      a0 = a1
      a1 = a2
      phiA0 = phiA1
      pt = p.at(a1)
      phiA1 = pt.value
      dA0 = dA1
    }
    // scipy returns the last bracketing step without a slope when the bracketing loop runs out.
    return { ...p.finish(a1, phiA1, Number.NaN, 'bracketing did not converge'), success: false }
  }
}

/**
 * Try one line search and fall back to another when it fails: scipy's BFGS and conjugate gradients run More-Thuente,
 * then the strong Wolfe zoom.
 *
 * @param primary - The first line search.
 * @param fallback - The line search to try when the first fails.
 * @returns The combined line search (its evaluation counts and trials include both).
 * @example
 * ```ts
 * import { moreThuente, strongWolfe, withFallback } from 'omai-engine/optim'
 * // phi(alpha) = (alpha - 0.25)^2: the first trial (alpha = 1) overshoots, interpolation finds the minimum
 * const phi = { at: (a: number) => ({ value: (a - 0.25) ** 2, slope: 2 * (a - 0.25) }) }
 * withFallback(moreThuente(), strongWolfe())(phi, { value: 0.0625, slope: -0.5 }).alpha // => 0.25
 * ```
 */
export function withFallback(primary: LineSearch, fallback: LineSearch): LineSearch {
  return (phi, start) => {
    const first = primary(phi, start)
    if (first.success) return first
    const second = fallback(phi, start)
    return {
      ...second,
      evaluations: first.evaluations + second.evaluations,
      gradEvaluations: first.gradEvaluations + second.gradEvaluations,
      trials: [...first.trials, ...second.trials],
      message: second.success ? `${second.message} (after: ${first.message})` : second.message,
    }
  }
}

/**
 * The line function of an objective along a direction: phi(alpha) = f(x + alpha p) with phi'(alpha) = grad f . p, and
 * the gradient carried along so the caller can reuse it.
 *
 * @param objective - A flat objective.
 * @param x - The point.
 * @param direction - The search direction.
 * @returns The line function.
 * @example
 * ```ts
 * import { flatObjective, lineFunction } from 'omai-engine/optim'
 * import { square, sum, type Value } from 'omai-engine/tensor'
 * const phi = lineFunction(flatObjective((x) => sum(square(x as Value)), [1, 1]), [1, 1], [-1, -1])
 * phi.at(0.5).value // => 0.5
 * ```
 */
export function lineFunction(
  objective: FlatObjective,
  x: ArrayLike<number>,
  direction: ArrayLike<number>,
): LineFunction {
  return {
    at: (alpha) => {
      const { value, grad } = objective.valueAndGrad(daxpy(alpha, direction, x))
      return { value, slope: ddot(grad, direction), grad }
    },
    valueAt: (alpha) => objective.value(daxpy(alpha, direction, x)),
  }
}

/** Options of {@link lineSearch}. */
export interface LineSearchOptions {
  /** The line search (default strong Wolfe, as scipy.optimize.line_search). */
  readonly method?: LineSearch
  /** f(x), if known. */
  readonly value?: number
  /** The gradient at x, if known: a vector (migration, D-009: was `ArrayLike<number>`). */
  readonly grad?: TensorLike
  /** The objective at the previous iterate, for the first-step heuristic. */
  readonly previousValue?: number
  /** The first trial step. */
  readonly alpha?: number
}

/** Result of {@link lineSearch}. */
export interface VectorLineSearchResult extends LineSearchResult {
  /** The new point x + alpha p (the start when the search failed). */
  readonly x: Float64Array
}

/**
 * Search along a direction for a step satisfying a line search's conditions (scipy.optimize.line_search).
 *
 * @param f - The objective.
 * @param x - The point (any parameters; flattened).
 * @param direction - The search direction (flattened like x): a tensor-like or a tree like `x` (migration, D-009: also
 *   accepted any `ArrayLike<number>`, which a tensor-like now covers).
 * @param options - The line search, known value and gradient, the previous value and the first step.
 * @returns The step, the new point and value, the gradient there, evaluation counts and the trial points.
 * @throws OmaiError with code `value` when the direction is not a descent direction.
 * @concept line-search
 * @example
 * ```ts
 * import { lineSearch } from 'omai-engine/optim'
 * import { square, sum, type Value } from 'omai-engine/tensor'
 * const r = lineSearch((x) => sum(square(x as Value)), [1, 2], [-1, -2])
 * [r.success, r.alpha, Array.from(r.x)] // => [true, 1, [0, 0]]
 * ```
 */
export function lineSearch(
  f: Objective,
  x: Parameters,
  direction: Parameters,
  options: LineSearchOptions = {},
): VectorLineSearchResult {
  const obj = flatObjective(f, x)
  const p = flatObjective(f, direction as Parameters).x0
  const start =
    options.value !== undefined && options.grad !== undefined
      ? { value: options.value, grad: Float64Array.from(toVector(options.grad, 'lineSearch')) }
      : obj.valueAndGrad(obj.x0)
  const slope = ddot(start.grad, p)
  if (!(slope < 0)) throw new OmaiError('value', `lineSearch: direction is not a descent direction (slope ${slope})`)
  const r = (options.method ?? strongWolfe())(lineFunction(obj, obj.x0, p), {
    value: start.value,
    slope,
    ...(options.alpha === undefined ? {} : { alpha: options.alpha }),
    ...(options.previousValue === undefined ? {} : { previousValue: options.previousValue }),
  })
  return { ...r, x: r.success ? daxpy(r.alpha, p, obj.x0) : Float64Array.from(obj.x0) }
}
