/**
 * Line searches as steppable algorithms: backtracking (Armijo) and Hager-Zhang, one trial step of the line function per
 * step, with the step, phi, the slope, the sufficient-decrease and curvature flags and the interval being searched on
 * the tape. A search is written as a procedure that asks for evaluations; the state is the list of trials made so far
 * (JSON), and a step feeds the trials back to the procedure and answers its next request, so a stored state resumes
 * anywhere. `backtracking` and `hagerZhang` are these run to the end.
 */

import type { Json } from '../base/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  numberFromJson,
  numberToJson,
  scalar,
  tensor,
} from '../trace/index.ts'
import { checkConstants, firstStep } from './_line-search.ts'
import type {
  BacktrackingOptions,
  LineFunction,
  LinePoint,
  LineSearchResult,
  LineStart,
  LineTrial,
} from './line-search.ts'
import type { HagerZhangOptions } from './line-search-interval.ts'

/** The paths the line-search algorithms log. */
export type LineSearchPath =
  | 'ls/alpha'
  | 'ls/phi'
  | 'ls/slope'
  | 'ls/armijo'
  | 'ls/curvature'
  | 'ls/bracket'
  | 'ls/interval'

/** What a stepped line search logs: the trial step and what the line function said there, and the tests it passed. */
export const lineSearchContract: Contract<unknown, LineSearchPath> = defineContract({
  'ls/alpha': scalar({ description: 'the trial step' }),
  'ls/phi': scalar({ description: 'the line function at the trial step', direction: 'lower' }),
  'ls/slope': scalar({ description: 'the slope of the line function there (absent for a value-only evaluation)' }),
  'ls/armijo': scalar({ description: '1 when the trial step gives sufficient decrease, else 0' }),
  'ls/curvature': scalar({ description: '1 when the slope there meets the curvature condition, else 0' }),
  'ls/bracket': scalar({ description: 'the width of the interval the trial was drawn from' }),
  'ls/interval': tensor({
    role: 'state',
    shape: [2],
    description: 'the ends of the interval the trial was drawn from',
  }),
})

/** What a line-search procedure asks for: the line function at a step, its value only or with the slope. */
export interface LineRequest {
  readonly alpha: number
  /** True when the value is enough (the evaluation may skip the slope). */
  readonly valueOnly: boolean
  /** The interval the trial is drawn from, for the tape. */
  readonly interval: readonly [number, number]
}

/** How a line search ended. */
export interface LineOutcome {
  /** The accepted step (NaN or absent when it failed). */
  readonly alpha?: number
  readonly value: number
  readonly slope: number
  readonly message: string
  /** Overrides the success of an outcome whose step is finite (a search that ran out of evaluations). */
  readonly success?: boolean
}

/** A line search written as a procedure: it asks for evaluations and is told what they were, then returns the outcome. */
export type LineProcedure = Generator<LineRequest, LineOutcome, LinePoint>

/**
 * The state of a stepped line search: every trial so far (a value-only trial has a `NaN` slope), the evaluation counts,
 * and the outcome once the search has finished. Steps push onto `trials` in place; keep a copy of a state you want to
 * look at later.
 */
export interface LineSearchState {
  readonly trials: LineTrial[]
  evaluations: number
  gradEvaluations: number
  outcome: LineOutcome | null
}

/** A stepped line search with the means to read its result as a {@link LineSearchResult}. */
export interface SteppedLineSearch {
  readonly algorithm: Algorithm<LineSearchState, LineSearchPath>
  /** The result of a finished state (the gradient is only known to a search run in this process). */
  readonly result: (state: LineSearchState) => LineSearchResult
}

interface Cursor {
  owner: LineSearchState | undefined
  gen: LineProcedure
  count: number
  next: IteratorResult<LineRequest, LineOutcome>
}

/**
 * Wrap a line-search procedure as an algorithm. The step makes the evaluation the procedure asked for and logs it.
 *
 * @param name - The algorithm's name.
 * @param config - Its configuration.
 * @param procedure - A fresh run of the procedure.
 * @param phi - The line function.
 * @param flags - The constants the logged tests use: sufficient decrease at `phi0` with slope `d0` and constant `c1`,
 * curvature with `c2`.
 * @returns The algorithm and its result reader.
 * @example
 * ```ts
 * import { type LineProcedure, steppedLineSearch } from 'omai-engine/optim'
 * import { runAlgorithm } from 'omai-engine/trace'
 * // A procedure that accepts the unit step after one evaluation.
 * function* procedure(): LineProcedure {
 *   const point = yield { alpha: 1, valueOnly: false, interval: [0, 1] }
 *   return { alpha: 1, value: point.value, slope: point.slope, message: 'accepted' }
 * }
 * const phi = { at: (a: number) => ({ value: (a - 0.5) ** 2, slope: 2 * (a - 0.5) }) }
 * const search = steppedLineSearch('unit', {}, procedure, phi, { phi0: 0.25, d0: -1, c1: 1e-4, c2: 0.9 })
 * const run = runAlgorithm(search.algorithm, { maxSteps: 5 })
 * [run.status, search.result(run.state).alpha] // => ['converged', 1]
 * ```
 */
export function steppedLineSearch(
  name: string,
  config: Json,
  procedure: () => LineProcedure,
  phi: LineFunction,
  flags: { readonly phi0: number; readonly d0: number; readonly c1: number; readonly c2: number },
): SteppedLineSearch {
  const last = new Map<number, LinePoint>()
  let cursor: Cursor | undefined
  const fresh = (): Cursor => {
    const gen = procedure()
    return { owner: undefined, gen, count: 0, next: gen.next() }
  }
  // Position the cursor on the procedure's next request after the trials of `s` (replaying them when it is elsewhere).
  const settle = (s: LineSearchState): IteratorResult<LineRequest, LineOutcome> => {
    if (cursor === undefined || cursor.owner !== s || cursor.count !== s.trials.length) {
      cursor = fresh()
      cursor.owner = s
      for (const t of s.trials) {
        if (cursor.next.done === true) break
        cursor.next = cursor.gen.next({ value: t.value, slope: t.slope })
        cursor.count += 1
      }
    }
    if (cursor.next.done === true && s.outcome === null) s.outcome = cursor.next.value
    return cursor.next
  }
  const algorithm: Algorithm<LineSearchState, LineSearchPath> = {
    name,
    config,
    init: () => {
      const s: LineSearchState = { trials: [], evaluations: 0, gradEvaluations: 0, outcome: null }
      settle(s)
      return s
    },
    step: (s, ctx) => {
      const next = settle(s)
      if (next.done === true) return s
      const req = next.value
      let point: LinePoint
      if (req.valueOnly && phi.valueAt !== undefined) {
        point = { value: phi.valueAt(req.alpha), slope: Number.NaN }
        s.evaluations += 1
      } else {
        point = phi.at(req.alpha)
        last.clear()
        last.set(req.alpha, point)
        s.evaluations += 1
        s.gradEvaluations += 1
      }
      s.trials.push({ alpha: req.alpha, value: point.value, slope: point.slope })
      if (cursor !== undefined && cursor.owner === s && cursor.count === s.trials.length - 1) {
        cursor.next = cursor.gen.next({ value: point.value, slope: point.slope })
        cursor.count += 1
      }
      ctx.log('ls/alpha', req.alpha)
      ctx.log('ls/phi', point.value)
      ctx.log('ls/armijo', point.value <= flags.phi0 + flags.c1 * req.alpha * flags.d0 ? 1 : 0)
      if (!Number.isNaN(point.slope)) {
        ctx.log('ls/slope', point.slope)
        ctx.log('ls/curvature', Math.abs(point.slope) <= flags.c2 * Math.abs(flags.d0) ? 1 : 0)
      }
      ctx.log('ls/bracket', Math.abs(req.interval[1] - req.interval[0]))
      if (ctx.wants('ls/interval')) ctx.log('ls/interval', [req.interval[0], req.interval[1]])
      settle(s)
      return s
    },
    done: (s) =>
      s.outcome === null ? undefined : s.outcome.success === false || !isOk(s.outcome) ? 'failed' : 'converged',
    serialise: (s) => ({
      trials: s.trials.map((t) => [numberToJson(t.alpha), numberToJson(t.value), numberToJson(t.slope)]),
      evaluations: s.evaluations,
      gradEvaluations: s.gradEvaluations,
      outcome:
        s.outcome === null
          ? null
          : {
              alpha: numberToJson(s.outcome.alpha ?? Number.NaN),
              value: numberToJson(s.outcome.value),
              slope: numberToJson(s.outcome.slope),
              message: s.outcome.message,
              ...(s.outcome.success === undefined ? {} : { success: s.outcome.success }),
            },
    }),
    deserialise: (j) => {
      const o = j as {
        trials: Json[][]
        evaluations: number
        gradEvaluations: number
        outcome: { alpha: Json; value: Json; slope: Json; message: string; success?: boolean } | null
      }
      return {
        trials: o.trials.map((t) => ({
          alpha: numberFromJson(t[0]),
          value: numberFromJson(t[1]),
          slope: numberFromJson(t[2]),
        })),
        evaluations: o.evaluations,
        gradEvaluations: o.gradEvaluations,
        outcome:
          o.outcome === null
            ? null
            : {
                alpha: numberFromJson(o.outcome.alpha),
                value: numberFromJson(o.outcome.value),
                slope: numberFromJson(o.outcome.slope),
                message: o.outcome.message,
                ...(o.outcome.success === undefined ? {} : { success: o.outcome.success }),
              },
      }
    },
  }
  const result = (s: LineSearchState): LineSearchResult => {
    const o = s.outcome ?? { value: Number.NaN, slope: Number.NaN, message: 'unfinished' }
    const ok = o.alpha !== undefined && Number.isFinite(o.alpha)
    const grad = ok ? last.get(o.alpha as number)?.grad : undefined
    return {
      alpha: ok ? (o.alpha as number) : Number.NaN,
      value: o.value,
      slope: o.slope,
      ...(grad === undefined ? {} : { grad }),
      success: o.success ?? ok,
      evaluations: s.evaluations,
      gradEvaluations: s.gradEvaluations,
      trials: s.trials,
      message: o.message,
    }
  }
  return { algorithm, result }
}

const isOk = (o: LineOutcome): boolean => o.alpha !== undefined && Number.isFinite(o.alpha)

const fail = (value: number, slope: number, message: string): LineOutcome => ({ value, slope, message })

/** The backtracking procedure: each `yield` is one value evaluation. */
function* backtrackingProcedure(start: LineStart, options: BacktrackingOptions): LineProcedure {
  const c1 = options.c1 ?? 1e-4
  const amin = options.alphaMin ?? 0
  const phi0 = start.value
  const d0 = start.slope
  if (!(d0 < 0)) return fail(phi0, Number.NaN, 'not a descent direction')
  const ok = (a: number, v: number): boolean => v <= phi0 + c1 * a * d0
  const accept = (a: number, pt: LinePoint): LineOutcome => ({
    alpha: a,
    value: pt.value,
    slope: pt.slope,
    message: 'Armijo condition satisfied',
  })
  if (options.interpolation === 'halving') {
    const rho = options.rho ?? 0.5
    let a = start.alpha ?? 1
    for (let i = 0; i < (options.maxEvaluations ?? 50) && a > amin; i++) {
      const pt = yield { alpha: a, valueOnly: true, interval: [0, a] }
      if (ok(a, pt.value)) return accept(a, pt)
      a *= rho
    }
    return fail(phi0, Number.NaN, 'step became too small')
  }
  let a0 = start.alpha ?? 1
  const p0 = yield { alpha: a0, valueOnly: true, interval: [0, a0] }
  let f0 = p0.value
  if (ok(a0, f0)) return accept(a0, p0)
  const guard = options.safeguard
  const clip = (a: number, prev: number): number =>
    guard === undefined || !Number.isFinite(a)
      ? guard === undefined
        ? a
        : guard[1] * prev
      : Math.min(Math.max(a, guard[0] * prev), guard[1] * prev)
  let a1 = clip((-d0 * a0 * a0) / 2 / (f0 - phi0 - d0 * a0), a0)
  const p1 = yield { alpha: a1, valueOnly: true, interval: [0, a0] }
  let f1 = p1.value
  if (ok(a1, f1)) return accept(a1, p1)
  while (a1 > amin) {
    const factor = a0 * a0 * a1 * a1 * (a1 - a0)
    const a = (a0 * a0 * (f1 - phi0 - d0 * a1) - a1 * a1 * (f0 - phi0 - d0 * a0)) / factor
    const b = (-(a0 ** 3) * (f1 - phi0 - d0 * a1) + a1 ** 3 * (f0 - phi0 - d0 * a0)) / factor
    let a2 = clip((-b + Math.sqrt(Math.abs(b * b - 3 * a * d0))) / (3 * a), a1)
    const p2 = yield { alpha: a2, valueOnly: true, interval: [0, a1] }
    const f2 = p2.value
    if (ok(a2, f2)) return accept(a2, p2)
    if (guard === undefined && (a1 - a2 > a1 / 2 || 1 - a2 / a1 < 0.96)) a2 = a1 / 2
    a0 = a1
    a1 = a2
    f0 = f1
    f1 = f2
    if (!Number.isFinite(a1)) break
  }
  return fail(f1, Number.NaN, 'step became too small')
}

/**
 * Backtracking (Armijo) line search as a stepped algorithm: each step evaluates the line function at one trial step
 * (the first, then quadratic and cubic interpolations of the rejected values as scipy's `scalar_search_armijo`, or
 * successive contractions by `rho` with `interpolation: 'halving'`) and logs the step, phi there, and whether
 * phi(alpha) <= phi(0) + c1 alpha phi'(0). It is done when a step is accepted (`converged`) or the step becomes too
 * small (`failed`). The state, the list of trials, serialises and resumes.
 *
 * @param phi - The line function.
 * @param start - phi(0), phi'(0) < 0 and the first step.
 * @param options - c1, smallest step, interpolation, contraction, evaluation cap and safeguard.
 * @returns The algorithm.
 * @concept line-search
 * @example
 * ```ts
 * import { backtrackingAlgorithm } from 'omai-engine/optim'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const phi = { at: (a: number) => ({ value: (a - 0.3) ** 2, slope: 2 * (a - 0.3) }) }
 * const run = runAlgorithm(backtrackingAlgorithm(phi, { value: 0.09, slope: -0.6 }, { interpolation: 'halving' }), {
 *   maxSteps: 60,
 * })
 * [run.status, run.state.trials.map((t) => t.alpha)] // => ['converged', [1, 0.5]]
 * ```
 */
export function backtrackingAlgorithm(
  phi: LineFunction,
  start: LineStart,
  options: BacktrackingOptions = {},
): Algorithm<LineSearchState, LineSearchPath> {
  return backtrackingSearch(phi, start, options).algorithm
}

/**
 * {@link backtrackingAlgorithm} with the reader of its result (what `backtracking` runs).
 *
 * @param phi - The line function.
 * @param start - phi(0), phi'(0) < 0 and the first step.
 * @param options - c1, smallest step, interpolation, contraction, evaluation cap and safeguard.
 * @returns The algorithm and the reader of a finished state as a line-search result.
 * @example
 * ```ts
 * import { backtrackingSearch } from 'omai-engine/optim'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const phi = { at: (a: number) => ({ value: (a - 0.3) ** 2, slope: 2 * (a - 0.3) }) }
 * const search = backtrackingSearch(phi, { value: 0.09, slope: -0.6 }, { interpolation: 'halving' })
 * const run = runAlgorithm(search.algorithm, { maxSteps: 60 })
 * search.result(run.state).alpha // => 0.5
 * ```
 */
export function backtrackingSearch(
  phi: LineFunction,
  start: LineStart,
  options: BacktrackingOptions = {},
): SteppedLineSearch {
  return steppedLineSearch(
    'optim/backtracking',
    { c1: options.c1 ?? 1e-4, interpolation: options.interpolation ?? 'cubic', rho: options.rho ?? 0.5 },
    () => backtrackingProcedure(start, options),
    phi,
    { phi0: start.value, d0: start.slope, c1: options.c1 ?? 1e-4, c2: 0.9 },
  )
}

/** The Hager-Zhang procedure: each `yield` is one evaluation of value and slope. */
function* hagerZhangProcedure(start: LineStart, options: HagerZhangOptions): LineProcedure {
  const delta = options.delta ?? 0.1
  const sigma = options.sigma ?? 0.9
  const epsilon = options.epsilon ?? 1e-6
  const gamma = options.gamma ?? 0.66
  const rho = options.rho ?? 5
  const maxEval = options.maxEvaluations ?? 50
  const phi0 = start.value
  const d0 = start.slope
  if (!(d0 < 0)) return fail(phi0, Number.NaN, 'not a descent direction')
  let alphaMax = options.alphaMax ?? Number.POSITIVE_INFINITY
  const phiLim = phi0 + epsilon * Math.abs(phi0)
  const alphas: number[] = [0]
  const values: number[] = [phi0]
  const slopes: number[] = [d0]
  let evals = 0
  const wolfe = (c: number, v: number, d: number): boolean =>
    (delta * d0 >= (v - phi0) / c && d >= sigma * d0) || ((2 * delta - 1) * d0 >= d && d >= sigma * d0 && v <= phiLim)
  const evaluate = function* (c: number, lo: number, hi: number): Generator<LineRequest, number, LinePoint> {
    const pt = yield { alpha: c, valueOnly: false, interval: [lo, hi] }
    evals += 1
    alphas.push(c)
    values.push(pt.value)
    slopes.push(pt.slope)
    return alphas.length - 1
  }
  const done = (i: number, message: string): LineOutcome => ({
    alpha: alphas[i] ?? Number.NaN,
    value: values[i] ?? Number.NaN,
    slope: slopes[i] ?? Number.NaN,
    message,
  })
  const at = (i: number): number => alphas[i] ?? Number.NaN
  const finite = (i: number): boolean => Number.isFinite(values[i]) && Number.isFinite(slopes[i])
  const bisect = function* (ia0: number, ib0: number): Generator<LineRequest, [number, number], LinePoint> {
    let ia = ia0
    let ib = ib0
    while (at(ib) - at(ia) > Number.EPSILON * at(ib) && evals < maxEval) {
      const id = yield* evaluate((at(ia) + at(ib)) / 2, at(ia), at(ib))
      if ((slopes[id] ?? 0) >= 0) return [ia, id]
      if ((values[id] ?? Number.POSITIVE_INFINITY) <= phiLim) ia = id
      else ib = id
    }
    return [ia, ib]
  }
  const update = function* (ia: number, ib: number, ic: number): Generator<LineRequest, [number, number], LinePoint> {
    const c = at(ic)
    if (c < at(ia) || c > at(ib)) return [ia, ib]
    if ((slopes[ic] ?? 0) >= 0) return [ia, ic]
    if ((values[ic] ?? Number.POSITIVE_INFINITY) <= phiLim) return [ic, ib]
    return yield* bisect(ia, ic)
  }
  const secant = (a: number, b: number, da: number, db: number): number => (a * db - b * da) / (db - da)
  // Initial step, shrunk until finite.
  let c = Math.min(firstStep(start), alphaMax)
  let ic = yield* evaluate(c, 0, c)
  for (let k = 0; k < 20 && !finite(ic); k++) {
    c *= 0.1
    ic = yield* evaluate(c, 0, c)
  }
  if (!finite(ic)) return fail(phi0, d0, 'no finite step')
  // Bracketing (HZ stages B0-B3).
  let ia = 0
  let ib = ic
  let bracketed = false
  while (!bracketed && evals < maxEval) {
    const lastIndex = alphas.length - 1
    if ((slopes[lastIndex] ?? 0) >= 0) {
      ib = lastIndex
      ia = 0
      for (let i = lastIndex - 1; i >= 0; i--) {
        if ((values[i] ?? Number.POSITIVE_INFINITY) <= phiLim) {
          ia = i
          break
        }
      }
      bracketed = true
    } else if ((values[lastIndex] ?? Number.POSITIVE_INFINITY) > phiLim) {
      ;[ia, ib] = yield* bisect(0, lastIndex)
      bracketed = true
    } else {
      const cold = c
      c = Math.min(c * rho, alphaMax)
      if (c === cold) return done(lastIndex, 'step at alphaMax')
      let k = yield* evaluate(c, 0, c)
      for (let j = 0; j < 20 && !finite(k) && c > cold; j++) {
        alphaMax = c
        c = (cold + c) / 2
        k = yield* evaluate(c, 0, c)
      }
      if (!finite(k)) return done(lastIndex, 'no finite larger step')
    }
  }
  // Secant^2 with bisection safeguards.
  while (evals < maxEval) {
    const a = at(ia)
    const b = at(ib)
    if (b - a <= Number.EPSILON * b) return done(ia, 'bracket collapsed')
    // secant2
    let iA: number
    let iB: number
    {
      const cs = secant(a, b, slopes[ia] ?? 0, slopes[ib] ?? 0)
      const icx = yield* evaluate(cs, a, b)
      if (wolfe(cs, values[icx] ?? Number.POSITIVE_INFINITY, slopes[icx] ?? Number.NaN))
        return done(icx, 'Wolfe conditions satisfied')
      ;[iA, iB] = yield* update(ia, ib, icx)
      let c2: number | undefined
      if (iB === icx) c2 = secant(b, at(iB), slopes[ib] ?? 0, slopes[iB] ?? 0)
      else if (iA === icx) c2 = secant(a, at(iA), slopes[ia] ?? 0, slopes[iA] ?? 0)
      if (c2 !== undefined && (iA === icx || iB === icx) && at(iA) <= c2 && c2 <= at(iB)) {
        const id = yield* evaluate(c2, at(iA), at(iB))
        if (wolfe(c2, values[id] ?? Number.POSITIVE_INFINITY, slopes[id] ?? Number.NaN))
          return done(id, 'Wolfe conditions satisfied')
        ;[iA, iB] = yield* update(iA, iB, id)
      }
    }
    if (at(iB) - at(iA) < gamma * (b - a)) {
      // So flat that function differences are rounding noise: the secant steps cannot do better.
      const flat = (i: number, j: number): boolean => {
        const u = values[i] ?? 0
        const v = values[j] ?? 0
        return Math.abs(u - v) <= 4 * Number.EPSILON * Math.max(Math.abs(u), Math.abs(v))
      }
      if (flat(ia, ib) && flat(iA, iB)) return done(iA, 'flat: secant made no progress')
      ia = iA
      ib = iB
    } else {
      const id = yield* evaluate((at(iA) + at(iB)) / 2, at(iA), at(iB))
      ;[ia, ib] = yield* update(iA, iB, id)
    }
  }
  return { ...done(ia, 'evaluation cap reached'), success: false }
}

/**
 * The Hager-Zhang line search (CG_DESCENT) as a stepped algorithm: each step is one evaluation of the line function,
 * made while bracketing a step that satisfies the (approximate) Wolfe conditions, by a secant step of the secant^2
 * pair, or by the bisection that safeguards them. It logs the step, phi and its slope there, the sufficient-decrease
 * and curvature flags, and the interval the trial was drawn from. It is done when a step is accepted (`converged`) or
 * the search gives up (`failed`). The state, the list of trials, serialises and resumes.
 *
 * @param phi - The line function.
 * @param start - phi(0), phi'(0) < 0 and the first step.
 * @param options - delta, sigma, epsilon, gamma, rho, the largest step and the evaluation cap.
 * @returns The algorithm.
 * @throws OmaiError with code `value` unless 0 < delta < sigma < 1.
 * @concept line-search
 * @reference W. W. Hager and H. Zhang, A new conjugate gradient method with guaranteed descent and an efficient line
 * search, SIAM J. Optim. 16 (2005); Algorithm 851: CG_DESCENT, ACM TOMS 32 (2006).
 * @example
 * ```ts
 * import { hagerZhangAlgorithm } from 'omai-engine/optim'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const phi = { at: (a: number) => ({ value: (a - 0.25) ** 2, slope: 2 * (a - 0.25) }) }
 * const run = runAlgorithm(hagerZhangAlgorithm(phi, { value: 0.0625, slope: -0.5 }), { maxSteps: 60 })
 * [run.status, run.state.trials.length, run.state.outcome?.alpha] // => ['converged', 2, 0.25]
 * ```
 */
export function hagerZhangAlgorithm(
  phi: LineFunction,
  start: LineStart,
  options: HagerZhangOptions = {},
): Algorithm<LineSearchState, LineSearchPath> {
  return hagerZhangSearch(phi, start, options).algorithm
}

/**
 * {@link hagerZhangAlgorithm} with the reader of its result (what `hagerZhang` runs).
 *
 * @param phi - The line function.
 * @param start - phi(0), phi'(0) < 0 and the first step.
 * @param options - delta, sigma, epsilon, gamma, rho, the largest step and the evaluation cap.
 * @returns The algorithm and the reader of a finished state as a line-search result.
 * @throws OmaiError with code `value` unless 0 < delta < sigma < 1.
 * @example
 * ```ts
 * import { hagerZhangSearch } from 'omai-engine/optim'
 * import { runAlgorithm } from 'omai-engine/trace'
 * const phi = { at: (a: number) => ({ value: (a - 0.25) ** 2, slope: 2 * (a - 0.25) }) }
 * const search = hagerZhangSearch(phi, { value: 0.0625, slope: -0.5 })
 * const run = runAlgorithm(search.algorithm, { maxSteps: 60 })
 * search.result(run.state).alpha // => 0.25
 * ```
 */
export function hagerZhangSearch(
  phi: LineFunction,
  start: LineStart,
  options: HagerZhangOptions = {},
): SteppedLineSearch {
  const delta = options.delta ?? 0.1
  const sigma = options.sigma ?? 0.9
  checkConstants(delta, sigma)
  return steppedLineSearch(
    'optim/hager-zhang',
    { delta, sigma, epsilon: options.epsilon ?? 1e-6, gamma: options.gamma ?? 0.66, rho: options.rho ?? 5 },
    () => hagerZhangProcedure(start, options),
    phi,
    { phi0: start.value, d0: start.slope, c1: delta, c2: sigma },
  )
}
