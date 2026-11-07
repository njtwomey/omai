/**
 * The scalar root finders as steppable algorithms: bisection, Brent's method, Ridders' method, TOMS 748, Newton (with
 * Halley's correction) and the secant method. One step is one iteration, the state is plain numbers and serialises, and
 * all of them share one state (`RootState`: the estimate, the bracket when there is one, the last step and the rule that
 * made it, the evaluations so far) and one set of log paths, so a figure of "the iterate and the bracket over the
 * iterations" works for every method. `scalar.ts` runs them to the end and returns the root with its history;
 * `root-options.ts` is the table that selects one by name.
 */

import { MdmlError } from '../../base/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  message,
  type RunStatus,
  runAlgorithm,
  runStatusOf,
  type StepContext,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'

const EPS = Number.EPSILON

/** One evaluation of a root finder, for convergence figures. */
export interface RootStep {
  readonly x: number
  readonly fx: number
}

/** Result of a scalar root finder (scipy's RootResults plus the evaluation history). */
export interface RootResult {
  readonly root: number
  readonly iterations: number
  readonly functionCalls: number
  readonly converged: boolean
  /** How the finder ended in the one vocabulary: `flag` mapped by {@link runStatusOf}. */
  readonly status: RunStatus
  readonly flag: 'converged' | 'sign error' | 'maximum iterations' | 'derivative was zero'
  readonly history: readonly RootStep[]
}

/** Tolerances of the bracketing methods (scipy's defaults: xtol 2e-12, rtol 4 eps, maxiter 100). */
export interface BracketOptions {
  readonly xtol?: number
  readonly rtol?: number
  readonly maxIterations?: number
}

/** Options of {@link newton}. */
export interface NewtonOptions {
  /** The derivative; without it, the secant method is used (as scipy). */
  readonly fprime?: (x: number) => number
  /** The second derivative; with fprime, Halley's method is used. */
  readonly fprime2?: (x: number) => number
  /** A second starting point for the secant method. */
  readonly x1?: number
  /** Stop when the step is below tol + rtol |x| (scipy's defaults: tol 1.48e-8, rtol 0). */
  readonly tol?: number
  readonly rtol?: number
  readonly maxIterations?: number
}

/** Why a root finder stopped, when it did: the flags of {@link RootResult} it can reach by stepping. */
export type RootFlag = RootResult['flag']

/**
 * The state of the scalar root finders. `x` is the estimate the method would return if it stopped now (the latest
 * midpoint, interpolant or Newton iterate), `lo` and `hi` the bracket for the bracketing methods (`null` for the open
 * ones), `step` how far the estimate moved in the last step, `rule` what made the step (`bisection`, `interpolation`,
 * `newton`, ...), `flag` why it finished (`null` while it runs), `history` every evaluation of `f` so far, and `internal`
 * the numbers the method itself carries from step to step.
 */
export interface RootState {
  readonly x: number
  /** f at the last point evaluated (`null` before the first evaluation). */
  readonly fx: number | null
  readonly lo: number | null
  readonly hi: number | null
  readonly step: number
  readonly rule: string
  readonly iteration: number
  readonly flag: RootFlag | null
  readonly history: readonly RootStep[]
  readonly internal: Readonly<Record<string, number>>
}

/** The paths the scalar root finders log, as a type. */
export type RootPath = 'root/x' | 'root/fx' | 'root/step' | 'root/bracket' | 'root/bracket-width' | 'root/method'

/**
 * What the scalar root finders log, by path: the estimate, f at the last point evaluated, how far the estimate moved,
 * the bracket and its width (bracketing methods only), and the rule that made the step.
 */
export const rootContract: Contract<RootState, RootPath> = defineContract({
  'root/x': scalar({ description: 'the current estimate of the root' }),
  'root/fx': scalar({ description: 'f at the last point evaluated', direction: 'lower' }),
  'root/step': scalar({ description: 'how far the estimate moved in this step' }),
  'root/bracket': tensor({ role: 'state', shape: [2], description: 'the bracket [lo, hi] (bracketing methods)' }),
  'root/bracket-width': scalar({ description: 'the width of the bracket (bracketing methods)', direction: 'lower' }),
  'root/method': message({
    description: 'the rule that made this step: bisection, interpolation, newton, secant, ...',
  }),
})

/** True when a and b are both non-zero with the same sign (compared by sign: a product of two small values underflows). */
const sameSign = (a: number, b: number): boolean => a !== 0 && b !== 0 && Math.sign(a) === Math.sign(b)

function checkBracket(fa: number, fb: number, name: string): void {
  if (sameSign(fa, fb)) throw new MdmlError('value', `${name}: f(a) and f(b) must have different signs`)
}

/**
 * Evaluate f, appending to a copy of the history. With a `name` (the bracketing methods), a NaN value throws: no sign
 * test can decide on it, and carrying on would report a root of a function that has none.
 */
function evaluator(
  f: (x: number) => number,
  past: readonly RootStep[],
  name?: string,
): { call: (x: number) => number; history: RootStep[] } {
  const history = [...past]
  return {
    history,
    call: (x) => {
      const fx = f(x)
      if (name !== undefined && Number.isNaN(fx)) throw new MdmlError('value', `${name}: f(${x}) is NaN`)
      history.push({ x, fx })
      return fx
    },
  }
}

interface Parts {
  readonly x: number
  readonly fx: number | null
  readonly lo?: number
  readonly hi?: number
  readonly rule: string
  readonly flag?: RootFlag
  readonly internal: Readonly<Record<string, number>>
}

const next = (s: RootState, history: readonly RootStep[], p: Parts): RootState => ({
  x: p.x,
  fx: p.fx,
  lo: p.lo ?? null,
  hi: p.hi ?? null,
  step: Math.abs(p.x - s.x),
  rule: p.rule,
  iteration: s.iteration + 1,
  flag: p.flag ?? null,
  history,
  internal: p.internal,
})

const first = (history: readonly RootStep[], p: Parts): RootState => ({
  x: p.x,
  fx: p.fx,
  lo: p.lo ?? null,
  hi: p.hi ?? null,
  step: 0,
  rule: p.rule,
  iteration: 0,
  flag: p.flag ?? null,
  history,
  internal: p.internal,
})

function logState(ctx: StepContext<RootPath>, s: RootState): void {
  if (ctx.wants('root/x')) ctx.log('root/x', s.x)
  if (ctx.wants('root/fx') && s.fx !== null) ctx.log('root/fx', s.fx)
  if (ctx.wants('root/step')) ctx.log('root/step', s.step)
  if (s.lo !== null && s.hi !== null) {
    if (ctx.wants('root/bracket')) ctx.log('root/bracket', Float64Array.of(s.lo, s.hi))
    if (ctx.wants('root/bracket-width')) ctx.log('root/bracket-width', Math.abs(s.hi - s.lo))
  }
  if (ctx.wants('root/method')) ctx.log('root/method', { rule: s.rule })
}

function assemble(
  name: string,
  config: Record<string, number | string | boolean>,
  maxIterations: number,
  init: () => RootState,
  advance: (s: RootState) => RootState,
): Algorithm<RootState, RootPath> {
  return {
    name,
    config,
    init,
    step: (s, ctx) => {
      const n = advance(s)
      logState(ctx, n)
      return n
    },
    done: (s): RunStatus | undefined =>
      s.flag === 'converged'
        ? 'converged'
        : s.flag !== null
          ? 'failed'
          : s.iteration >= maxIterations
            ? 'completed'
            : undefined,
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as RootState,
  }
}

/**
 * The result of a root finder's state, as the functions of `scalar.ts` return it: the root, the iterations done, the
 * evaluations, and why it stopped (`maximum iterations` when it did not finish by itself).
 *
 * @param s - The state.
 * @returns The result.
 * @example
 * ```ts
 * import { bisectAlgorithm, rootResultOf } from 'mdml-engine/numerics/roots'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const s = runAlgorithm(bisectAlgorithm((x) => x - 1, 0, 4), { maxSteps: 100 }).state
 * rootResultOf(s).root // => 1
 * ```
 */
export function rootResultOf(s: RootState): RootResult {
  const flag = s.flag ?? 'maximum iterations'
  return {
    root: s.x,
    iterations: s.iteration,
    functionCalls: s.history.length,
    converged: flag === 'converged',
    status: runStatusOf(flag),
    flag,
    history: s.history,
  }
}

/**
 * Run a root finder's algorithm to its end (at most `maxIterations` steps) and return the result.
 *
 * @param algorithm - The algorithm of one of the finders.
 * @param maxIterations - The step cap.
 * @returns The root with its history.
 * @example
 * ```ts
 * import { bisectAlgorithm, solveRoot } from 'mdml-engine/numerics/roots'
 * solveRoot(bisectAlgorithm((x) => x - 1, 0, 4), 100).converged // => true
 * ```
 */
export function solveRoot(algorithm: Algorithm<RootState, RootPath>, maxIterations: number): RootResult {
  return rootResultOf(runAlgorithm(algorithm, { maxSteps: maxIterations }).state)
}

/**
 * Bisection (scipy.optimize.bisect) as a steppable algorithm: each step halves the bracket and keeps the half with the
 * sign change, until it is shorter than xtol + rtol |x|. The evaluations at the two ends happen in `init`; a root at an
 * end finishes at once.
 *
 * @param f - The function.
 * @param a - One end of the bracket.
 * @param b - The other end (f(a) and f(b) of opposite signs).
 * @param options - Tolerances and iteration cap.
 * @returns The algorithm; its state is a {@link RootState}.
 * @throws MdmlError with code `value` when f(a) and f(b) have the same sign, or f returns NaN.
 * @concept root-finding
 * @example
 * ```ts
 * import { bisectAlgorithm } from 'mdml-engine/numerics/roots'
 * import { iterate } from 'mdml-engine/trace'
 * Array.from(iterate(bisectAlgorithm((x) => x - 1, 0, 4), { maxSteps: 3 }), (i) => i.state.x) // => [4, 2, 1]
 * ```
 */
export function bisectAlgorithm(
  f: (x: number) => number,
  a: number,
  b: number,
  options: BracketOptions = {},
): Algorithm<RootState, RootPath> {
  const xtol = options.xtol ?? 2e-12
  const rtol = options.rtol ?? 4 * EPS
  const max = options.maxIterations ?? 100
  return assemble(
    'bisect',
    { xtol, rtol, maxIterations: max },
    max,
    () => {
      const { call, history } = evaluator(f, [], 'bisect')
      const fa = call(a)
      const fb = call(b)
      checkBracket(fa, fb, 'bisect')
      if (fa === 0) return first(history, { x: a, fx: fa, rule: 'end point', flag: 'converged', internal: {} })
      if (fb === 0) return first(history, { x: b, fx: fb, rule: 'end point', flag: 'converged', internal: {} })
      return first(history, { x: a + (b - a), fx: fb, lo: a, hi: b, rule: 'start', internal: { lo: a, dm: b - a, fa } })
    },
    (s) => {
      const { call, history } = evaluator(f, s.history, 'bisect')
      let { lo, dm, fa } = s.internal as Record<'lo' | 'dm' | 'fa', number>
      dm *= 0.5
      const xm = lo + dm
      const fm = call(xm)
      if (fm === 0 || Math.sign(fm) === Math.sign(fa)) {
        lo = xm
        fa = fm
      }
      const converged = fm === 0 || Math.abs(dm) < xtol + rtol * Math.abs(xm)
      return next(s, history, {
        x: converged ? xm : lo + dm,
        fx: fm,
        lo,
        hi: lo + dm,
        rule: 'bisection',
        ...(converged ? { flag: 'converged' as const } : {}),
        internal: { lo, dm, fa },
      })
    },
  )
}

/**
 * Brent's method (scipy.optimize.brentq) as a steppable algorithm: inverse quadratic interpolation and secant steps,
 * falling back to bisection when the interpolant is not trusted. One step checks the stopping rule at the current
 * iterate and, if it does not hold, evaluates the next one; the `rule` says which kind of step it took.
 *
 * @param f - The function.
 * @param a - One end of the bracket.
 * @param b - The other end.
 * @param options - Tolerances and iteration cap.
 * @returns The algorithm; its state is a {@link RootState}.
 * @throws MdmlError with code `value` when f(a) and f(b) have the same sign, or f returns NaN.
 * @concept root-finding
 * @example
 * ```ts
 * import { brentqAlgorithm } from 'mdml-engine/numerics/roots'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * runAlgorithm(brentqAlgorithm(Math.cos, 0, 2), { maxSteps: 100 }).state.flag // => 'converged'
 * ```
 */
export function brentqAlgorithm(
  f: (x: number) => number,
  a: number,
  b: number,
  options: BracketOptions = {},
): Algorithm<RootState, RootPath> {
  const xtol = options.xtol ?? 2e-12
  const rtol = options.rtol ?? 4 * EPS
  const max = options.maxIterations ?? 100
  return assemble(
    'brentq',
    { xtol, rtol, maxIterations: max },
    max,
    () => {
      const { call, history } = evaluator(f, [], 'brentq')
      const fpre = call(a)
      const fcur = call(b)
      checkBracket(fpre, fcur, 'brentq')
      if (fpre === 0) return first(history, { x: a, fx: fpre, rule: 'end point', flag: 'converged', internal: {} })
      if (fcur === 0) return first(history, { x: b, fx: fcur, rule: 'end point', flag: 'converged', internal: {} })
      return first(history, {
        x: b,
        fx: fcur,
        lo: Math.min(a, b),
        hi: Math.max(a, b),
        rule: 'start',
        internal: { xpre: a, xcur: b, fpre, fcur, xblk: 0, fblk: 0, spre: 0, scur: 0 },
      })
    },
    (s) => {
      const { call, history } = evaluator(f, s.history, 'brentq')
      let { xpre, xcur, fpre, fcur, xblk, fblk, spre, scur } = s.internal as Record<
        'xpre' | 'xcur' | 'fpre' | 'fcur' | 'xblk' | 'fblk' | 'spre' | 'scur',
        number
      >
      if (fpre !== 0 && fcur !== 0 && Math.sign(fpre) !== Math.sign(fcur)) {
        xblk = xpre
        fblk = fpre
        spre = scur = xcur - xpre
      }
      if (Math.abs(fblk) < Math.abs(fcur)) {
        xpre = xcur
        xcur = xblk
        xblk = xpre
        fpre = fcur
        fcur = fblk
        fblk = fpre
      }
      const delta = (xtol + rtol * Math.abs(xcur)) / 2
      const sbis = (xblk - xcur) / 2
      const bracket = { lo: Math.min(xcur, xblk), hi: Math.max(xcur, xblk) }
      if (fcur === 0 || Math.abs(sbis) < delta)
        return next(s, history, {
          x: xcur,
          fx: fcur,
          ...bracket,
          rule: 'converged',
          flag: 'converged',
          internal: { xpre, xcur, fpre, fcur, xblk, fblk, spre, scur },
        })
      let rule = 'bisection'
      if (Math.abs(spre) > delta && Math.abs(fcur) < Math.abs(fpre)) {
        let stry: number
        if (xpre === xblk) {
          rule = 'secant'
          stry = (-fcur * (xcur - xpre)) / (fcur - fpre)
        } else {
          rule = 'inverse quadratic interpolation'
          const dpre = (fpre - fcur) / (xpre - xcur)
          const dblk = (fblk - fcur) / (xblk - xcur)
          stry = (-fcur * (fblk * dblk - fpre * dpre)) / (dblk * dpre * (fblk - fpre))
        }
        if (2 * Math.abs(stry) < Math.min(Math.abs(spre), 3 * Math.abs(sbis) - delta)) {
          spre = scur
          scur = stry
        } else {
          rule = 'bisection'
          spre = sbis
          scur = sbis
        }
      } else {
        spre = sbis
        scur = sbis
      }
      xpre = xcur
      fpre = fcur
      xcur += Math.abs(scur) > delta ? scur : sbis > 0 ? delta : -delta
      fcur = call(xcur)
      return next(s, history, {
        x: xcur,
        fx: fcur,
        lo: Math.min(xcur, xblk),
        hi: Math.max(xcur, xblk),
        rule,
        internal: { xpre, xcur, fpre, fcur, xblk, fblk, spre, scur },
      })
    },
  )
}

/**
 * Ridders' method (scipy.optimize.ridder) as a steppable algorithm: each step evaluates the midpoint of the bracket and
 * the point given by exponential interpolation through it, and keeps the sub-bracket with the sign change.
 *
 * @param f - The function.
 * @param a - One end of the bracket.
 * @param b - The other end.
 * @param options - Tolerances and iteration cap.
 * @returns The algorithm; its state is a {@link RootState}.
 * @throws MdmlError with code `value` when f(a) and f(b) have the same sign, or f returns NaN.
 * @example
 * ```ts
 * import { ridderAlgorithm } from 'mdml-engine/numerics/roots'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * runAlgorithm(ridderAlgorithm((x) => x * x - 2, 0, 2), { maxSteps: 100 }).state.flag // => 'converged'
 * ```
 */
export function ridderAlgorithm(
  f: (x: number) => number,
  a: number,
  b: number,
  options: BracketOptions = {},
): Algorithm<RootState, RootPath> {
  const xtol = options.xtol ?? 2e-12
  const rtol = options.rtol ?? 4 * EPS
  const max = options.maxIterations ?? 100
  return assemble(
    'ridder',
    { xtol, rtol, maxIterations: max },
    max,
    () => {
      const { call, history } = evaluator(f, [], 'ridder')
      const fa = call(a)
      const fb = call(b)
      checkBracket(fa, fb, 'ridder')
      if (fa === 0) return first(history, { x: a, fx: fa, rule: 'end point', flag: 'converged', internal: {} })
      if (fb === 0) return first(history, { x: b, fx: fb, rule: 'end point', flag: 'converged', internal: {} })
      return first(history, {
        x: a,
        fx: fa,
        lo: Math.min(a, b),
        hi: Math.max(a, b),
        rule: 'start',
        internal: { xa: a, xb: b, fa, fb, tol: xtol + rtol * Math.max(Math.abs(a), Math.abs(b)) },
      })
    },
    (s) => {
      const { call, history } = evaluator(f, s.history, 'ridder')
      let { xa, xb, fa, fb, tol } = s.internal as Record<'xa' | 'xb' | 'fa' | 'fb' | 'tol', number>
      const dm = 0.5 * (xb - xa)
      const xm = xa + dm
      const fm = call(xm)
      // The root of the square is taken of values scaled to at most 1 once they are far from unit scale, where fm * fm underflows.
      const unit = Math.max(Math.abs(fa), Math.abs(fb), Math.abs(fm))
      const k = unit > 1e-100 && unit < 1e100 ? 1 : unit
      const dn = (Math.sign(fb - fa) * dm * (fm / k)) / Math.sqrt((fm / k) * (fm / k) - (fa / k) * (fb / k))
      const xn = xm - Math.sign(dn) * Math.min(Math.abs(dn), Math.abs(dm) - 0.5 * tol)
      const fn = call(xn)
      if (Math.sign(fn) * Math.sign(fm) < 0) {
        xa = xn
        fa = fn
        xb = xm
        fb = fm
      } else if (Math.sign(fn) * Math.sign(fa) < 0) {
        xb = xn
        fb = fn
      } else {
        xa = xn
        fa = fn
      }
      tol = xtol + rtol * Math.abs(xn)
      const converged = fn === 0 || Math.abs(xb - xa) < tol
      return next(s, history, {
        x: xn,
        fx: fn,
        lo: Math.min(xa, xb),
        hi: Math.max(xa, xb),
        rule: 'exponential interpolation',
        ...(converged ? { flag: 'converged' as const } : {}),
        internal: { xa, xb, fa, fb, tol },
      })
    },
  )
}

/** A number kept in a state as a finite value and a flag, because JSON has no NaN. */
const present = (v: number): number => (Number.isNaN(v) ? 0 : v)

/**
 * The method of Alefeld, Potra and Shi (TOMS 748; scipy.optimize.toms748) in its essentials, as a steppable algorithm:
 * each step tries inverse cubic interpolation (falling back to a Newton step on the quadratic through three points) twice,
 * then a double-length secant step, and a bisection whenever the bracket has not shrunk by half. The `rule` is the last
 * kind of point it evaluated.
 *
 * @param f - The function.
 * @param a0 - One end of the bracket.
 * @param b0 - The other end.
 * @param options - Tolerances and iteration cap.
 * @returns The algorithm; its state is a {@link RootState}.
 * @throws MdmlError with code `value` when f(a) and f(b) have the same sign, or f returns NaN.
 * @example
 * ```ts
 * import { toms748Algorithm } from 'mdml-engine/numerics/roots'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * runAlgorithm(toms748Algorithm((x) => x ** 3 - x - 1, 1, 2), { maxSteps: 100 }).state.flag // => 'converged'
 * ```
 */
export function toms748Algorithm(
  f: (x: number) => number,
  a0: number,
  b0: number,
  options: BracketOptions = {},
): Algorithm<RootState, RootPath> {
  const xtol = options.xtol ?? 2e-12
  const rtol = options.rtol ?? 4 * EPS
  const max = options.maxIterations ?? 100
  const pick = (a: number, fa: number, b: number, fb: number): number => (Math.abs(fa) < Math.abs(fb) ? a : b)
  return assemble(
    'toms748',
    { xtol, rtol, maxIterations: max },
    max,
    () => {
      const { call, history } = evaluator(f, [], 'toms748')
      const a = Math.min(a0, b0)
      const b = Math.max(a0, b0)
      const fa = call(a)
      const fb = call(b)
      checkBracket(fa, fb, 'toms748')
      if (fa === 0) return first(history, { x: a, fx: fa, rule: 'end point', flag: 'converged', internal: {} })
      if (fb === 0) return first(history, { x: b, fx: fb, rule: 'end point', flag: 'converged', internal: {} })
      return first(history, {
        x: pick(a, fa, b, fb),
        fx: fb,
        lo: a,
        hi: b,
        rule: 'start',
        internal: { a, b, fa, fb, d: 0, fd: 0, e: 0, fe: 0, hasD: 0, hasE: 0 },
      })
    },
    (s) => {
      const { call, history } = evaluator(f, s.history, 'toms748')
      const m = s.internal as Record<'a' | 'b' | 'fa' | 'fb' | 'd' | 'fd' | 'e' | 'fe' | 'hasD' | 'hasE', number>
      let { a, b, fa, fb } = m
      let d = m.hasD === 1 ? m.d : Number.NaN
      let fd = m.hasD === 1 ? m.fd : Number.NaN
      let e = m.hasE === 1 ? m.e : Number.NaN
      let fe = m.hasE === 1 ? m.fe : Number.NaN
      let rule = 'start'
      let lastFx = s.fx
      const inside = (c: number): boolean => c > a && c < b
      const tolerance = (): number => 2 * (xtol + rtol * Math.min(Math.abs(a), Math.abs(b)))
      const bracket = (c: number): boolean => {
        const fc = call(c)
        lastFx = fc
        if (fc === 0) {
          a = b = c
          fa = fb = 0
          return true
        }
        if (Math.sign(fc) === Math.sign(fa)) {
          e = d
          fe = fd
          d = a
          fd = fa
          a = c
          fa = fc
        } else {
          e = d
          fe = fd
          d = b
          fd = fb
          b = c
          fb = fc
        }
        return false
      }
      const secantPoint = (): number => a - (fa * (b - a)) / (fb - fa)
      const newtonQuadratic = (steps: number): number => {
        const B = (fb - fa) / (b - a)
        const A = ((fd - fb) / (d - b) - B) / (d - a)
        if (!Number.isFinite(A) || A === 0) return secantPoint()
        let r = Math.sign(A) === Math.sign(fa) ? a : b
        for (let i = 0; i < steps; i++) r -= (fa + (B + A * (r - b)) * (r - a)) / (B + A * (2 * r - a - b))
        return inside(r) ? r : secantPoint()
      }
      const inverseCubic = (): number => {
        const xs = [a, b, d, e]
        const fs = [fa, fb, fd, fe]
        const p = [...xs]
        for (let j = 1; j < 4; j++) {
          for (let i = 3; i >= j; i--)
            p[i] = ((p[i] ?? 0) * (fs[i - j] ?? 0) - (p[i - 1] ?? 0) * (fs[i] ?? 0)) / ((fs[i - j] ?? 0) - (fs[i] ?? 0))
        }
        return p[3] ?? Number.NaN
      }
      const finish = (converged: boolean): RootState =>
        next(s, history, {
          x: pick(a, fa, b, fb),
          fx: lastFx,
          lo: a,
          hi: b,
          rule,
          ...(converged ? { flag: 'converged' as const } : {}),
          internal: {
            a,
            b,
            fa,
            fb,
            d: present(d),
            fd: present(fd),
            e: present(e),
            fe: present(fe),
            hasD: Number.isNaN(d) ? 0 : 1,
            hasE: Number.isNaN(e) ? 0 : 1,
          },
        })
      const width = b - a
      for (let pass = 0; pass < 2; pass++) {
        const distinct = Number.isFinite(fe) && new Set([fa, fb, fd, fe]).size === 4
        let c = distinct ? inverseCubic() : Number.NaN
        rule = 'inverse cubic interpolation'
        if (!inside(c)) {
          rule = Number.isFinite(fd) ? 'quadratic interpolation' : 'secant'
          c = Number.isFinite(fd) ? newtonQuadratic(pass + 2) : secantPoint()
        }
        if (!inside(c)) {
          rule = 'bisection'
          c = (a + b) / 2
        }
        if (bracket(c) || b - a < tolerance()) return finish(true)
      }
      const [u, fu] = Math.abs(fa) < Math.abs(fb) ? [a, fa] : [b, fb]
      let c = u - (2 * fu * (b - a)) / (fb - fa)
      rule = 'double-length secant'
      if (!inside(c) || Math.abs(c - u) > (b - a) / 2) {
        rule = 'bisection'
        c = (a + b) / 2
      }
      if (bracket(c) || b - a < tolerance()) return finish(true)
      if (b - a > 0.5 * width) {
        rule = 'bisection'
        if (bracket((a + b) / 2)) return finish(true)
      }
      if (b - a < tolerance()) return finish(true)
      return finish(false)
    },
  )
}

/**
 * Newton's method (scipy.optimize.newton) as a steppable algorithm: with `fprime`, Newton steps (with Halley's
 * correction when `fprime2` is also given); without it, the secant method from `x0` and `x1`. One step evaluates f at the
 * iterate and moves it; the derivative being zero ends the run with the flag `derivative was zero`.
 *
 * @param f - The function.
 * @param x0 - Starting point.
 * @param options - Derivatives, a second start, tolerances and iteration cap.
 * @returns The algorithm; its state is a {@link RootState}.
 * @concept root-finding
 * @example
 * ```ts
 * import { newtonAlgorithm } from 'mdml-engine/numerics/roots'
 * import { iterate } from 'mdml-engine/trace'
 * const alg = newtonAlgorithm((x) => x * x - 4, 3, { fprime: (x) => 2 * x })
 * Array.from(iterate(alg, { maxSteps: 2 }), (i) => Math.round(i.state.x * 1e6) / 1e6) // => [3, 2.166667, 2.006410]
 * ```
 */
export function newtonAlgorithm(
  f: (x: number) => number,
  x0: number,
  options: NewtonOptions = {},
): Algorithm<RootState, RootPath> {
  const tol = options.tol ?? 1.48e-8
  const rtol = options.rtol ?? 0
  const max = options.maxIterations ?? 50
  const fprime = options.fprime
  const fprime2 = options.fprime2
  const name = fprime === undefined ? 'secant' : fprime2 === undefined ? 'newton' : 'halley'
  if (fprime !== undefined) {
    return assemble(
      name,
      { tol, rtol, maxIterations: max },
      max,
      () => first([], { x: x0, fx: null, rule: 'start', internal: {} }),
      (s) => {
        const { call, history } = evaluator(f, s.history)
        const p0 = s.x
        const fval = call(p0)
        if (fval === 0) return next(s, history, { x: p0, fx: fval, rule: 'root', flag: 'converged', internal: {} })
        const d1 = fprime(p0)
        if (d1 === 0)
          return next(s, history, {
            x: p0,
            fx: fval,
            rule: 'derivative was zero',
            flag: 'derivative was zero',
            internal: {},
          })
        let step = fval / d1
        let rule = 'newton'
        if (fprime2 !== undefined) {
          const adj = (step * fprime2(p0)) / d1 / 2
          if (Math.abs(adj) < 1) {
            step /= 1 - adj
            rule = 'halley'
          }
        }
        const p = p0 - step
        const converged = Math.abs(p - p0) <= tol + rtol * Math.abs(p)
        return next(s, history, {
          x: p,
          fx: fval,
          rule,
          ...(converged ? { flag: 'converged' as const } : {}),
          internal: {},
        })
      },
    )
  }
  return assemble(
    name,
    { tol, rtol, maxIterations: max },
    max,
    () => {
      const { call, history } = evaluator(f, [])
      let p0 = x0
      let p1 = options.x1 ?? x0 * (1 + 1e-4) + (x0 >= 0 ? 1e-4 : -1e-4)
      let q0 = call(p0)
      let q1 = call(p1)
      if (Math.abs(q1) < Math.abs(q0)) {
        ;[p0, p1, q0, q1] = [p1, p0, q1, q0]
      }
      return first(history, { x: p1, fx: q1, rule: 'start', internal: { p0, p1, q0, q1 } })
    },
    (s) => {
      const { call, history } = evaluator(f, s.history)
      let { p0, p1, q0, q1 } = s.internal as Record<'p0' | 'p1' | 'q0' | 'q1', number>
      if (q1 === q0)
        return next(s, history, {
          x: (p1 + p0) / 2,
          fx: q1,
          rule: 'derivative was zero',
          flag: p1 === p0 ? 'converged' : 'derivative was zero',
          internal: { p0, p1, q0, q1 },
        })
      const p =
        Math.abs(q1) > Math.abs(q0) ? ((-q0 / q1) * p1 + p0) / (1 - q0 / q1) : ((-q1 / q0) * p0 + p1) / (1 - q1 / q0)
      if (Math.abs(p - p1) <= tol + rtol * Math.abs(p))
        return next(s, history, { x: p, fx: q1, rule: 'secant', flag: 'converged', internal: { p0, p1, q0, q1 } })
      p0 = p1
      q0 = q1
      p1 = p
      q1 = call(p1)
      return next(s, history, { x: p1, fx: q1, rule: 'secant', internal: { p0, p1, q0, q1 } })
    },
  )
}

/**
 * The secant method from two starting points, as a steppable algorithm.
 *
 * @param f - The function.
 * @param x0 - First point.
 * @param x1 - Second point.
 * @param options - Tolerances and iteration cap.
 * @returns The algorithm; its state is a {@link RootState}.
 * @example
 * ```ts
 * import { secantAlgorithm } from 'mdml-engine/numerics/roots'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * runAlgorithm(secantAlgorithm((x) => x - 3, 0, 1), { maxSteps: 50 }).state.x // => 3
 * ```
 */
export function secantAlgorithm(
  f: (x: number) => number,
  x0: number,
  x1: number,
  options: Omit<NewtonOptions, 'fprime' | 'fprime2' | 'x1'> = {},
): Algorithm<RootState, RootPath> {
  return newtonAlgorithm(f, x0, { ...options, x1 })
}

/**
 * Halley's method (cubically convergent with first and second derivatives) as a steppable algorithm.
 *
 * @param f - The function.
 * @param x0 - Starting point.
 * @param fprime - First derivative.
 * @param fprime2 - Second derivative.
 * @param options - Tolerances and iteration cap.
 * @returns The algorithm; its state is a {@link RootState}.
 * @example
 * ```ts
 * import { halleyAlgorithm } from 'mdml-engine/numerics/roots'
 * halleyAlgorithm((x) => x ** 3 - 8, 3, (x) => 3 * x * x, (x) => 6 * x).name // => 'halley'
 * ```
 */
export function halleyAlgorithm(
  f: (x: number) => number,
  x0: number,
  fprime: (x: number) => number,
  fprime2: (x: number) => number,
  options: Omit<NewtonOptions, 'fprime' | 'fprime2' | 'x1'> = {},
): Algorithm<RootState, RootPath> {
  return newtonAlgorithm(f, x0, { ...options, fprime, fprime2 })
}
