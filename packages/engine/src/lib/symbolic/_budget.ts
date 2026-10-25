// Work budgets: every public symbolic operation runs inside a frame holding a number of work units, which the inner
// loops spend (one unit is roughly one exact rational operation, one expression node built or one term of polynomial
// arithmetic), and a cap on the terms of an intermediate expansion. Running out raises `WorkExhausted`, which the
// operation that opened the frame turns into its documented fallback. Work is counted, not timed, so the same input
// gives the same result on any machine. Frames nest: an operation called inside another, or a search attempt that
// may fail on its own (an integration method, a simplification candidate), opens a frame that never has more work
// than its parent has left; what it spends is charged to the parent when it closes, so a parent whose child used up
// everything is exhausted too (exhaustion is sticky: every later charge fails again).
import { MdmlError } from '../base/index.ts'
import type { WorkBudget } from './expr.ts'

/** Work units of a top-level operation unless its budget says otherwise. @internal */
export const DEFAULT_STEPS = 2_000_000
/** Terms of an intermediate expansion unless the budget says otherwise. @internal */
export const DEFAULT_TERMS = 2_000

interface Frame {
  /** Work left; negative once exhausted. */
  left: number
  readonly limit: number
  readonly terms: number
}

const stack: Frame[] = []
let last = 0

/** Raised when a frame runs out of work or an expansion would exceed the term cap. @internal */
export class WorkExhausted extends MdmlError {
  /** The frame whose operation takes its fallback. */
  readonly frame: object
  constructor(frame: Frame, message: string) {
    super('convergence', message, { steps: frame.limit, terms: frame.terms })
    this.frame = frame
  }
}

/** Spend `n` work units of the innermost frame (free outside any frame). @internal */
export function charge(n: number): void {
  const f = stack[stack.length - 1]
  if (f === undefined) return
  f.left -= n
  if (f.left < 0) throw new WorkExhausted(f, `symbolic: the work budget of ${f.limit} steps is exhausted`)
}

/** Spend `n` units for an intermediate result of `n` terms, refused when that exceeds the term cap. @internal */
export function chargeTerms(n: number): void {
  charge(n)
  const f = stack[stack.length - 1]
  if (f !== undefined && n > f.terms) {
    throw new WorkExhausted(f, `symbolic: an intermediate result would have more than ${f.terms} terms`)
  }
}

/** The work spent by the last top-level operation (for tests and calibration). @internal */
export function lastWork(): number {
  return last
}

/** Work left in the innermost frame (Infinity outside any frame). @internal */
export function workLeft(): number {
  const f = stack[stack.length - 1]
  return f === undefined ? Number.POSITIVE_INFINITY : Math.max(0, f.left)
}

/**
 * Run `body` in a frame of its own and return its value, or `fallback()` when the frame runs out (the fallback runs
 * after the frame is closed, in the enclosing one). The frame has `budget.steps` units, or by default the enclosing
 * frame's remainder (DEFAULT_STEPS at top level), and never more than the enclosing frame has left.
 * @internal
 */
export function withBudget<T>(budget: WorkBudget | undefined, body: () => T, fallback: () => T): T {
  const parent = stack[stack.length - 1]
  const avail = parent === undefined ? Number.POSITIVE_INFINITY : Math.max(0, parent.left)
  const limit = Math.min(budget?.steps ?? (parent === undefined ? DEFAULT_STEPS : avail), avail)
  const frame: Frame = { left: limit, limit, terms: budget?.terms ?? parent?.terms ?? DEFAULT_TERMS }
  stack.push(frame)
  try {
    return body()
  } catch (err) {
    if (!(err instanceof WorkExhausted) || err.frame !== frame) throw err
  } finally {
    stack.pop()
    // The work refused by the charge that ran out was never done, so a frame spends at most its limit.
    const spent = Math.min(limit, limit - frame.left)
    if (parent !== undefined) parent.left -= spent
    else last = spent
  }
  return fallback()
}

/**
 * Run a public operation without a fallback of its own: running out is a MdmlError with code `convergence`.
 * @internal
 */
export function bounded<T>(name: string, body: () => T, budget?: WorkBudget): T {
  // Inside another operation, a frame that had all the work left exhausts its caller as well, which takes its own
  // fallback; otherwise running out is the caller's error to handle.
  const inherited = stack.length > 0 && (budget?.steps ?? Number.POSITIVE_INFINITY) >= workLeft()
  return withBudget(budget, body, () => {
    const caller = stack[stack.length - 1]
    if (inherited && caller !== undefined) {
      caller.left = Math.min(caller.left, -1)
      throw new WorkExhausted(caller, `symbolic: the work budget of ${caller.limit} steps is exhausted`)
    }
    throw new MdmlError('convergence', `${name}: the work budget is exhausted (raise it with withWorkBudget)`)
  })
}

/** Run a search attempt on at most `share` of the work left; undefined when it runs out. @internal */
export function attempt<T>(share: number, body: () => T | undefined): T | undefined {
  return withBudget({ steps: Math.floor(workLeft() * share) }, body, () => undefined)
}
