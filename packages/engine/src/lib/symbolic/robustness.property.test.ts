// Every public operation on random expressions either succeeds or fails with a MdmlError (never a TypeError or
// a hang), within a bounded and deterministic amount of work, and the results that are claimed to be equal to the
// input are equal in value.
import fc from 'fast-check'
import { describe, it } from 'vitest'
import { exprArb } from '../../testing/symbolic.ts'
import { isMdmlError } from '../base/index.ts'
import { DEFAULT_STEPS, lastWork } from './_budget.ts'
import {
  apart,
  cancel,
  diff,
  type Expr,
  evaluateComplex,
  expand,
  factor,
  integrate,
  isExpr,
  limit,
  parseLatex,
  series,
  simplify,
  sym,
  together,
  toLatex,
  toText,
} from './index.ts'

const x = sym('x')

function guarded(f: () => unknown): boolean {
  try {
    f()
    return true
  } catch (e) {
    if (isMdmlError(e)) return true
    throw e
  }
}

function valueAt(e: Expr): [number, number] | undefined {
  try {
    return evaluateComplex(e, { x: 0.41, y: 0.67 })
  } catch {
    return undefined
  }
}

const OPERATIONS: readonly (readonly [string, (e: Expr) => unknown])[] = [
  ['simplify', (e) => simplify(e)],
  ['factor', (e) => factor(e)],
  ['expand', (e) => expand(e)],
  ['together', (e) => together(e)],
  ['cancel', (e) => cancel(e)],
  ['apart', (e) => apart(e, x)],
  ['latex', (e) => toText(parseLatex(toLatex(e)))],
  ['diff', (e) => diff(e, x)],
  ['limit', (e) => limit(e, x, 0)],
  ['integrate', (e) => integrate(e, x)],
  ['series', (e) => series(e, x, 0, 3)],
]

/** The printed result (or error code) of an operation and the work it spent. */
function outcome(f: () => unknown): [string, number] {
  let out: string
  try {
    const r = f()
    out = isExpr(r) ? toText(r) : JSON.stringify(r, (_, v) => (typeof v === 'bigint' ? String(v) : v))
  } catch (e) {
    if (!isMdmlError(e)) throw e
    out = `error ${e.code}`
  }
  return [out, lastWork()]
}

describe('robustness on random expressions', () => {
  it('never throws anything but MdmlError', () => {
    fc.assert(
      fc.property(exprArb, (e) => OPERATIONS.every(([, op]) => guarded(() => op(e)))),
      { numRuns: 40 },
    )
  })

  it('every operation stays within its work budget, deterministically', () => {
    // Work is counted, not timed: the same input spends the same work and gives the same result on any machine,
    // and no operation spends more than the default budget (a fraction of a second).
    fc.assert(
      fc.property(exprArb, (e) =>
        OPERATIONS.every(([, op]) => {
          const [first, work] = outcome(() => op(e))
          const [again, workAgain] = outcome(() => op(e))
          return first === again && work === workAgain && work <= DEFAULT_STEPS
        }),
      ),
      { numRuns: 20 },
    )
  })

  it('rewrites keep their value', () => {
    fc.assert(
      fc.property(exprArb, (e) => {
        const v = valueAt(e)
        if (v === undefined || !v.every(Number.isFinite)) return true
        return [expand, factor, together, cancel].every((f) => {
          let r: Expr
          try {
            r = f(e)
          } catch (err) {
            if (isMdmlError(err)) return true
            throw err
          }
          const w = valueAt(r)
          return w === undefined || Math.hypot(w[0] - v[0], w[1] - v[1]) <= 1e-8 * Math.max(1, Math.hypot(...v))
        })
      }),
      { numRuns: 40 },
    )
  })
})
