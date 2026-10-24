// Every public operation on random expressions either succeeds or fails with a WhiteboxError (never a TypeError or
// a hang), and the results that are claimed to be equal to the input are equal in value.
import fc from 'fast-check'
import { describe, it } from 'vitest'
import { exprArb } from '../../testing/symbolic.ts'
import { isWhiteboxError } from '../base/index.ts'
import {
  apart,
  cancel,
  diff,
  type Expr,
  evaluateComplex,
  expand,
  factor,
  integrate,
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
    if (isWhiteboxError(e)) return true
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

describe('robustness on random expressions', () => {
  it('never throws anything but WhiteboxError', () => {
    fc.assert(
      fc.property(exprArb, (e) =>
        [
          () => simplify(e),
          () => factor(e),
          () => expand(e),
          () => together(e),
          () => cancel(e),
          () => apart(e, x),
          () => toText(parseLatex(toLatex(e))),
          () => diff(e, x),
          () => integrate(e, x),
          () => series(e, x, 0, 3),
          () => limit(e, x, 0),
        ].every(guarded),
      ),
      { numRuns: 40 },
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
            if (isWhiteboxError(err)) return true
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
