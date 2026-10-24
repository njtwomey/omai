// Expansion (SymPy's `expand` with its default hints: mul, multinomial, power_exp, log with assumptions, deep).

import { isNegative, isNonnegative, isNonzero, isPositive, isReal } from './_assume.ts'
import { chargeTerms } from './_budget.ts'
import { isNumeric, MINUS_ONE, makeAdd, makeMul, makePow, mapChildren, ONE } from './_canon.ts'
import { makeFn } from './_functions.ts'
import type { Expr, Step } from './expr.ts'

const MAX_MULTINOMIAL = 200

function isNonpositive(e: Expr): boolean {
  return isNegative(e) === true || (e.kind === 'num' && e.n === 0n)
}

function record(steps: Step[] | undefined, rule: string, before: Expr, after: Expr): Expr {
  if (steps !== undefined && before.key !== after.key) steps.push({ rule, before, after })
  return after
}

/** Distribute a product over the sums among its factors. */
function distribute(factors: readonly Expr[]): Expr {
  let terms: Expr[] = [ONE]
  for (const f of factors) {
    if (f.kind === 'add') {
      chargeTerms(terms.length * f.args.length)
      const next: Expr[] = []
      for (const t of terms) for (const a of f.args) next.push(makeMul([t, a]))
      terms = next
    } else terms = terms.map((t) => makeMul([t, f]))
  }
  return makeAdd(terms)
}

function needsMore(e: Expr): boolean {
  const terms = e.kind === 'add' ? e.args : [e]
  return terms.some(
    (t) =>
      (t.kind === 'mul' &&
        t.args.some(
          (f) =>
            f.kind === 'add' ||
            (f.kind === 'pow' && f.base.kind === 'add' && f.exp.kind === 'num' && f.exp.d === 1n && f.exp.n > 0n),
        )) ||
      (t.kind === 'pow' && t.base.kind === 'add' && t.exp.kind === 'num' && t.exp.d === 1n && t.exp.n > 0n),
  )
}

function expandNode(e: Expr, steps: Step[] | undefined, depth: number): Expr {
  const x = mapChildren(e, (c) => expandNode(c, steps, depth))
  switch (x.kind) {
    case 'mul': {
      if (!x.args.some((f) => f.kind === 'add')) return x
      const out = distribute(x.args)
      record(steps, 'distribute', x, out)
      return depth < 4 && needsMore(out) ? expandNode(out, steps, depth + 1) : out
    }
    case 'pow': {
      const { base, exp } = x
      if (base.kind === 'add' && exp.kind === 'num' && exp.d === 1n && exp.n !== 0n) {
        const n = Number(exp.n < 0n ? -exp.n : exp.n)
        if (n > MAX_MULTINOMIAL) return x
        let acc: Expr = base
        for (let i = 1; i < n; i++) acc = distribute([acc, base])
        if (depth < 4 && needsMore(acc)) acc = expandNode(acc, steps, depth + 1)
        const out = exp.n < 0n ? makePow(acc, MINUS_ONE) : acc
        return record(steps, exp.n < 0n ? 'expand denominator' : 'multinomial', x, out)
      }
      const sameSign =
        exp.kind === 'add' &&
        (exp.args.every((t) => isNonnegative(t) === true) || exp.args.every((t) => isNonpositive(t)))
      if (exp.kind === 'add' && (isNonzero(base) === true || sameSign)) {
        // x^(a + b) = x^a * x^b, valid when x is non-zero or the exponents share a sign (as SymPy's power_exp).
        const out = makeMul(exp.args.map((t) => makePow(base, t)))
        return record(steps, 'power of a sum', x, out)
      }
      if (exp.kind === 'mul' && isNumeric(exp.args[0] as Expr) && base.kind === 'mul') return x
      return x
    }
    case 'fn': {
      const a = x.args[0] as Expr
      if (x.name === 'exp' && a.kind === 'add') {
        const out = makeMul(a.args.map((t) => makeFn('exp', [t])))
        return record(steps, 'exp of a sum', x, out)
      }
      if (x.name === 'log' && a.kind === 'mul' && a.args.every((f) => isPositive(f) === true)) {
        const out = makeAdd(a.args.map((f) => expandNode(makeFn('log', [f]), steps, depth)))
        return record(steps, 'log of a product', x, out)
      }
      if (x.name === 'log' && a.kind === 'pow' && isPositive(a.base) === true && isReal(a.exp) === true) {
        const out = makeMul([a.exp, makeFn('log', [a.base])])
        return record(steps, 'log of a power', x, out)
      }
      return x
    }
    default:
      return x
  }
}

/** Expand an expression, recording rewrites when `steps` is given. @internal */
export function expandExpr(e: Expr, steps?: Step[]): Expr {
  return expandNode(e, steps, 0)
}
