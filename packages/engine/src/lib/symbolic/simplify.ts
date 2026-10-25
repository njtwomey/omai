/**
 * Simplification and rewriting, as SymPy's `simplify`: expanding, collecting, factoring over the rationals, combining
 * over a common denominator, cancelling, partial fractions and trigonometric simplification, and `simplify`, which
 * tries these rewrites and keeps whichever has the fewest operations. Most have `Steps` variants that return the
 * derivation, and `compareExprs` gives SymPy's canonical ordering.
 */

import { attempt, bounded, withBudget } from './_budget.ts'
import {
  asCoeffMul,
  isNumeric,
  keepCoeff,
  MINUS_ONE,
  makeAdd,
  makeMul,
  makePow,
  mapChildren,
  numFromRational,
  ONE,
} from './_canon.ts'
import { expandExpr } from './_expand.ts'
import { factorMultivariate } from './_factor.ts'
import { makeFn } from './_functions.ts'
import { complexity, countOps as ops } from './_measure.ts'
import { compareKeys, sortKey } from './_order.ts'
import { exprToP, findGens, pToExpr } from './_poly.ts'
import { rat } from './_rational.ts'
import { apart as apartRaw, cancel as cancelRaw, numerDenom, together as togetherRaw } from './_rational-functions.ts'
import { hasTrig, trigSimplifyExpr } from './_trigsimp.ts'
import { type Derivation, type Expr, type ExprSym, type Step, sym, type WorkBudget } from './expr.ts'

/**
 * Compare two expressions in SymPy's canonical sort order (`default_sort_key`): numbers, then symbols, then
 * compound expressions, then functions. The printers use this order for factors; sorting terms of a sum in printed
 * order uses the monomial order, see {@link toText}.
 *
 * @param a - First expression.
 * @param b - Second expression.
 * @returns Negative, zero or positive.
 * @example
 * ```ts
 * import { compareExprs, parse } from 'whitebox-engine/symbolic'
 * [parse('sin(x)'), parse('y'), parse('2')].sort(compareExprs).map(String).length // => 3
 * compareExprs(parse('x'), parse('sin(x)')) < 0 // => true
 * ```
 */
export function compareExprs(a: Expr, b: Expr): number {
  return compareKeys(sortKey(a), sortKey(b))
}

/**
 * The number of operations in an expression (SymPy's `count_ops`, the measure `simplify` minimises).
 *
 * @param e - The expression.
 * @returns The count.
 * @example
 * ```ts
 * import { countOps, parse } from 'whitebox-engine/symbolic'
 * countOps(parse('x^2 + 2*x + 1')) // => 4
 * ```
 */
export function countOps(e: Expr): number {
  return ops(e)
}

/**
 * Expand products and powers of sums, sums in exponents (x^(a+b) = x^a x^b, exp(a+b) = exp(a) exp(b)) and logarithms
 * of products and powers of positive quantities, recursively (SymPy's `expand`).
 *
 * @param e - The expression.
 * @returns The expanded expression.
 * @example
 * ```ts
 * import { expand, parse, toText } from 'whitebox-engine/symbolic'
 * toText(expand(parse('(x + y)^2'))) // => 'x**2 + 2*x*y + y**2'
 * ```
 */
export function expand(e: Expr): Expr {
  return bounded('expand', () => expandExpr(e))
}

/**
 * {@link expand} with the rewrites it made.
 *
 * @param e - The expression.
 * @returns The expanded expression and its steps.
 * @example
 * ```ts
 * import { expandSteps, parse } from 'whitebox-engine/symbolic'
 * expandSteps(parse('(x + 1)^2')).steps[0]?.rule // => 'multinomial'
 * ```
 */
export function expandSteps(e: Expr): Derivation {
  return bounded('expandSteps', () => {
    const steps: Step[] = []
    const result = expandExpr(e, steps)
    return { result, steps }
  })
}

function powerIn(f: Expr, s: Expr): Expr | undefined {
  if (f.key === s.key) return ONE
  if (f.kind === 'pow' && f.base.key === s.key) return f.exp
  return undefined
}

/**
 * Collect the terms of a sum by powers of the given expressions (SymPy's `collect`): a*x + b*x + c becomes
 * x*(a + b) + c. The sum is not expanded first.
 *
 * @param e - The expression.
 * @param syms - The symbols or subexpressions to collect by.
 * @returns The collected expression.
 * @example
 * ```ts
 * import { collect, parse, toText } from 'whitebox-engine/symbolic'
 * toText(collect(parse('a*x^2 + b*x^2 + a*x - b*x + c'), parse('x'))) // => 'c + x**2*(a + b) + x*(a - b)'
 * ```
 */
export function collect(e: Expr, ...syms: readonly Expr[]): Expr {
  if (e.kind !== 'add') return e
  const groups = new Map<string, { key: Expr; coeffs: Expr[] }>()
  for (const t of e.args) {
    const factors = t.kind === 'mul' ? t.args : [t]
    const keyParts: Expr[] = []
    const rest: Expr[] = []
    for (const f of factors) {
      const hit = syms.find((s) => powerIn(f, s) !== undefined)
      if (hit !== undefined) keyParts.push(f)
      else rest.push(f)
    }
    const key = makeMul(keyParts)
    const g = groups.get(key.key)
    if (g === undefined) groups.set(key.key, { key, coeffs: [makeMul(rest)] })
    else g.coeffs.push(makeMul(rest))
  }
  return makeAdd([...groups.values()].map((g) => makeMul([g.key, makeAdd(g.coeffs)])))
}

function factorPolynomial(e: Expr): Expr {
  const gens = findGens(e)
  const p = exprToP(e, gens)
  if (p === undefined || gens.length === 0) return e
  const [c, factors] = factorMultivariate(p)
  const parts = factors.map(([f, k]) => makePow(pToExpr(f, gens), numFromRational(rat(BigInt(k)))))
  return keepCoeff(numFromRational(c), makeMul(parts))
}

/**
 * Factor into irreducibles over the rationals (SymPy's `factor`): sums are put over a common denominator and the
 * numerator and denominator factored as polynomials in their symbols, functions and radicals, with any number of
 * variables. Products and powers are factored part by part; function arguments are left alone. When the work budget
 * runs out, the expression is returned as it was (the best form found so far that is certainly equal to it).
 *
 * @param e - The expression.
 * @param budget - Work limits (see {@link WorkBudget}).
 * @returns The factored form.
 * @example
 * ```ts
 * import { factor, parse, toText } from 'whitebox-engine/symbolic'
 * toText(factor(parse('x^3 - x^2*y - x*y^2 + y^3'))) // => '(x - y)**2*(x + y)'
 * toText(factor(parse('2*x^2 - 2'))) // => '2*(x - 1)*(x + 1)'
 * toText(factor(parse('x^4 - 1'), { steps: 50 })) // => 'x**4 - 1'
 * ```
 */
export function factor(e: Expr, budget?: WorkBudget): Expr {
  return withBudget(
    budget,
    () => factorExpr(e),
    () => e,
  )
}

function factorExpr(e: Expr): Expr {
  if (e.kind === 'mul') return makeMul(e.args.map(factorExpr))
  if (e.kind === 'pow') return makePow(factorExpr(e.base), e.exp)
  if (e.kind !== 'add') return e
  const t = togetherRaw(e)
  const [n, d] = numerDenom(t)
  const fn = factorPolynomial(n)
  const fd = factorPolynomial(d)
  if (fd.kind === 'num') return fd.n === 1n && fd.d === 1n ? fn : keepCoeff(makePow(fd, MINUS_ONE), fn)
  const [c, body] = asCoeffMul(fn)
  const [cd, bodyD] = asCoeffMul(fd)
  return keepCoeff(makeMul([c, makePow(cd, MINUS_ONE)]), makeMul([body, makePow(bodyD, MINUS_ONE)]))
}

/**
 * Put a sum over a common denominator without expanding (SymPy's `together`).
 *
 * @param e - The expression.
 * @returns One fraction.
 * @example
 * ```ts
 * import { parse, together, toText } from 'whitebox-engine/symbolic'
 * toText(together(parse('1/x + 1/y'))) // => '(x + y)/(x*y)'
 * ```
 */
export function together(e: Expr): Expr {
  return bounded('together', () => togetherRaw(e))
}

/**
 * Cancel common factors in a rational function (SymPy's `cancel`): numerator and denominator are expanded, coprime
 * and primitive, with a rational constant in front.
 *
 * @param e - The expression.
 * @returns p/q in lowest terms.
 * @example
 * ```ts
 * import { cancel, parse, toText } from 'whitebox-engine/symbolic'
 * toText(cancel(parse('(x^2 - 1)/(x - 1)'))) // => 'x + 1'
 * ```
 */
export function cancel(e: Expr): Expr {
  return bounded('cancel', () => cancelRaw(e))
}

/**
 * Partial fractions in x (SymPy's `apart`): a polynomial part plus N(x)/f(x)^j over the irreducible factors f of the
 * denominator over the rationals; other symbols are parameters.
 *
 * @param e - A rational function of x.
 * @param x - The variable.
 * @returns The decomposition.
 * @throws WhiteboxError with code `value` when `e` is not a rational function of x, `convergence` when the work
 *   budget runs out.
 * @example
 * ```ts
 * import { apart, parse, sym, toText } from 'whitebox-engine/symbolic'
 * toText(apart(parse('1/(x^2 - 1)'), sym('x'))) // => '-1/(2*(x + 1)) + 1/(2*(x - 1))'
 * ```
 */
export function apart(e: Expr, x: ExprSym): Expr {
  return bounded('apart', () => apartRaw(e, x))
}

/**
 * Simplify trigonometric and hyperbolic expressions (Pythagorean identities, double angles, products to sums,
 * tan = sin/cos), choosing the form with the fewest operations, as SymPy's `trigsimp`. When the work budget runs out
 * the expression is returned unchanged.
 *
 * @param e - The expression.
 * @returns The simplified expression.
 * @example
 * ```ts
 * import { parse, toText, trigSimplify } from 'whitebox-engine/symbolic'
 * toText(trigSimplify(parse('sin(x)^2 + cos(x)^2'))) // => '1'
 * toText(trigSimplify(parse('2*sin(x)*cos(x)'))) // => 'sin(2*x)'
 * ```
 */
export function trigSimplify(e: Expr): Expr {
  return withBudget(
    undefined,
    () => trigSimplifyExpr(e),
    () => e,
  )
}

/** Combine powers: x^a*x^b = x^(a+b) and exp(a)*exp(b) = exp(a+b) (SymPy's `powsimp`). */
function powsimp(e: Expr): Expr {
  const x = mapChildren(e, powsimp)
  if (x.kind !== 'mul') return x
  const bases = new Map<string, { base: Expr; exps: Expr[] }>()
  const expArgs: Expr[] = []
  const rest: Expr[] = []
  for (const f of x.args) {
    if (f.kind === 'fn' && f.name === 'exp') expArgs.push(f.args[0] as Expr)
    else if (f.kind === 'pow' && !isNumeric(f.base)) {
      const g = bases.get(f.base.key)
      if (g === undefined) bases.set(f.base.key, { base: f.base, exps: [f.exp] })
      else g.exps.push(f.exp)
    } else if (f.kind === 'sym' || f.kind === 'fn') {
      const g = bases.get(f.key)
      if (g === undefined) bases.set(f.key, { base: f, exps: [ONE] })
      else g.exps.push(ONE)
    } else rest.push(f)
  }
  const out = [...rest]
  for (const g of bases.values()) out.push(makePow(g.base, makeAdd(g.exps)))
  if (expArgs.length > 0) out.push(makeFn('exp', [makeAdd(expArgs)]))
  return makeMul(out)
}

const CANDIDATES: readonly [string, (e: Expr) => Expr][] = [
  ['expand', expandExpr],
  ['cancel', cancelRaw],
  ['factor', factorExpr],
  ['together', togetherRaw],
  ['combine powers', powsimp],
  ['trigonometric simplification', trigSimplifyExpr],
]

/** The best form of the whole expression so far and the steps that led to it, kept when the work runs out. */
interface Progress {
  best: Expr
  steps: number
}

function simplifyNode(e: Expr, steps: Step[], progress?: Progress): Expr {
  // Simplify inside function arguments and exponents first.
  let cur =
    e.kind === 'fn' || e.kind === 'pow' || e.kind === 'piecewise' ? mapChildren(e, (c) => simplifyNode(c, steps)) : e
  if (cur.kind === 'add' || cur.kind === 'mul') {
    const inner = mapChildren(cur, (c) => (c.kind === 'fn' || c.kind === 'pow' ? simplifyNode(c, steps) : c))
    if (inner.key !== cur.key) cur = inner
  }
  const keep = (): void => {
    if (progress === undefined) return
    progress.best = cur
    progress.steps = steps.length
  }
  keep()
  for (let round = 0; round < 4; round++) {
    let best = cur
    let bestRule = ''
    for (const [rule, f] of CANDIDATES) {
      if (rule === 'trigonometric simplification' && !hasTrig(cur)) continue
      let out: Expr | undefined
      try {
        // Each rewrite may use half of the work left, so that one that blows up leaves some for the others.
        out = attempt(0.5, () => f(cur))
      } catch {
        continue
      }
      if (out !== undefined && complexity(out) < complexity(best)) {
        best = out
        bestRule = rule
      }
    }
    if (best.key === cur.key) break
    steps.push({ rule: bestRule, before: cur, after: best })
    cur = best
    keep()
  }
  return cur
}

function simplifyWithin(e: Expr, budget: WorkBudget | undefined): Derivation {
  const steps: Step[] = []
  const progress: Progress = { best: e, steps: 0 }
  return withBudget(
    budget,
    () => ({ result: simplifyNode(e, steps, progress), steps }),
    () => ({ result: progress.best, steps: steps.slice(0, progress.steps) }),
  )
}

/**
 * Simplify by trying the rewrites (expand, cancel, factor, together, power combination, trigonometric
 * simplification) and keeping whichever gives the fewest operations, repeatedly, after simplifying function
 * arguments (in the spirit of SymPy's `simplify`). When the work budget runs out, the best form found so far is
 * returned (at worst the expression itself); a rewrite that runs out of its share of the work is skipped.
 *
 * @param e - The expression.
 * @param budget - Work limits (see {@link WorkBudget}).
 * @returns The simplest form found.
 * @example
 * ```ts
 * import { parse, simplify, toText } from 'whitebox-engine/symbolic'
 * toText(simplify(parse('(x^2 - 1)/(x - 1) + sin(x)^2 + cos(x)^2'))) // => 'x + 2'
 * toText(simplify(parse('(x^2 - 1)/(x - 1)'), { steps: 10 })) // => '(x**2 - 1)/(x - 1)'
 * ```
 */
export function simplify(e: Expr, budget?: WorkBudget): Expr {
  return simplifyWithin(e, budget).result
}

/**
 * {@link simplify} with the chain of rules that improved the expression.
 *
 * @param e - The expression.
 * @param budget - Work limits, as for simplify.
 * @returns The result and its steps.
 * @example
 * ```ts
 * import { parse, simplifySteps } from 'whitebox-engine/symbolic'
 * simplifySteps(parse('(x^2 - 1)/(x - 1)')).steps.map((s) => s.rule) // => ['cancel']
 * ```
 */
export function simplifySteps(e: Expr, budget?: WorkBudget): Derivation {
  return simplifyWithin(e, budget)
}

/**
 * {@link trigSimplify} with the rewrites that led to the result.
 *
 * @param e - The expression.
 * @returns The result and its steps.
 * @example
 * ```ts
 * import { parse, trigSimplifySteps } from 'whitebox-engine/symbolic'
 * trigSimplifySteps(parse('tan(x)*cos(x)')).steps.length > 0 // => true
 * ```
 */
export function trigSimplifySteps(e: Expr): Derivation {
  const steps: Step[] = []
  return withBudget(
    undefined,
    () => ({ result: trigSimplifyExpr(e, steps), steps }),
    () => ({ result: e, steps: [] }),
  )
}

/** The rewrites of this module, by name, for {@link rewriteSteps}. */
export type Rewrite = 'expand' | 'factor' | 'collect' | 'together' | 'cancel' | 'apart' | 'trigSimplify' | 'simplify'

/**
 * Apply one of the rewrites by name and return its derivation: the detailed steps of expand, trigSimplify and
 * simplify, or a single step named after the rewrite for factor, collect, together, cancel and apart (none when the
 * expression is already in that form).
 *
 * @param rule - The rewrite.
 * @param e - The expression.
 * @param x - The variable for collect and apart (the symbol x by default).
 * @returns The result and its steps.
 * @example
 * ```ts
 * import { parse, rewriteSteps } from 'whitebox-engine/symbolic'
 * rewriteSteps('factor', parse('x^2 - 1')).steps.map((s) => s.rule) // => ['factor']
 * ```
 */
export function rewriteSteps(rule: Rewrite, e: Expr, x: ExprSym = sym('x')): Derivation {
  if (rule === 'expand') return expandSteps(e)
  if (rule === 'trigSimplify') return trigSimplifySteps(e)
  if (rule === 'simplify') return simplifySteps(e)
  const rewrites: Readonly<Record<string, () => Expr>> = {
    factor: () => factor(e),
    collect: () => collect(e, x),
    together: () => togetherRaw(e),
    cancel: () => cancelRaw(e),
    apart: () => apartRaw(e, x),
  }
  const result = (rewrites[rule] as () => Expr)()
  return { result, steps: result.key === e.key ? [] : [{ rule, before: e, after: result }] }
}
