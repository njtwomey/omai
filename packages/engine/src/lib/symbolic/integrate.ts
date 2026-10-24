import { WhiteboxError } from '../base/index.ts'
import { isInfinite, MINUS_ONE, makeAdd, makeIntegral, makeMul, makePow, NAN, numFromFloat } from './_canon.ts'
import { expandExpr } from './_expand.ts'
import { isRationalIn, rationalIntegral, reductionIntegral, table, trigIntegral } from './_integrals.ts'
import { numericallyEqual } from './_measure.ts'
import { children, hasSymbol } from './_node.ts'
import { evalReal, freeNames, type NumericFunctions, quadrature } from './_numeric.ts'
import { cancel, numerDenom } from './_rational-functions.ts'
import { diff } from './differentiate.ts'
import { asExpr, type Derivation, type Expr, type ExprLike, type ExprSym, type Step, substitute, sym } from './expr.ts'
import { toText } from './latex.ts'
import { limit } from './limits.ts'
import { countRealRoots } from './polynomials.ts'

const MAX_DEPTH = 8

interface Ctx {
  readonly steps: Step[] | undefined
  depth: number
}

function record(ctx: Ctx, rule: string, e: Expr, x: ExprSym, F: Expr): Expr {
  if (ctx.steps !== undefined) ctx.steps.push({ rule, before: makeIntegral(e, x), after: F })
  return F
}

/** LIATE priority of a factor as the part to differentiate (higher first); -1 when it should not be. */
function liate(f: Expr, x: ExprSym): number {
  const [b] = f.kind === 'pow' ? [f.base] : [f]
  if (b.kind === 'fn' && b.name === 'log') return 5
  if (b.kind === 'fn' && ['asin', 'acos', 'atan', 'acot', 'asinh', 'acosh', 'atanh', 'erf'].includes(b.name)) return 4
  if (
    f.key === x.key ||
    (f.kind === 'pow' && f.base.key === x.key && f.exp.kind === 'num' && f.exp.d === 1n && f.exp.n > 0n)
  )
    return 3
  if (b.kind === 'fn' && ['sin', 'cos', 'sinh', 'cosh'].includes(b.name)) return 2
  if (b.kind === 'fn' && b.name === 'exp') return 1
  return -1
}

function candidates(e: Expr, x: ExprSym): Expr[] {
  const out = new Map<string, Expr>()
  const walk = (y: Expr): void => {
    for (const c of children(y)) {
      if (hasSymbol(c, x) && c.key !== x.key && c.kind !== 'add') out.set(c.key, c)
      if (hasSymbol(c, x) && c.kind === 'add') out.set(c.key, c)
      walk(c)
    }
  }
  walk(e)
  if (e.kind === 'fn' || e.kind === 'pow') out.set(e.key, e)
  return [...out.values()].sort((a, b) => b.key.length - a.key.length)
}

/** Derivative-divides substitution: e = h(g(x)) g'(x) gives ∫ h(U) dU at U = g(x). */
function substitution(e: Expr, x: ExprSym, ctx: Ctx): Expr | undefined {
  const U = sym(`u_sub${ctx.depth}__`)
  for (const g of candidates(e, x)) {
    const dg = diff(g, x)
    if (dg.kind === 'num' && dg.n === 0n) continue
    const quotient = cancel(makeMul([e, makePow(dg, MINUS_ONE)]))
    const inU = substitute(quotient, [[g, U]])
    if (hasSymbol(inU, x)) continue
    if (inU.key === e.key) continue
    const F = core(inU, U, ctx)
    if (F === undefined) continue
    return record(ctx, `substitution u = ${toText(g)}`, e, x, substitute(F, [[U, g]]))
  }
  return undefined
}

function byParts(e: Expr, x: ExprSym, ctx: Ctx): Expr | undefined {
  const factors = e.kind === 'mul' ? e.args : [e]
  const ranked = factors
    .map((f, i) => ({ f, i, p: liate(f, x) }))
    .filter((r) => r.p >= 0)
    .sort((a, b) => b.p - a.p)
  for (const { f, i } of ranked) {
    const dv = makeMul(factors.filter((_, j) => j !== i))
    if (dv.kind === 'num' && factors.length === 1 && liate(f, x) < 4) continue
    const v = core(dv, x, ctx)
    if (v === undefined) continue
    const rest = core(makeMul([v, diff(f, x)]), x, ctx)
    if (rest === undefined) continue
    return record(ctx, 'integration by parts', e, x, makeAdd([makeMul([f, v]), makeMul([MINUS_ONE, rest])]))
  }
  return undefined
}

function core(e: Expr, x: ExprSym, ctx: Ctx): Expr | undefined {
  if (!hasSymbol(e, x)) return record(ctx, 'constant', e, x, makeMul([e, x]))
  if (ctx.depth > MAX_DEPTH) return undefined
  ctx.depth++
  try {
    return attempt(e, x, ctx)
  } finally {
    ctx.depth--
  }
}

function verified(F: Expr | undefined, e: Expr, x: ExprSym): Expr | undefined {
  if (F === undefined) return undefined
  return numericallyEqual(diff(F, x), e, 1e-7) ? F : undefined
}

function attempt(e: Expr, x: ExprSym, ctx: Ctx): Expr | undefined {
  // Linearity.
  if (e.kind === 'add') {
    const parts: Expr[] = []
    for (const t of e.args) {
      const F = core(t, x, ctx)
      if (F === undefined) return undefined
      parts.push(F)
    }
    return record(ctx, 'sum rule', e, x, makeAdd(parts))
  }
  if (e.kind === 'mul') {
    const constant = e.args.filter((f) => !hasSymbol(f, x))
    if (constant.length > 0) {
      const body = makeMul(e.args.filter((f) => hasSymbol(f, x)))
      const F = core(body, x, ctx)
      return F === undefined ? undefined : record(ctx, 'constant multiple', e, x, makeMul([...constant, F]))
    }
  }
  const t = table(e, x)
  if (verified(t, e, x) !== undefined) return record(ctx, 'table', e, x, t as Expr)
  const methods: [string, () => Expr | undefined][] = [
    ['partial fractions', () => rationalIntegral(e, x)],
    ['trigonometric powers', () => trigIntegral(e, x, (f, u) => core(f, u, ctx))],
    ['reduction formula', () => reductionIntegral(e, x, (f) => core(f, x, ctx))],
  ]
  for (const [rule, m] of methods) {
    let F: Expr | undefined
    try {
      F = m()
    } catch (err) {
      if (!(err instanceof WhiteboxError)) throw err
      F = undefined
    }
    if (verified(F, e, x) !== undefined) return record(ctx, rule, e, x, F as Expr)
  }
  const sub = verified(substitution(e, x, ctx), e, x)
  if (sub !== undefined) return sub
  const parts = verified(byParts(e, x, ctx), e, x)
  if (parts !== undefined) return parts
  // Expand products of sums and powers of sums, then retry.
  const ex = expandExpr(e)
  if (ex.key !== e.key && ex.kind === 'add') {
    const F = core(ex, x, ctx)
    if (F !== undefined) return record(ctx, 'expand', e, x, F)
  }
  if (!isRationalIn(e, x)) {
    const cancelled = cancel(e)
    if (cancelled.key !== e.key && cancelled.key !== ex.key) {
      const F = core(cancelled, x, ctx)
      if (F !== undefined) return record(ctx, 'cancel', e, x, F)
    }
  }
  return undefined
}

function run(e: Expr, x: ExprSym, steps: Step[] | undefined): Expr {
  const ctx: Ctx = { steps, depth: 0 }
  // Integrate term by term so that what fails stays as an unevaluated integral of its own.
  const ex = e.kind === 'add' ? e : expandExpr(e)
  const terms = ex.kind === 'add' && e.kind === 'add' ? ex.args : [e]
  const done: Expr[] = []
  const left: Expr[] = []
  for (const t of terms) {
    const F = core(t, x, ctx)
    if (F === undefined) left.push(t)
    else done.push(F)
  }
  if (left.length > 0) done.push(makeIntegral(makeAdd(left), x))
  return makeAdd(done)
}

/**
 * An antiderivative (SymPy's `integrate(f, x)`), by a heuristic integrator: linearity, a table of elementary
 * integrals with linear arguments (powers, exponentials, trigonometric, hyperbolic and inverse functions,
 * logarithms, Gaussians via erf, quadratic radicals), rational functions by partial fractions (logarithms,
 * arctangents and the reduction formula for repeated quadratic factors), powers and products of sines and cosines,
 * reduction formulas for tan, cot, sec and csc, derivative-divides substitution and integration by parts (LIATE).
 * Every candidate is checked by differentiating it. What no rule handles is returned as an unevaluated `Integral`.
 * TODO: the full Risch algorithm, and Rothstein-Trager for rational functions with irreducible denominators of
 * degree above two.
 *
 * @param e - The integrand.
 * @param x - The variable.
 * @returns The antiderivative, possibly containing an unevaluated integral.
 * @example
 * ```ts
 * import { integrate, parse, sym, toText } from 'whitebox-engine/symbolic'
 * const x = sym('x')
 * toText(integrate(parse('x*exp(x)'), x)) // => 'x*exp(x) - exp(x)'
 * toText(integrate(parse('1/(x^2 + 1)'), x)) // => 'atan(x)'
 * toText(integrate(parse('exp(x^2)'), x)) // => 'Integral(exp(x**2), x)'
 * ```
 */
export function integrate(e: Expr, x: ExprSym): Expr {
  return run(e, x, undefined)
}

/**
 * {@link integrate} with the rules applied (each step takes an unevaluated integral to its antiderivative).
 *
 * @param e - The integrand.
 * @param x - The variable.
 * @returns The antiderivative and its steps.
 * @example
 * ```ts
 * import { integrateSteps, parse, sym } from 'whitebox-engine/symbolic'
 * integrateSteps(parse('x*cos(x)'), sym('x')).steps.some((s) => s.rule === 'integration by parts') // => true
 * ```
 */
export function integrateSteps(e: Expr, x: ExprSym): Derivation {
  const steps: Step[] = []
  const result = run(e, x, steps)
  return { result, steps }
}

function containsIntegral(e: Expr): boolean {
  return e.kind === 'integral' || children(e).some(containsIntegral)
}

function numericValue(e: Expr): number | undefined {
  if (freeNames(e).length > 0) return undefined
  try {
    return evalReal(e, {})
  } catch {
    return undefined
  }
}

/** Options for {@link definiteIntegral}. */
export interface DefiniteOptions {
  /** Numeric implementations of functions (such as erf) for the numeric check and fallback. */
  readonly functions?: NumericFunctions
  /** Relative tolerance of the agreement between the symbolic value and quadrature (default 1e-6). */
  readonly tol?: number
}

function definite(
  e: Expr,
  x: ExprSym,
  av: ExprLike,
  bv: ExprLike,
  options: DefiniteOptions,
  steps: Step[] | undefined,
): Expr {
  const a = asExpr(av)
  const b = asExpr(bv)
  const fns = options.functions ?? {}
  const tol = options.tol ?? 1e-6
  const numericBounds = numericValue(a) !== undefined && numericValue(b) !== undefined
  const onlyX = freeNames(e).every((n) => n === x.name)
  // Poles of a rational integrand inside the interval make it divergent.
  if (numericBounds && onlyX && isRationalIn(e, x)) {
    const [, d] = numerDenom(cancel(e))
    if (hasSymbol(d, x)) {
      const lo = Math.min(numericValue(a) as number, numericValue(b) as number)
      const hi = Math.max(numericValue(a) as number, numericValue(b) as number)
      const k = countRealRoots(
        d,
        x,
        Number.isFinite(lo) ? lo : Number.NEGATIVE_INFINITY,
        Number.isFinite(hi) ? hi : Number.POSITIVE_INFINITY,
      )
      if (k > 0) throw new WhiteboxError('domain', 'definiteIntegral: the integrand has a pole in the interval')
    }
  }
  const quad = (): number | undefined => {
    if (!numericBounds || !onlyX) return undefined
    const lo = numericValue(a) as number
    const hi = numericValue(b) as number
    const v = quadrature((t) => evalReal(e, { [x.name]: t }, fns), lo, hi)
    return Number.isFinite(v) ? v : undefined
  }
  const F = run(e, x, steps)
  let value: Expr | undefined
  if (!containsIntegral(F)) {
    try {
      const upper = limit(F, x, b, isInfinite(b) ? '+' : '-')
      const lower = limit(F, x, a, '+')
      value = cancel(makeAdd([upper, makeMul([MINUS_ONE, lower])]))
      if (value.key === NAN.key) value = undefined
      if (steps !== undefined && value !== undefined) {
        steps.push({
          rule: 'fundamental theorem of calculus: F(b) - F(a)',
          before: makeIntegral(e, x, [a, b]),
          after: value,
        })
      }
    } catch (err) {
      if (!(err instanceof WhiteboxError)) throw err
      value = undefined
    }
  }
  const numeric = (() => {
    try {
      return quad()
    } catch {
      return undefined
    }
  })()
  if (value !== undefined && numeric !== undefined) {
    const sv = numericValue(value)
    if (sv !== undefined && Math.abs(sv - numeric) <= tol * Math.max(1, Math.abs(numeric))) return value
    if (sv !== undefined && !Number.isFinite(sv)) return value
    // The antiderivative is discontinuous on the interval (or a rule misfired): trust quadrature.
    if (steps !== undefined)
      steps.push({
        rule: 'numeric fallback (antiderivative check failed)',
        before: makeIntegral(e, x, [a, b]),
        after: numFromFloat(numeric),
      })
    return numFromFloat(numeric)
  }
  if (value !== undefined) return value
  if (numeric !== undefined) {
    if (steps !== undefined)
      steps.push({
        rule: 'numeric fallback (adaptive Gauss-Kronrod)',
        before: makeIntegral(e, x, [a, b]),
        after: numFromFloat(numeric),
      })
    return numFromFloat(numeric)
  }
  return makeIntegral(e, x, [a, b])
}

/**
 * A definite integral: F(b) - F(a) from the antiderivative (one-sided limits at the ends, so infinite bounds work),
 * checked against adaptive Gauss-Kronrod quadrature when the bounds are numbers; when there is no closed form, or
 * the check fails (an antiderivative discontinuous on the interval), the numeric value is returned as a float.
 *
 * @param e - The integrand.
 * @param x - The variable.
 * @param a - Lower bound (may be -oo).
 * @param b - Upper bound (may be oo).
 * @param options - Numeric implementations for the check and the fallback, and its tolerance.
 * @returns The value: exact when possible, a float from quadrature otherwise, or an unevaluated integral.
 * @throws WhiteboxError with code `domain` when a rational integrand has a pole in the interval.
 * @example
 * ```ts
 * import { definiteIntegral, oo, parse, sym, toText } from 'whitebox-engine/symbolic'
 * const x = sym('x')
 * toText(definiteIntegral(parse('x^2'), x, 0, 3)) // => '9'
 * toText(definiteIntegral(parse('exp(-x)'), x, 0, oo)) // => '1'
 * ```
 */
export function definiteIntegral(e: Expr, x: ExprSym, a: ExprLike, b: ExprLike, options: DefiniteOptions = {}): Expr {
  return definite(e, x, a, b, options, undefined)
}

/**
 * {@link definiteIntegral} with its steps.
 *
 * @param e - The integrand.
 * @param x - The variable.
 * @param a - Lower bound.
 * @param b - Upper bound.
 * @param options - As for definiteIntegral.
 * @returns The value and its steps.
 * @example
 * ```ts
 * import { definiteIntegralSteps, parse, sym } from 'whitebox-engine/symbolic'
 * definiteIntegralSteps(parse('exp(x^2)'), sym('x'), 0, 1).steps.at(-1)?.rule // => 'numeric fallback (adaptive Gauss-Kronrod)'
 * ```
 */
export function definiteIntegralSteps(
  e: Expr,
  x: ExprSym,
  a: ExprLike,
  b: ExprLike,
  options: DefiniteOptions = {},
): Derivation {
  const steps: Step[] = []
  const result = definite(e, x, a, b, options, steps)
  return { result, steps }
}
