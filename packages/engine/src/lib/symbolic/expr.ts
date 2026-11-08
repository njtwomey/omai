/**
 * Symbolic expressions: the immutable expression tree in canonical form (exact rationals, floats, symbols with
 * assumptions, constants, sums, products, powers, function applications, relations, unevaluated derivatives and
 * integrals), its constructors and arithmetic, substitution and free symbols. Also the derivation and step types every
 * rewrite reports, and work budgets that bound symbolic operations deterministically.
 *
 * @index structure Immutable symbolic expression tree in canonical form.
 */

import { OmaiError } from '../base/index.ts'
import { bounded } from './_budget.ts'
import {
  E as CE,
  I as CI,
  NAN as CNAN,
  MINUS_ONE,
  makeAdd,
  makeMul,
  makePow,
  mapChildren,
  NEG_INF,
  numFromRational,
  OO,
  PI,
  ZOO,
} from './_canon.ts'
import { children, hasSymbol, rawFloat, rawSym } from './_node.ts'
import { rat, rfromDecimal } from './_rational.ts'

/** Assumptions attached to a symbol (as in SymPy's `Symbol('x', positive=True)`); unstated means unknown. */
export interface Assumptions {
  readonly real?: boolean
  readonly positive?: boolean
  readonly nonnegative?: boolean
  readonly integer?: boolean
  readonly nonzero?: boolean
}

/** Named constants: π, e, the imaginary unit, the infinities, NaN and the booleans. */
export type ConstName = 'pi' | 'E' | 'I' | 'oo' | '-oo' | 'zoo' | 'nan' | 'true' | 'false'

/** Relation operators, as in SymPy's `rel_op`. */
export type RelOp = '==' | '!=' | '<' | '<=' | '>' | '>='

interface Node {
  /** The structural key: two canonical expressions are equal exactly when their keys are equal. */
  readonly key: string
}

/** An exact rational number n / d (d > 0, coprime). */
export interface ExprNum extends Node {
  readonly kind: 'num'
  readonly n: bigint
  readonly d: bigint
}
/** An inexact (floating-point) number. */
export interface ExprFloat extends Node {
  readonly kind: 'float'
  readonly value: number
}
/** A symbol with optional assumptions. */
export interface ExprSym extends Node {
  readonly kind: 'sym'
  readonly name: string
  readonly assumptions: Assumptions
}
/** A named constant. */
export interface ExprConst extends Node {
  readonly kind: 'const'
  readonly name: ConstName
}
/** A sum of at least two terms (flattened, like terms collected). */
export interface ExprAdd extends Node {
  readonly kind: 'add'
  readonly args: readonly Expr[]
}
/** A product of at least two factors (flattened, numeric coefficient first, like bases collected). */
export interface ExprMul extends Node {
  readonly kind: 'mul'
  readonly args: readonly Expr[]
}
/** A power `base^exp` (square roots are powers with exponent 1/2). */
export interface ExprPow extends Node {
  readonly kind: 'pow'
  readonly base: Expr
  readonly exp: Expr
}
/** A function application: a known function (`sin`, `exp`, ...) or an undefined one (`f(x)`). */
export interface ExprFn extends Node {
  readonly kind: 'fn'
  readonly name: string
  readonly args: readonly Expr[]
}
/** A piecewise expression: the first piece whose condition holds. */
export interface ExprPiecewise extends Node {
  readonly kind: 'piecewise'
  readonly pieces: readonly (readonly [Expr, Expr])[]
}
/** A relation between two expressions (an equation or inequality). */
export interface ExprRel extends Node {
  readonly kind: 'rel'
  readonly op: RelOp
  readonly lhs: Expr
  readonly rhs: Expr
}
/** A logical connective over conditions. */
export interface ExprLogic extends Node {
  readonly kind: 'logic'
  readonly op: 'and' | 'or' | 'not'
  readonly args: readonly Expr[]
}
/** Landau's O(expr) as `variable` tends to `point` (the remainder of a series). */
export interface ExprOrder extends Node {
  readonly kind: 'order'
  readonly expr: Expr
  readonly variable: ExprSym
  readonly point: Expr
}
/** An unevaluated derivative, for undefined functions. */
export interface ExprDerivative extends Node {
  readonly kind: 'derivative'
  readonly expr: Expr
  readonly variables: readonly (readonly [ExprSym, number])[]
}
/** An unevaluated integral, returned when no rule applies. */
export interface ExprIntegral extends Node {
  readonly kind: 'integral'
  readonly expr: Expr
  readonly variable: ExprSym
  readonly bounds?: readonly [Expr, Expr]
}

/**
 * A symbolic expression: an immutable tree in canonical form. Constructors flatten sums and products, combine numeric
 * coefficients, collect like terms and equal bases, and evaluate functions at special values (as SymPy does on
 * construction), so structural equality is a comparison of keys.
 */
export type Expr =
  | ExprNum
  | ExprFloat
  | ExprSym
  | ExprConst
  | ExprAdd
  | ExprMul
  | ExprPow
  | ExprFn
  | ExprPiecewise
  | ExprRel
  | ExprLogic
  | ExprOrder
  | ExprDerivative
  | ExprIntegral

/** Anything accepted where an expression is expected: expressions, numbers (integers exact) and bigints. */
export type ExprLike = Expr | number | bigint

/** One rewrite in a derivation: a named rule taking `before` to `after`. */
export interface Step {
  readonly rule: string
  readonly before: Expr
  readonly after: Expr
}

/** The result of a rewrite together with the steps that produced it. */
export interface Derivation {
  readonly result: Expr
  readonly steps: readonly Step[]
}

/**
 * Limits on the work of one symbolic operation, counted in deterministic work units rather than time (one unit is
 * roughly one exact rational operation, one expression node built or one term of polynomial arithmetic), so that
 * any input returns after a bounded, machine-independent amount of work. Every operation has a budget; when it
 * runs out the operation returns its documented fallback (`integrate` the terms it finished plus an unevaluated
 * integral, `simplify` the best form found so far, `factor` its input) or throws a OmaiError with code
 * `convergence`. Raise the limits for inputs that need more; operations called inside another draw on its budget.
 */
export interface WorkBudget {
  /** Work units (default 2,000,000, a fraction of a second; ordinary inputs use from tens to about 100,000). */
  readonly steps?: number
  /** The most terms an intermediate expansion may have (default 2,000). */
  readonly terms?: number
}

/**
 * Run `f` with a work budget shared by every symbolic operation it calls (each draws on what is left), for example
 * to raise the limits for an expansion that needs more terms than the default allows. Operations with a fallback
 * (integrate, simplify, factor) still take it when they run out; others throw.
 *
 * @param budget - Work limits (see {@link WorkBudget}).
 * @param f - The computation.
 * @returns What `f` returns.
 * @throws OmaiError with code `convergence` when the budget runs out outside an operation with a fallback.
 * @example
 * ```ts
 * import { expand, parse, toText, withWorkBudget } from 'omai-engine/symbolic'
 * const big = parse('(a + b)*(c + d)*(e + f)*(g + h)*(i + j)*(k + l)*(m + n)*(o + p)*(q + r)*(s + t)*(u + v)')
 * expand(big) // => throws convergence
 * withWorkBudget({ terms: 4096 }, () => expand(big)).kind // => 'add'
 * ```
 */
export function withWorkBudget<T>(budget: WorkBudget, f: () => T): T {
  return bounded('withWorkBudget', f, budget)
}

/**
 * π.
 *
 * @example
 * ```ts
 * import { pi, sin, toText } from 'omai-engine/symbolic'
 * toText(sin(pi)) // => '0'
 * ```
 */
export const pi: Expr = PI
/**
 * Euler's number e (exp(1)).
 *
 * @example
 * ```ts
 * import { E, log, toText } from 'omai-engine/symbolic'
 * toText(log(E)) // => '1'
 * ```
 */
export const E: Expr = CE
/**
 * The imaginary unit, I^2 = -1.
 *
 * @example
 * ```ts
 * import { I, mul, toText } from 'omai-engine/symbolic'
 * toText(mul(I, I)) // => '-1'
 * ```
 */
export const I: Expr = CI
/**
 * Positive infinity.
 *
 * @example
 * ```ts
 * import { add, oo, toText } from 'omai-engine/symbolic'
 * toText(add(oo, 1)) // => 'oo'
 * ```
 */
export const oo: Expr = OO
/**
 * Complex infinity, the value of 1/0.
 *
 * @example
 * ```ts
 * import { div, toText } from 'omai-engine/symbolic'
 * toText(div(1, 0)) // => 'zoo'
 * ```
 */
export const zoo: Expr = ZOO
/**
 * Not a number, as from oo - oo or 0 * oo.
 *
 * @example
 * ```ts
 * import { oo, sub, toText } from 'omai-engine/symbolic'
 * toText(sub(oo, oo)) // => 'nan'
 * ```
 */
export const nan: Expr = CNAN

/**
 * Whether a value is an expression.
 *
 * @param x - Any value.
 * @returns True for expression nodes.
 * @example
 * ```ts
 * import { isExpr, sym } from 'omai-engine/symbolic'
 * isExpr(sym('x')) // => true
 * isExpr(2) // => false
 * ```
 */
export function isExpr(x: unknown): x is Expr {
  return typeof x === 'object' && x !== null && 'kind' in x && 'key' in x && typeof x.key === 'string'
}

/**
 * Convert a number, bigint or expression to an expression. Integers become exact; other numbers become floats.
 *
 * @param x - The value.
 * @returns The expression.
 * @throws OmaiError with code `value` for anything else.
 * @example
 * ```ts
 * import { asExpr, toText } from 'omai-engine/symbolic'
 * toText(asExpr(3)) // => '3'
 * toText(asExpr(0.5)) // => '0.5'
 * ```
 */
export function asExpr(x: ExprLike): Expr {
  if (typeof x === 'bigint') return numFromRational(rat(x))
  if (typeof x === 'number') {
    if (Number.isNaN(x)) return CNAN
    if (x === Number.POSITIVE_INFINITY) return OO
    if (x === Number.NEGATIVE_INFINITY) return NEG_INF
    return Number.isInteger(x) ? numFromRational(rat(BigInt(x))) : rawFloat(x)
  }
  if (isExpr(x)) return x
  throw new OmaiError('value', 'asExpr: expected an expression, number or bigint')
}

/**
 * A symbol.
 *
 * @param name - An identifier (letters, digits, `_` and `'`; Greek letters allowed).
 * @param assumptions - Optional assumptions such as `{ positive: true }`; symbols with different assumptions differ.
 * @returns The symbol.
 * @example
 * ```ts
 * import { sym, toText } from 'omai-engine/symbolic'
 * toText(sym('x')) // => 'x'
 * ```
 */
export function sym(name: string, assumptions: Assumptions = {}): ExprSym {
  const a: Assumptions = assumptions.positive
    ? { ...assumptions, real: true, nonnegative: true, nonzero: true }
    : assumptions.integer || assumptions.nonnegative
      ? { ...assumptions, real: true }
      : assumptions
  return rawSym(name, a)
}

/**
 * Several symbols from a space- or comma-separated list, as in SymPy's `symbols('x y z')`.
 *
 * @param names - The names.
 * @param assumptions - Assumptions shared by all of them.
 * @returns The symbols, in order.
 * @example
 * ```ts
 * import { symbols } from 'omai-engine/symbolic'
 * symbols('x, y z').map((s) => s.name) // => ['x', 'y', 'z']
 * ```
 */
export function symbols(names: string, assumptions: Assumptions = {}): ExprSym[] {
  return names
    .split(/[\s,]+/)
    .filter((s) => s.length > 0)
    .map((s) => sym(s, assumptions))
}

/**
 * An exact number from an integer, a bigint or a decimal string (`'0.25'` becomes 1/4).
 *
 * @param value - The value; a non-integer JavaScript number is converted through its shortest decimal form.
 * @returns The rational number.
 * @example
 * ```ts
 * import { num, toText } from 'omai-engine/symbolic'
 * toText(num('0.25')) // => '1/4'
 * toText(num(0.1)) // => '1/10'
 * ```
 */
export function num(value: number | bigint | string): Expr {
  if (typeof value === 'bigint') return numFromRational(rat(value))
  if (typeof value === 'string') return numFromRational(rfromDecimal(value))
  if (!Number.isFinite(value)) return asExpr(value)
  return numFromRational(rfromDecimal(String(value)))
}

/**
 * The exact rational p / q.
 *
 * @param p - Numerator (an integer).
 * @param q - Denominator (a non-zero integer).
 * @returns The normalised rational.
 * @throws OmaiError with code `value` for non-integers and `domain` for a zero denominator.
 * @example
 * ```ts
 * import { rational, toText } from 'omai-engine/symbolic'
 * toText(rational(6, -4)) // => '-3/2'
 * ```
 */
export function rational(p: number | bigint, q: number | bigint = 1): Expr {
  if ((typeof p === 'number' && !Number.isInteger(p)) || (typeof q === 'number' && !Number.isInteger(q))) {
    throw new OmaiError('value', 'rational: numerator and denominator must be integers')
  }
  return numFromRational(rat(BigInt(p), BigInt(q)))
}

/**
 * Sum of expressions, in canonical form.
 *
 * @param terms - The terms.
 * @returns Their sum.
 * @example
 * ```ts
 * import { add, sym, toText } from 'omai-engine/symbolic'
 * const x = sym('x')
 * toText(add(x, 1, x)) // => '2*x + 1'
 * ```
 */
export function add(...terms: readonly ExprLike[]): Expr {
  return makeAdd(terms.map(asExpr))
}

/**
 * Difference a - b.
 *
 * @param a - Minuend.
 * @param b - Subtrahend.
 * @returns a - b.
 * @example
 * ```ts
 * import { sub, sym, toText } from 'omai-engine/symbolic'
 * toText(sub(sym('x'), sym('y'))) // => 'x - y'
 * ```
 */
export function sub(a: ExprLike, b: ExprLike): Expr {
  return makeAdd([asExpr(a), makeMul([MINUS_ONE, asExpr(b)])])
}

/**
 * Product of expressions, in canonical form.
 *
 * @param factors - The factors.
 * @returns Their product.
 * @example
 * ```ts
 * import { mul, sym, toText } from 'omai-engine/symbolic'
 * const x = sym('x')
 * toText(mul(2, x, x)) // => '2*x**2'
 * ```
 */
export function mul(...factors: readonly ExprLike[]): Expr {
  return makeMul(factors.map(asExpr))
}

/**
 * Quotient a / b.
 *
 * @param a - Numerator.
 * @param b - Denominator.
 * @returns a * b^-1.
 * @example
 * ```ts
 * import { div, sym, toText } from 'omai-engine/symbolic'
 * toText(div(sym('x'), 2)) // => 'x/2'
 * ```
 */
export function div(a: ExprLike, b: ExprLike): Expr {
  return makeMul([asExpr(a), makePow(asExpr(b), MINUS_ONE)])
}

/**
 * Power a^b, with exact evaluation of numeric powers and extraction of perfect powers from radicals.
 *
 * @param base - The base.
 * @param exponent - The exponent.
 * @returns The power.
 * @example
 * ```ts
 * import { pow, rational, toText } from 'omai-engine/symbolic'
 * toText(pow(8, rational(1, 2))) // => '2*sqrt(2)'
 * ```
 */
export function pow(base: ExprLike, exponent: ExprLike): Expr {
  return makePow(asExpr(base), asExpr(exponent))
}

/**
 * Negation -a.
 *
 * @param a - The expression.
 * @returns -1 * a.
 * @example
 * ```ts
 * import { neg, sym, toText } from 'omai-engine/symbolic'
 * toText(neg(sym('x'))) // => '-x'
 * ```
 */
export function neg(a: ExprLike): Expr {
  return makeMul([MINUS_ONE, asExpr(a)])
}

/**
 * Square root, the power with exponent 1/2.
 *
 * @param a - The radicand.
 * @returns a^(1/2).
 * @example
 * ```ts
 * import { sqrt, toText } from 'omai-engine/symbolic'
 * toText(sqrt(12)) // => '2*sqrt(3)'
 * ```
 */
export function sqrt(a: ExprLike): Expr {
  return makePow(asExpr(a), numFromRational(rat(1n, 2n)))
}

/**
 * The free symbols of an expression (variables of definite integrals are bound), sorted by name.
 *
 * @param e - The expression.
 * @returns The symbols.
 * @example
 * ```ts
 * import { add, free, mul, sym } from 'omai-engine/symbolic'
 * free(add(mul(sym('y'), sym('x')), 1)).map((s) => s.name) // => ['x', 'y']
 * ```
 */
export function free(e: Expr): ExprSym[] {
  const seen = new Map<string, ExprSym>()
  const walk = (x: Expr, bound: readonly string[]): void => {
    if (x.kind === 'sym') {
      if (!bound.includes(x.key)) seen.set(x.key, x)
      return
    }
    if (x.kind === 'integral' && x.bounds !== undefined) {
      walk(x.expr, [...bound, x.variable.key])
      walk(x.bounds[0], bound)
      walk(x.bounds[1], bound)
      return
    }
    for (const c of children(x)) walk(c, bound)
  }
  walk(e, [])
  return [...seen.values()].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : a.key < b.key ? -1 : 1))
}

/**
 * Whether an expression depends on a symbol.
 *
 * @param e - The expression.
 * @param s - The symbol.
 * @returns True when `s` occurs in `e`.
 * @example
 * ```ts
 * import { dependsOn, sym } from 'omai-engine/symbolic'
 * dependsOn(sym('x'), sym('y')) // => false
 * ```
 */
export function dependsOn(e: Expr, s: ExprSym): boolean {
  return hasSymbol(e, s)
}

/**
 * Substitute subexpressions, then re-canonicalise (SymPy's `subs`). Keys of a record are symbol names; pairs replace
 * any subexpression structurally equal to their first element. Substitution is simultaneous.
 *
 * @param e - The expression.
 * @param replacements - A record from symbol names, or a list of `[target, value]` pairs.
 * @returns The substituted expression.
 * @example
 * ```ts
 * import { add, mul, substitute, sym, toText } from 'omai-engine/symbolic'
 * const x = sym('x')
 * toText(substitute(add(mul(x, x), x), { x: 3 })) // => '12'
 * ```
 */
export function substitute(
  e: Expr,
  replacements: Readonly<Record<string, ExprLike>> | readonly (readonly [Expr, ExprLike])[],
): Expr {
  const table = new Map<string, Expr>()
  const byName = new Map<string, Expr>()
  if (Array.isArray(replacements)) {
    for (const [target, value] of replacements as readonly (readonly [Expr, ExprLike])[]) {
      table.set(target.key, asExpr(value))
    }
  } else {
    for (const [name, value] of Object.entries(replacements as Readonly<Record<string, ExprLike>>)) {
      byName.set(name, asExpr(value))
    }
  }
  const go = (x: Expr): Expr => {
    const hit = table.get(x.key)
    if (hit !== undefined) return hit
    if (x.kind === 'sym') return byName.get(x.name) ?? x
    return mapChildren(x, go)
  }
  return go(e)
}

/**
 * Structural equality of canonical expressions (cheap: a key comparison). It does not prove mathematical equality;
 * for that, compare `simplify(sub(a, b))` with zero.
 *
 * @param a - First expression.
 * @param b - Second expression.
 * @returns True when the trees are identical.
 * @example
 * ```ts
 * import { add, equals, sym } from 'omai-engine/symbolic'
 * const [x, y] = [sym('x'), sym('y')]
 * equals(add(x, y), add(y, x)) // => true
 * ```
 */
export function equals(a: ExprLike, b: ExprLike): boolean {
  return asExpr(a).key === asExpr(b).key
}
