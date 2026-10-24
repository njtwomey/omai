import { WhiteboxError } from '../base/index.ts'
import { MINUS_ONE, makeMul, makePow, numFromRational } from './_canon.ts'
import { makeFn } from './_functions.ts'
import { rat } from './_rational.ts'
import { asExpr, type Expr, type ExprLike } from './expr.ts'

/**
 * Apply a function by name. Known names (`sin`, `exp`, `Abs`, `gamma`, ...) are evaluated where a rule applies;
 * any other name is an undefined function such as `f(x)`, which differentiates to an unevaluated derivative.
 *
 * @param name - The function name (an identifier).
 * @param args - The arguments.
 * @returns The application.
 * @throws WhiteboxError with code `value` for an invalid name.
 * @example
 * ```ts
 * import { fn, sym, toText } from 'whitebox-engine/symbolic'
 * toText(fn('f', sym('x'), 2)) // => 'f(x, 2)'
 * toText(fn('sin', 0)) // => '0'
 * ```
 */
export function fn(name: string, ...args: readonly ExprLike[]): Expr {
  if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(name)) throw new WhiteboxError('value', `fn: '${name}' is not a function name`)
  return makeFn(name, args.map(asExpr))
}

/**
 * The exponential function.
 *
 * @param x - The argument.
 * @returns exp(x), with exp(0) = 1, exp(log(y)) = y and exp(n log y) = y^n.
 * @example
 * ```ts
 * import { exp, log, sym, toText } from 'whitebox-engine/symbolic'
 * toText(exp(log(sym('y')))) // => 'y'
 * ```
 */
export function exp(x: ExprLike): Expr {
  return makeFn('exp', [asExpr(x)])
}

/**
 * The natural logarithm, or the logarithm to a base as log(x)/log(base) (as SymPy's `log(x, b)`).
 *
 * @param x - The argument.
 * @param base - Optional base.
 * @returns The logarithm.
 * @example
 * ```ts
 * import { E, log, toText } from 'whitebox-engine/symbolic'
 * toText(log(E)) // => '1'
 * toText(log(8, 2)) // => 'log(8)/log(2)'
 * ```
 */
export function log(x: ExprLike, base?: ExprLike): Expr {
  const l = makeFn('log', [asExpr(x)])
  if (base === undefined) return l
  return makeMul([l, makePow(makeFn('log', [asExpr(base)]), MINUS_ONE)])
}

/**
 * The principal n-th root x^(1/n).
 *
 * @param x - The radicand.
 * @param n - The index (a positive integer).
 * @returns x^(1/n).
 * @example
 * ```ts
 * import { root, toText } from 'whitebox-engine/symbolic'
 * toText(root(16, 4)) // => '2'
 * ```
 */
export function root(x: ExprLike, n: number): Expr {
  if (!Number.isInteger(n) || n < 1) throw new WhiteboxError('value', 'root: the index must be a positive integer')
  return makePow(asExpr(x), numFromRational(rat(1n, BigInt(n))))
}

/**
 * Absolute value (SymPy's `Abs`).
 *
 * @param x - The argument.
 * @returns |x|.
 * @example
 * ```ts
 * import { abs, mul, sym, toText } from 'whitebox-engine/symbolic'
 * toText(abs(mul(-3, sym('x')))) // => '3*Abs(x)'
 * ```
 */
export function abs(x: ExprLike): Expr {
  return makeFn('Abs', [asExpr(x)])
}

/**
 * The sign function: -1, 0 or 1 for reals.
 *
 * @param x - The argument.
 * @returns sign(x).
 * @example
 * ```ts
 * import { sign, toText } from 'whitebox-engine/symbolic'
 * toText(sign(-5)) // => '-1'
 * ```
 */
export function sign(x: ExprLike): Expr {
  return makeFn('sign', [asExpr(x)])
}

/**
 * Floor (the greatest integer not above x).
 *
 * @param x - The argument.
 * @returns floor(x).
 * @example
 * ```ts
 * import { floor, rational, toText } from 'whitebox-engine/symbolic'
 * toText(floor(rational(-7, 2))) // => '-4'
 * ```
 */
export function floor(x: ExprLike): Expr {
  return makeFn('floor', [asExpr(x)])
}

/**
 * Ceiling (the least integer not below x).
 *
 * @param x - The argument.
 * @returns ceiling(x).
 * @example
 * ```ts
 * import { ceiling, rational, toText } from 'whitebox-engine/symbolic'
 * toText(ceiling(rational(7, 2))) // => '4'
 * ```
 */
export function ceiling(x: ExprLike): Expr {
  return makeFn('ceiling', [asExpr(x)])
}

/**
 * Factorial (exact for integers up to 1000).
 *
 * @param x - The argument.
 * @returns x!.
 * @example
 * ```ts
 * import { factorial, toText } from 'whitebox-engine/symbolic'
 * toText(factorial(5)) // => '120'
 * ```
 */
export function factorial(x: ExprLike): Expr {
  return makeFn('factorial', [asExpr(x)])
}

/**
 * The gamma function (exact at integers and half-integers).
 *
 * @param x - The argument.
 * @returns Γ(x).
 * @example
 * ```ts
 * import { gamma, rational, toText } from 'whitebox-engine/symbolic'
 * toText(gamma(rational(1, 2))) // => 'sqrt(pi)'
 * ```
 */
export function gamma(x: ExprLike): Expr {
  return makeFn('gamma', [asExpr(x)])
}

/**
 * The error function (numeric evaluation needs an implementation passed to `evaluate` or `compile`).
 *
 * @param x - The argument.
 * @returns erf(x).
 * @example
 * ```ts
 * import { erf, neg, sym, toText } from 'whitebox-engine/symbolic'
 * toText(erf(neg(sym('x')))) // => '-erf(x)'
 * ```
 */
export function erf(x: ExprLike): Expr {
  return makeFn('erf', [asExpr(x)])
}

/**
 * Real part.
 *
 * @param x - The argument.
 * @returns re(x) (x itself when x is known to be real).
 * @example
 * ```ts
 * import { re, sym, toText } from 'whitebox-engine/symbolic'
 * toText(re(sym('t', { real: true }))) // => 't'
 * ```
 */
export function re(x: ExprLike): Expr {
  return makeFn('re', [asExpr(x)])
}

/**
 * Imaginary part.
 *
 * @param x - The argument.
 * @returns im(x) (zero when x is known to be real).
 * @example
 * ```ts
 * import { im, toText } from 'whitebox-engine/symbolic'
 * toText(im(3)) // => '0'
 * ```
 */
export function im(x: ExprLike): Expr {
  return makeFn('im', [asExpr(x)])
}
