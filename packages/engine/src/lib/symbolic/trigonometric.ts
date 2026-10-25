/**
 * Trigonometric and hyperbolic functions on symbolic expressions, with their inverses and `atan2`: sine, cosine and
 * tangent are exact at rational multiples of pi with small denominators, and the rest evaluate exactly where they can
 * and stay symbolic otherwise.
 */

import { makeFn } from './_functions.ts'
import { asExpr, type Expr, type ExprLike } from './expr.ts'

const apply =
  (name: string) =>
  (x: ExprLike): Expr =>
    makeFn(name, [asExpr(x)])

/**
 * Sine, exact at rational multiples of π with denominators 1, 2, 3, 4, 6 and 12.
 *
 * @param x - The argument.
 * @returns sin(x).
 * @example
 * ```ts
 * import { mul, pi, rational, sin, toText } from 'mdml-engine/symbolic'
 * toText(sin(mul(rational(1, 6), pi))) // => '1/2'
 * ```
 */
export const sin: (x: ExprLike) => Expr = apply('sin')

/**
 * Cosine.
 *
 * @param x - The argument.
 * @returns cos(x).
 * @example
 * ```ts
 * import { cos, mul, pi, rational, toText } from 'mdml-engine/symbolic'
 * toText(cos(mul(rational(1, 4), pi))) // => 'sqrt(2)/2'
 * ```
 */
export const cos: (x: ExprLike) => Expr = apply('cos')

/**
 * Tangent.
 *
 * @param x - The argument.
 * @returns tan(x).
 * @example
 * ```ts
 * import { mul, pi, rational, tan, toText } from 'mdml-engine/symbolic'
 * toText(tan(mul(rational(1, 3), pi))) // => 'sqrt(3)'
 * ```
 */
export const tan: (x: ExprLike) => Expr = apply('tan')

/**
 * Cotangent.
 *
 * @param x - The argument.
 * @returns cot(x).
 * @example
 * ```ts
 * import { cot, mul, pi, rational, toText } from 'mdml-engine/symbolic'
 * toText(cot(mul(rational(1, 4), pi))) // => '1'
 * ```
 */
export const cot: (x: ExprLike) => Expr = apply('cot')

/**
 * Secant.
 *
 * @param x - The argument.
 * @returns sec(x).
 * @example
 * ```ts
 * import { sec, toText } from 'mdml-engine/symbolic'
 * toText(sec(0)) // => '1'
 * ```
 */
export const sec: (x: ExprLike) => Expr = apply('sec')

/**
 * Cosecant.
 *
 * @param x - The argument.
 * @returns csc(x).
 * @example
 * ```ts
 * import { csc, sym, toText } from 'mdml-engine/symbolic'
 * toText(csc(sym('x'))) // => 'csc(x)'
 * ```
 */
export const csc: (x: ExprLike) => Expr = apply('csc')

/**
 * Inverse sine (principal branch).
 *
 * @param x - The argument.
 * @returns asin(x).
 * @example
 * ```ts
 * import { asin, rational, toText } from 'mdml-engine/symbolic'
 * toText(asin(rational(1, 2))) // => 'pi/6'
 * ```
 */
export const asin: (x: ExprLike) => Expr = apply('asin')

/**
 * Inverse cosine (principal branch).
 *
 * @param x - The argument.
 * @returns acos(x).
 * @example
 * ```ts
 * import { acos, toText } from 'mdml-engine/symbolic'
 * toText(acos(-1)) // => 'pi'
 * ```
 */
export const acos: (x: ExprLike) => Expr = apply('acos')

/**
 * Inverse tangent.
 *
 * @param x - The argument.
 * @returns atan(x).
 * @example
 * ```ts
 * import { atan, toText } from 'mdml-engine/symbolic'
 * toText(atan(1)) // => 'pi/4'
 * ```
 */
export const atan: (x: ExprLike) => Expr = apply('atan')

/**
 * Inverse cotangent.
 *
 * @param x - The argument.
 * @returns acot(x).
 * @example
 * ```ts
 * import { acot, toText } from 'mdml-engine/symbolic'
 * toText(acot(0)) // => 'pi/2'
 * ```
 */
export const acot: (x: ExprLike) => Expr = apply('acot')

/**
 * Two-argument inverse tangent of y/x, in (-π, π].
 *
 * @param y - Ordinate.
 * @param x - Abscissa.
 * @returns atan2(y, x).
 * @example
 * ```ts
 * import { atan2, toText } from 'mdml-engine/symbolic'
 * toText(atan2(1, 1)) // => 'pi/4'
 * ```
 */
export function atan2(y: ExprLike, x: ExprLike): Expr {
  return makeFn('atan2', [asExpr(y), asExpr(x)])
}

/**
 * Hyperbolic sine.
 *
 * @param x - The argument.
 * @returns sinh(x).
 * @example
 * ```ts
 * import { sinh, toText } from 'mdml-engine/symbolic'
 * toText(sinh(0)) // => '0'
 * ```
 */
export const sinh: (x: ExprLike) => Expr = apply('sinh')

/**
 * Hyperbolic cosine.
 *
 * @param x - The argument.
 * @returns cosh(x).
 * @example
 * ```ts
 * import { cosh, neg, sym, toText } from 'mdml-engine/symbolic'
 * toText(cosh(neg(sym('x')))) // => 'cosh(x)'
 * ```
 */
export const cosh: (x: ExprLike) => Expr = apply('cosh')

/**
 * Hyperbolic tangent.
 *
 * @param x - The argument.
 * @returns tanh(x).
 * @example
 * ```ts
 * import { tanh, toText } from 'mdml-engine/symbolic'
 * toText(tanh(0)) // => '0'
 * ```
 */
export const tanh: (x: ExprLike) => Expr = apply('tanh')

/**
 * Hyperbolic cotangent.
 *
 * @param x - The argument.
 * @returns coth(x).
 * @example
 * ```ts
 * import { coth, sym, toText } from 'mdml-engine/symbolic'
 * toText(coth(sym('x'))) // => 'coth(x)'
 * ```
 */
export const coth: (x: ExprLike) => Expr = apply('coth')

/**
 * Inverse hyperbolic sine.
 *
 * @param x - The argument.
 * @returns asinh(x).
 * @example
 * ```ts
 * import { asinh, toText } from 'mdml-engine/symbolic'
 * toText(asinh(0)) // => '0'
 * ```
 */
export const asinh: (x: ExprLike) => Expr = apply('asinh')

/**
 * Inverse hyperbolic cosine.
 *
 * @param x - The argument.
 * @returns acosh(x).
 * @example
 * ```ts
 * import { acosh, toText } from 'mdml-engine/symbolic'
 * toText(acosh(1)) // => '0'
 * ```
 */
export const acosh: (x: ExprLike) => Expr = apply('acosh')

/**
 * Inverse hyperbolic tangent.
 *
 * @param x - The argument.
 * @returns atanh(x).
 * @example
 * ```ts
 * import { atanh, toText } from 'mdml-engine/symbolic'
 * toText(atanh(0)) // => '0'
 * ```
 */
export const atanh: (x: ExprLike) => Expr = apply('atanh')
