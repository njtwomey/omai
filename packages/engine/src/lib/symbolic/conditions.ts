/**
 * Conditions and piecewise expressions: equations and inequalities (SymPy's `Eq`, `Lt` and friends), which evaluate to
 * true or false when they can, the logical connectives `and`, `or` and `not` in simplified form, and `piecewise` from
 * value and condition pairs.
 */

import { MdmlError } from '../base/index.ts'
import { makeLogic, makePiecewise, makeRel, TRUE } from './_canon.ts'
import { asExpr, type Expr, type ExprLike, type RelOp } from './expr.ts'

const rel =
  (op: RelOp) =>
  (lhs: ExprLike, rhs: ExprLike): Expr =>
    makeRel(op, asExpr(lhs), asExpr(rhs))

/**
 * The equation lhs = rhs (SymPy's `Eq`); evaluates to true or false for numbers and identical sides.
 *
 * @param lhs - Left side.
 * @param rhs - Right side.
 * @returns The relation.
 * @example
 * ```ts
 * import { eq, sym, toText } from 'mdml-engine/symbolic'
 * toText(eq(sym('x'), 1)) // => 'Eq(x, 1)'
 * ```
 */
export const eq: (lhs: ExprLike, rhs: ExprLike) => Expr = rel('==')

/**
 * The relation lhs ≠ rhs.
 *
 * @param lhs - Left side.
 * @param rhs - Right side.
 * @returns The relation.
 * @example
 * ```ts
 * import { ne, toText } from 'mdml-engine/symbolic'
 * toText(ne(1, 2)) // => 'True'
 * ```
 */
export const ne: (lhs: ExprLike, rhs: ExprLike) => Expr = rel('!=')

/**
 * The relation lhs < rhs.
 *
 * @param lhs - Left side.
 * @param rhs - Right side.
 * @returns The relation.
 * @example
 * ```ts
 * import { lt, sym, toText } from 'mdml-engine/symbolic'
 * toText(lt(sym('x'), 0)) // => 'x < 0'
 * ```
 */
export const lt: (lhs: ExprLike, rhs: ExprLike) => Expr = rel('<')

/**
 * The relation lhs ≤ rhs.
 *
 * @param lhs - Left side.
 * @param rhs - Right side.
 * @returns The relation.
 * @example
 * ```ts
 * import { le, pi, toText } from 'mdml-engine/symbolic'
 * toText(le(3, pi)) // => 'True'
 * ```
 */
export const le: (lhs: ExprLike, rhs: ExprLike) => Expr = rel('<=')

/**
 * The relation lhs > rhs.
 *
 * @param lhs - Left side.
 * @param rhs - Right side.
 * @returns The relation.
 * @example
 * ```ts
 * import { gt, sym, toText } from 'mdml-engine/symbolic'
 * toText(gt(sym('x'), 0)) // => 'x > 0'
 * ```
 */
export const gt: (lhs: ExprLike, rhs: ExprLike) => Expr = rel('>')

/**
 * The relation lhs ≥ rhs.
 *
 * @param lhs - Left side.
 * @param rhs - Right side.
 * @returns The relation.
 * @example
 * ```ts
 * import { ge, sym, toText } from 'mdml-engine/symbolic'
 * toText(ge(sym('x'), 1)) // => 'x >= 1'
 * ```
 */
export const ge: (lhs: ExprLike, rhs: ExprLike) => Expr = rel('>=')

/**
 * Conjunction of conditions (flattened, true dropped, false absorbing).
 *
 * @param conditions - The conditions.
 * @returns Their conjunction.
 * @example
 * ```ts
 * import { and, gt, lt, sym, toText } from 'mdml-engine/symbolic'
 * const x = sym('x')
 * toText(and(gt(x, 0), lt(x, 1))) // => '(x > 0) & (x < 1)'
 * ```
 */
export function and(...conditions: readonly Expr[]): Expr {
  return makeLogic('and', conditions)
}

/**
 * Disjunction of conditions.
 *
 * @param conditions - The conditions.
 * @returns Their disjunction.
 * @example
 * ```ts
 * import { gt, lt, or, sym, toText } from 'mdml-engine/symbolic'
 * const x = sym('x')
 * toText(or(lt(x, -1), gt(x, 1))) // => '(x > 1) | (x < -1)'
 * ```
 */
export function or(...conditions: readonly Expr[]): Expr {
  return makeLogic('or', conditions)
}

/**
 * Negation of a condition (a relation is negated in place: not(x < 0) is x >= 0).
 *
 * @param condition - The condition.
 * @returns Its negation.
 * @example
 * ```ts
 * import { lt, not, sym, toText } from 'mdml-engine/symbolic'
 * toText(not(lt(sym('x'), 0))) // => 'x >= 0'
 * ```
 */
export function not(condition: Expr): Expr {
  return makeLogic('not', [condition])
}

/**
 * A piecewise expression from (value, condition) pairs; a final pair may use `true` (or omit the condition) for
 * "otherwise". Pieces after an always-true condition are dropped.
 *
 * @param pieces - Pairs `[value, condition]`, or a bare value for the last piece.
 * @returns The piecewise expression.
 * @throws MdmlError with code `value` when no pieces are given.
 * @example
 * ```ts
 * import { gt, neg, piecewise, sym, toText } from 'mdml-engine/symbolic'
 * const x = sym('x')
 * toText(piecewise([x, gt(x, 0)], neg(x))) // => 'Piecewise((x, x > 0), (-x, True))'
 * ```
 */
export function piecewise(...pieces: readonly (readonly [ExprLike, Expr] | ExprLike)[]): Expr {
  if (pieces.length === 0) throw new MdmlError('value', 'piecewise: no pieces')
  return makePiecewise(
    pieces.map((p) => (Array.isArray(p) ? [asExpr(p[0] as ExprLike), p[1] as Expr] : [asExpr(p as ExprLike), TRUE])),
  )
}
