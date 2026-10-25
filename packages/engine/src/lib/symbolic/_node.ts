// Raw expression nodes: construction without canonicalisation, the structural key, and small accessors. The
// canonicalising constructors live in _canon.ts; everything public goes through them.
import { MdmlError } from '../base/index.ts'
import { charge } from './_budget.ts'
import type { Rational } from './_rational.ts'
import type {
  Assumptions,
  ConstName,
  Expr,
  ExprAdd,
  ExprConst,
  ExprDerivative,
  ExprFloat,
  ExprFn,
  ExprIntegral,
  ExprLogic,
  ExprMul,
  ExprNum,
  ExprOrder,
  ExprPiecewise,
  ExprPow,
  ExprRel,
  ExprSym,
  RelOp,
} from './expr.ts'

const NAME = /^[A-Za-zͰ-Ͽ][A-Za-z0-9_'Ͱ-Ͽ]*$/

function joinKeys(xs: readonly Expr[]): string {
  let s = ''
  for (let i = 0; i < xs.length; i++) s += (i === 0 ? '' : ',') + (xs[i] as Expr).key
  return s
}

/** Order expressions by key (the internal canonical argument order). @internal */
export function byKey(a: Expr, b: Expr): number {
  return a.key < b.key ? -1 : a.key > b.key ? 1 : 0
}

/** A rational number node. @internal */
export function rawNum(r: Rational): ExprNum {
  return { kind: 'num', n: r.n, d: r.d, key: `#${r.n}/${r.d}` }
}

/** A floating-point number node. @internal */
export function rawFloat(value: number): ExprFloat {
  return { kind: 'float', value, key: `#f${Object.is(value, -0) ? '-0' : value}` }
}

/** A symbol node. @internal */
export function rawSym(name: string, assumptions: Assumptions = {}): ExprSym {
  if (!NAME.test(name)) throw new MdmlError('value', `sym: '${name}' is not a valid symbol name`)
  const flags = Object.entries(assumptions)
    .filter(([, v]) => v === true)
    .map(([k]) => k)
    .sort()
  const a: Assumptions = Object.fromEntries(flags.map((k) => [k, true]))
  return { kind: 'sym', name, assumptions: a, key: flags.length === 0 ? `$${name}` : `$${name}{${flags.join(' ')}}` }
}

/** A named constant node. @internal */
export function rawConst(name: ConstName): ExprConst {
  return { kind: 'const', name, key: `&${name}` }
}

/** A sum node over already canonical, sorted terms. @internal */
export function rawAdd(args: readonly Expr[]): ExprAdd {
  charge(args.length + 1)
  return { kind: 'add', args, key: `+[${joinKeys(args)}]` }
}

/** A product node over already canonical, sorted factors. @internal */
export function rawMul(args: readonly Expr[]): ExprMul {
  charge(args.length + 1)
  return { kind: 'mul', args, key: `*[${joinKeys(args)}]` }
}

/** A power node. @internal */
export function rawPow(base: Expr, exp: Expr): ExprPow {
  charge(2)
  return { kind: 'pow', base, exp, key: `^[${base.key},${exp.key}]` }
}

/** A function application node. @internal */
export function rawFn(name: string, args: readonly Expr[]): ExprFn {
  charge(args.length + 1)
  return { kind: 'fn', name, args, key: `@${name}[${joinKeys(args)}]` }
}

/** A piecewise node. @internal */
export function rawPiecewise(pieces: readonly (readonly [Expr, Expr])[]): ExprPiecewise {
  return { kind: 'piecewise', pieces, key: `?[${joinKeys(pieces.flat())}]` }
}

/** A relation node. @internal */
export function rawRel(op: RelOp, lhs: Expr, rhs: Expr): ExprRel {
  return { kind: 'rel', op, lhs, rhs, key: `${op}[${lhs.key},${rhs.key}]` }
}

/** A logical connective node. @internal */
export function rawLogic(op: 'and' | 'or' | 'not', args: readonly Expr[]): ExprLogic {
  return { kind: 'logic', op, args, key: `!${op}[${joinKeys(args)}]` }
}

/** A big-O node. @internal */
export function rawOrder(expr: Expr, variable: ExprSym, point: Expr): ExprOrder {
  return { kind: 'order', expr, variable, point, key: `O[${expr.key},${variable.key},${point.key}]` }
}

/** An unevaluated derivative node. @internal */
export function rawDerivative(expr: Expr, variables: readonly (readonly [ExprSym, number])[]): ExprDerivative {
  const vs = variables.map(([v, k]) => `${v.key}:${k}`).join(',')
  return { kind: 'derivative', expr, variables, key: `D[${expr.key};${vs}]` }
}

/** An unevaluated integral node. @internal */
export function rawIntegral(expr: Expr, variable: ExprSym, bounds?: readonly [Expr, Expr]): ExprIntegral {
  const b = bounds === undefined ? '' : `;${bounds[0].key},${bounds[1].key}`
  return bounds === undefined
    ? { kind: 'integral', expr, variable, key: `S[${expr.key};${variable.key}]` }
    : { kind: 'integral', expr, variable, bounds, key: `S[${expr.key};${variable.key}${b}]` }
}

/** The direct children of a node, in order. @internal */
export function children(e: Expr): readonly Expr[] {
  switch (e.kind) {
    case 'add':
    case 'mul':
    case 'fn':
    case 'logic':
      return e.args
    case 'pow':
      return [e.base, e.exp]
    case 'piecewise':
      return e.pieces.flat()
    case 'rel':
      return [e.lhs, e.rhs]
    case 'order':
      return [e.expr, e.variable, e.point]
    case 'derivative':
      return [e.expr, ...e.variables.map(([v]) => v)]
    case 'integral':
      return e.bounds === undefined ? [e.expr, e.variable] : [e.expr, e.variable, e.bounds[0], e.bounds[1]]
    default:
      return []
  }
}

/** Whether `e` contains `target` as a subexpression. @internal */
export function contains(e: Expr, target: Expr): boolean {
  if (e.key === target.key) return true
  for (const c of children(e)) if (contains(c, target)) return true
  return false
}

/** Whether `e` mentions the symbol (free or bound). @internal */
export function hasSymbol(e: Expr, s: ExprSym): boolean {
  if (e.kind === 'sym') return e.key === s.key
  if (!e.key.includes(s.key)) return false
  for (const c of children(e)) if (hasSymbol(c, s)) return true
  return false
}
