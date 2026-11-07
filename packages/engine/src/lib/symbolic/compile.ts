/**
 * Turning symbolic expressions into numbers: `compile` builds a numeric function over tensors from an expression
 * (SymPy's `lambdify`), with implementations for special functions passed in, and `evaluate` and `evaluateComplex`
 * compute an expression's value as a real number or as a complex pair on principal branches.
 */

import { MdmlError } from '../base/index.ts'
import type { Complex } from '../exposition/index.ts'
import {
  abs,
  add,
  arccos,
  arccosh,
  arcsin,
  arcsinh,
  arctan,
  arctan2,
  arctanh,
  asFloat,
  asTensor,
  broadcastShapes,
  broadcastTo,
  ceil,
  cos,
  cosh,
  div,
  equal,
  exp,
  floor,
  greater,
  greaterEqual,
  less,
  lessEqual,
  log,
  logicalAnd,
  logicalNot,
  logicalOr,
  mul,
  notEqual,
  pow,
  reciprocal,
  type Shape,
  sign,
  sin,
  sinh,
  sqrt,
  square,
  type TensorLike,
  tan,
  tanh,
  type Value,
  where,
} from '../tensor/index.ts'
import { evalComplex, evalReal, freeNames, type NumericFunctions } from './_numeric.ts'
import type { Expr, ExprSym } from './expr.ts'

export type { NumericFunctions } from './_numeric.ts'

/** Implementations of functions for {@link compile}: tensor functions of tensors (for erf, gamma, f, ...). */
export type TensorFunctions = Readonly<Record<string, (...args: Value[]) => Value>>

/** Options for {@link compile}. */
export interface CompileOptions {
  /** Tensor implementations of functions symbolic cannot compile itself (`erf`, `gamma`, undefined functions). */
  readonly functions?: TensorFunctions
}

const UNARY: Readonly<Record<string, (x: Value) => Value>> = {
  exp,
  log,
  sin,
  cos,
  tan,
  asin: arcsin,
  acos: arccos,
  atan: arctan,
  sinh,
  cosh,
  tanh,
  asinh: arcsinh,
  acosh: arccosh,
  atanh: arctanh,
  Abs: abs,
  sign,
  floor,
  ceiling: ceil,
  sec: (x) => reciprocal(cos(x)),
  csc: (x) => reciprocal(sin(x)),
  cot: (x) => reciprocal(tan(x)),
  coth: (x) => reciprocal(tanh(x)),
  acot: (x) => arctan(reciprocal(x)),
  re: (x) => x,
  conjugate: (x) => x,
}

function constant(v: number): Value {
  return asTensor(v)
}

function compileNode(e: Expr, env: ReadonlyMap<string, Value>, fns: TensorFunctions, memo: Map<string, Value>): Value {
  const hit = memo.get(e.key)
  if (hit !== undefined) return hit
  const go = (x: Expr): Value => compileNode(x, env, fns, memo)
  let out: Value
  switch (e.kind) {
    case 'num':
      out = constant(Number(e.n) / Number(e.d))
      break
    case 'float':
      out = constant(e.value)
      break
    case 'sym': {
      const v = env.get(e.key)
      if (v === undefined) throw new MdmlError('value', `compile: ${e.name} is not among the symbols`)
      out = v
      break
    }
    case 'const':
      if (e.name === 'pi') out = constant(Math.PI)
      else if (e.name === 'E') out = constant(Math.E)
      else if (e.name === 'oo') out = constant(Number.POSITIVE_INFINITY)
      else if (e.name === '-oo') out = constant(Number.NEGATIVE_INFINITY)
      else if (e.name === 'nan') out = constant(Number.NaN)
      else if (e.name === 'true' || e.name === 'false') out = asTensor(e.name === 'true')
      else throw new MdmlError('not-implemented', `compile: ${e.name} needs complex arithmetic`)
      break
    case 'add':
      out = e.args.map(go).reduce((a, b) => add(a, b))
      break
    case 'mul': {
      // Factors with negative integer powers become one division.
      const num: Value[] = []
      const den: Value[] = []
      for (const f of e.args) {
        if (f.kind === 'pow' && f.exp.kind === 'num' && f.exp.n === -1n && f.exp.d === 1n) den.push(go(f.base))
        else num.push(go(f))
      }
      const top = num.length === 0 ? constant(1) : num.reduce((a, b) => mul(a, b))
      out =
        den.length === 0
          ? top
          : div(
              top,
              den.reduce((a, b) => mul(a, b)),
            )
      break
    }
    case 'pow': {
      const b = go(e.base)
      const x = e.exp
      if (x.kind === 'num' && x.d === 1n && x.n === 2n) out = square(b)
      else if (x.kind === 'num' && x.d === 2n && x.n === 1n) out = sqrt(b)
      else if (x.kind === 'num' && x.d === 1n && x.n === -1n) out = reciprocal(b)
      else if (x.kind === 'num' && x.d === 2n && x.n === -1n) out = reciprocal(sqrt(b))
      else out = pow(b, go(x))
      break
    }
    case 'fn': {
      const user = fns[e.name]
      const args = e.args.map(go)
      if (user !== undefined) out = user(...args)
      else if (e.name === 'atan2') out = arctan2(args[0] as Value, args[1] as Value)
      else {
        const f = UNARY[e.name]
        if (f === undefined || args.length !== 1) {
          throw new MdmlError('not-implemented', `compile: no tensor implementation of ${e.name}; pass functions`)
        }
        out = f(args[0] as Value)
      }
      break
    }
    case 'piecewise': {
      const pieces = e.pieces
      let acc: Value = constant(Number.NaN)
      for (let i = pieces.length - 1; i >= 0; i--) {
        const [v, c] = pieces[i] as readonly [Expr, Expr]
        acc = c.kind === 'const' && c.name === 'true' ? go(v) : where(go(c), go(v), acc)
      }
      out = acc
      break
    }
    case 'rel': {
      const ops = { '==': equal, '!=': notEqual, '<': less, '<=': lessEqual, '>': greater, '>=': greaterEqual }
      out = ops[e.op](go(e.lhs), go(e.rhs))
      break
    }
    case 'logic': {
      const args = e.args.map(go)
      out =
        e.op === 'not'
          ? logicalNot(args[0] as Value)
          : args.reduce((a, b) => (e.op === 'and' ? logicalAnd(a, b) : logicalOr(a, b)))
      break
    }
    default:
      throw new MdmlError('not-implemented', `compile: cannot compile ${e.kind}`)
  }
  memo.set(e.key, out)
  return out
}

/**
 * Compile an expression into a numeric function over tensors (SymPy's `lambdify`). The function takes one
 * `TensorLike` per symbol, broadcasts them elementwise and returns a float tensor. It is built from tensor
 * primitives, so it accepts traced values: it can be differentiated with `grad`, vectorised with `vmap`, plotted and
 * used as an objective.
 *
 * @param e - The expression.
 * @param symbols - The parameters, in argument order (symbols or names).
 * @param options - Tensor implementations for functions such as `erf` and `gamma` (from numerics) or undefined ones.
 * @returns The compiled function.
 * @throws MdmlError with code `value` for a free symbol missing from `symbols`, `not-implemented` for complex
 *   constants or functions without an implementation.
 * @example
 * ```ts
 * import { compile, parse } from 'mdml-engine/symbolic'
 * import { toArray } from 'mdml-engine/tensor'
 * const f = compile(parse('x^2 + sin(y)'), ['x', 'y'])
 * toArray(f([1, 2, 3], 0)) // => [1, 4, 9]
 * ```
 */
export function compile(
  e: Expr,
  symbols: readonly (ExprSym | string)[],
  options: CompileOptions = {},
): (...args: TensorLike[]) => Value {
  const names = symbols.map((s) => (typeof s === 'string' ? s : s.name))
  const missing = freeNames(e).filter((n) => !names.includes(n))
  if (missing.length > 0) throw new MdmlError('value', `compile: free symbols ${missing.join(', ')} are not parameters`)
  const fns = options.functions ?? {}
  return (...args: TensorLike[]): Value => {
    if (args.length !== names.length) {
      throw new MdmlError('value', `compile: expected ${names.length} arguments, got ${args.length}`)
    }
    const env = new Map<string, Value>()
    const values = args.map((a) => asFloat(asTensor(a)))
    symbols.forEach((s, i) => {
      env.set(typeof s === 'string' ? `$${s}` : s.key, values[i] as Value)
    })
    const out = asFloat(compileNode(e, env, fns, new Map()))
    let shape: Shape = []
    for (const v of values) shape = broadcastShapes(shape, v.shape)
    return out.shape.length === shape.length && out.shape.every((n, i) => n === shape[i])
      ? out
      : broadcastTo(out, broadcastShapes(out.shape, shape))
  }
}

/**
 * Evaluate an expression to a real number (NaN where the value is complex, as numpy).
 *
 * @param e - The expression.
 * @param values - Values of the free symbols, by name.
 * @param functions - Numeric implementations of functions such as `erf` or undefined ones.
 * @returns The value.
 * @throws MdmlError with code `value` for an unbound symbol, `not-implemented` for a function without a numeric
 *   implementation.
 * @example
 * ```ts
 * import { evaluate, parse } from 'mdml-engine/symbolic'
 * evaluate(parse('x^2 + 1'), { x: 3 }) // => 10
 * ```
 */
export function evaluate(
  e: Expr,
  values: Readonly<Record<string, number>> = {},
  functions: NumericFunctions = {},
): number {
  return evalReal(e, values, functions)
}

/**
 * Evaluate an expression as a complex number [re, im] (principal branches for roots and logarithms).
 *
 * @param e - The expression.
 * @param values - Values of the free symbols: reals or [re, im] pairs.
 * @param functions - Numeric implementations of functions such as `erf` (used at real arguments).
 * @returns The value as [re, im].
 * @throws MdmlError with code `value` when the expression cannot be evaluated.
 * @example
 * ```ts
 * import { evaluateComplex, parse } from 'mdml-engine/symbolic'
 * evaluateComplex(parse('sqrt(-4)')) // => [0, 2]
 * ```
 */
export function evaluateComplex(
  e: Expr,
  values: Readonly<Record<string, number | readonly [number, number]>> = {},
  functions: NumericFunctions = {},
): [number, number] {
  const env = new Map<string, Complex>()
  for (const [k, v] of Object.entries(values)) env.set(k, typeof v === 'number' ? [v, 0] : [v[0], v[1]])
  const z = evalComplex(e, env, functions)
  if (z === undefined) throw new MdmlError('value', 'evaluateComplex: cannot evaluate the expression')
  return [z[0], z[1]]
}
