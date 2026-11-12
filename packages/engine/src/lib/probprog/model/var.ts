/**
 * The values a model is written with: `Var`, a handle on an expression with its named dimensions and event rank known,
 * so `X.matmul(w).add(b)` builds a checked expression tree instead of computing anything, and `ops`, the same
 * operators as functions (`ops.probit(eta)`). A `Var` never holds data or a tensor; it is a node reference or an
 * operator application of the registry, and the model it ends up in compiles it. Numbers and nested arrays are accepted
 * wherever an operand is and become constants.
 *
 * @index structure Var: a checked expression handle with named dims, and ops, the operators as functions.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { isValue, toArray, type Value } from '../../tensor/index.ts'
import { defaultOperators, type NamedShape, type OperatorTable, type OpParams, operatorShape } from './operators.ts'
import type { ExprSpec } from './spec.ts'

/**
 * What an operator accepts as an argument: a `Var`, a number, a boolean, a nested array of them or a tensor (a constant:
 * a tensor made with the tensor creation functions is read once, when the expression is built; a traced value is not a
 * constant and is refused).
 */
export type Operand = Var | number | boolean | Value | readonly Operand[]

/** The registry the fluent methods use (the default one). */
const REGISTRY: OperatorTable = defaultOperators()

function nestedRank(value: Json): number {
  let rank = 0
  let v: Json = value
  while (Array.isArray(v)) {
    rank += 1
    v = v[0] ?? 0
  }
  return rank
}

function jsonOf(o: readonly Operand[] | number | boolean | Value): Json {
  if (isValue(o)) return jsonOf(toArray(o))
  if (Array.isArray(o)) return o.map((x) => jsonOf(x as number | boolean | readonly Operand[] | Value))
  if (typeof o === 'number') {
    if (!Number.isFinite(o)) throw new OmaiError('value', `a constant must be finite, got ${o}`)
    return o
  }
  if (typeof o === 'boolean') return o
  throw new OmaiError('value', 'a constant array may hold only numbers and booleans, not variables')
}

/**
 * An expression with its named dimensions: a reference to a node of the model under construction, or an operator
 * applied to other expressions. Immutable; every operator method returns a new `Var`.
 *
 * @concept model-spec
 */
export class Var {
  readonly expr: ExprSpec
  readonly dims: readonly string[]
  readonly eventRank: number

  constructor(expr: ExprSpec, dims: readonly string[], eventRank: number) {
    this.expr = expr
    this.dims = dims
    this.eventRank = eventRank
  }

  /** The name of the node this refers to, or undefined for a computed expression. */
  get name(): string | undefined {
    return this.expr.kind === 'ref' ? this.expr.name : undefined
  }

  /** The shape (dimension names and event rank) of the expression. */
  get shape(): NamedShape {
    return { dims: this.dims, eventRank: this.eventRank }
  }

  add(o: Operand): Var {
    return apply('add', [this, o])
  }
  sub(o: Operand): Var {
    return apply('sub', [this, o])
  }
  mul(o: Operand): Var {
    return apply('mul', [this, o])
  }
  div(o: Operand): Var {
    return apply('div', [this, o])
  }
  pow(o: Operand): Var {
    return apply('pow', [this, o])
  }
  neg(): Var {
    return apply('neg', [this])
  }
  exp(): Var {
    return apply('exp', [this])
  }
  log(): Var {
    return apply('log', [this])
  }
  sqrt(): Var {
    return apply('sqrt', [this])
  }
  square(): Var {
    return apply('square', [this])
  }
  abs(): Var {
    return apply('abs', [this])
  }
  cos(): Var {
    return apply('cos', [this])
  }
  sigmoid(): Var {
    return apply('sigmoid', [this])
  }
  probit(): Var {
    return apply('probit', [this])
  }
  softplus(): Var {
    return apply('softplus', [this])
  }
  gt(o: Operand): Var {
    return apply('gt', [this, o])
  }
  lt(o: Operand): Var {
    return apply('lt', [this, o])
  }
  /** Sum over a named dimension (every named dimension when none is given). */
  sum(dim?: string | { readonly name: string }): Var {
    return apply('sum', [this], dimParams(dim))
  }
  /** Mean over a named dimension (every named dimension when none is given). */
  mean(dim?: string | { readonly name: string }): Var {
    return apply('mean', [this], dimParams(dim))
  }
  /** Matrix product: the last dimension of this against the first of `o`. */
  matmul(o: Operand): Var {
    return apply('matmul', [this, o])
  }
  /** Gather: the entries of this (along its first dimension) at the integer codes `index`. */
  at(index: Operand): Var {
    return apply('at', [this, index])
  }
}

const dimParams = (dim: string | { readonly name: string } | undefined): OpParams =>
  dim === undefined ? {} : { dim: typeof dim === 'string' ? dim : dim.name }

/**
 * A constant as an expression: a number, or nested arrays whose leading axes are the named `dims` (the rest are event
 * axes).
 *
 * @param value - The number, boolean or nested array.
 * @param dims - Names of the leading axes.
 * @returns The constant as a `Var`.
 * @throws OmaiError with code `shape` when the array has fewer axes than `dims`, and `value` for non-finite numbers.
 * @concept model-spec
 * @example
 * ```ts
 * import { constant } from 'omai-engine/probprog/model'
 * constant([1, 2, 3], ['feature']).dims // => ['feature']
 * ```
 */
export function constant(value: readonly Operand[] | number | boolean | Value, dims: readonly string[] = []): Var {
  const json = jsonOf(value)
  const rank = nestedRank(json)
  if (rank < dims.length)
    throw new OmaiError('shape', `a constant of rank ${rank} cannot have ${dims.length} named dimensions`, { dims })
  return new Var(
    dims.length === 0 ? { kind: 'const', value: json } : { kind: 'const', value: json, dims },
    dims,
    rank - dims.length,
  )
}

/**
 * Anything an operator accepts as a `Var`.
 *
 * @param o - A `Var`, number, boolean or nested array.
 * @returns A `Var` (the operand itself when it is one).
 * @example
 * ```ts
 * import { toVar } from 'omai-engine/probprog/model'
 * toVar(2).expr // => { kind: 'const', value: 2 }
 * ```
 */
export function toVar(o: Operand): Var {
  return o instanceof Var ? o : constant(o)
}

/**
 * Apply a registry operator to operands, checking arity and dimensions as the expression is built.
 *
 * @param op - The operator's name (in the default registry).
 * @param args - The operands.
 * @param params - The operator's parameters (the `dim` of a reduction).
 * @returns The expression.
 * @throws OmaiError with code `value` for an unknown operator, `shape` when the dimensions do not fit.
 * @concept model-spec
 * @example
 * ```ts
 * import { apply, constant } from 'omai-engine/probprog/model'
 * apply('add', [constant(1), constant(2)]).expr.kind // => 'op'
 * ```
 */
export function apply(op: string, args: readonly Operand[], params: OpParams = {}): Var {
  const vs = args.map(toVar)
  const shape = operatorShape(
    REGISTRY,
    op,
    vs.map((v) => v.shape),
    params,
  )
  const expr: ExprSpec = {
    kind: 'op',
    op,
    args: vs.map((v) => v.expr),
    ...(Object.keys(params).length === 0 ? {} : { params }),
  }
  return new Var(expr, shape.dims, shape.eventRank)
}

/** The operators of the registry as functions of operands, for `ops.probit(eta)` style. */
export interface Ops {
  add(a: Operand, b: Operand): Var
  sub(a: Operand, b: Operand): Var
  mul(a: Operand, b: Operand): Var
  div(a: Operand, b: Operand): Var
  pow(a: Operand, b: Operand): Var
  maximum(a: Operand, b: Operand): Var
  minimum(a: Operand, b: Operand): Var
  neg(a: Operand): Var
  exp(a: Operand): Var
  log(a: Operand): Var
  sqrt(a: Operand): Var
  square(a: Operand): Var
  abs(a: Operand): Var
  tanh(a: Operand): Var
  cos(a: Operand): Var
  sigmoid(a: Operand): Var
  /** The standard normal CDF (the inverse probit link). */
  probit(a: Operand): Var
  softplus(a: Operand): Var
  gt(a: Operand, b: Operand): Var
  lt(a: Operand, b: Operand): Var
  eq(a: Operand, b: Operand): Var
  where(condition: Operand, a: Operand, b: Operand): Var
  clip(a: Operand, lo: Operand, hi: Operand): Var
  sum(a: Operand, dim?: string | { readonly name: string }): Var
  mean(a: Operand, dim?: string | { readonly name: string }): Var
  matmul(a: Operand, b: Operand): Var
  at(a: Operand, index: Operand): Var
  /** A known matrix (two event axes) times a vector (one event axis): the vector one linear step maps it to. */
  matvec(a: Operand, x: Operand): Var
  /** A known vector (one event axis) dotted with a vector (one event axis): a scalar linear reading of it. */
  dot(a: Operand, x: Operand): Var
  /**
   * Along the LAST dimension of the first operand, the entry the integer codes of the second name (the prediction of each
   * row's class). The dimension picked along is the last one the first operand names, so the order of its dimensions
   * matters: an elementwise operator lists the dimensions of its first operand first, so `ops.mul(x, slope)` is
   * `[row, group]` and picks along `group`, while `ops.mul(slope, x)` is `[group, row]` and would pick along `row`, the
   * dimension the index itself runs over. That would give wrong numbers without a word, so it is an error that says which
   * order to write.
   *
   * @example
   * ```ts
   * import { defineModel, ops } from 'omai-engine/probprog/model'
   * const model = (order: 'slopeLast' | 'slopeFirst') =>
   *   defineModel('lines', (m) => {
   *     const row = m.range('row')
   *     const group = m.range('group', { size: 2 })
   *     const x = m.data('x', { dims: [row] })
   *     const slope = m.data('slope', { dims: [group] })
   *     const z = m.data('z', { dims: [row], support: 'index' })
   *     m.derived('mean', ops.pick(order === 'slopeLast' ? ops.mul(x, slope) : ops.mul(slope, x), z))
   *   })
   * const fails = (order: 'slopeLast' | 'slopeFirst'): boolean => {
   *   try {
   *     model(order)
   *     return false
   *   } catch {
   *     return true
   *   }
   * }
   * [fails('slopeLast'), fails('slopeFirst')] // => [false, true]
   * ```
   */
  pick(a: Operand, index: Operand): Var
  /** The stick-breaking weights of break fractions over the last dimension (one more weight than breaks, as an event axis). */
  stickBreaking(a: Operand): Var
}

/**
 * The operators of the registry as functions: `ops.probit(X.matmul(w))`.
 *
 * @concept model-spec
 * @example
 * ```ts
 * import { constant, ops } from 'omai-engine/probprog/model'
 * ops.where(constant(1).gt(0), 1, 2).expr.kind // => 'op'
 * ```
 */
export const ops: Ops = {
  add: (a, b) => apply('add', [a, b]),
  sub: (a, b) => apply('sub', [a, b]),
  mul: (a, b) => apply('mul', [a, b]),
  div: (a, b) => apply('div', [a, b]),
  pow: (a, b) => apply('pow', [a, b]),
  maximum: (a, b) => apply('maximum', [a, b]),
  minimum: (a, b) => apply('minimum', [a, b]),
  neg: (a) => apply('neg', [a]),
  exp: (a) => apply('exp', [a]),
  log: (a) => apply('log', [a]),
  sqrt: (a) => apply('sqrt', [a]),
  square: (a) => apply('square', [a]),
  abs: (a) => apply('abs', [a]),
  tanh: (a) => apply('tanh', [a]),
  cos: (a) => apply('cos', [a]),
  sigmoid: (a) => apply('sigmoid', [a]),
  probit: (a) => apply('probit', [a]),
  softplus: (a) => apply('softplus', [a]),
  gt: (a, b) => apply('gt', [a, b]),
  lt: (a, b) => apply('lt', [a, b]),
  eq: (a, b) => apply('eq', [a, b]),
  where: (c, a, b) => apply('where', [c, a, b]),
  clip: (a, lo, hi) => apply('clip', [a, lo, hi]),
  sum: (a, dim) => apply('sum', [a], dimParams(dim)),
  mean: (a, dim) => apply('mean', [a], dimParams(dim)),
  matmul: (a, b) => apply('matmul', [a, b]),
  at: (a, index) => apply('at', [a, index]),
  matvec: (a, x) => apply('matvec', [a, x]),
  dot: (a, x) => apply('dot', [a, x]),
  pick: (a, index) => apply('pick', [a, index]),
  stickBreaking: (a) => apply('stickBreaking', [a]),
}
