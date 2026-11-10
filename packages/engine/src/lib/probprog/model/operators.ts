/**
 * The operator registry of the model layer: a closed set of named operators over named dimensions (arithmetic, links,
 * reductions, `matmul`, `at` for group lookups, `where`), each with a shape rule (which dimensions the result has, so a
 * model is checked as it is built) and an evaluation on tensors that is the library's own tensor ops. The registry is a
 * value, not a side effect of importing: `defaultOperators()` returns it, and an expression names an operator by string,
 * which is what lets a spec be JSON. Values carry their dimension names (`NamedValue`) and elementwise operators align
 * operands by name, so the order axes were declared in never matters. The `custom` expression of the spec is the escape
 * for anything the registry lacks.
 *
 * @index structure The closed operator registry: shape rules and tensor evaluation over named dimensions.
 */

import { OmaiError } from '../../base/index.ts'
import { ndtr } from '../../numerics/special/index.ts'
import {
  abs,
  add,
  clip,
  div,
  equal,
  exp,
  greater,
  less,
  log,
  matmul,
  maximum,
  mean,
  minimum,
  mul,
  neg,
  pow,
  reshape,
  shapesEqual,
  sigmoid,
  softplus,
  sqrt,
  square,
  sub,
  sum,
  take,
  tanh,
  transpose,
  type Value,
  where,
} from '../../tensor/index.ts'

/** The dimension names and event rank of a value, known without computing it: what a shape rule works on. */
export interface NamedShape {
  readonly dims: readonly string[]
  readonly eventRank: number
}

/**
 * A tensor with the names of its leading axes. The value's shape is `[sizes of dims..., ...event axes]`; the event
 * rank is whatever follows the named axes.
 */
export interface NamedValue {
  readonly value: Value
  readonly dims: readonly string[]
}

/** The JSON parameters of an operator application (the dimension a reduction runs over). */
export type OpParams = { readonly [key: string]: import('../../base/index.ts').Json }

/** One operator of the registry. */
export interface OperatorDef {
  readonly name: string
  /** The number of arguments: exactly `arity`. */
  readonly arity: number
  readonly description: string
  /** The name of the message-passing factor operator that implements this one, when it differs from `name`. */
  readonly factor?: string
  /** The dimensions of the result for arguments of these dimensions; throws `OmaiError('shape')` when they do not fit. */
  shape(args: readonly NamedShape[], params: OpParams): NamedShape
  /** The value of the operator on tensors, by the library's tensor ops. */
  evaluate(args: readonly NamedValue[], params: OpParams): NamedValue
}

/** A registry of operators by name. */
export type OperatorTable = Readonly<Record<string, OperatorDef>>

/**
 * The names of a value's axes joined: every dimension of any operand, in order of first appearance, and the largest
 * event rank. The result shape of an elementwise operator.
 *
 * @param shapes - The operands' shapes.
 * @returns The joint shape.
 * @example
 * ```ts
 * import { joinShapes } from 'omai-engine/probprog/model'
 * joinShapes([{ dims: ['row'], eventRank: 0 }, { dims: ['feature', 'row'], eventRank: 0 }]).dims // => ['row', 'feature']
 * ```
 */
export function joinShapes(shapes: readonly NamedShape[]): NamedShape {
  const dims: string[] = []
  for (const s of shapes) for (const d of s.dims) if (!dims.includes(d)) dims.push(d)
  return { dims, eventRank: shapes.reduce((m, s) => Math.max(m, s.eventRank), 0) }
}

/**
 * A named value laid out for broadcasting against `dims`: axes permuted into the order of `dims`, size-1 axes inserted
 * for the dimensions the value lacks, and size-1 axes appended after its event axes up to `eventRank`. Plain tensor
 * broadcasting then does the rest, so two operands that name the same dimension line up whatever order they declared.
 *
 * @param x - The value.
 * @param dims - The dimensions to align to, in the order of the result.
 * @param eventRank - The event rank to pad to (default the value's own).
 * @returns The tensor, with `dims.length + eventRank` axes.
 * @throws OmaiError with code `shape` when the value has a dimension that `dims` lacks.
 * @example
 * ```ts
 * import { alignTo } from 'omai-engine/probprog/model'
 * import { tensor } from 'omai-engine/tensor'
 * alignTo({ value: tensor([1, 2, 3]), dims: ['b'] }, ['a', 'b']).shape // => [1, 3]
 * ```
 */
export function alignTo(x: NamedValue, dims: readonly string[], eventRank?: number): Value {
  const own = x.value.shape.length - x.dims.length
  const want = eventRank ?? own
  for (const d of x.dims)
    if (!dims.includes(d))
      throw new OmaiError('shape', `a value over [${x.dims.join(', ')}] does not fit dimensions [${dims.join(', ')}]`, {
        dim: d,
      })
  const present = dims.filter((d) => x.dims.includes(d))
  const perm = [...present.map((d) => x.dims.indexOf(d)), ...Array.from({ length: own }, (_, i) => x.dims.length + i)]
  const v = perm.every((p, i) => p === i) ? x.value : transpose(x.value, perm)
  const sizes = dims.map((d) => (x.dims.includes(d) ? (v.shape[present.indexOf(d)] ?? 1) : 1))
  const shape = [...sizes, ...v.shape.slice(present.length), ...new Array<number>(Math.max(0, want - own)).fill(1)]
  return shapesEqual(v.shape, shape) ? v : reshape(v, shape)
}

/** The shape of a named value: its dimension names and the number of axes after them. */
export function shapeOfValue(x: NamedValue): NamedShape {
  return { dims: x.dims, eventRank: x.value.shape.length - x.dims.length }
}

function elementwise(
  name: string,
  arity: number,
  description: string,
  fn: (...xs: Value[]) => Value,
  extra: { readonly factor?: string } = {},
): OperatorDef {
  return {
    name,
    arity,
    description,
    ...extra,
    shape: (args) => joinShapes(args),
    evaluate: (args) => {
      const out = joinShapes(args.map(shapeOfValue))
      const xs = args.map((a) => alignTo(a, out.dims, out.eventRank))
      return { value: fn(...xs), dims: out.dims }
    },
  }
}

function dimParam(name: string, params: OpParams, dims: readonly string[]): string | undefined {
  const d = params['dim']
  if (d === undefined) return undefined
  if (typeof d !== 'string' || !dims.includes(d))
    throw new OmaiError('shape', `${name}: the dimension '${String(d)}' is not one of [${dims.join(', ')}]`, { dim: d })
  return d
}

function reduction(name: string, description: string, fn: (x: Value, axis: number | number[]) => Value): OperatorDef {
  return {
    name,
    arity: 1,
    description,
    shape: ([a], params) => {
      const x = a as NamedShape
      const d = dimParam(name, params, x.dims)
      return { dims: d === undefined ? [] : x.dims.filter((e) => e !== d), eventRank: x.eventRank }
    },
    evaluate: ([a], params) => {
      const x = a as NamedValue
      const d = dimParam(name, params, x.dims)
      if (d === undefined) {
        if (x.dims.length === 0) return x
        return {
          value: fn(
            x.value,
            x.dims.map((_, i) => i),
          ),
          dims: [],
        }
      }
      return { value: fn(x.value, x.dims.indexOf(d)), dims: x.dims.filter((e) => e !== d) }
    },
  }
}

const matmulOperator: OperatorDef = {
  name: 'matmul',
  arity: 2,
  description: 'matrix product contracting the last dimension of the left operand with the first of the right',
  factor: 'matmul',
  shape: ([a, b]) => {
    const x = a as NamedShape
    const y = b as NamedShape
    if (x.eventRank !== 0 || y.eventRank !== 0) throw new OmaiError('shape', 'matmul: operands must have no event axes')
    const inner = x.dims[x.dims.length - 1]
    if (inner === undefined || y.dims[0] === undefined || y.dims.length > 2)
      throw new OmaiError('shape', `matmul: need [..., d] times [d] or [d, k], got [${x.dims}] and [${y.dims}]`)
    if (inner !== y.dims[0])
      throw new OmaiError('shape', `matmul: the contracted dimensions differ ('${inner}' and '${y.dims[0]}')`, {
        left: inner,
        right: y.dims[0],
      })
    const dims = [...x.dims.slice(0, -1), ...y.dims.slice(1)]
    if (new Set(dims).size !== dims.length)
      throw new OmaiError('shape', `matmul: the result repeats a dimension [${dims}]`)
    return { dims, eventRank: 0 }
  },
  evaluate: ([a, b], params) => {
    const out = matmulOperator.shape(
      [
        { dims: (a as NamedValue).dims, eventRank: 0 },
        { dims: (b as NamedValue).dims, eventRank: 0 },
      ],
      params,
    )
    return { value: matmul((a as NamedValue).value, (b as NamedValue).value), dims: out.dims }
  },
}

const atOperator: OperatorDef = {
  name: 'at',
  arity: 2,
  description:
    'gather: the entries of the first operand (along its first dimension) at the integer codes of the second',
  factor: 'at',
  shape: ([a, i]) => {
    const x = a as NamedShape
    const idx = i as NamedShape
    if (x.dims.length === 0) throw new OmaiError('shape', 'at: the first operand needs a dimension to index')
    if (idx.eventRank !== 0) throw new OmaiError('shape', 'at: the index must have no event axes')
    const dims = [...idx.dims, ...x.dims.slice(1)]
    if (new Set(dims).size !== dims.length) throw new OmaiError('shape', `at: the result repeats a dimension [${dims}]`)
    return { dims, eventRank: x.eventRank }
  },
  evaluate: ([a, i]) => {
    const x = a as NamedValue
    const idx = i as NamedValue
    return { value: take(x.value, idx.value, 0), dims: [...idx.dims, ...x.dims.slice(1)] }
  },
}

/**
 * The registry the model layer ships: arithmetic (`add`, `sub`, `mul`, `div`, `pow`, `maximum`, `minimum`, `neg`),
 * elementwise functions (`exp`, `log`, `sqrt`, `square`, `abs`, `tanh`), links (`sigmoid`, `probit` (the standard normal
 * CDF, the inverse probit link), `softplus`), comparisons (`gt`, `lt`, `eq`), `where` and `clip`, reductions over a named
 * dimension (`sum`, `mean`), `matmul` and the gather `at`. Elementwise operators align operands by dimension name.
 * Message-passing factors for these are keyed by the same names.
 *
 * @returns The operator table (a fresh object each call, so it can be extended without side effects).
 * @concept model-spec
 * @example
 * ```ts
 * import { defaultOperators } from 'omai-engine/probprog/model'
 * Object.keys(defaultOperators()).length // => 28
 * ```
 */
export function defaultOperators(): Record<string, OperatorDef> {
  const defs: OperatorDef[] = [
    elementwise('add', 2, 'a + b', (a, b) => add(a, b), { factor: 'plus' }),
    elementwise('sub', 2, 'a - b', (a, b) => sub(a, b), { factor: 'difference' }),
    elementwise('mul', 2, 'a * b', (a, b) => mul(a, b), { factor: 'times' }),
    elementwise('div', 2, 'a / b', (a, b) => div(a, b)),
    elementwise('pow', 2, 'a to the power b', (a, b) => pow(a, b)),
    elementwise('maximum', 2, 'the larger of a and b', (a, b) => maximum(a, b), { factor: 'max' }),
    elementwise('minimum', 2, 'the smaller of a and b', (a, b) => minimum(a, b), { factor: 'min' }),
    elementwise('neg', 1, '-a', (a) => neg(a)),
    elementwise('exp', 1, 'exp(a)', (a) => exp(a)),
    elementwise('log', 1, 'log(a)', (a) => log(a)),
    elementwise('sqrt', 1, 'square root of a', (a) => sqrt(a)),
    elementwise('square', 1, 'a squared', (a) => square(a)),
    elementwise('abs', 1, 'absolute value of a', (a) => abs(a)),
    elementwise('tanh', 1, 'hyperbolic tangent of a', (a) => tanh(a)),
    elementwise('sigmoid', 1, 'the logistic function 1 / (1 + exp(-a))', (a) => sigmoid(a), { factor: 'logistic' }),
    elementwise('probit', 1, 'the standard normal CDF of a (the inverse probit link)', (a) => ndtr(a)),
    elementwise('softplus', 1, 'log(1 + exp(a))', (a) => softplus(a)),
    elementwise('gt', 2, 'a > b (a boolean)', (a, b) => greater(a, b), { factor: 'greater-than' }),
    elementwise('lt', 2, 'a < b (a boolean)', (a, b) => less(a, b)),
    elementwise('eq', 2, 'a == b (a boolean)', (a, b) => equal(a, b)),
    elementwise('where', 3, 'a where the first operand is true, else the second', (c, a, b) => where(c, a, b)),
    elementwise('clip', 3, 'a limited to the range [lo, hi]', (a, lo, hi) => clip(a, lo, hi)),
    reduction('sum', 'sum over the dimension in params.dim (every named dimension when absent)', (x, axis) =>
      sum(x, axis),
    ),
    reduction('mean', 'mean over the dimension in params.dim (every named dimension when absent)', (x, axis) =>
      mean(x, axis),
    ),
    matmulOperator,
    atOperator,
  ]
  return Object.fromEntries(defs.map((d) => [d.name, d]))
}

/**
 * The result shape of applying an operator to arguments of the given shapes, checking arity and dimensions.
 *
 * @param table - The operator registry.
 * @param op - The operator's name.
 * @param args - The arguments' shapes.
 * @param params - The operator's parameters.
 * @returns The result's shape.
 * @throws OmaiError with code `value` for an unknown operator or the wrong number of arguments, and `shape` when the
 *   dimensions do not fit.
 * @example
 * ```ts
 * import { defaultOperators, operatorShape } from 'omai-engine/probprog/model'
 * operatorShape(defaultOperators(), 'matmul', [{ dims: ['row', 'feature'], eventRank: 0 }, { dims: ['feature'], eventRank: 0 }]).dims // => ['row']
 * ```
 */
export function operatorShape(
  table: OperatorTable,
  op: string,
  args: readonly NamedShape[],
  params: OpParams = {},
): NamedShape {
  const def = table[op]
  if (def === undefined)
    throw new OmaiError('value', `unknown operator '${op}' (the registry has ${Object.keys(table).join(', ')})`)
  if (args.length !== def.arity)
    throw new OmaiError('value', `operator '${op}' takes ${def.arity} argument(s), got ${args.length}`)
  return def.shape(args, params)
}
