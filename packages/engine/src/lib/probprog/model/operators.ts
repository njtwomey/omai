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
import type { OperatorName } from '../../message-passing/index.ts'
import { ndtr } from '../../numerics/special/index.ts'
import {
  abs,
  add,
  all,
  arange,
  clip,
  cos,
  div,
  equal,
  exp,
  floor,
  greater,
  greaterEqual,
  less,
  log,
  logicalAnd,
  matmul,
  maximum,
  mean,
  minimum,
  mul,
  neg,
  ones,
  pow,
  reshape,
  shapesEqual,
  sigmoid,
  softplus,
  sqrt,
  square,
  stack,
  sub,
  sum,
  take,
  tanh,
  tensor,
  toFlat,
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
  /**
   * The message-passing factor operator that implements this one, when it differs from `name` (an `OperatorName` of the engine's registry,
   * so a name that is not one fails `tsc`). Absent: the operator has the factor of its own name, or is mapped by the compiler's own rules
   * (a product, a gate, a matrix product), which `factors.ts` does not read from here.
   */
  readonly factor?: OperatorName
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
  const perm = [...present.map((d) => x.dims.indexOf(d)), ...toFlat(arange(x.dims.length, x.dims.length + own))]
  const v = perm.every((p, i) => p === i) ? x.value : transpose(x.value, perm)
  const sizes = dims.map((d) => (x.dims.includes(d) ? (v.shape[present.indexOf(d)] ?? 1) : 1))
  const shape = [...sizes, ...v.shape.slice(present.length), ...toFlat(ones([Math.max(0, want - own)]))]
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
  extra: { readonly factor?: OperatorName } = {},
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
 * The dimension `pick` goes along: the last of the first operand's, which must not be a dimension of the index (the index
 * names the entry along the dimension, so it cannot also run over it). The silent alternative, picking along an axis the
 * index spans, gives the wrong numbers, so it is refused with the order that works.
 */
function pickDimension(x: readonly string[], idx: readonly string[]): string {
  const last = x[x.length - 1]
  if (last === undefined) throw new OmaiError('shape', 'pick: the first operand needs a last dimension to pick along')
  if (idx.includes(last))
    throw new OmaiError(
      'shape',
      `pick: the last dimension '${last}' of the first operand [${x.join(', ')}] is also a dimension of the index ` +
        `[${idx.join(', ')}], and pick goes along the last dimension. Put the dimension to pick along last: ` +
        `ops.mul(a, b) has the dimensions of a first, so write ops.mul(b, a) when b is the one over the ${last}.`,
      { dim: last, dims: [...x], index: [...idx] },
    )
  return last
}

const pickOperator: OperatorDef = {
  name: 'pick',
  arity: 2,
  description:
    'along the last dimension of the first operand (which the index must not run over), the entry that the integer codes of the second name (the prediction of the class each row belongs to)',
  shape: ([a, i]) => {
    const x = a as NamedShape
    const idx = i as NamedShape
    if (x.eventRank !== 0 || idx.eventRank !== 0)
      throw new OmaiError('shape', 'pick: the operands must have no event axes')
    pickDimension(x.dims, idx.dims)
    return joinShapes([{ dims: x.dims.slice(0, -1), eventRank: 0 }, idx])
  },
  evaluate: ([a, i]) => {
    const x = a as NamedValue
    const idx = i as NamedValue
    const along = pickDimension(x.dims, idx.dims)
    const axis = x.value.shape.length - 1
    const kept = x.dims.slice(0, -1)
    const out = joinShapes([{ dims: kept, eventRank: 0 }, shapeOfValue(idx)])
    const code = alignTo(idx, out.dims, 0)
    // The nested where would answer the last entry for a code outside the table; refuse it, as `at` (take) and the factor-graph
    // compiler do. A code is valid when it is an integer in 0..K-1 (NaN is neither).
    const size = x.value.shape[axis] as number
    const inRange = logicalAnd(greaterEqual(idx.value, 0), less(idx.value, size))
    if (toFlat(all(logicalAnd(inRange, equal(floor(idx.value), idx.value))))[0] === 0) {
      const bad = toFlat(idx.value).find((c) => !Number.isInteger(c) || c < 0 || c >= size)
      throw new OmaiError(
        'value',
        `pick: the index ${bad} is not one of the ${size} entries (0 to ${size - 1}) along '${along}'`,
        { index: bad, size, dim: along },
      )
    }
    let picked: Value | undefined
    for (let k = (x.value.shape[axis] as number) - 1; k >= 0; k--) {
      const entry = alignTo({ value: take(x.value, k, axis), dims: kept }, out.dims, 0)
      picked = picked === undefined ? entry : where(equal(code, k), entry, picked)
    }
    return { value: picked as Value, dims: out.dims }
  },
}

const matvecOperator: OperatorDef = {
  name: 'matvec',
  arity: 2,
  description:
    "a known matrix (two event axes) times a vector (one event axis): the vector one linear step maps it to, x -> A x (a linear dynamical system's transition)",
  factor: 'gaussian-transition',
  shape: ([a, x]) => {
    const m = a as NamedShape
    const v = x as NamedShape
    if (m.dims.length > 0 || m.eventRank !== 2 || v.eventRank !== 1)
      throw new OmaiError(
        'shape',
        'matvec: needs a matrix with no named dimension (two event axes) and a vector (one event axis)',
      )
    return { dims: v.dims, eventRank: 1 }
  },
  evaluate: ([a, x]) => {
    const m = a as NamedValue
    const v = x as NamedValue
    // x [..., c] times A' [c, r]: the leading axes are the vector's own named dimensions.
    return { value: matmul(v.value, transpose(m.value, [1, 0])), dims: v.dims }
  },
}

const dotOperator: OperatorDef = {
  name: 'dot',
  arity: 2,
  description:
    'a known vector (one event axis) dotted with a vector (one event axis): the scalar a linear reading of it gives',
  factor: 'dot-constant',
  shape: ([a, x]) => {
    const m = a as NamedShape
    const v = x as NamedShape
    if (m.dims.length > 0 || m.eventRank !== 1 || v.eventRank !== 1)
      throw new OmaiError('shape', 'dot: needs a vector with no named dimension and a vector (one event axis each)')
    // A vector written as one event may stand over a dimension that names its axis (`event#2`): the dot product consumes it.
    return { dims: v.dims.filter((d) => !d.startsWith('event#')), eventRank: 0 }
  },
  evaluate: ([a, x]) => {
    const m = a as NamedValue
    const v = x as NamedValue
    return { value: matmul(v.value, m.value), dims: v.dims.filter((d) => !d.startsWith('event#')) }
  },
}

const stickBreakingOperator: OperatorDef = {
  name: 'stickBreaking',
  arity: 1,
  description:
    'the stick-breaking weights of break fractions along the last dimension: w_k = v_k prod_{j<k} (1 - v_j), the last weight taking what is left (one more weight than breaks, as an event axis)',
  factor: 'stick-breaking',
  shape: ([a]) => {
    const x = a as NamedShape
    if (x.eventRank !== 0 || x.dims.length === 0)
      throw new OmaiError('shape', 'stickBreaking: the operand must be a vector of breaks over a named dimension')
    return { dims: x.dims.slice(0, -1), eventRank: 1 }
  },
  evaluate: ([a]) => {
    const x = a as NamedValue
    const axis = x.value.shape.length - 1
    const sticks = x.value.shape[axis] as number
    const parts: Value[] = []
    let rest: Value | undefined
    for (let i = 0; i < sticks; i++) {
      const v = take(x.value, i, axis)
      parts.push(rest === undefined ? v : mul(v, rest))
      rest = mul(rest ?? tensor(1), sub(1, v))
    }
    parts.push(rest ?? tensor(1))
    return { value: stack(parts, axis), dims: x.dims.slice(0, -1) }
  },
}

/**
 * The range the `probit` operator keeps its value in. The normal CDF rounds to exactly 1 from about 8.3 and underflows to 0
 * below about -38, where the logit of a Bernoulli on it is infinite and its gradient not a number: a gradient method that
 * wanders there (a sampler's leapfrog, a variational guide) ends on a failure. Held to these, the value is the same to the
 * last digit away from the tails and the gradient there is zero.
 */
const PROBIT_FLOOR = 1e-300
const PROBIT_CEILING = 1 - Number.EPSILON / 2

/**
 * The registry the model layer ships: arithmetic (`add`, `sub`, `mul`, `div`, `pow`, `maximum`, `minimum`, `neg`),
 * elementwise functions (`exp`, `log`, `sqrt`, `square`, `abs`, `tanh`), links (`sigmoid`, `probit` (the standard normal
 * CDF, the inverse probit link), `softplus`), comparisons (`gt`, `lt`, `eq`), `where` and `clip`, reductions over a named
 * dimension (`sum`, `mean`), `matmul`, the gather `at` and `matvec` and `dot` (a known matrix or vector applied to a vector that is one event: a linear dynamical system's step and reading), `pick` (the entry of the last dimension that an index names: the branch a class selects) and `stickBreaking` (the weights of a stick-breaking process from its break fractions). Elementwise operators align operands by dimension name.
 * The factor operator that implements an operator is `OperatorDef.factor` where it differs from the name (`add` is `plus`, `sub` is
 * `difference`, `sigmoid` is `logistic`, `gt` is `greater-than`); the arithmetic that has none (`mul`, `matmul`, `at`, `pick`) the compiler maps itself.
 *
 * @returns The operator table (a fresh object each call, so it can be extended without side effects).
 * @concept model-spec
 * @example
 * ```ts
 * import { defaultOperators } from 'omai-engine/probprog/model'
 * Object.keys(defaultOperators()).length // => 31
 * ```
 */
export function defaultOperators(): Record<string, OperatorDef> {
  const defs: OperatorDef[] = [
    elementwise('add', 2, 'a + b', (a, b) => add(a, b), { factor: 'plus' }),
    elementwise('sub', 2, 'a - b', (a, b) => sub(a, b), { factor: 'difference' }),
    elementwise('mul', 2, 'a * b', (a, b) => mul(a, b)),
    elementwise('div', 2, 'a / b', (a, b) => div(a, b)),
    elementwise('pow', 2, 'a to the power b', (a, b) => pow(a, b)),
    elementwise('maximum', 2, 'the larger of a and b', (a, b) => maximum(a, b), { factor: 'max' }),
    elementwise('minimum', 2, 'the smaller of a and b', (a, b) => minimum(a, b), { factor: 'min' }),
    elementwise('neg', 1, 'minus a', (a) => neg(a)),
    elementwise('exp', 1, 'exp(a)', (a) => exp(a)),
    elementwise('log', 1, 'log(a)', (a) => log(a)),
    elementwise('sqrt', 1, 'square root of a', (a) => sqrt(a)),
    elementwise('square', 1, 'a squared', (a) => square(a)),
    elementwise('abs', 1, 'absolute value of a', (a) => abs(a)),
    elementwise('tanh', 1, 'hyperbolic tangent of a', (a) => tanh(a)),
    elementwise('cos', 1, 'cosine of a (radians)', (a) => cos(a)),
    elementwise('sigmoid', 1, 'the logistic function 1 / (1 + exp(-a))', (a) => sigmoid(a), { factor: 'logistic' }),
    elementwise(
      'probit',
      1,
      'the standard normal CDF of a (the inverse probit link), kept inside (0, 1) so that a Bernoulli on it has a finite logit',
      (a) => clip(ndtr(a), PROBIT_FLOOR, PROBIT_CEILING),
    ),
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
    stickBreakingOperator,
    pickOperator,
    matvecOperator,
    dotOperator,
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
