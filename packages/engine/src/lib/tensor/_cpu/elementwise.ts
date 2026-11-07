import { roundHalfEven } from '../../base/index.ts'
import type { BinaryKernel, CompareKernel, PredicateKernel, UnaryKernel } from '../backend.ts'
import { allocate, type DType, type Storage } from '../dtype.ts'
import { broadcastShapes, sizeOf } from '../shape.ts'
import type { Tensor } from '../tensor.ts'
import { assertSameDtype, broadcastIndex, wrap } from './_util.ts'

// Round half to even, as numpy does.
const unaryKernels: Record<UnaryKernel, (x: number) => number> = {
  neg: (x) => -x,
  abs: Math.abs,
  sign: (x) => (x > 0 ? 1 : x < 0 ? -1 : x),
  exp: Math.exp,
  expm1: Math.expm1,
  log: Math.log,
  log1p: Math.log1p,
  log2: Math.log2,
  log10: Math.log10,
  sqrt: Math.sqrt,
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  asin: Math.asin,
  acos: Math.acos,
  atan: Math.atan,
  sinh: Math.sinh,
  cosh: Math.cosh,
  tanh: Math.tanh,
  asinh: Math.asinh,
  acosh: Math.acosh,
  atanh: Math.atanh,
  // Split by sign so exp never overflows.
  sigmoid: (x) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x))),
  floor: Math.floor,
  ceil: Math.ceil,
  round: roundHalfEven,
  trunc: Math.trunc,
}

/** Python's `%`: the exact remainder of `a % b` moved to the sign of the divisor (numpy `mod`, no `a / b` rounding). */
function pyMod(a: number, b: number): number {
  const r = a % b
  return r !== 0 && r < 0 !== b < 0 ? r + b : r
}

/** Python's `//` as numpy computes it: `(a - fmod(a, b)) / b`, corrected for the sign and rounded to the nearest whole. */
function floorDivide(a: number, b: number): number {
  if (b === 0) return a / b
  const r = a % b
  let div = (a - r) / b
  if (r !== 0 && r < 0 !== b < 0) div -= 1
  if (div === 0) return a / b < 0 ? -0 : 0
  const floor = Math.floor(div)
  return div - floor > 0.5 ? floor + 1 : floor
}

const binaryKernels: Record<BinaryKernel, (a: number, b: number) => number> = {
  add: (a, b) => a + b,
  sub: (a, b) => a - b,
  mul: (a, b) => a * b,
  div: (a, b) => a / b,
  // C99 and numpy: 1 ** y and x ** 0 are 1 even for NaN and Infinity (JavaScript's `**` gives NaN).
  pow: (a, b) => (a === 1 || b === 0 || (a === -1 && Math.abs(b) === Infinity) ? 1 : a ** b),
  max: (a, b) => (Number.isNaN(a) || Number.isNaN(b) ? Number.NaN : Math.max(a, b)),
  min: (a, b) => (Number.isNaN(a) || Number.isNaN(b) ? Number.NaN : Math.min(a, b)),
  atan2: Math.atan2,
  floorDiv: (a, b) => floorDivide(a, b),
  // Python semantics: the result takes the sign of the divisor.
  mod: pyMod,
}

const compareKernels: Record<CompareKernel, (a: number, b: number) => number> = {
  eq: (a, b) => (a === b ? 1 : 0),
  ne: (a, b) => (a !== b ? 1 : 0),
  lt: (a, b) => (a < b ? 1 : 0),
  le: (a, b) => (a <= b ? 1 : 0),
  gt: (a, b) => (a > b ? 1 : 0),
  ge: (a, b) => (a >= b ? 1 : 0),
}

const predicateKernels: Record<PredicateKernel, (x: number) => boolean> = {
  isnan: Number.isNaN,
  isfinite: Number.isFinite,
  isinf: (x) => x === Infinity || x === -Infinity,
}

// The most used unary kernels get a loop of their own: the per-element call of `unaryKernels[op]` is shared by every op
// and so megamorphic (negating a million elements takes 9.6 ms through it, 1.5 ms in a loop of its own).
const unaryLoops: Partial<Record<UnaryKernel, (out: Storage<DType>, data: Storage<DType>) => void>> = {
  neg: (out, data) => {
    for (let i = 0; i < out.length; i++) out[i] = -(data[i] ?? 0)
  },
  abs: (out, data) => {
    for (let i = 0; i < out.length; i++) out[i] = Math.abs(data[i] ?? 0)
  },
  exp: (out, data) => {
    for (let i = 0; i < out.length; i++) out[i] = Math.exp(data[i] ?? 0)
  },
  log: (out, data) => {
    for (let i = 0; i < out.length; i++) out[i] = Math.log(data[i] ?? 0)
  },
  sqrt: (out, data) => {
    for (let i = 0; i < out.length; i++) out[i] = Math.sqrt(data[i] ?? 0)
  },
  log1p: (out, data) => {
    for (let i = 0; i < out.length; i++) out[i] = Math.log1p(data[i] ?? 0)
  },
  tanh: (out, data) => {
    for (let i = 0; i < out.length; i++) out[i] = Math.tanh(data[i] ?? 0)
  },
  sigmoid: (out, data) => {
    for (let i = 0; i < out.length; i++) {
      const v = data[i] ?? 0
      out[i] = v >= 0 ? 1 / (1 + Math.exp(-v)) : Math.exp(v) / (1 + Math.exp(v))
    }
  },
}

export function unary<D extends DType>(op: UnaryKernel, x: Tensor<D>): Tensor<D> {
  const out = allocate(x.dtype, x.data.length)
  const loop = unaryLoops[op]
  if (loop !== undefined) loop(out, x.data)
  else {
    const kernel = unaryKernels[op]
    for (let i = 0; i < out.length; i++) out[i] = kernel(x.data[i] ?? 0)
  }
  return wrap(x.dtype, x.shape, out)
}

export function predicate(op: PredicateKernel, x: Tensor): Tensor<'bool'> {
  const kernel = predicateKernels[op]
  const out = new Uint8Array(x.data.length)
  for (let i = 0; i < out.length; i++) out[i] = kernel(x.data[i] ?? 0) ? 1 : 0
  return wrap('bool', x.shape, out)
}

export function map<D extends DType>(x: Tensor, fn: (v: number) => number, dtype: D): Tensor<D> {
  const out = allocate(dtype, x.data.length)
  for (let i = 0; i < out.length; i++) out[i] = fn(x.data[i] ?? 0)
  return wrap(dtype, x.shape, out)
}

/**
 * How an operand of an elementwise kernel is read: `1` when it has the output's size (every element in turn; an equal
 * shape, since a broadcast that expands nothing keeps the layout), `0` when it has a single element, and `null` when it
 * needs an index map.
 */
function readStride(data: ArrayLike<number>, size: number): 0 | 1 | null {
  return data.length === size ? 1 : data.length === 1 ? 0 : null
}

export function map2<D extends DType>(a: Tensor, b: Tensor, fn: (x: number, y: number) => number, dtype: D): Tensor<D> {
  const shape = broadcastShapes(a.shape, b.shape)
  const size = sizeOf(shape)
  const out = allocate(dtype, size)
  const sa = readStride(a.data, size)
  const sb = readStride(b.data, size)
  if (sa !== null && sb !== null) {
    // Equal shapes or a one-element operand: read in place, no index maps.
    for (let i = 0; i < size; i++) out[i] = fn(a.data[i * sa] ?? 0, b.data[i * sb] ?? 0)
    return wrap(dtype, shape, out)
  }
  const ia = broadcastIndex(shape, a.shape)
  const ib = broadcastIndex(shape, b.shape)
  for (let i = 0; i < out.length; i++) out[i] = fn(a.data[ia[i] ?? 0] ?? 0, b.data[ib[i] ?? 0] ?? 0)
  return wrap(dtype, shape, out)
}

/** How two operands of the output's size map onto it: both in place, or one of them a single element. */
type Layout = 'same' | 'left' | 'right'

function layoutOf(a: Tensor, b: Tensor, size: number): Layout | null {
  if (a.data.length === size && b.data.length === size) return 'same'
  if (a.data.length === 1 && b.data.length === size) return 'left'
  if (b.data.length === 1 && a.data.length === size) return 'right'
  return null
}

// The four field operations get loops of their own: a single call site per element would be shared by every binary
// kernel, which makes it megamorphic (11 ms against 2 ms for a million elements). The results are the same IEEE
// operations as `binaryKernels`.
const fieldLoops: Partial<
  Record<BinaryKernel, (out: Storage<DType>, ad: Storage<DType>, bd: Storage<DType>, layout: Layout) => void>
> = {
  add: (out, ad, bd, layout) => {
    const n = out.length
    if (layout === 'same') for (let i = 0; i < n; i++) out[i] = (ad[i] ?? 0) + (bd[i] ?? 0)
    else if (layout === 'left') for (let i = 0, x = ad[0] ?? 0; i < n; i++) out[i] = x + (bd[i] ?? 0)
    else for (let i = 0, y = bd[0] ?? 0; i < n; i++) out[i] = (ad[i] ?? 0) + y
  },
  sub: (out, ad, bd, layout) => {
    const n = out.length
    if (layout === 'same') for (let i = 0; i < n; i++) out[i] = (ad[i] ?? 0) - (bd[i] ?? 0)
    else if (layout === 'left') for (let i = 0, x = ad[0] ?? 0; i < n; i++) out[i] = x - (bd[i] ?? 0)
    else for (let i = 0, y = bd[0] ?? 0; i < n; i++) out[i] = (ad[i] ?? 0) - y
  },
  mul: (out, ad, bd, layout) => {
    const n = out.length
    if (layout === 'same') for (let i = 0; i < n; i++) out[i] = (ad[i] ?? 0) * (bd[i] ?? 0)
    else if (layout === 'left') for (let i = 0, x = ad[0] ?? 0; i < n; i++) out[i] = x * (bd[i] ?? 0)
    else for (let i = 0, y = bd[0] ?? 0; i < n; i++) out[i] = (ad[i] ?? 0) * y
  },
  div: (out, ad, bd, layout) => {
    const n = out.length
    if (layout === 'same') for (let i = 0; i < n; i++) out[i] = (ad[i] ?? 0) / (bd[i] ?? 0)
    else if (layout === 'left') for (let i = 0, x = ad[0] ?? 0; i < n; i++) out[i] = x / (bd[i] ?? 0)
    else for (let i = 0, y = bd[0] ?? 0; i < n; i++) out[i] = (ad[i] ?? 0) / y
  },
}

export function binary<D extends DType>(op: BinaryKernel, a: Tensor<D>, b: Tensor<D>): Tensor<D> {
  assertSameDtype(a, b, op)
  const loop = fieldLoops[op]
  const shape = broadcastShapes(a.shape, b.shape)
  const size = sizeOf(shape)
  const layout = layoutOf(a, b, size)
  if (loop === undefined || layout === null) return map2(a, b, binaryKernels[op], a.dtype)
  const out = allocate(a.dtype, size)
  loop(out, a.data, b.data, layout)
  return wrap(a.dtype, shape, out)
}

export function compare(op: CompareKernel, a: Tensor, b: Tensor): Tensor<'bool'> {
  return map2(a, b, compareKernels[op], 'bool')
}

export function where<D extends DType>(condition: Tensor<'bool'>, a: Tensor<D>, b: Tensor<D>): Tensor<D> {
  assertSameDtype(a, b, 'where')
  const shape = broadcastShapes(condition.shape, broadcastShapes(a.shape, b.shape))
  const size = sizeOf(shape)
  const out = allocate(a.dtype, size)
  const sc = readStride(condition.data, size)
  const sa = readStride(a.data, size)
  const sb = readStride(b.data, size)
  if (sc !== null && sa !== null && sb !== null) {
    for (let i = 0; i < size; i++) out[i] = condition.data[i * sc] ? (a.data[i * sa] ?? 0) : (b.data[i * sb] ?? 0)
    return wrap(a.dtype, shape, out)
  }
  const ic = broadcastIndex(shape, condition.shape)
  const ia = broadcastIndex(shape, a.shape)
  const ib = broadcastIndex(shape, b.shape)
  for (let i = 0; i < out.length; i++) {
    out[i] = condition.data[ic[i] ?? 0] ? (a.data[ia[i] ?? 0] ?? 0) : (b.data[ib[i] ?? 0] ?? 0)
  }
  return wrap(a.dtype, shape, out)
}

export function cast<D extends DType>(x: Tensor, dtype: D): Tensor<D> {
  const out = allocate(dtype, x.data.length)
  if (dtype === 'int32') for (let i = 0; i < out.length; i++) out[i] = Math.trunc(x.data[i] ?? 0)
  else if (dtype === 'bool') for (let i = 0; i < out.length; i++) out[i] = x.data[i] !== 0 ? 1 : 0
  else out.set(x.data)
  return wrap(dtype, x.shape, out)
}
