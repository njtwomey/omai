import type { BinaryKernel, CompareKernel, PredicateKernel, UnaryKernel } from '../backend.ts'
import { allocate, type DType } from '../dtype.ts'
import { broadcastShapes, sizeOf } from '../shape.ts'
import type { Tensor } from '../tensor.ts'
import { assertSameDtype, broadcastIndex, wrap } from './_util.ts'

// Round half to even, as numpy does.
function roundHalfEven(x: number): number {
  const r = Math.round(x)
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r
}

const unaryKernels: Record<UnaryKernel, (x: number) => number> = {
  neg: (x) => -x,
  abs: Math.abs,
  sign: (x) => (x > 0 ? 1 : x < 0 ? -1 : x),
  exp: Math.exp,
  expm1: Math.expm1,
  log: Math.log,
  log1p: Math.log1p,
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
  not: (x) => (x ? 0 : 1),
}

const binaryKernels: Record<BinaryKernel, (a: number, b: number) => number> = {
  add: (a, b) => a + b,
  sub: (a, b) => a - b,
  mul: (a, b) => a * b,
  div: (a, b) => a / b,
  pow: (a, b) => a ** b,
  max: (a, b) => (Number.isNaN(a) || Number.isNaN(b) ? Number.NaN : Math.max(a, b)),
  min: (a, b) => (Number.isNaN(a) || Number.isNaN(b) ? Number.NaN : Math.min(a, b)),
  atan2: Math.atan2,
  floorDiv: (a, b) => Math.floor(a / b),
  // Python semantics: the result takes the sign of the divisor.
  mod: (a, b) => a - b * Math.floor(a / b),
  and: (a, b) => (a && b ? 1 : 0),
  or: (a, b) => (a || b ? 1 : 0),
  xor: (a, b) => (a ? 1 : 0) ^ (b ? 1 : 0),
}

const compareKernels: Record<CompareKernel, (a: number, b: number) => boolean> = {
  eq: (a, b) => a === b,
  ne: (a, b) => a !== b,
  lt: (a, b) => a < b,
  le: (a, b) => a <= b,
  gt: (a, b) => a > b,
  ge: (a, b) => a >= b,
}

const predicateKernels: Record<PredicateKernel, (x: number) => boolean> = {
  isnan: Number.isNaN,
  isfinite: Number.isFinite,
  isinf: (x) => x === Infinity || x === -Infinity,
}

export function unary<D extends DType>(op: UnaryKernel, x: Tensor<D>): Tensor<D> {
  const kernel = unaryKernels[op]
  const out = allocate(x.dtype, x.data.length)
  for (let i = 0; i < out.length; i++) out[i] = kernel(x.data[i] ?? 0)
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

export function map2<D extends DType>(a: Tensor, b: Tensor, fn: (x: number, y: number) => number, dtype: D): Tensor<D> {
  const shape = broadcastShapes(a.shape, b.shape)
  const ia = broadcastIndex(shape, a.shape)
  const ib = broadcastIndex(shape, b.shape)
  const out = allocate(dtype, sizeOf(shape))
  for (let i = 0; i < out.length; i++) out[i] = fn(a.data[ia[i] ?? 0] ?? 0, b.data[ib[i] ?? 0] ?? 0)
  return wrap(dtype, shape, out)
}

export function binary<D extends DType>(op: BinaryKernel, a: Tensor<D>, b: Tensor<D>): Tensor<D> {
  assertSameDtype(a, b, op)
  return map2(a, b, binaryKernels[op], a.dtype)
}

export function compare(op: CompareKernel, a: Tensor, b: Tensor): Tensor<'bool'> {
  const kernel = compareKernels[op]
  return map2(a, b, (x, y) => (kernel(x, y) ? 1 : 0), 'bool')
}

export function where<D extends DType>(condition: Tensor<'bool'>, a: Tensor<D>, b: Tensor<D>): Tensor<D> {
  assertSameDtype(a, b, 'where')
  const shape = broadcastShapes(condition.shape, broadcastShapes(a.shape, b.shape))
  const ic = broadcastIndex(shape, condition.shape)
  const ia = broadcastIndex(shape, a.shape)
  const ib = broadcastIndex(shape, b.shape)
  const out = allocate(a.dtype, sizeOf(shape))
  for (let i = 0; i < out.length; i++) {
    out[i] = condition.data[ic[i] ?? 0] ? (a.data[ia[i] ?? 0] ?? 0) : (b.data[ib[i] ?? 0] ?? 0)
  }
  return wrap(a.dtype, shape, out)
}

export function cast<D extends DType>(x: Tensor, dtype: D): Tensor<D> {
  const out = allocate(dtype, x.data.length)
  if (dtype === 'int32') for (let i = 0; i < out.length; i++) out[i] = Math.trunc(x.data[i] ?? 0)
  else if (dtype === 'bool') for (let i = 0; i < out.length; i++) out[i] = x.data[i] ? 1 : 0
  else out.set(x.data)
  return wrap(dtype, x.shape, out)
}
