import { describe, it } from 'vitest'
import { decode, expectMatches, loadFixture } from '../../../testing/golden.ts'
import type { Shape, TensorLike, Value } from '../index.ts'
import * as T from '../index.ts'

type Params = Readonly<Record<string, unknown>>
type Op = (inputs: TensorLike[], p: Params) => Value | number

const axis = (p: Params): number | number[] | undefined => p['axis'] as number | number[] | undefined
const num = (p: Params, k: string, d: number): number => (p[k] === undefined ? d : (p[k] as number))

const elementwise = Object.fromEntries(
  [
    'exp',
    'expm1',
    'log',
    'log1p',
    'log2',
    'log10',
    'sqrt',
    'sin',
    'cos',
    'tan',
    'arcsin',
    'arccos',
    'arctan',
    'sinh',
    'cosh',
    'tanh',
    'arcsinh',
    'arccosh',
    'arctanh',
    'sigmoid',
    'softplus',
    'relu',
    'floor',
    'ceil',
    'trunc',
    'round',
    'neg',
    'abs',
    'sign',
    'square',
  ].map((name) => [
    name,
    ((xs: TensorLike[]) =>
      (T as unknown as Record<string, (x: TensorLike) => Value>)[name]?.(xs[0] as TensorLike)) as Op,
  ]),
)

const binaryOps = Object.fromEntries(
  ['add', 'sub', 'mul', 'div', 'pow', 'maximum', 'minimum', 'mod', 'floorDiv', 'arctan2'].map((name) => [
    name,
    ((xs: TensorLike[]) =>
      (T as unknown as Record<string, (a: TensorLike, b: TensorLike) => Value>)[name]?.(
        xs[0] as TensorLike,
        xs[1] as TensorLike,
      )) as Op,
  ]),
)

const ops: Record<string, Op> = {
  ...elementwise,
  ...binaryOps,
  sum: ([x], p) => T.sum(x as TensorLike, axis(p), (p['keepDims'] as boolean | undefined) ?? false),
  prod: ([x], p) => T.prod(x as TensorLike, axis(p)),
  mean: ([x], p) => T.mean(x as TensorLike, axis(p)),
  max: ([x], p) => T.max(x as TensorLike, axis(p)),
  min: ([x], p) => T.min(x as TensorLike, axis(p)),
  variance: ([x], p) => T.variance(x as TensorLike, axis(p), { ddof: num(p, 'ddof', 0) }),
  std: ([x], p) => T.std(x as TensorLike, axis(p)),
  logsumexp: ([x], p) => T.logsumexp(x as TensorLike, axis(p)),
  argmax: ([x], p) => T.argmax(x as TensorLike, p['axis'] as number),
  argmin: ([x], p) => T.argmin(x as TensorLike, p['axis'] as number),
  cumsum: ([x], p) => T.cumsum(x as TensorLike, p['axis'] as number),
  cumprod: ([x], p) => T.cumprod(x as TensorLike, p['axis'] as number),
  softmax: ([x], p) => T.softmax(x as TensorLike, p['axis'] as number),
  logSoftmax: ([x], p) => T.logSoftmax(x as TensorLike, p['axis'] as number),
  norm: ([x], p) => T.norm(x as TensorLike, decodeNum(p['ord'])),
  reshape: ([x], p) => T.reshape(x as TensorLike, p['shape'] as Shape),
  transpose: ([x], p) => T.transpose(x as TensorLike, p['perm'] as number[] | undefined),
  swapaxes: ([x], p) => T.swapaxes(x as TensorLike, p['a'] as number, p['b'] as number),
  moveaxis: ([x], p) => T.moveaxis(x as TensorLike, p['source'] as number, p['destination'] as number),
  expandDims: ([x], p) => T.expandDims(x as TensorLike, p['axis'] as number),
  squeeze: ([x]) => T.squeeze(x as TensorLike),
  broadcastTo: ([x], p) => T.broadcastTo(x as TensorLike, p['shape'] as Shape),
  concat: (xs, p) => T.concat(xs, num(p, 'axis', 0)),
  stack: (xs, p) => T.stack(xs, num(p, 'axis', 0)),
  flip: ([x], p) => T.flip(x as TensorLike, axis(p)),
  roll: ([x], p) => T.roll(x as TensorLike, p['shift'] as number, p['axis'] as number | undefined),
  tile: ([x], p) => T.tile(x as TensorLike, p['reps'] as number[]),
  repeat: ([x], p) => T.repeat(x as TensorLike, p['repeats'] as number | number[], p['axis'] as number | undefined),
  pad: ([x], p) => T.pad(x as TensorLike, p['widths'] as [number, number][], num(p, 'value', 0)),
  slice: ([x], p) =>
    T.slice(x as TensorLike, nulls(p['starts']), nulls(p['stops']), (p['steps'] as number[] | undefined) ?? []),
  take: ([x, i], p) => T.take(x as TensorLike, i as TensorLike, num(p, 'axis', 0)),
  takeAlongAxis: ([x, i], p) => T.takeAlongAxis(x as TensorLike, i as TensorLike, p['axis'] as number),
  gather: ([x, i]) => T.gather(x as TensorLike, i as TensorLike),
  scatter: ([x, i, u], p) => T.scatter(x as TensorLike, i as TensorLike, u as TensorLike, p['mode'] as T.ScatterMode),
  slidingWindowView: ([x], p) =>
    T.slidingWindowView(x as TensorLike, p['window'] as number | number[], {
      ...(p['axis'] === undefined ? {} : { axis: p['axis'] as number | number[] }),
      step: num(p, 'step', 1),
      dilation: num(p, 'dilation', 1),
    }),
  indices: (_xs, p) => T.indices(p['shape'] as Shape),
  sort: ([x], p) => T.sort(x as TensorLike, num(p, 'axis', -1)),
  argsort: ([x], p) => T.argsort(x as TensorLike, num(p, 'axis', -1)),
  oneHot: ([x], p) => T.oneHot(x as TensorLike, p['n'] as number),
  searchsorted: ([s, v], p) =>
    T.searchsorted(s as TensorLike, v as TensorLike, (p['side'] as 'left' | 'right' | undefined) ?? 'left'),
  bincount: ([x, w]) => T.bincount(x as TensorLike, 0, w),
  matmul: ([a, b]) => T.matmul(a as TensorLike, b as TensorLike),
  dot: ([a, b]) => T.dot(a as TensorLike, b as TensorLike),
  vdot: ([a, b]) => T.vdot(a as TensorLike, b as TensorLike),
  outer: ([a, b]) => T.outer(a as TensorLike, b as TensorLike),
  inner: ([a, b]) => T.inner(a as TensorLike, b as TensorLike),
  tensordot: ([a, b], p) => T.tensordot(a as TensorLike, b as TensorLike, p['axes'] as number | [number[], number[]]),
  einsum: (xs, p) => T.einsum(p['spec'] as string, ...xs),
  kron: ([a, b]) => T.kron(a as TensorLike, b as TensorLike),
  diagonal: ([x], p) => T.diagonal(x as TensorLike, num(p, 'offset', 0)),
  trace: ([x]) => T.trace(x as TensorLike),
}

function decodeNum(x: unknown): number {
  return x === 'inf' ? Infinity : x === '-inf' ? -Infinity : (x as number)
}

function nulls(x: unknown): (number | undefined)[] {
  return ((x as (number | null)[] | undefined) ?? []).map((v) => (v === null ? undefined : v))
}

for (const file of ['arithmetic', 'math', 'reduce', 'shape', 'indexing', 'linear']) {
  const fixture = loadFixture(`tensor/ops/${file}`)
  describe(`${file} matches ${Object.entries(fixture.reference)
    .map(([k, v]) => `${k} ${v}`)
    .join(', ')}`, () => {
    for (const c of fixture.cases) {
      it(c.name, () => {
        const op = ops[c.fn]
        if (op === undefined) throw new Error(`no omai function mapped for '${c.fn}'`)
        const inputs = c.inputs.map(decode) as TensorLike[]
        expectMatches(op(inputs, c.params), c.output, c.rtol, c.atol, c.name)
      })
    }
  })
}
