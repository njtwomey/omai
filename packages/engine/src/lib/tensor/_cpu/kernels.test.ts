import { describe, expect, it } from 'vitest'
import { cpu } from '../cpu.ts'
import { allocate, type DType } from '../dtype.ts'
import { type Shape, sizeOf, stridesOf } from '../shape.ts'
import type { Tensor } from '../tensor.ts'

// The CPU kernels against straightforward reference loops written here, element by element and bit for bit (Object.is,
// so -0 and NaN count): the fast paths of the kernels (equal shapes, a size-1 operand, trailing-axis reductions, runs
// copied whole) must give exactly what the general path gives.

const SPECIAL = [0.5, -1.25, Number.NaN, 3, -0, Infinity, 2, -Infinity, 7, 0.25, -4, 1.5]

function make<D extends DType>(dtype: D, shape: Shape, offset = 0): Tensor<D> {
  const out = allocate(dtype, sizeOf(shape))
  for (let i = 0; i < out.length; i++)
    out[i] = dtype === 'bool' ? ((i + offset) % 3 === 0 ? 1 : 0) : (SPECIAL[(i + offset) % SPECIAL.length] ?? 0)
  return cpu.wrap(dtype, shape, out)
}

// Row-major multi-index of flat index i.
function unravel(i: number, shape: Shape): number[] {
  const idx: number[] = []
  for (let a = shape.length - 1; a >= 0; a--) {
    idx[a] = i % (shape[a] ?? 1)
    i = Math.floor(i / (shape[a] ?? 1))
  }
  return idx
}

// The value of a tensor at an index of the broadcast output (trailing axes aligned, size-1 axes repeat).
function broadcastAt(t: Tensor, idx: readonly number[]): number {
  const lead = idx.length - t.shape.length
  const strides = stridesOf(t.shape)
  let flat = 0
  t.shape.forEach((n, a) => {
    flat += (n === 1 ? 0 : (idx[lead + a] ?? 0)) * (strides[a] ?? 0)
  })
  return t.data[flat] ?? 0
}

function broadcastOf(...shapes: Shape[]): Shape {
  const rank = Math.max(...shapes.map((s) => s.length))
  return Array.from(
    { length: rank },
    (_, a) => shapes.map((s) => s[a - (rank - s.length)] ?? 1).find((n) => n !== 1) ?? 1,
  )
}

function same(got: ArrayLike<number>, want: ArrayLike<number>): void {
  expect(got.length).toBe(want.length)
  for (let i = 0; i < want.length; i++)
    if (!Object.is(got[i], want[i])) expect.fail(`element ${i}: ${got[i]} is not ${want[i]}`)
}

const SHAPE_PAIRS: readonly (readonly [Shape, Shape])[] = [
  [[12], [12]], // equal shapes
  [
    [3, 4],
    [3, 4],
  ],
  [[1], [12]], // a size-1 left operand
  [[12], [1]], // a size-1 right operand
  [
    [1, 1],
    [3, 4],
  ],
  [[3, 4], [1]],
  [[3, 4], [4]], // broadcast rows
  [
    [3, 1],
    [1, 4],
  ], // broadcast both
  [
    [2, 3, 4],
    [3, 1],
  ],
  [
    [0, 3],
    [0, 3],
  ], // empty
  [[0, 3], [1]],
  [[1], [1]],
]

describe('cpu elementwise kernels: every broadcast case against a reference loop', () => {
  it.each(SHAPE_PAIRS)('binary, compare and where on %j and %j in float64, int32 and bool', (sa, sb) => {
    const shape = broadcastOf(sa, sb)
    for (const dtype of ['float64', 'int32', 'bool'] as const) {
      const a = make(dtype, sa)
      const b = make(dtype, sb, 5)
      const want = (f: (x: number, y: number) => number, out: DType): Storage => {
        const r = allocate(out, sizeOf(shape))
        for (let i = 0; i < r.length; i++) {
          const idx = unravel(i, shape)
          r[i] = f(broadcastAt(a, idx), broadcastAt(b, idx))
        }
        return r
      }
      same(
        cpu.binary('add', a, b).data,
        want((x, y) => x + y, dtype),
      )
      same(
        cpu.binary('sub', a, b).data,
        want((x, y) => x - y, dtype),
      )
      same(
        cpu.binary('mul', a, b).data,
        want((x, y) => x * y, dtype),
      )
      same(
        cpu.compare('lt', a, b).data,
        want((x, y) => (x < y ? 1 : 0), 'bool'),
      )
      same(
        cpu.compare('eq', a, b).data,
        want((x, y) => (x === y ? 1 : 0), 'bool'),
      )
      expect(cpu.binary('add', a, b).shape).toEqual(shape)
      // where: the condition has the shape of the first operand here, then of the second, then both operands scalar
      for (const [sc, sx, sy] of [
        [sa, sa, sb],
        [sb, sa, sb],
        [sa, sb, [1]],
        [[1], sa, sb],
      ] as const) {
        const c = make('bool', sc, 1)
        const x = make(dtype, sx, 2)
        const y = make(dtype, sy, 7)
        const wshape = broadcastOf(sc, sx, sy)
        const w = cpu.where(c, x, y)
        const r = allocate(dtype, sizeOf(wshape))
        for (let i = 0; i < r.length; i++) {
          const idx = unravel(i, wshape)
          r[i] = broadcastAt(c, idx) ? broadcastAt(x, idx) : broadcastAt(y, idx)
        }
        expect(w.shape).toEqual(wshape)
        same(w.data, r)
      }
    }
  })
})

type Storage = ReturnType<typeof allocate>

describe('cpu unary kernels', () => {
  const REFERENCE: Readonly<Record<string, (x: number) => number>> = {
    neg: (x) => -x,
    abs: Math.abs,
    exp: Math.exp,
    log: Math.log,
    sqrt: Math.sqrt,
    log1p: Math.log1p,
    tanh: Math.tanh,
    sigmoid: (x) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x))),
    sin: Math.sin, // no loop of its own: the shared path
  }
  it.each(Object.keys(REFERENCE))('%s equals its reference on specials in float64, float32 and int32', (op) => {
    const f = REFERENCE[op] as (x: number) => number
    for (const dtype of ['float64', 'float32', 'int32'] as const) {
      const x = make(dtype, [3, 4])
      const want = allocate(dtype, x.data.length)
      for (let i = 0; i < want.length; i++) want[i] = f(x.data[i] ?? 0)
      same(cpu.unary(op as never, x).data, want)
    }
  })
})

describe('cpu structure kernels against a reference loop', () => {
  const REDUCES: readonly (readonly [Shape, readonly number[], boolean])[] = [
    [[12], [0], false], // full reduction of a vector
    [[3, 4], [0, 1], false], // full reduction of a matrix
    [[3, 4], [1], false], // trailing axis
    [[3, 4], [1], true],
    [[2, 3, 4], [2], false],
    [[2, 3, 4], [1, 2], true], // several trailing axes
    [[3, 4], [0], false], // leading axis
    [[2, 3, 4], [1], false], // middle axis
    [[2, 3, 4], [0, 2], false], // split axes
    [[3, 0], [1], false], // an empty trailing axis (sum has an identity)
    [[0, 3], [0], false],
    [[], [], false], // scalar
  ]
  it.each(REDUCES)('reduce of shape %j over %j (keepDims %s), per op and dtype', (shape, axes, keepDims) => {
    const kept = shape.map((n, a) => (axes.includes(a) ? 1 : n))
    const outShape = keepDims ? kept : shape.filter((_, a) => !axes.includes(a))
    const identity = { sum: 0, prod: 1, max: -Infinity, min: Infinity }
    const step = {
      sum: (p: number, x: number) => p + x,
      prod: (p: number, x: number) => p * x,
      max: (p: number, x: number) => (Number.isNaN(p) || Number.isNaN(x) ? Number.NaN : Math.max(x, p)),
      min: (p: number, x: number) => (Number.isNaN(p) || Number.isNaN(x) ? Number.NaN : Math.min(x, p)),
    }
    for (const dtype of ['float64', 'float32', 'int32'] as const) {
      const x = make(dtype, shape)
      for (const op of ['sum', 'prod', 'max', 'min'] as const) {
        if ((op === 'max' || op === 'min') && x.data.length === 0 && sizeOf(outShape) > 0) continue
        const acc = new Float64Array(sizeOf(outShape)).fill(identity[op])
        const outStrides = stridesOf(kept)
        for (let i = 0; i < x.data.length; i++) {
          const idx = unravel(i, shape)
          let o = 0
          idx.forEach((v, a) => {
            o += axes.includes(a) ? 0 : v * (outStrides[a] ?? 0)
          })
          acc[o] = step[op](acc[o] ?? 0, x.data[i] ?? 0)
        }
        const want = allocate(dtype, acc.length)
        for (let i = 0; i < acc.length; i++) want[i] = acc[i] ?? 0
        const got = cpu.reduce(op, x, axes, keepDims)
        expect(got.shape).toEqual(outShape)
        same(got.data, want)
      }
    }
  })

  it.each([
    [
      [
        [2, 3],
        [2, 3],
      ],
      0,
    ],
    [
      [
        [2, 3],
        [2, 3],
      ],
      1,
    ],
    [
      [
        [2, 3, 4],
        [2, 1, 4],
        [2, 5, 4],
      ],
      1,
    ],
    [
      [
        [2, 3, 4],
        [1, 3, 4],
      ],
      0,
    ],
    [
      [
        [2, 3, 4],
        [2, 3, 2],
      ],
      2,
    ],
    [
      [
        [0, 3],
        [2, 3],
      ],
      0,
    ],
    [[[4]], 0],
  ] as const)('concat of %j along axis %i, and transpose, slice and pad of its result', (shapes, axis) => {
    const xs = shapes.map((s, k) => make('float64', s as unknown as Shape, k * 3))
    const got = cpu.concat(xs, axis)
    const total = xs.reduce((n, t) => n + (t.shape[axis] ?? 0), 0)
    const shape = (xs[0] as Tensor).shape.map((n, a) => (a === axis ? total : n))
    const want = allocate('float64', sizeOf(shape))
    for (let i = 0; i < want.length; i++) {
      const idx = unravel(i, shape)
      let pos = idx[axis] ?? 0
      let k = 0
      while (pos >= ((xs[k] as Tensor).shape[axis] ?? 0)) pos -= (xs[k++] as Tensor).shape[axis] ?? 0
      const src = (xs[k] as Tensor).shape
      const sstr = stridesOf(src)
      want[i] = (xs[k] as Tensor).data[idx.reduce((acc, v, a) => acc + (a === axis ? pos : v) * (sstr[a] ?? 0), 0)] ?? 0
    }
    expect(got.shape).toEqual(shape)
    same(got.data, want)
    // transpose by the reversed permutation
    const perm = shape.map((_, a) => shape.length - 1 - a)
    const tr = cpu.transpose(got, perm)
    const tshape = perm.map((p) => shape[p] ?? 1)
    const tw = allocate('float64', sizeOf(shape))
    const gstr = stridesOf(shape)
    for (let i = 0; i < tw.length; i++) {
      const idx = unravel(i, tshape)
      tw[i] = got.data[idx.reduce((acc, v, a) => acc + v * (gstr[perm[a] ?? 0] ?? 0), 0)] ?? 0
    }
    same(tr.data, tw)
    // a strided slice and a pad of the same tensor
    if (sizeOf(shape) > 0 && shape.length > 0) {
      const starts = shape.map((n) => (n > 1 ? 1 : 0))
      const steps = shape.map(() => 1)
      const sl = cpu.slice(got, starts, shape, steps)
      const slShape = shape.map((n, a) => n - (starts[a] ?? 0))
      const slWant = allocate('float64', sizeOf(slShape))
      for (let i = 0; i < slWant.length; i++) {
        const idx = unravel(i, slShape)
        slWant[i] = got.data[idx.reduce((acc, v, a) => acc + (v + (starts[a] ?? 0)) * (gstr[a] ?? 0), 0)] ?? 0
      }
      same(sl.data, slWant)
      const widths = shape.map((_, a) => [a % 2, 1 - (a % 2)] as const)
      const padded = cpu.pad(got, widths, 9)
      const pShape = shape.map((n) => n + 1)
      const pWant = allocate('float64', sizeOf(pShape)).fill(9)
      const pstr = stridesOf(pShape)
      for (let i = 0; i < got.data.length; i++) {
        const idx = unravel(i, shape)
        pWant[idx.reduce((acc, v, a) => acc + (v + (widths[a]?.[0] ?? 0)) * (pstr[a] ?? 0), 0)] = got.data[i] ?? 0
      }
      same(padded.data, pWant)
    }
  })
})
