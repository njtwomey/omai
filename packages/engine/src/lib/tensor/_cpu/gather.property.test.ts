import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { cpu } from '../cpu.ts'
import { allocate, type DType } from '../dtype.ts'
import { type Shape, sizeOf } from '../shape.ts'
import type { Tensor } from '../tensor.ts'
import { broadcastToIndexed } from './structure.ts'

// The gather and scatter kernels (`take`, `takeAlongAxis` and their adjoints, with a loop of their own where the indexed axis
// is the last one) and `broadcastTo` (a layout plan) against references written here from the definition: one element at a
// time through the multi-index. Bit for bit (`Object.is`), with NaN, infinities and signed zeros in the data, random shapes and
// axes, repeated indices, and out-of-range indices (which must throw the same error as before).

const RUNS = 300
const SPECIAL = [0, -0, 1, -1, 0.5, -2.5, 3, Number.NaN, Infinity, -Infinity, 7, 1e-300, 2]

function make(dtype: DType, shape: Shape, seed: number): Tensor {
  const out = allocate(dtype, sizeOf(shape))
  for (let i = 0; i < out.length; i++) {
    const k = (i * 7 + seed * 13 + ((i * i) % 5)) % (SPECIAL.length + 5)
    out[i] = dtype === 'bool' ? (k % 3 === 0 ? 1 : 0) : (SPECIAL[k] ?? ((k * 37) % 11) - 4.5)
  }
  return cpu.wrap(dtype, shape, out)
}

function same(got: ArrayLike<number>, want: ArrayLike<number>, what: string): void {
  expect(got.length).toBe(want.length)
  for (let i = 0; i < want.length; i++)
    if (!Object.is(got[i], want[i])) expect.fail(`${what}: element ${i} is ${got[i]}, the reference gives ${want[i]}`)
}

const lanesOf = (shape: Shape, axis: number): { outer: number; n: number; inner: number } => ({
  outer: sizeOf(shape.slice(0, axis)),
  n: shape[axis] ?? 1,
  inner: sizeOf(shape.slice(axis + 1)),
})

/** A shape, one of its axes, indices into the axis (repeats allowed) and a seed. */
const gatherCase = fc
  .array(fc.constantFrom(1, 2, 3, 4, 7), { minLength: 1, maxLength: 4 })
  .chain((shape) =>
    fc.tuple(
      fc.constant(shape),
      fc.nat(shape.length - 1),
      fc.array(fc.nat(100), { minLength: 0, maxLength: 9 }),
      fc.nat(5),
    ),
  )
  .map(([shape, axis, raw, seed]) => ({
    shape,
    axis,
    indices: Int32Array.from(raw.map((r) => r % (shape[axis] ?? 1))),
    seed,
  }))

describe('gather and scatter kernels against the definition', () => {
  for (const dtype of ['float64', 'float32', 'int32'] as const) {
    it(`take in ${dtype}`, () => {
      fc.assert(
        fc.property(gatherCase, ({ shape, axis, indices, seed }) => {
          const x = make(dtype, shape, seed)
          const { outer, n, inner } = lanesOf(shape, axis)
          const want = allocate(dtype, outer * indices.length * inner)
          for (let o = 0; o < outer; o++)
            for (let t = 0; t < indices.length; t++)
              for (let k = 0; k < inner; k++)
                want[(o * indices.length + t) * inner + k] = x.data[
                  (o * n + (indices[t] as number)) * inner + k
                ] as number
          const got = cpu.take(x, indices, axis)
          expect(got.shape).toEqual(shape.map((s, a) => (a === axis ? indices.length : s)))
          same(got.data, want, `take ${JSON.stringify(shape)} axis ${axis}`)
        }),
        { numRuns: RUNS },
      )
    })

    it(`takeScatter in ${dtype} (repeated indices accumulate in order)`, () => {
      fc.assert(
        fc.property(gatherCase, fc.integer({ min: 0, max: 3 }), ({ shape, axis, indices, seed }, extra) => {
          const size = (shape[axis] ?? 1) + extra
          const updatesShape = shape.map((s, a) => (a === axis ? indices.length : s))
          const updates = make(dtype, updatesShape, seed)
          const outShape = shape.map((s, a) => (a === axis ? size : s))
          const { outer, inner } = lanesOf(outShape, axis)
          const want = allocate(dtype, sizeOf(outShape))
          for (let o = 0; o < outer; o++)
            for (let t = 0; t < indices.length; t++)
              for (let k = 0; k < inner; k++) {
                const dst = (o * size + (indices[t] as number)) * inner + k
                want[dst] = (want[dst] as number) + (updates.data[(o * indices.length + t) * inner + k] as number)
              }
          const got = cpu.takeScatter(updates, indices, axis, size)
          expect(got.shape).toEqual(outShape)
          same(got.data, want, `takeScatter ${JSON.stringify(shape)} axis ${axis}`)
        }),
        { numRuns: RUNS },
      )
    })

    it(`takeAlongAxis and putAlongAxisAdd in ${dtype}`, () => {
      fc.assert(
        fc.property(gatherCase, fc.integer({ min: 1, max: 5 }), ({ shape, axis, seed }, m) => {
          const { outer, n, inner } = lanesOf(shape, axis)
          const indexShape = shape.map((s, a) => (a === axis ? m : s))
          const indices = cpu.wrap(
            'int32',
            indexShape,
            Int32Array.from({ length: sizeOf(indexShape) }, (_, i) => (i * 5 + seed) % n),
          )
          const x = make(dtype, shape, seed)
          const picked = allocate(dtype, sizeOf(indexShape))
          const added = allocate(dtype, sizeOf(shape))
          const u = make(dtype, indexShape, seed + 2)
          for (let o = 0; o < outer; o++)
            for (let t = 0; t < m; t++)
              for (let k = 0; k < inner; k++) {
                const pos = (o * m + t) * inner + k
                const idx = indices.data[pos] as number
                picked[pos] = x.data[(o * n + idx) * inner + k] as number
                const dst = (o * n + idx) * inner + k
                added[dst] = (added[dst] as number) + (u.data[pos] as number)
              }
          same(cpu.takeAlongAxis(x, indices, axis).data, picked, `takeAlongAxis ${JSON.stringify(shape)} axis ${axis}`)
          same(
            cpu.putAlongAxisAdd(u, indices, axis, shape).data,
            added,
            `putAlongAxisAdd ${JSON.stringify(shape)} axis ${axis}`,
          )
        }),
        { numRuns: RUNS },
      )
    })
  }

  it('out-of-range indices throw the same errors, along the last axis and elsewhere', () => {
    for (const shape of [[4], [3, 4], [4, 3]] as const) {
      const x = make('float64', [...shape], 1)
      const axis = shape.length === 2 && shape[0] === 4 ? 0 : shape.length - 1
      const n = shape[axis] as number
      expect(() => cpu.take(x, Int32Array.of(0, n), axis)).toThrow(`take: index ${n} out of range for length ${n}`)
      expect(() => cpu.take(x, Int32Array.of(-1), axis)).toThrow('take: index -1 out of range')
      const idx = cpu.wrap(
        'int32',
        shape.map((s, a) => (a === axis ? 1 : s)),
        new Int32Array(sizeOf(shape) / n).fill(n),
      )
      expect(() => cpu.takeAlongAxis(x, idx, axis)).toThrow(`takeAlongAxis: index ${n} out of range for length ${n}`)
    }
  })

  it('an empty outer extent or no indices gives the empty result and does not validate', () => {
    expect(cpu.take(make('float64', [0, 3], 1), Int32Array.of(99), 1).shape).toEqual([0, 1])
    expect(cpu.take(make('float64', [2, 3], 1), new Int32Array(0), 1).shape).toEqual([2, 0])
  })
})

describe('broadcastTo against the index map', () => {
  it('is bit-identical on random broadcasts, planned or not', () => {
    fc.assert(
      fc.property(
        fc.array(fc.constantFrom(0, 1, 1, 2, 3, 4), { minLength: 0, maxLength: 6 }),
        fc.array(fc.boolean(), { minLength: 6, maxLength: 6 }),
        fc.nat(6),
        fc.nat(5),
        (out, ones, drop, seed) => {
          const from = out.map((n, a) => (ones[a] ? 1 : n)).slice(Math.min(drop, out.length))
          const x = make('float64', from, seed)
          const got = cpu.broadcastTo(x, out)
          same(
            got.data,
            broadcastToIndexed(x, out).data,
            `broadcastTo ${JSON.stringify(from)} to ${JSON.stringify(out)}`,
          )
          expect(got.shape).toEqual(out)
        },
      ),
      { numRuns: RUNS },
    )
  })
})
