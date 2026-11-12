import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { cpu } from '../cpu.ts'
import { allocate, type DType } from '../dtype.ts'
import { type Shape, sizeOf, stridesOf } from '../shape.ts'
import type { Tensor } from '../tensor.ts'

// The reductions of `reduce.ts` (plain loops over adjacent axes, four running values for a maximum or minimum) against a
// reference written here the way the strided walk is: every input element in row-major order folded into its output cell with
// the original combine functions. Compared with `Object.is`, so NaN, infinities and signed zeros count, over random shapes and
// random axis sets, adjacent or not (an axis set with a gap keeps the strided walk and must still agree).

const RUNS = 400
const SPECIAL = [0, -0, 1, -1, 0.5, -2.5, 3, Number.NaN, Infinity, -Infinity, 7, 1e-300, 1e300, 2]

const identity = { sum: 0, prod: 1, max: -Infinity, min: Infinity } as const
const combine = {
  sum: (acc: number, x: number) => acc + x,
  prod: (acc: number, x: number) => acc * x,
  max: (acc: number, x: number) => (Number.isNaN(acc) || Number.isNaN(x) ? Number.NaN : Math.max(x, acc)),
  min: (acc: number, x: number) => (Number.isNaN(acc) || Number.isNaN(x) ? Number.NaN : Math.min(x, acc)),
}

function make(dtype: DType, shape: Shape, seed: number, special: boolean): Tensor {
  const out = allocate(dtype, sizeOf(shape))
  for (let i = 0; i < out.length; i++) {
    const k = (i * 7 + seed * 13 + ((i * i) % 5)) % (SPECIAL.length + 6)
    const v = special ? (SPECIAL[k] ?? ((k * 37) % 11) - 4.5) : ((k * 37) % 11) - 4.5
    out[i] = dtype === 'bool' ? (k % 3 === 0 ? 1 : 0) : v
  }
  return cpu.wrap(dtype, shape, out)
}

function reference(op: keyof typeof identity, x: Tensor, axes: readonly number[], keepDims: boolean): Tensor {
  const reduced = new Set(axes)
  const kept = x.shape.map((n, a) => (reduced.has(a) ? 1 : n))
  const outShape = keepDims ? kept : x.shape.filter((_, a) => !reduced.has(a))
  const acc = new Float64Array(sizeOf(outShape)).fill(identity[op])
  const strides = stridesOf(kept)
  const rank = x.shape.length
  const index = new Array<number>(rank).fill(0)
  for (let i = 0; i < x.data.length; i++) {
    let cell = 0
    for (let a = 0; a < rank; a++) cell += reduced.has(a) ? 0 : (index[a] ?? 0) * (strides[a] ?? 0)
    acc[cell] = combine[op](acc[cell] ?? 0, x.data[i] ?? 0)
    for (let a = rank - 1; a >= 0; a--) {
      index[a] = (index[a] ?? 0) + 1
      if ((index[a] ?? 0) < (x.shape[a] ?? 1)) break
      index[a] = 0
    }
  }
  const out = allocate(x.dtype, acc.length)
  for (let i = 0; i < acc.length; i++) out[i] = acc[i] ?? 0
  return cpu.wrap(x.dtype, outShape, out)
}

function same(got: ArrayLike<number>, want: ArrayLike<number>, what: string): void {
  expect(got.length).toBe(want.length)
  for (let i = 0; i < want.length; i++)
    if (!Object.is(got[i], want[i])) expect.fail(`${what}: element ${i} is ${got[i]}, the reference gives ${want[i]}`)
}

/** A shape and a non-empty set of its axes. */
const reduction = fc.array(fc.constantFrom(1, 2, 3, 4, 5, 8), { minLength: 1, maxLength: 4 }).chain((shape) =>
  fc
    .subarray(
      shape.map((_, a) => a),
      { minLength: 1 },
    )
    .map((axes) => ({ shape, axes })),
)

describe('reductions against the strided reference', () => {
  for (const dtype of ['float64', 'float32', 'int32'] as const) {
    for (const op of ['sum', 'prod', 'max', 'min'] as const) {
      it(`${op} in ${dtype}: bit-identical over random shapes and axis sets`, () => {
        fc.assert(
          fc.property(reduction, fc.nat(5), fc.boolean(), fc.boolean(), ({ shape, axes }, seed, keepDims, special) => {
            const x = make(dtype, shape, seed, special)
            const got = cpu.reduce(op, x, axes, keepDims)
            const want = reference(op, x, axes, keepDims)
            expect(got.shape).toEqual(want.shape)
            same(got.data, want.data, `${op} of ${JSON.stringify(shape)} over ${JSON.stringify(axes)}`)
          }),
          { numRuns: RUNS },
        )
      })
    }
  }

  for (const op of ['max', 'min'] as const) {
    it(`arg${op}: the first NaN, else the first extreme, over random shapes and axes`, () => {
      fc.assert(
        fc.property(reduction, fc.nat(5), fc.boolean(), fc.boolean(), ({ shape, axes }, seed, keepDims, special) => {
          const axis = axes[0] ?? 0
          const x = make('float64', shape, seed, special)
          const got = cpu.argReduce(op, x, axis, keepDims)
          const n = shape[axis] ?? 1
          const inner = sizeOf(shape.slice(axis + 1))
          const want = new Int32Array(got.data.length)
          for (let cell = 0; cell < want.length; cell++) {
            const o = Math.floor(cell / inner)
            const k = cell % inner
            let best = 0
            for (let i = 0; i < n; i++) {
              const v = x.data[(o * n + i) * inner + k] as number
              const b = x.data[(o * n + best) * inner + k] as number
              if (Number.isNaN(b)) break
              if (Number.isNaN(v) || (op === 'max' ? v > b : v < b)) best = i
            }
            want[cell] = best
          }
          same(got.data, want, `arg${op} of ${JSON.stringify(shape)} axis ${axis}`)
        }),
        { numRuns: RUNS },
      )
    })
  }

  it('a maximum with a NaN anywhere is NaN, and +0 beats -0, in every position of a row', () => {
    for (let n = 1; n <= 9; n++)
      for (let at = 0; at < n; at++) {
        const nan = new Float64Array(n).fill(1)
        nan[at] = Number.NaN
        expect(cpu.reduce('max', cpu.wrap('float64', [n], nan), [0], false).data[0]).toBeNaN()
        expect(cpu.reduce('min', cpu.wrap('float64', [n], nan), [0], false).data[0]).toBeNaN()
        const zeros = new Float64Array(n).fill(-0)
        zeros[at] = 0
        expect(Object.is(cpu.reduce('max', cpu.wrap('float64', [n], zeros), [0], false).data[0], 0)).toBe(true)
        const mins = new Float64Array(n).fill(0)
        mins[at] = -0
        expect(Object.is(cpu.reduce('min', cpu.wrap('float64', [n], mins), [0], false).data[0], -0)).toBe(true)
      }
  })
})
