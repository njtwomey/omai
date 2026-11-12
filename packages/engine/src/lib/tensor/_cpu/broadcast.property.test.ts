import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { cpu } from '../cpu.ts'
import { allocate, type DType } from '../dtype.ts'
import { type Shape, sizeOf } from '../shape.ts'
import type { Tensor } from '../tensor.ts'
import { planBroadcast } from './broadcast.ts'
import { binaryKernels, compareKernels, map2Indexed, whereIndexed } from './elementwise.ts'

// The broadcast fast paths (`broadcast.ts`: row loops over a layout plan) against the general path (`map2Indexed`,
// `whereIndexed`: an index map per operand and one closure call per element), element by element with `Object.is` so NaN,
// infinities and signed zeros count. The shapes are random broadcasts of a random output shape, with axes of extent one,
// leading axes missing, empty axes and enough axes that no plan exists, so that the layouts the fast path claims and the
// ones it must hand back both run.

const RUNS = 400
const SPECIAL = [0, -0, 1, -1, 0.5, -2.5, 3, Number.NaN, Infinity, -Infinity, 7, 1e-300, 1e300, 2]

/** An output shape and two or three operand shapes that broadcast to it. */
const broadcastCase = (operands: number): fc.Arbitrary<{ shapes: Shape[]; out: Shape }> =>
  fc.array(fc.integer({ min: 1, max: 4 }), { minLength: 0, maxLength: 7 }).chain((out) =>
    fc
      .tuple(
        ...Array.from({ length: operands }, () =>
          fc.tuple(fc.array(fc.boolean(), { minLength: out.length, maxLength: out.length }), fc.nat(out.length)),
        ),
      )
      .map((specs) => ({
        out,
        shapes: specs.map(([ones, drop]) => out.map((n, a) => (ones[a] ? 1 : n)).slice(drop)),
      })),
  )

function make(dtype: DType, shape: Shape, seed: number): Tensor {
  const out = allocate(dtype, sizeOf(shape))
  for (let i = 0; i < out.length; i++) {
    const k = (i * 7 + seed * 13 + ((i * i) % 5)) % (SPECIAL.length + 6)
    out[i] = dtype === 'bool' ? (k % 3 === 0 ? 1 : 0) : (SPECIAL[k] ?? ((k * 37) % 11) - 4.5)
  }
  return cpu.wrap(dtype, shape, out)
}

function same(got: ArrayLike<number>, want: ArrayLike<number>, what: string): void {
  expect(got.length).toBe(want.length)
  for (let i = 0; i < want.length; i++)
    if (!Object.is(got[i], want[i]))
      expect.fail(`${what}: element ${i} is ${got[i]}, the general path gives ${want[i]}`)
}

const ARITHMETIC = ['add', 'sub', 'mul', 'div', 'max', 'min', 'pow', 'atan2', 'mod', 'floorDiv'] as const
const COMPARISONS = ['eq', 'ne', 'lt', 'le', 'gt', 'ge'] as const

describe('broadcast fast paths against the general path', () => {
  for (const dtype of ['float64', 'float32', 'int32'] as const) {
    it(`binary ops in ${dtype}: bit-identical on random broadcasts`, () => {
      fc.assert(
        fc.property(broadcastCase(2), fc.nat(5), ({ shapes }, seed) => {
          const a = make(dtype, shapes[0] as Shape, seed)
          const b = make(dtype, shapes[1] as Shape, seed + 3)
          for (const op of ARITHMETIC) {
            const got = cpu.binary(op, a, b)
            const want = map2Indexed(a, b, binaryKernels[op], dtype)
            expect(got.shape).toEqual(want.shape)
            same(got.data, want.data, `${op} ${JSON.stringify(shapes)}`)
          }
        }),
        { numRuns: RUNS },
      )
    })
    it(`comparisons in ${dtype}: bit-identical on random broadcasts`, () => {
      fc.assert(
        fc.property(broadcastCase(2), fc.nat(5), ({ shapes }, seed) => {
          const a = make(dtype, shapes[0] as Shape, seed)
          const b = make(dtype, shapes[1] as Shape, seed + 3)
          for (const op of COMPARISONS) {
            const got = cpu.compare(op, a, b)
            const want = map2Indexed(a, b, compareKernels[op], 'bool')
            expect(got.shape).toEqual(want.shape)
            same(got.data, want.data, `${op} ${JSON.stringify(shapes)}`)
          }
        }),
        { numRuns: RUNS },
      )
    })
  }

  for (const dtype of ['float64', 'int32', 'bool'] as const) {
    it(`where in ${dtype}: bit-identical on random broadcasts`, () => {
      fc.assert(
        fc.property(broadcastCase(3), fc.nat(5), ({ shapes }, seed) => {
          const c = make('bool', shapes[0] as Shape, seed) as Tensor<'bool'>
          const a = make(dtype, shapes[1] as Shape, seed + 1)
          const b = make(dtype, shapes[2] as Shape, seed + 4)
          const got = cpu.where(c, a, b)
          const want = whereIndexed(c, a, b)
          expect(got.shape).toEqual(want.shape)
          same(got.data, want.data, `where ${JSON.stringify(shapes)}`)
        }),
        { numRuns: RUNS },
      )
    })
  }

  it('map2 with an arbitrary function matches the index maps', () => {
    fc.assert(
      fc.property(broadcastCase(2), fc.nat(5), ({ shapes }, seed) => {
        const a = make('float64', shapes[0] as Shape, seed)
        const b = make('float64', shapes[1] as Shape, seed + 3)
        const fn = (x: number, y: number): number => x * 3 - y / 2
        same(cpu.map2(a, b, fn, 'float64').data, map2Indexed(a, b, fn, 'float64').data, 'map2')
      }),
      { numRuns: RUNS },
    )
  })
})

describe('the layout plan', () => {
  // The layouts the fast path exists for have a plan; shapes of five or more alternating axes do not (they fall back).
  const planned: readonly (readonly [string, Shape, Shape])[] = [
    ['outer product', [8, 1], [1, 9]],
    ['row against matrix', [8, 9], [1, 9]],
    ['column against matrix', [8, 9], [8, 1]],
    ['trailing axis', [4, 5, 6], [6]],
    ['leading axis', [4, 5, 6], [4, 1, 1]],
    ['rank-3 grid, two axes against one', [4, 1, 1], [1, 5, 1]],
    ['rank-3 grid, all three', [4, 1, 6], [1, 5, 1]],
    ['scalar-like operand', [3, 4], [1, 1]],
    ['equal shapes', [3, 4, 2], [3, 4, 2]],
  ]
  it.each(planned)('%s has a plan', (_name, a, b) => {
    const shape = [...a.map((n, i) => Math.max(n, b[i - (a.length - b.length)] ?? 1))]
    expect(planBroadcast(shape, [a, b])).not.toBeNull()
  })

  it('alternating extents of one and many (more than four axes after merging) have none and still agree', () => {
    const a: Shape = [2, 1, 2, 1, 2, 1, 2]
    const b: Shape = [1, 2, 1, 2, 1, 2, 1]
    expect(planBroadcast([2, 2, 2, 2, 2, 2, 2], [a, b])).toBeNull()
    const x = make('float64', a, 1)
    const y = make('float64', b, 2)
    same(cpu.binary('add', x, y).data, map2Indexed(x, y, binaryKernels.add, 'float64').data, 'add, no plan')
    same(cpu.compare('lt', x, y).data, map2Indexed(x, y, compareKernels.lt, 'bool').data, 'lt, no plan')
  })

  it('merges the axes every operand treats alike', () => {
    const plan = planBroadcast([4, 5, 6], [[4, 5, 6], [6]])
    expect(plan?.extents).toEqual([1, 1, 20, 6])
    expect(plan?.strides).toEqual([
      [0, 0, 6, 1],
      [0, 0, 0, 1],
    ])
  })
})
