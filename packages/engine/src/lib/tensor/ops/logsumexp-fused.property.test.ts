import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { cpu } from '../cpu.ts'
import { allocate } from '../dtype.ts'
import { type Shape, sizeOf } from '../shape.ts'
import type { Tensor } from '../tensor.ts'
import { add, sub } from './arithmetic.ts'
import { isfinite, where } from './compare.ts'
import { exp, log } from './math.ts'
import { logsumexp, max, softmax, sum } from './reduce.ts'
import { reshape } from './shape.ts'

// The fused `logsumexp` kernel of the CPU backend (a maximum pass and an exp-sum pass) against the composition of operations
// that `logsumexp` builds for traced values and for the cases the kernel does not take: `max`, `where(isfinite)`, `sub`,
// `exp`, `sum`, `log`, `add`. Bit for bit (`Object.is`), over random shapes and axes and rows of special values (all `-Infinity`,
// a `+Infinity`, a NaN, huge and tiny magnitudes), float64 where the kernel applies and float32 where it must hand back.

const RUNS = 300
const SPECIAL = [0, -0, 1, -1, 0.5, -2.5, 3, Number.NaN, Infinity, -Infinity, 700, -800, 1e-300, 1e300, 2]

function make(dtype: 'float64' | 'float32', shape: Shape, seed: number, special: boolean): Tensor {
  const out = allocate(dtype, sizeOf(shape))
  for (let i = 0; i < out.length; i++) {
    const k = (i * 7 + seed * 13 + ((i * i) % 5)) % (SPECIAL.length + 8)
    out[i] = special ? (SPECIAL[k] ?? ((k * 37) % 11) - 4.5) : ((k * 37) % 11) - 4.5 + (i % 3) * 0.1
  }
  return cpu.wrap(dtype, shape, out)
}

/** `logsumexp` as the composition, whatever the backend offers. */
function composed(x: Tensor, axes: readonly number[], keepDims: boolean): Tensor {
  const m = max(x, axes, true)
  const shift = where(isfinite(m), m, 0)
  const out = add(log(sum(exp(sub(x, shift)), axes, true)), shift)
  return (
    keepDims
      ? out
      : reshape(
          out,
          x.shape.filter((_, a) => !axes.includes(a)),
        )
  ) as Tensor
}

function same(got: ArrayLike<number>, want: ArrayLike<number>, what: string): void {
  expect(got.length).toBe(want.length)
  for (let i = 0; i < want.length; i++)
    if (!Object.is(got[i], want[i])) expect.fail(`${what}: element ${i} is ${got[i]}, the composition gives ${want[i]}`)
}

const reduction = fc.array(fc.constantFrom(1, 2, 3, 4, 5, 9), { minLength: 1, maxLength: 4 }).chain((shape) =>
  fc
    .subarray(
      shape.map((_, a) => a),
      { minLength: 1 },
    )
    .map((axes) => ({ shape, axes })),
)

describe('fused logsumexp against the composition of operations', () => {
  for (const dtype of ['float64', 'float32'] as const) {
    it(`${dtype}: bit-identical over random shapes and axes (axes with a gap fall back to the composition)`, () => {
      fc.assert(
        fc.property(reduction, fc.nat(5), fc.boolean(), fc.boolean(), ({ shape, axes }, seed, keepDims, special) => {
          const x = make(dtype, shape, seed, special)
          const got = logsumexp(x, axes, keepDims) as Tensor
          const want = composed(x, axes, keepDims)
          expect(got.shape).toEqual(want.shape)
          same(got.data, want.data, `${dtype} ${JSON.stringify(shape)} over ${JSON.stringify(axes)}`)
        }),
        { numRuns: RUNS },
      )
    })
  }

  it('the kernel is taken for float64 over adjacent axes and declines the rest', () => {
    const x = make('float64', [3, 4, 5], 1, false)
    expect(cpu.logsumexp?.(x, [1], false)).toBeDefined()
    expect(cpu.logsumexp?.(x, [0, 2], false)).toBeUndefined()
    expect(cpu.logsumexp?.(make('float32', [3, 4], 1, false), [1], false)).toBeUndefined()
    expect(cpu.logsumexp?.(make('float64', [3, 0], 1, false), [1], false)).toBeUndefined()
  })

  it('an empty reduction is log(0) = -Infinity, and softmax over a row of -Infinity is NaN, as before', () => {
    const empty = make('float64', [3, 0], 1, false)
    same((logsumexp(empty, 1) as Tensor).data, [-Infinity, -Infinity, -Infinity], 'empty')
    const row = cpu.wrap('float64', [1, 3], Float64Array.of(-Infinity, -Infinity, -Infinity))
    same((logsumexp(row, 1) as Tensor).data, [-Infinity], 'all -Infinity')
    expect((softmax(row, 1) as Tensor).data.every(Number.isNaN)).toBe(true)
  })
})
