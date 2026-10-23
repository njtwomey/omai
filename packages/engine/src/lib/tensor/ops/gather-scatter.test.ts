import { describe, expect, it } from 'vitest'
import { checkPrimitive, grad } from '../../autodiff/index.ts'
import { supportsDType } from '../backend.ts'
import { cpu } from '../cpu.ts'
import { indices } from '../create.ts'
import { iinfo } from '../dtype.ts'
import { tensor, toArray, toFlat } from '../tensor.ts'
import { lift, type Trace, type Traced, type Value } from '../traced.ts'
import { mul } from './arithmetic.ts'
import { gather, type ScatterMode, scatter, take } from './indexing.ts'
import { sum } from './reduce.ts'
import { slidingWindowView } from './shape.ts'

const cube = [
  [
    [0.3, -1.2],
    [2.1, 0.7],
    [-0.4, 1.9],
  ],
  [
    [1.1, 0.2],
    [-2.3, 0.9],
    [0.6, -0.8],
  ],
]
const pairs = [
  [1, 2],
  [0, 0],
  [1, 2],
  [0, -1],
]
const updates = [
  [0.5, 2.5],
  [-1.5, 0.4],
  [1.7, -2.2],
  [0.8, 3.1],
]

describe('gather and scatter', () => {
  it('are differentiable in every mode', () => {
    expect(checkPrimitive((x) => gather(x, pairs), [cube]).failures).toEqual([])
    for (const mode of ['set', 'add', 'mul', 'max', 'min'] as ScatterMode[])
      expect(checkPrimitive((x, u) => scatter(x, pairs, u, mode), [cube, updates]).failures, mode).toEqual([])
  })

  it('set keeps the last write and routes its gradient only to the winner', () => {
    const out = scatter(tensor([0, 0, 0]), [[1], [1], [1]], [5, 6, 7])
    expect(toArray(out)).toEqual([0, 7, 0])
    const g = grad((u: Value) => sum(mul(scatter(tensor([0, 0, 0]), [[1], [1], [1]], u), tensor([1, 2, 3]))))(
      tensor([5, 6, 7]),
    )
    expect(toFlat(g as Value)).toEqual([0, 0, 2])
  })

  it('max ties go to the original value and NaN propagates', () => {
    expect(toArray(scatter(tensor([1, 2]), [[0], [1]], [1, Number.NaN], 'max'))).toEqual([1, Number.NaN])
    const g = grad((u: Value) => sum(scatter(tensor([1, 2]), [[0], [1]], u, 'max')))(tensor([1, 3]))
    expect(toFlat(g as Value)).toEqual([0, 1])
  })

  it('rejects bad coordinates and shapes', () => {
    expect(() => gather(cube, [[2, 0]])).toThrow(/out of range/)
    expect(() => gather(cube, [[0, 0, 0, 0]])).toThrow(/rank 3/)
    expect(() => scatter(cube, pairs, [[1, 2, 3]])).toThrow(/broadcast/)
    expect(
      toArray(
        scatter(
          tensor([1, 2]),
          cpu.wrap('int32', [0, 1], new Int32Array(0)),
          cpu.wrap('float64', [0], new Float64Array(0)),
        ),
      ),
    ).toEqual([1, 2])
  })

  it('take keeps the shape of multi-dimensional indices', () => {
    expect(
      take(tensor([10, 20, 30]), [
        [2, -1],
        [0, 1],
      ]).shape,
    ).toEqual([2, 2])
  })
})

describe('sliding windows', () => {
  it('are differentiable, accumulating overlaps', () => {
    const x = [0.2, -1.1, 0.7, 1.4, -0.3, 2.2, 0.9]
    expect(checkPrimitive((v) => slidingWindowView(v, 3, { step: 2, dilation: 1 }), [x]).failures).toEqual([])
    const g = grad((v: Value) => sum(slidingWindowView(v, 3)))(tensor(x))
    expect(toFlat(g as Value)).toEqual([1, 2, 3, 3, 3, 2, 1])
  })

  it('reject windows that do not fit', () => {
    expect(() => slidingWindowView([1, 2, 3], 2, { dilation: 3 })).toThrow(/exceeds/)
    expect(() => slidingWindowView([1, 2, 3], 0)).toThrow(/positive/)
  })
})

describe('dtype, creation, tracing and backend helpers', () => {
  it('indices, iinfo, lift and capabilities', () => {
    expect(indices([]).shape).toEqual([0])
    expect(toArray(indices([3]))).toEqual([[0, 1, 2]])
    expect(iinfo('int32').min).toBe(-2147483648)
    expect(iinfo('bool')).toEqual({ min: 0, max: 1, bits: 1 })
    expect(supportsDType(cpu, 'bool')).toBe(true)
    const trace: Trace = {
      level: 3,
      lift: (x): Traced => ({ kind: 'traced', dtype: x.dtype, shape: x.shape, trace }),
      process: () => [],
    }
    const t = lift(trace, tensor([1]))
    expect(lift(trace, t)).toBe(t)
  })
})
