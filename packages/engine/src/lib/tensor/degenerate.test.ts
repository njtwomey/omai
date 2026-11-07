import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import {
  all,
  any,
  argmax,
  argmin,
  asTensor,
  cast,
  cumprod,
  cumsum,
  floorDiv,
  full,
  log2,
  log10,
  logsumexp,
  matmul,
  max,
  min,
  mod,
  type NestedArray,
  norm,
  pow,
  prod,
  scatterAdd,
  segmentSum,
  std,
  sum,
  type Tensor,
  tensor,
  toArray,
  tril,
  triu,
  unique,
  type Value,
  zeros,
} from './index.ts'

const inf = Number.POSITIVE_INFINITY
const nan = Number.NaN

describe('degenerate inputs agree with numpy', () => {
  // [name, value under test, expected] (toEqual treats NaN as equal to NaN)
  const rows: [string, () => Value, NestedArray][] = [
    ['matmul: 0 x Infinity is NaN, not skipped', () => matmul(tensor([[0, 1]]), tensor([[inf], [1]])), [[nan]]],
    ['matmul: 0 x NaN is NaN, not skipped', () => matmul(tensor([[0, 1]]), tensor([[nan], [1]])), [[nan]]],
    ['matmul: zero rows are still zero', () => matmul(tensor([[0, 0]]), tensor([[1], [2]])), [[0]]],
    [
      'tril: Infinity above the diagonal is dropped',
      () =>
        tril(
          tensor([
            [1, inf],
            [3, 4],
          ]),
        ),
      [
        [1, 0],
        [3, 4],
      ],
    ],
    [
      'tril: NaN above the diagonal is dropped',
      () =>
        tril(
          tensor([
            [1, nan],
            [3, 4],
          ]),
        ),
      [
        [1, 0],
        [3, 4],
      ],
    ],
    [
      'triu: -Infinity below the diagonal is dropped',
      () =>
        triu(
          tensor([
            [1, 2],
            [-inf, 4],
          ]),
        ),
      [
        [1, 2],
        [0, 4],
      ],
    ],
    [
      'triu: an offset moves the diagonal',
      () =>
        triu(
          tensor([
            [1, 2],
            [3, 4],
          ]),
          1,
        ),
      [
        [0, 2],
        [0, 0],
      ],
    ],
    ['all of nothing is true', () => all(zeros([0])), 1],
    ['any of nothing is false', () => any(zeros([0])), 0],
    ['all over an empty axis keeps the other axis', () => all(zeros([2, 0]), 1), [1, 1]],
    ['any with a NaN is true', () => any(tensor([0, nan])), 1],
    ['sum of nothing is 0', () => sum(zeros([0])), 0],
    ['prod of nothing is 1', () => prod(zeros([0])), 1],
    ['logsumexp of nothing is -Infinity', () => logsumexp(zeros([0])), -inf],
    ['max over an axis that is not empty, with no rows', () => max(zeros([0, 3]), 1), []],
    ['argmax over an axis that is not empty, with no rows', () => argmax(zeros([0, 3]), 1), []],
    [
      'pow(1, y) and pow((-1), Infinity) are 1',
      () => pow(tensor([1, 1, -1, -1]), tensor([inf, nan, inf, -inf])),
      [1, 1, 1, 1],
    ],
    ['pow(x, 0) is 1 for NaN and Infinity', () => pow(tensor([nan, inf, 0]), 0), [1, 1, 1]],
    ['pow keeps the ordinary values', () => pow(tensor([2, -8, 0]), tensor([3, 1 / 3, -1])), [8, nan, inf]],
    [
      'floorDiv from the exact remainder: 1 // 0.1 is 9',
      () => floorDiv(tensor([1, 7, -7, 5, -5]), tensor([0.1, 2, 2, -2, inf])),
      [9, 3, -4, -3, -1],
    ],
    [
      'mod from the exact remainder: 1 % 0.1',
      () => mod(tensor([1, 7, -7, 5, -5, 5]), tensor([0.1, 3, 3, -3, inf, inf])),
      [0.09999999999999995, 1, 2, -1, inf, 5],
    ],
    ['log10 is exact on powers of ten', () => log10(tensor([1000, 1e-3, 1])), [3, -3, 0]],
    ['log2 is exact on powers of two', () => log2(tensor([8, 0.25])), [3, -2]],
    ['cumprod of bools counts as int32', () => cumprod(tensor([true, true, false])), [1, 1, 0]],
  ]
  it('gives the numpy value at zeros, non-finite values and empty inputs', () => {
    for (const [name, thunk, want] of rows) expect(toArray(thunk()), name).toEqual(want)
  })

  it('keeps dtype integrity at the boundaries', () => {
    // 300 trues: a bool running total would wrap at 256
    const running = cumsum(full([300], 1, { dtype: 'bool' })) as Tensor
    expect([running.dtype, running.data[299]]).toEqual(['int32', 300])
    // bool storage is 0 or 1 whatever went in; nonzero and NaN are true, as in `cast`
    expect(toArray(tensor([2, 256, nan, 0], { dtype: 'bool' }))).toEqual([1, 1, 1, 0])
    expect(toArray(asTensor(new Uint8Array([0, 2, 255])))).toEqual([0, 1, 1])
    expect(toArray(cast(tensor([nan, 0, -2]), 'bool'))).toEqual([1, 0, 1])
    // unique keeps the storage class of the dtype
    const u = unique(tensor([3, 1, 3], { dtype: 'int32' }))
    expect([u.values.dtype, u.values.data.constructor.name, toArray(u.values)]).toEqual(['int32', 'Int32Array', [1, 3]])
  })
})

describe('invalid reductions and indices throw a value error', () => {
  const empty = zeros([0])
  const rows: [string, () => Value][] = [
    ['max of an empty axis', () => max(empty)],
    ['min of an empty axis', () => min(empty)],
    ['max over a zero-length axis with outputs', () => max(zeros([2, 0]), 1)],
    ['argmax of an empty axis', () => argmax(empty)],
    ['argmin over a zero-length axis with outputs', () => argmin(zeros([3, 0]), 1)],
    ['the infinity norm of nothing', () => norm(empty, inf)],
    ['scatterAdd past the end along axis 1', () => scatterAdd([2, 2], [[2], [0]], [[1], [1]], 1)],
    ['scatterAdd before the start', () => scatterAdd([3], [-4], [1], 0)],
    ['segmentSum past the last segment', () => segmentSum([1, 2, 3], [0, 5, 1], 2)],
    ['segmentSum before the first segment', () => segmentSum([1, 2], [0, -3], 2)],
  ]
  it('throws on empty max, min and argmax and on out-of-range scatter indices', () => {
    for (const [name, thunk] of rows) expect(thunk, name).toThrow(expect.objectContaining({ code: 'value' }))
  })

  it('wraps negative scatter indices like take', () => {
    expect(toArray(scatterAdd([3], [-1, 0], [5, 7], 0))).toEqual([7, 0, 5])
    expect(toArray(segmentSum([1, 2, 3], [0, -1, 0], 2))).toEqual([4, 2])
  })
})

describe('derivatives at points where they exist', () => {
  const rows: [string, (x: Value) => Value, NestedArray, NestedArray][] = [
    ['prod with one zero', (x) => prod(x), [0, 2], [2, 0]],
    ['prod with two zeros', (x) => prod(x), [0, 0], [0, 0]],
    ['prod with no zero is out / x', (x) => prod(x), [2, 3, 4], [12, 8, 6]],
    [
      'prod over an axis',
      (x) => sum(prod(x, 1)),
      [
        [0, 2],
        [3, 4],
      ],
      [
        [2, 0],
        [4, 3],
      ],
    ],
    ['norm at the zero vector', (x) => norm(x), [0, 0], [0, 0]],
    ['norm elsewhere', (x) => norm(x), [0, 5], [0, 1]],
    ['std of a constant', (x) => std(x), [1, 1], [0, 0]],
    ['pow(x, 0) at zero', (x) => sum(pow(x, 0)), [0, 1], [0, 0]],
    ['pow(x, 3) at zero', (x) => sum(pow(x, 3)), [0, 2], [0, 12]],
  ]
  it('are finite and right (not NaN) for prod, norm, std and pow', () => {
    for (const [name, f, x, want] of rows) expect(toArray(grad(f)(tensor(x))), name).toEqual(want)
  })
})
