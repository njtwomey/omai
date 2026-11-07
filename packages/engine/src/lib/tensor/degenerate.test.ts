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
import * as T from './index.ts'

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

const t32 = (x: number[]): Tensor => T.tensor(x, { dtype: 'float32' })
const t32i = (x: number[]): Tensor => T.tensor(x, { dtype: 'int32' })
const flat = (x: NestedArray): number[] => (Array.isArray(x) ? x.flatMap(flat) : [Number(x)])
const shapeOf = (x: NestedArray): number[] =>
  Array.isArray(x) ? [x.length, ...(x.length > 0 ? shapeOf(x[0] as NestedArray) : [])] : []

describe('more degenerate inputs agree with numpy', () => {
  // [name, value under test, numpy's answer] (numpy 2.5; each row is one numpy expression)
  const rows: [string, () => Value, NestedArray][] = [
    ['add: an empty axis broadcasts against 1', () => T.add(T.zeros([0]), T.tensor([5])), []],
    ['add: [0, 3] with [3] stays [0, 3]', () => T.add(T.zeros([0, 3]), T.tensor([1, 2, 3])), []],
    ['mul: [2, 1] with [1, 0] is [2, 0]', () => T.mul(T.ones([2, 1]), T.zeros([1, 0])), [[], []]],
    ['sum over the zero axis of [0, 3]', () => T.sum(T.zeros([0, 3]), 0), [0, 0, 0]],
    ['sum over the empty axis of [2, 0] keeps [2]', () => T.sum(T.zeros([2, 0]), 1), [0, 0]],
    ['mean of nothing is NaN', () => T.mean(T.zeros([0])), nan],
    ['mean over an empty axis is NaN per row', () => T.mean(T.zeros([2, 0]), 1), [nan, nan]],
    ['logsumexp over an empty axis is -Infinity per row', () => T.logsumexp(T.zeros([2, 0]), 1), [-inf, -inf]],
    [
      'logsumexp of equal large values does not overflow',
      () => T.logsumexp(T.tensor([1000, 1000])),
      1000.6931471805599,
    ],
    ['logsumexp of all -Infinity is -Infinity', () => T.logsumexp(T.tensor([-inf, -inf])), -inf],
    ['logsumexp with +Infinity is Infinity', () => T.logsumexp(T.tensor([inf, 1])), inf],
    ['sum: Infinity minus Infinity is NaN', () => T.sum(T.tensor([inf, -inf, 1])), nan],
    ['prod: 0 times Infinity is NaN', () => T.prod(T.tensor([0, inf])), nan],
    ['max propagates NaN', () => T.max(T.tensor([1, nan, 3])), nan],
    ['min propagates NaN', () => T.min(T.tensor([1, nan, 3])), nan],
    ['argmax returns the first NaN', () => T.argmax(T.tensor([1, nan, 3, nan])), 1],
    ['argmin returns the first NaN', () => T.argmin(T.tensor([1, nan, 0])), 1],
    ['argmax of ties is the first index', () => T.argmax(T.tensor([2, 5, 5, 1])), 1],
    ['norm of nothing is 0', () => T.norm(T.zeros([0])), 0],
    [
      'norm with an Infinity is Infinity, with a NaN too is NaN',
      () => T.stack([T.norm(T.tensor([3, inf])), T.norm(T.tensor([nan, inf]))]),
      [inf, nan],
    ],
    ['cumsum of nothing is nothing', () => T.cumsum(T.zeros([0])), []],
    ['cumsum passes an Infinity on', () => T.cumsum(T.tensor([1, inf, 2])), [1, inf, inf]],
    ['log: 0 is -Infinity, negatives NaN', () => T.log(T.tensor([0, -1, inf])), [-inf, nan, inf]],
    ['log1p(-1) is -Infinity', () => T.log1p(T.tensor([-1, -2, 0])), [-inf, nan, 0]],
    ['sqrt: negative is NaN, Infinity stays', () => T.sqrt(T.tensor([-1, inf, 0])), [nan, inf, 0]],
    ['exp overflows to Infinity and underflows to 0', () => T.exp(T.tensor([710, -750, -inf])), [inf, 0, 0]],
    ['exp of float32 overflows at 89', () => T.exp(t32([88, 89, 100])), [1.6516362661361307e38, inf, inf]],
    ['expm1 is accurate near 0', () => T.expm1(T.tensor([1e-10, -1e-10])), [1.00000000005e-10, -9.999999999500001e-11]],
    ['tanh saturates at +-Infinity', () => T.tanh(T.tensor([inf, -inf, 1000])), [1, -1, 1]],
    ['sigmoid saturates without NaN', () => T.sigmoid(T.tensor([1000, -1000, -inf])), [1, 0, 0]],
    ['softplus is linear for large x and 0 for very negative x', () => T.softplus(T.tensor([1000, -1000])), [1000, 0]],
    ['softmax of equal huge values is uniform', () => T.softmax(T.tensor([1000, 1000])), [0.5, 0.5]],
    ['softmax with a -Infinity entry gives it 0', () => T.softmax(T.tensor([0, -inf])), [1, 0]],
    ['div: x / 0 is +-Infinity, 0 / 0 NaN', () => T.div(T.tensor([1, -1, 0]), T.tensor([0, 0, 0])), [inf, -inf, nan]],
    ['div of int32 by zero is Infinity (float64)', () => T.div(t32i([1, -1]), t32i([0, 0])), [inf, -inf]],
    ['mod by zero is NaN', () => T.mod(T.tensor([5, -5]), T.tensor([0, 0])), [nan, nan]],
    [
      'floorDiv by zero is +-Infinity, 0 // 0 NaN',
      () => T.floorDiv(T.tensor([5, -5, 0]), T.tensor([0, 0, 0])),
      [inf, -inf, nan],
    ],
    [
      'mod takes the sign of the divisor, with Infinity',
      () => T.mod(T.tensor([-1, 1, 3]), T.tensor([inf, -inf, -2])),
      [inf, -inf, -1],
    ],
    ['pow: 0 ** 0 is 1 and 0 ** -1 is Infinity', () => T.pow(T.tensor([0, 0]), T.tensor([0, -1])), [1, inf]],
    [
      'arctan2 at signed zeros and Infinity',
      () => T.arctan2(T.tensor([0, 0, inf, 1]), T.tensor([-0, 0, inf, -inf])),
      [3.141592653589793, 0, 0.7853981633974483, 3.141592653589793],
    ],
    [
      'maximum and minimum propagate NaN',
      () =>
        T.stack([T.maximum(T.tensor([nan, 1]), T.tensor([1, nan])), T.minimum(T.tensor([nan, 1]), T.tensor([1, nan]))]),
      [
        [nan, nan],
        [nan, nan],
      ],
    ],
    ['clip keeps a NaN', () => T.clip(T.tensor([nan, 5, -5]), -1, 1), [nan, 1, -1]],
    ['sign of NaN is NaN, of Infinity +-1', () => T.sign(T.tensor([nan, inf, -inf, 0])), [nan, 1, -1, 0]],
    ['round halves to even', () => T.round(T.tensor([0.5, 1.5, 2.5, -1.5, 2.5000001])), [0, 2, 2, -2, 3]],
    [
      'floor, ceil, trunc at negative fractions',
      () => T.stack([T.floor(T.tensor([-0.5, 0.5])), T.ceil(T.tensor([-0.5, 0.5])), T.trunc(T.tensor([-1.7, 1.7]))]),
      [
        [-1, 0],
        [0, 1],
        [-1, 1],
      ],
    ],
    [
      'equal and notEqual with NaN',
      () =>
        T.stack([T.equal(T.tensor([nan, 1]), T.tensor([nan, 1])), T.notEqual(T.tensor([nan, 1]), T.tensor([nan, 1]))]),
      [
        [0, 1],
        [1, 0],
      ],
    ],
    [
      'less and greaterEqual with NaN are false',
      () => T.stack([T.less(T.tensor([nan, 1]), 2), T.greaterEqual(T.tensor([nan, 1]), 0)]),
      [
        [0, 1],
        [0, 1],
      ],
    ],
    [
      'isnan, isinf, isfinite',
      () =>
        T.stack([
          T.isnan(T.tensor([nan, inf, 1])),
          T.isinf(T.tensor([nan, inf, -inf])),
          T.isfinite(T.tensor([nan, inf, 1])),
        ]),
      [
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 1],
      ],
    ],
    [
      'logicalXor and logicalNot treat NaN as true',
      () => T.stack([T.logicalXor(T.tensor([nan, 0, 2]), T.tensor([1, 0, 0])), T.logicalNot(T.tensor([nan, 0, 2]))]),
      [
        [0, 0, 1],
        [0, 1, 0],
      ],
    ],
    [
      'where takes a NaN condition as true',
      () => T.where(T.tensor([nan, 0]), T.tensor([1, 2]), T.tensor([3, 4])),
      [1, 4],
    ],
    ['reshape [0, 3] to [-1] is empty', () => T.reshape(T.zeros([0, 3]), [-1]), []],
    [
      'reshape [6] to [2, -1]',
      () => T.reshape(T.arange(6), [2, -1]),
      [
        [0, 1, 2],
        [3, 4, 5],
      ],
    ],
    ['concat with an empty array', () => T.concat([T.zeros([0]), T.tensor([1, 2])]), [1, 2]],
    ['concat along axis 1 of [2, 0] and [2, 1]', () => T.concat([T.zeros([2, 0]), T.ones([2, 1])], 1), [[1], [1]]],
    ['slice: reversed', () => T.slice(T.arange(5), [undefined], [undefined], [-1]), [4, 3, 2, 1, 0]],
    ['slice: out of range bounds clamp', () => T.slice(T.arange(5), [-100], [100], [2]), [0, 2, 4]],
    ['slice: an empty range', () => T.slice(T.arange(5), [3], [1]), []],
    ['slice: a negative step with bounds', () => T.slice(T.arange(6), [4], [0], [-2]), [4, 2]],
    ['take: negative indices wrap', () => T.take(T.tensor([10, 20, 30]), [-1, -3, 0]), [30, 10, 10]],
    ['take: no indices gives an empty result', () => T.take(T.tensor([10, 20, 30]), []), []],
    [
      'roll by more than the length wraps; negative shifts back',
      () => T.stack([T.roll(T.tensor([1, 2, 3]), 4), T.roll(T.tensor([1, 2, 3]), -1)]),
      [
        [3, 1, 2],
        [2, 3, 1],
      ],
    ],
    [
      'repeat 0 times and tile 0 times are empty',
      () => T.concat([T.repeat(T.tensor([1, 2]), 0), T.tile(T.tensor([1, 2]), [0])]),
      [],
    ],
    ['repeat with counts per element', () => T.repeat(T.tensor([1, 2, 3]), [2, 0, 1]), [1, 1, 3]],
    ['pad with a constant on both sides', () => T.pad(T.tensor([1, 2]), [[2, 1]], 9), [9, 9, 1, 2, 9]],
    ['flip and sort of an empty array', () => T.concat([T.flip(T.zeros([0]), 0), T.sort(T.zeros([0]))]), []],
    [
      'sort puts NaN last, descending too (numpy flips: NaN first)',
      () => T.sort(T.tensor([3, nan, 1, -inf, inf])),
      [-inf, 1, 3, inf, nan],
    ],
    ['argsort is stable on ties', () => T.argsort(T.tensor([2, 1, 2, 1])), [1, 3, 0, 2]],
    ['argsort with NaN puts it last', () => T.argsort(T.tensor([nan, 2, 1])), [2, 1, 0]],
    [
      'unique collapses NaNs and keeps the counts',
      () =>
        T.stack([
          T.unique(T.tensor([nan, 1, nan, 1, 1])).values,
          T.cast(T.unique(T.tensor([nan, 1, nan, 1, 1])).counts, 'float64'),
        ]),
      [
        [1, nan],
        [3, 2],
      ],
    ],
    [
      'searchsorted left and right on ties and out of range',
      () =>
        T.concat([
          T.searchsorted(T.tensor([1, 2, 2, 3]), T.tensor([2, 0, 4])),
          T.searchsorted(T.tensor([1, 2, 2, 3]), T.tensor([2, 0, 4]), 'right'),
        ]),
      [1, 0, 4, 3, 0, 4],
    ],
    [
      'bincount of nothing is empty; with minLength pads',
      () =>
        T.concat([T.bincount(T.tensor([], { dtype: 'int32' })), T.bincount(T.tensor([1, 1], { dtype: 'int32' }), 4)]),
      [0, 2, 0, 0],
    ],
    ['topk with ties keeps the lower index first', () => T.topk(T.tensor([3, 5, 5, 1]), 2).indices, [1, 2]],
    [
      'nonzero of a matrix: one index array per axis',
      () =>
        T.stack(
          T.nonzero(
            T.tensor([
              [0, 2, 0],
              [3, 0, 4],
            ]),
          ),
        ),
      [
        [0, 1, 1],
        [1, 0, 2],
      ],
    ],
    ['nonzero of all zeros is empty per axis', () => T.concat(T.nonzero(T.zeros([2, 2]))), []],
    [
      'maskedSelect picks the true positions',
      () =>
        T.maskedSelect(
          T.tensor([
            [1, 2],
            [3, 4],
          ]),
          T.tensor(
            [
              [1, 0],
              [0, 1],
            ],
            { dtype: 'bool' },
          ),
        ),
      [1, 4],
    ],
    [
      'matmul with an empty inner axis is zeros',
      () => T.matmul(T.zeros([2, 0]), T.zeros([0, 3])),
      [
        [0, 0, 0],
        [0, 0, 0],
      ],
    ],
    ['matmul of empty vectors is 0', () => T.matmul(T.zeros([0]), T.zeros([0])), 0],
    [
      'matmul vector with matrix',
      () =>
        T.matmul(
          T.tensor([1, 2]),
          T.tensor([
            [1, 0, 1],
            [0, 1, 1],
          ]),
        ),
      [1, 2, 3],
    ],
    ['matmul broadcasts a zero-sized batch', () => T.matmul(T.zeros([0, 2, 2]), T.eye(2)), []],
    [
      'trace of a rectangular matrix, and an offset',
      () =>
        T.stack([
          T.trace(
            T.tensor([
              [1, 2, 3],
              [4, 5, 6],
            ]),
          ),
          T.trace(
            T.tensor([
              [1, 2, 3],
              [4, 5, 6],
            ]),
            1,
          ),
        ]),
      [6, 8],
    ],
    [
      'diagonal with a negative offset',
      () =>
        T.diagonal(
          T.tensor([
            [1, 2],
            [3, 4],
            [5, 6],
          ]),
          -1,
        ),
      [3, 6],
    ],
    [
      'linspace with one point is the start',
      () => T.concat([T.linspace(2, 5, 1), T.linspace(0, 1, 3)]),
      [2, 0, 0.5, 1],
    ],
    [
      'arange: negative step, empty, and a fractional step length',
      () => T.concat([T.arange(3, 0, -1), T.arange(5, 0), T.arange(0, 1, 0.25)]),
      [3, 2, 1, 0, 0.25, 0.5, 0.75],
    ],
    [
      'eye: non-square with an offset, and 0 by 0',
      () => T.concat([T.reshape(T.eye(2, 3, 1), [-1]), T.reshape(T.eye(0), [-1])]),
      [0, 1, 0, 0, 0, 1],
    ],
    [
      'triu of a rectangular matrix, offset -1',
      () =>
        T.triu(
          T.tensor([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
          ]),
          -1,
        ),
      [
        [1, 2, 3],
        [4, 5, 6],
        [0, 8, 9],
        [0, 0, 12],
      ],
    ],
    [
      'einsum with a repeated index takes the diagonal',
      () =>
        T.einsum(
          'ii->i',
          T.tensor([
            [1, 2],
            [3, 4],
          ]),
        ),
      [1, 4],
    ],
    [
      'allclose follows |a - b| <= atol + rtol |b|',
      () =>
        T.tensor(
          [
            T.allclose(T.tensor([1e-8]), T.tensor([0])),
            T.allclose(T.tensor([1e-7]), T.tensor([0])),
            T.allclose(T.tensor([inf]), T.tensor([inf])),
            T.allclose(T.tensor([nan]), T.tensor([nan])),
          ].map(Number),
        ),
      [1, 0, 1, 0],
    ],
    [
      'int32 addition wraps at 2^31 like numpy',
      () => T.add(t32i([2147483647, -2147483648]), t32i([1, -1])),
      [-2147483648, 2147483647],
    ],
    [
      'int32 multiplication wraps like numpy',
      () => T.mul(t32i([65536, 46341]), t32i([65536, 46341])),
      [0, -2147479015],
    ],
    [
      'int32 floorDiv and mod by zero are 0 as in numpy',
      () => T.concat([T.floorDiv(t32i([5, -5]), t32i([0, 0])), T.mod(t32i([5, -5]), t32i([0, 0]))]),
      [0, 0, 0, 0],
    ],
    [
      'cast float to int32 truncates towards zero',
      () => T.cast(T.tensor([-1.7, 1.7, -0.2, 2147483000.9]), 'int32'),
      [-1, 1, 0, 2147483000],
    ],
    ['float32 keeps 24 bits', () => T.add(t32([16777216]), t32([1])), [16777216]],
    ['float32 rounds 0.1 + 0.2', () => T.add(t32([0.1]), t32([0.2])), [0.30000001192092896]],
    ['float32 overflow is Infinity', () => T.mul(t32([3e38]), t32([10])), [inf]],
  ]
  it('gives the numpy value, shape and NaN pattern for empty, non-finite, signed and out-of-range inputs', () => {
    const wrong: string[] = []
    for (const [name, thunk, want] of rows) {
      let got: Value
      try {
        got = thunk()
      } catch (e) {
        wrong.push(`${name}: threw ${e instanceof Error ? e.message : String(e)}`)
        continue
      }
      // an empty answer has no first row to read the inner shape from: compare the element count then
      if (flat(want).length > 0 && got.shape.join() !== shapeOf(want).join())
        wrong.push(`${name}: shape [${got.shape}], numpy [${shapeOf(want)}]`)
      const g = flat(toArray(got))
      const w = flat(want)
      if (g.length !== w.length) wrong.push(`${name}: ${g.length} values, numpy ${w.length}`)
      for (const [k, v] of g.entries()) {
        const b = w[k] as number
        const same =
          (Number.isNaN(v) && Number.isNaN(b)) || v === b || Math.abs(v - b) <= 1e-12 * Math.max(1, Math.abs(b))
        if (!same) wrong.push(`${name}: [${k}] got ${v}, numpy ${b}`)
      }
    }
    expect(wrong).toEqual([])
  })

  it('keeps the shape of an empty answer (numpy: add, reshape, matmul and take over zero-length axes)', () => {
    const shapes: [string, () => Value, number[]][] = [
      ['add [0] + [1]', () => T.add(T.zeros([0]), T.tensor([5])), [0]],
      ['add [0, 3] + [3]', () => T.add(T.zeros([0, 3]), T.tensor([1, 2, 3])), [0, 3]],
      ['mul [2, 1] * [1, 0]', () => T.mul(T.ones([2, 1]), T.zeros([1, 0])), [2, 0]],
      ['reshape [0, 3] to [-1]', () => T.reshape(T.zeros([0, 3]), [-1]), [0]],
      ['take with no indices on axis 1', () => T.take(T.ones([2, 3]), [], 1), [2, 0]],
      ['slice 3:1', () => T.slice(T.arange(5), [3], [1]), [0]],
      ['matmul [0, 2, 2] @ [2, 2]', () => T.matmul(T.zeros([0, 2, 2]), T.eye(2)), [0, 2, 2]],
      ['matmul [2, 0] @ [0, 3]', () => T.matmul(T.zeros([2, 0]), T.zeros([0, 3])), [2, 3]],
      ['sum [0, 3] over axis 1', () => T.sum(T.zeros([0, 3]), 1), [0]],
      ['concat [2, 0] and [2, 1] on axis 1', () => T.concat([T.zeros([2, 0]), T.ones([2, 1])], 1), [2, 1]],
    ]
    for (const [name, thunk, shape] of shapes) expect(thunk().shape, name).toEqual(shape)
  })

  it('refuses what numpy answers with NaN or an empty result, by a value error naming the cause', () => {
    // numpy returns NaN (variance of nothing, a degrees-of-freedom of 0) or an empty array (linspace with no points)
    const rows: [string, () => Value][] = [
      ['variance of nothing', () => T.variance(T.zeros([0]))],
      ['variance of one value with ddof 1', () => T.variance(T.tensor([5]), undefined, { ddof: 1 })],
      ['linspace with no points', () => T.linspace(0, 1, 0)],
      ['arange with step 0', () => T.arange(0, 1, 0)],
      ['slice with step 0', () => T.slice(T.arange(3), [0], [3], [0])],
      ['take past the end', () => T.take(T.tensor([1, 2, 3]), [3])],
      ['take before the start', () => T.take(T.tensor([1, 2, 3]), [-4])],
      ['reshape to the wrong size', () => T.reshape(T.arange(6), [4, -1])],
      ['squeeze an axis of length 2', () => T.squeeze(T.zeros([2, 1]), 0)],
      ['matmul with mismatched inner axes', () => T.matmul(T.zeros([2, 3]), T.zeros([2, 3]))],
      ['concat of nothing', () => T.concat([])],
    ]
    for (const [name, thunk] of rows)
      expect(thunk, name).toThrow(expect.objectContaining({ code: expect.stringMatching(/value|shape/) }))
  })
})
