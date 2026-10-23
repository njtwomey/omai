import fc from 'fast-check'
import { describe, expect, it } from 'vitest'
import { distinctTensor, expectConformant, RUNS, tensorOf } from '../../../testing/arbitrary.ts'
import type { Value } from '../index.ts'
import * as T from '../index.ts'
import {
  broadcastTo,
  concat,
  cumprod,
  cumsum,
  diagonal,
  dot,
  einsum,
  expandDims,
  flip,
  inner,
  kron,
  logSoftmax,
  logsumexp,
  matmul,
  max,
  mean,
  min,
  moveaxis,
  norm,
  outer,
  pad,
  prod,
  repeat,
  reshape,
  roll,
  slice,
  softmax,
  sort,
  split,
  squeeze,
  stack,
  std,
  sum,
  swapaxes,
  take,
  takeAlongAxis,
  tensordot,
  tile,
  topk,
  trace,
  transpose,
  variance,
} from '../index.ts'

function check(
  name: string,
  shapes: readonly (readonly number[])[],
  f: (...xs: Value[]) => Value,
  lo = -2,
  hi = 2,
  distinct = false,
): void {
  const gen = distinct ? distinctTensor : tensorOf
  it(name, () => {
    fc.assert(
      fc.property(fc.tuple(...shapes.map((s) => gen(s, lo, hi))), (xs) => expectConformant(f, xs)),
      { numRuns: RUNS },
    )
  })
}

describe('reductions conform', () => {
  check('sum over all axes', [[2, 3]], (x) => sum(x))
  check('sum over axis 0 keeping dims', [[2, 3]], (x) => sum(x, 0, true))
  check('sum over axes [0, 2]', [[2, 3, 2]], (x) => sum(x, [0, 2]))
  check('mean over axis -1', [[2, 3]], (x) => mean(x, -1))
  check('max over axis 1', [[3, 4]], (x) => max(x, 1), -2, 2, true)
  check('min over all axes', [[3, 4]], (x) => min(x), -2, 2, true)
  check('prod over axis 0', [[3, 2]], (x) => prod(x, 0), 0.5, 2)
  check('variance with ddof 1', [[5]], (x) => variance(x, undefined, { ddof: 1 }))
  check('std over axis 0', [[4, 2]], (x) => std(x, 0), -2, 2, true)
  check('logsumexp over axis 1', [[2, 4]], (x) => logsumexp(x, 1))
  check('softmax', [[2, 4]], (x) => softmax(x))
  check('logSoftmax over axis 0', [[3, 2]], (x) => logSoftmax(x, 0))
  check('norm 2', [[5]], (x) => norm(x), 0.3, 2)
  check('norm 1', [[5]], (x) => norm(x, 1), 0.3, 2)
  check('norm inf', [[5]], (x) => norm(x, Infinity), 0.3, 2, true)
  check('norm 3 over axis 1', [[2, 3]], (x) => norm(x, 3, 1), 0.3, 2)
  check('cumsum over axis 1', [[2, 4]], (x) => cumsum(x, 1))
  check('reverse cumsum', [[5]], (x) => cumsum(x, 0, true))
  check('cumprod', [[5]], (x) => cumprod(x), 0.5, 2)
})

describe('shape operations conform', () => {
  check('reshape', [[2, 3]], (x) => reshape(x, [3, 2]))
  check('transpose', [[2, 3, 4]], (x) => transpose(x, [2, 0, 1]))
  check('swapaxes', [[2, 3]], (x) => swapaxes(x, 0, 1))
  check('moveaxis', [[2, 3, 2]], (x) => moveaxis(x, 0, -1))
  check('expandDims and squeeze', [[3]], (x) => squeeze(expandDims(x, 0)))
  check('broadcastTo', [[3]], (x) => broadcastTo(x, [2, 3]))
  check(
    'concat on axis 1',
    [
      [2, 2],
      [2, 3],
    ],
    (a, b) => concat([a as Value, b as Value], 1),
  )
  check('stack', [[3], [3]], (a, b) => stack([a as Value, b as Value]))
  check('split', [[4, 2]], (x) => split(x, 2)[1] as Value)
  check('flip', [[2, 3]], (x) => flip(x, 1))
  check('roll', [[5]], (x) => roll(x, 2))
  check('tile', [[2]], (x) => tile(x, [3, 1]))
  check('repeat', [[3]], (x) => repeat(x, 2))
  check('pad', [[2, 2]], (x) =>
    pad(
      x,
      [
        [1, 0],
        [0, 2],
      ],
      5,
    ),
  )
})

describe('indexing operations conform', () => {
  check('strided slice', [[6]], (x) => slice(x, [1], [6], [2]))
  check('negative-step slice', [[3, 4]], (x) => slice(x, [undefined, 3], [undefined, 0], [1, -1]))
  check('take with repeats', [[4, 2]], (x) => take(x, [3, 0, 3]))
  check('take on axis 1', [[2, 3]], (x) => take(x, [2, 1], 1))
  check('takeAlongAxis', [[2, 3]], (x) =>
    takeAlongAxis(
      x,
      [
        [2, 0],
        [1, 1],
      ],
      1,
    ),
  )
  check('sort', [[5]], (x) => sort(x), -2, 2, true)
  check('topk values', [[6]], (x) => topk(x, 3).values, -2, 2, true)
})

describe('linear algebra operations conform', () => {
  check(
    'matmul',
    [
      [2, 3],
      [3, 4],
    ],
    matmul,
  )
  check(
    'batched matmul with broadcasting',
    [
      [2, 2, 3],
      [3, 2],
    ],
    matmul,
  )
  check('matrix times vector', [[2, 3], [3]], matmul)
  check('vector times matrix', [[3], [3, 2]], matmul)
  check('vector dot vector', [[4], [4]], dot)
  check('outer', [[2], [3]], outer)
  check('inner', [[2, 3], [3]], inner)
  check(
    'tensordot',
    [
      [2, 3, 2],
      [3, 2, 2],
    ],
    (a, b) => tensordot(a, b, 2),
  )
  check(
    'einsum matrix product',
    [
      [2, 3],
      [3, 2],
    ],
    (a, b) => einsum('ij,jk->ik', a as Value, b as Value),
  )
  check(
    'einsum batched',
    [
      [2, 2, 3],
      [2, 3, 2],
    ],
    (a, b) => einsum('bij,bjk->bik', a as Value, b as Value),
  )
  check('einsum trace', [[3, 3]], (a) => einsum('ii->', a))
  check('einsum outer and transpose', [[2], [3]], (a, b) => einsum('i,j->ji', a as Value, b as Value))
  check(
    'kron',
    [
      [2, 2],
      [2, 3],
    ],
    kron,
  )
  check('diagonal with offset', [[3, 4]], (x) => diagonal(x, 1))
  check('trace', [[3, 3]], (x) => trace(x))
})

describe('integer and boolean reductions', () => {
  it('reduce int32 and bool along an axis (a past bug: integer accumulators could not hold +-Infinity)', () => {
    const ints = T.tensor(
      [
        [3, 1],
        [-2, 5],
      ],
      { dtype: 'int32' },
    )
    expect(T.toArray(T.min(ints, -1))).toEqual([1, -2])
    expect(T.toArray(T.max(ints, -1))).toEqual([3, 5])
    const flags = T.greaterEqual(
      T.tensor([
        [1, 3],
        [0, -6],
      ]),
      0,
    )
    expect(T.toArray(T.all(flags, -1))).toEqual([1, 0])
    expect(T.toArray(T.any(flags, -1))).toEqual([1, 1])
  })
})
