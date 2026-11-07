import { describe, expect, it } from 'vitest'
import { expectRows } from '../../../testing/close.ts'
import * as V from '../../autodiff/index.ts'
import type { NestedArray, Value } from '../index.ts'
import * as T from '../index.ts'

const nan = Number.NaN
const M = [
  [3, 1, 2, 5],
  [0, -1, 4, 4],
  [2, 2, 2, 1],
]

describe('indexing operations agree with numpy', () => {
  // [name, value under test, numpy's answer] (numpy 2.5: argsort with kind="stable", take_along_axis, eye, bincount,
  // add.at, multiply.at, maximum.at, minimum.at, searchsorted; the vmap rows are the same call row by row)
  const rows: [string, () => Value, NestedArray][] = [
    ['sort descending with ties', () => T.sort(T.tensor([3, 1, 3, 2]), -1, true), [3, 3, 2, 1]],
    ['argsort descending is stable on ties', () => T.argsort(T.tensor([2, 1, 2, 1]), -1, true), [0, 2, 1, 3]],
    [
      'argsort along axis 0',
      () => T.argsort(T.tensor(M), 0),
      [
        [1, 1, 0, 2],
        [2, 0, 2, 1],
        [0, 2, 1, 0],
      ],
    ],
    [
      'argsort along the last axis, descending',
      () => T.argsort(T.tensor(M), -1, true),
      [
        [3, 0, 2, 1],
        [2, 3, 0, 1],
        [0, 1, 2, 3],
      ],
    ],
    [
      'sort along axis 0',
      () => T.sort(T.tensor(M), 0),
      [
        [0, -1, 2, 1],
        [2, 1, 2, 4],
        [3, 2, 4, 5],
      ],
    ],
    [
      'topk values and indices, largest',
      () =>
        T.concat([
          T.topk(T.tensor([3, 5, 5, 1]), 3).values,
          T.cast(T.topk(T.tensor([3, 5, 5, 1]), 3).indices, 'float64'),
        ]),
      [5, 5, 3, 1, 2, 0],
    ],
    [
      'topk smallest',
      () =>
        T.concat([
          T.topk(T.tensor([3, 5, 5, 1]), 2, -1, false).values,
          T.cast(T.topk(T.tensor([3, 5, 5, 1]), 2, -1, false).indices, 'float64'),
        ]),
      [1, 3, 3, 0],
    ],
    [
      'topk along axis 0 of a matrix',
      () => T.topk(T.tensor(M), 2, 0).values,
      [
        [3, 2, 4, 5],
        [2, 1, 2, 4],
      ],
    ],
    ['topk with k above the length returns everything', () => T.topk(T.tensor([2, 7]), 5).values, [7, 2]],
    [
      'takeAlongAxis along axis 1',
      () =>
        T.takeAlongAxis(
          T.tensor(M),
          T.tensor(
            [
              [3, 0],
              [1, 1],
              [2, 0],
            ],
            { dtype: 'int32' },
          ),
          1,
        ),
      [
        [5, 3],
        [-1, -1],
        [2, 2],
      ],
    ],
    [
      'takeAlongAxis along axis 0',
      () => T.takeAlongAxis(T.tensor(M), T.tensor([[2, 0, 1, 2]], { dtype: 'int32' }), 0),
      [[2, 1, 4, 1]],
    ],
    [
      'oneHot of a matrix of labels',
      () =>
        T.oneHot(
          T.tensor(
            [
              [0, 2],
              [3, 1],
            ],
            { dtype: 'int32' },
          ),
          4,
        ),
      [
        [
          [1, 0, 0, 0],
          [0, 0, 1, 0],
        ],
        [
          [0, 0, 0, 1],
          [0, 1, 0, 0],
        ],
      ],
    ],
    [
      'oneHot outside the classes is a row of zeros',
      () => T.oneHot(T.tensor([5, -1, 1], { dtype: 'int32' }), 3),
      [
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
      ],
    ],
    [
      'bincount with weights and minLength',
      () => T.bincount(T.tensor([0, 2, 2, 3], { dtype: 'int32' }), 6, T.tensor([0.5, 1, 2, -1])),
      [0.5, 0, 3, -1, 0, 0],
    ],
    [
      'segmentSum with repeated segments',
      () => T.segmentSum(T.tensor([1, 2, 3, 4, 5]), T.tensor([2, 0, 2, 0, 1], { dtype: 'int32' }), 3),
      [6, 5, 4],
    ],
    [
      'scatterAdd along axis 1 accumulates repeats',
      () =>
        T.scatterAdd(
          [2, 4],
          T.tensor(
            [
              [3, 3],
              [0, 3],
            ],
            { dtype: 'int32' },
          ),
          T.tensor([
            [1, 2],
            [3, 4],
          ]),
          1,
        ),
      [
        [0, 0, 0, 3],
        [3, 0, 0, 4],
      ],
    ],
    [
      'scatter mode set with a repeated index',
      () =>
        T.scatter(
          T.tensor([1, 1, 1]),
          T.tensor([[0], [2], [0], [1]], { dtype: 'int32' }),
          T.tensor([2, 3, 4, 5]),
          'set',
        ),
      [4, 5, 3],
    ],
    [
      'scatter mode add with a repeated index',
      () =>
        T.scatter(
          T.tensor([1, 1, 1]),
          T.tensor([[0], [2], [0], [1]], { dtype: 'int32' }),
          T.tensor([2, 3, 4, 5]),
          'add',
        ),
      [7, 6, 4],
    ],
    [
      'scatter mode mul with a repeated index',
      () =>
        T.scatter(
          T.tensor([1, 1, 1]),
          T.tensor([[0], [2], [0], [1]], { dtype: 'int32' }),
          T.tensor([2, 3, 4, 5]),
          'mul',
        ),
      [8, 5, 3],
    ],
    [
      'scatter mode max with a repeated index',
      () =>
        T.scatter(
          T.tensor([1, 1, 1]),
          T.tensor([[0], [2], [0], [1]], { dtype: 'int32' }),
          T.tensor([2, 3, 4, 5]),
          'max',
        ),
      [4, 5, 3],
    ],
    [
      'scatter mode min with a repeated index',
      () =>
        T.scatter(
          T.tensor([1, 1, 1]),
          T.tensor([[0], [2], [0], [1]], { dtype: 'int32' }),
          T.tensor([2, 3, 4, 5]),
          'min',
        ),
      [1, 1, 1],
    ],
    [
      'gather of coordinate pairs',
      () =>
        T.gather(
          T.tensor(M),
          T.tensor(
            [
              [0, 3],
              [2, 0],
              [1, 2],
            ],
            { dtype: 'int32' },
          ),
        ),
      [5, 2, 4],
    ],
    [
      'searchsorted of a matrix of values keeps its shape',
      () =>
        T.searchsorted(
          T.tensor([1, 3, 5]),
          T.tensor([
            [0, 3],
            [4, 9],
          ]),
        ),
      [
        [0, 1],
        [2, 3],
      ],
    ],
    [
      'vmap of argsort over rows is argsort along the last axis',
      () => V.vmap(T.argsort)(T.tensor(M)),
      [
        [1, 2, 0, 3],
        [1, 0, 2, 3],
        [3, 0, 1, 2],
      ],
    ],
    [
      'vmap of topk values over rows',
      () => V.vmap((r: T.Value) => T.topk(r, 2).values)(T.tensor(M)),
      [
        [5, 3],
        [4, 4],
        [2, 2],
      ],
    ],
    [
      'vmap of takeAlongAxis with batched indices',
      () =>
        V.vmap((r: T.Value, i: T.Value) => T.takeAlongAxis(r, i, 0))(
          T.tensor(M),
          T.tensor(
            [
              [3, 0],
              [1, 1],
              [2, 0],
            ],
            { dtype: 'int32' },
          ),
        ),
      [
        [5, 3],
        [-1, -1],
        [2, 2],
      ],
    ],
    [
      'vmap of take with batched data and shared indices',
      () => V.vmap((r: T.Value) => T.take(r, [3, 0, 3]))(T.tensor(M)),
      [
        [5, 3, 5],
        [4, 0, 4],
        [1, 2, 1],
      ],
    ],
  ]
  it('sorts, selects, scatters and counts like numpy, under vmap too, and rejects bad arguments', () => {
    expectRows(rows)
    const invalid: [string, () => Value][] = [
      ['bincount of a negative value', () => T.bincount(T.tensor([1, -1], { dtype: 'int32' }))],
      [
        'takeAlongAxis with indices of another rank',
        () => T.takeAlongAxis(T.tensor(M), T.tensor([0, 1], { dtype: 'int32' }), 0),
      ],
      [
        'takeAlongAxis with indices of another length',
        () => T.takeAlongAxis(T.tensor(M), T.tensor([[0, 1]], { dtype: 'int32' }), 1),
      ],
      ['gather with a scalar index array', () => T.gather(T.tensor(M), T.tensor(1, { dtype: 'int32' }))],
      ['sort along an axis that does not exist', () => T.sort(T.tensor(M), 2)],
    ]
    for (const [name, thunk] of invalid)
      expect(thunk, name).toThrow(expect.objectContaining({ code: expect.stringMatching(/value|shape/) }))
    // scatterAdd differentiates in its updates, and its second derivative is zero (it is linear)
    const spots = T.tensor(
      [
        [3, 3],
        [0, 3],
      ],
      { dtype: 'int32' },
    )
    expect(
      V.checkPrimitive(
        (u) => T.scatterAdd([2, 4], spots, u, 1),
        [
          T.tensor([
            [1, 2],
            [3, 4],
          ]),
        ],
        { order: 2 },
      ).failures,
    ).toEqual([])
    // NaN sorts last, as in numpy
    expect(T.toArray(T.sort(T.tensor([3, nan, 1])))).toEqual([1, 3, nan])
    expect(
      V.checkPrimitive((x) => T.sort(x, -1, true), [T.tensor([0.3, -1.2, 2.1, 0.7])], { order: 2 }).failures,
    ).toEqual([])
  })
})
