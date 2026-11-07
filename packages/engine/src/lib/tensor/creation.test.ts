import { describe, expect, it } from 'vitest'
import { expectRows } from '../../testing/close.ts'
import * as T from './index.ts'
import { type NestedArray, toArray, type Value } from './index.ts'

describe('creation helpers agree with numpy', () => {
  // [name, value under test, numpy's answer] (one-off numpy 2.5 run)
  const rows: [string, () => Value, NestedArray][] = [
    ['logspace(0, 3, 4)', () => T.logspace(0, 3, 4), [1, 10, 100, 1000]],
    ['logspace with base 2', () => T.logspace(0, 4, 5, 2), [1, 2, 4, 8, 16]],
    ['logspace with one point', () => T.logspace(2, 5, 1), [100]],
    [
      'geomspace(1, 1000, 4) is exact at the ends and powers of ten between',
      () => T.geomspace(1, 1000, 4),
      [1, 10, 100, 1000],
    ],
    ['geomspace with negative endpoints', () => T.geomspace(-1, -100, 3), [-1, -10, -100]],
    ['geomspace of 2 to 32 in 5 points', () => T.geomspace(2, 32, 5), [2, 4, 7.999999999999999, 16, 32]],
    ['geomspace with one point', () => T.geomspace(5, 50, 1), [5]],
    [
      'eye(3, 4, -1)',
      () => T.eye(3, 4, -1),
      [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
      ],
    ],
    [
      'eye with an offset past the width is zeros',
      () => T.eye(2, 2, 5),
      [
        [0, 0],
        [0, 0],
      ],
    ],
    [
      'tri(3, 4, 1)',
      () => T.tri(3, 4, 1),
      [
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
      ],
    ],
    [
      'tri(2, 3, -5) is zeros',
      () => T.tri(2, 3, -5),
      [
        [0, 0, 0],
        [0, 0, 0],
      ],
    ],
    [
      'diag(v, 1)',
      () => T.diag(T.tensor([1, 2, 3]), 1),
      [
        [0, 1, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 3],
        [0, 0, 0, 0],
      ],
    ],
    [
      'diag(v, -2)',
      () => T.diag(T.tensor([1, 2]), -2),
      [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 2, 0, 0],
      ],
    ],
    [
      'meshgrid xy: x varies along columns',
      () => T.stack(T.meshgrid([T.tensor([1, 2, 3]), T.tensor([4, 5])])),
      [
        [
          [1, 2, 3],
          [1, 2, 3],
        ],
        [
          [4, 4, 4],
          [5, 5, 5],
        ],
      ],
    ],
    [
      'meshgrid ij',
      () => T.stack(T.meshgrid([T.tensor([1, 2, 3]), T.tensor([4, 5])], 'ij')),
      [
        [
          [1, 1],
          [2, 2],
          [3, 3],
        ],
        [
          [4, 5],
          [4, 5],
          [4, 5],
        ],
      ],
    ],
    [
      'meshgrid of three axes, ij',
      () => T.stack(T.meshgrid([T.tensor([1, 2]), T.tensor([3, 4]), T.tensor([5, 6])], 'ij')),
      [
        [
          [
            [1, 1],
            [1, 1],
          ],
          [
            [2, 2],
            [2, 2],
          ],
        ],
        [
          [
            [3, 3],
            [4, 4],
          ],
          [
            [3, 3],
            [4, 4],
          ],
        ],
        [
          [
            [5, 6],
            [5, 6],
          ],
          [
            [5, 6],
            [5, 6],
          ],
        ],
      ],
    ],
    ['meshgrid of one axis', () => T.stack(T.meshgrid([T.tensor([1, 2])])), [[1, 2]]],
  ]
  it("gives numpy's values and shapes for logspace, geomspace, eye, tri, diag and meshgrid", () => {
    expectRows(rows)
  })

  it('keeps the dtype in onesLike, fullLike and zerosLike, and rejects endpoints geomspace cannot use', () => {
    for (const dtype of ['bool', 'int32', 'float32', 'float64'] as const) {
      const x = T.tensor([1, 0], { dtype })
      expect([T.onesLike(x), T.fullLike(x, 1), T.zerosLike(x)].map((t) => [t.dtype, toArray(t)])).toEqual([
        [dtype, [1, 1]],
        [dtype, [1, 1]],
        [dtype, [0, 0]],
      ])
    }
    for (const [a, b] of [
      [0, 5],
      [5, 0],
      [-1, 5],
    ] as const)
      expect(() => T.geomspace(a, b, 3), `${a} ${b}`).toThrow(expect.objectContaining({ code: 'value' }))
  })
})

describe('flatten and chunk', () => {
  it('flatten is row-major reshape(-1) for every rank; chunk makes pieces of ceil(n / k), as torch.chunk does', () => {
    const x = T.reshape(T.arange(24), [2, 3, 4])
    expect(toArray(T.flatten(x))).toEqual(toArray(T.arange(24)))
    expect([T.flatten(T.tensor(5)).shape, T.flatten(T.zeros([0, 3])).shape]).toEqual([[1], [0]])
    // [length, chunks asked for, sizes]: torch.chunk(torch.arange(n), k)
    const cases: [number, number, number[]][] = [
      [5, 2, [3, 2]],
      [6, 4, [2, 2, 2]],
      [6, 3, [2, 2, 2]],
      [7, 3, [3, 3, 1]],
      [1, 5, [1]],
    ]
    for (const [n, k, sizes] of cases)
      expect(
        T.chunk(T.arange(n), k).map((c) => c.shape[0]),
        `${n} in ${k}`,
      ).toEqual(sizes)
    const along = T.chunk(T.reshape(T.arange(12), [3, 4]), 2, 1)
    expect(along.map((c) => toArray(c))).toEqual([
      [
        [0, 1],
        [4, 5],
        [8, 9],
      ],
      [
        [2, 3],
        [6, 7],
        [10, 11],
      ],
    ])
  })
})
