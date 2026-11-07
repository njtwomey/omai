import { describe, it } from 'vitest'
import { expectRows } from '../../../testing/close.ts'
import type { NestedArray, Value } from '../../tensor/index.ts'
import * as T from '../../tensor/index.ts'
import * as D from '../index.ts'

describe('the LKJ distribution against closed forms', () => {
  // [name, value under test, reference]: the 2 x 2 case is a beta on the correlation (scipy.stats.beta), the 3 x 3 uniform
  // case has the volume pi^2 / 2 of the 3 x 3 correlation matrices, and the moments are those of Lewandowski et al.
  const rows: [string, () => Value, NestedArray][] = [
    [
      'lkj(2, 2.5): the correlation r is 2 Beta(2.5, 2.5) - 1',
      () =>
        D.lkj(2, 2.5).logProb(
          T.tensor([
            [
              [1, 0.3],
              [0.3, 1],
            ],
            [
              [1, -0.8],
              [-0.8, 1],
            ],
          ]),
        ),
      [-0.3053666520445363, -1.6963775041356461],
    ],
    [
      'lkj(3, 1) is uniform on the 3 x 3 correlation matrices, of volume pi^2 / 2',
      () =>
        D.lkj(3, 1).logProb(
          T.tensor([
            [1, 0.2, 0.1],
            [0.2, 1, -0.3],
            [0.1, -0.3, 1],
          ]),
        ),
      -1.596312591138855,
    ],
    [
      'lkj(3, 2): mean and mode are the identity, off-diagonal variance 1 / (2 eta + d - 1)',
      () => T.stack([D.lkj(3, 2).mean(), D.lkj(3, 2).mode(), D.lkj(3, 2).variance()]),
      [
        [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        [
          [0, 0.16666666666666666, 0.16666666666666666],
          [0.16666666666666666, 0, 0.16666666666666666],
          [0.16666666666666666, 0.16666666666666666, 0],
        ],
      ],
    ],
    [
      'wishart and inverse Wishart outside the support (not positive definite) are -Infinity, not an error',
      () =>
        T.stack([
          D.wishart(3, {
            covariance: [
              [1, 0],
              [0, 1],
            ],
          }).logProb(
            T.tensor([
              [1, 2],
              [2, 1],
            ]),
          ),
          D.inverseWishart(3, T.eye(2)).logProb(
            T.tensor([
              [1, 2],
              [2, 1],
            ]),
          ),
        ]),
      [-Infinity, -Infinity],
    ],
    [
      'lkj with no positive definite argument is outside the support',
      () =>
        D.lkj(2, 1).logProb(
          T.tensor([
            [1, 1.5],
            [1.5, 1],
          ]),
        ),
      -Infinity,
    ],
  ]
  it('has the beta density for 2 x 2, the uniform density for 3 x 3, the identity as mean and mode, and zero outside', () => {
    expectRows(rows, 1e-10)
  })
})
