/**
 * The migration rules of the option vectors of `optim` that take tensor-likes (D-009): the starting points and constants
 * of ADMM, the weights of the barycentres, the warm start of the lasso and the known gradient of a line search. A tensor,
 * a typed array and a nested array give the same result, a matrix where a vector is required is a `shape` error, and a
 * traced value is rejected with `not-differentiable`.
 */

import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import { square, sum, type TensorLike, tensor, toFlat, type Value } from '../tensor/index.ts'
import { admm } from './admm.ts'
import { lineSearch } from './line-search.ts'
import { lassoCoordinateDescent } from './sparse.ts'
import { freeSupportBarycenter, sinkhornBarycenter } from './wasserstein.ts'

const rules: [string, (v: TensorLike) => unknown, number[]][] = [
  [
    'admm x0',
    (x0) =>
      toFlat(
        admm(
          {
            xUpdate: (z, u, rho) => z.map((v, i) => (1 + rho * (v - (u[i] ?? 0))) / (1 + rho)),
            zUpdate: (ax, u, rho) => ax.map((v, i) => (-1 + rho * (v + (u[i] ?? 0))) / (1 + rho)),
          },
          { x0 },
        ).x,
      ),
    [0.5],
  ],
  [
    'admm c',
    (c) =>
      toFlat(
        admm(
          {
            xUpdate: (z, u, rho) => z.map((v, i) => (1 + rho * (v - (u[i] ?? 0))) / (1 + rho)),
            zUpdate: (ax, u, rho) => ax.map((v, i) => (-1 + rho * (v + (u[i] ?? 0))) / (1 + rho)),
            c,
          },
          { x0: [0] },
        ).x,
      ),
    [0],
  ],
  [
    'sinkhornBarycenter weights',
    (weights) =>
      toFlat(
        sinkhornBarycenter(
          [
            [0.5, 0.1],
            [0.5, 0.9],
          ],
          [
            [0, 1],
            [1, 0],
          ],
          0.5,
          { weights },
        ).barycenter,
      ),
    [0.3, 0.7],
  ],
  [
    'freeSupportBarycenter weights',
    (weights) => toFlat(freeSupportBarycenter([[[0]], [[2]]], [[1], [1]], [[5]], { weights }).support),
    [0.3, 0.7],
  ],
  [
    'lassoCoordinateDescent x0',
    (x0) =>
      toFlat(
        lassoCoordinateDescent(
          [
            [1, 0],
            [0, 1],
          ],
          [3, 0.5],
          1,
          { x0 },
        ).coef,
      ),
    [0.5, 0.5],
  ],
  [
    'lineSearch grad',
    (g) => lineSearch((x) => sum(square(x as Value)), [1, 2], [-1, -2], { value: 5, grad: g }).alpha,
    [2, 4],
  ],
]

describe('tensor-like option vectors (migration, D-009)', () => {
  it('a tensor, a typed array and a list agree; a matrix is a shape error; a traced one is not differentiable', () => {
    for (const [name, f, values] of rules) {
      const expected = f(values)
      expect(f(Float64Array.from(values)), name).toEqual(expected)
      expect(f(tensor(values)), name).toEqual(expected)
      expect(() => f(tensor([values, values])), name).toThrow(expect.objectContaining({ code: 'shape' }))
      expect(() => grad((a: Value) => (f(a as TensorLike), sum(a)))(tensor(values)), name).toThrow(
        expect.objectContaining({ code: 'not-differentiable' }),
      )
    }
  })
})
