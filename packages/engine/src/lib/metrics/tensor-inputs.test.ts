/**
 * The migration rules of the stats and metrics functions that take their vectors and matrices as tensor-likes (D-009):
 * a tensor, a typed array and a nested array give the same result, a matrix where a vector is required is a `shape`
 * error, and a traced value is rejected with `not-differentiable`.
 */

import { describe, expect, it } from 'vitest'
import { grad } from '../autodiff/index.ts'
import * as S from '../stats/index.ts'
import { neg, square, sum, type Tensor, type TensorLike, tensor, type Value } from '../tensor/index.ts'
import * as M from './index.ts'

/** How a vector, or a 2 by 2 matrix given as four numbers, arrives. */
interface Form {
  readonly vector: (values: readonly number[]) => TensorLike
  readonly matrix: (values: readonly number[]) => TensorLike
}

const forms: Record<string, Form> = {
  'nested array': {
    vector: (v) => [...v],
    matrix: (v) => [
      [v[0] ?? 0, v[1] ?? 0],
      [v[2] ?? 0, v[3] ?? 0],
    ],
  },
  'typed array': {
    vector: (v) => Float64Array.from(v),
    matrix: (v) => [
      [v[0] ?? 0, v[1] ?? 0],
      [v[2] ?? 0, v[3] ?? 0],
    ],
  },
  tensor: {
    vector: (v) => tensor([...v]),
    matrix: (v) =>
      tensor([
        [v[0] ?? 0, v[1] ?? 0],
        [v[2] ?? 0, v[3] ?? 0],
      ]),
  },
}

const t = (f: Form): ((v: readonly number[]) => TensorLike) => f.vector
const costs = [0, 1, 5, 0]
const bowl = (x: Value): Value => neg(sum(square(x)))
const table = {
  labels: [0, 1],
  matrix: [
    [3, 1],
    [2, 4],
  ],
}

/** Each row: the function called with its vector (or matrix) arguments built in the given form. */
const rows: readonly (readonly [string, (f: Form) => unknown])[] = [
  ['ranksOf', (f) => S.ranksOf(t(f)([3, 1, 2, 1]), 'min')],
  ['tieCounts', (f) => S.tieCounts(t(f)([3, 1, 3, 2]))],
  ['auc', (f) => M.auc(t(f)([0, 0.5, 1]), t(f)([0, 1, 1]))],
  ['weightedPercentile', (f) => M.weightedPercentile(t(f)([1, 2, 3, 4]), t(f)([1, 1, 2, 1]), 50)],
  ['meanDifference', (f) => S.meanDifference(t(f)([3, 5, 4]), t(f)([1, 2, 2]))],
  ['alphaSpending', (f) => S.tests.alphaSpending(t(f)([0.5, 1])).boundaries],
  ['proportionsZtest', (f) => S.tests.proportionsZtest(t(f)([15, 10]), t(f)([50, 50])).statistic],
  [
    'eppsSingleton',
    (f) => S.tests.eppsSingleton([1, 2, 3, 4, 5, 9], [2, 3, 6, 7, 8, 9], { t: t(f)([0.4, 0.8]) }).pValue,
  ],
  ['observedInformation', (f) => S.observedInformation(bowl, t(f)([1]))],
  ['newtonMle', (f) => S.newtonMle(bowl, t(f)([0.5]), { maxIterations: 1 }).iterations],
  [
    'bsplineDerivativeDesign',
    (f) => S.smoothing.bsplineDerivativeDesign(t(f)([0, 0, 0, 0, 0.5, 1, 1, 1, 1]), 3, t(f)([0.3]), 1),
  ],
  ['tableCost', (f) => M.tableCost(table, f.matrix(costs)).total],
  ['thresholdCosts', (f) => M.thresholdCosts([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], f.matrix(costs)).best],
]

const plain = (x: unknown): unknown =>
  typeof x === 'object' && x !== null && 'data' in x ? Array.from((x as Tensor).data) : x

describe('tensor-like inputs of the migrated stats and metrics functions', () => {
  it('a tensor, a typed array and a nested array agree; a matrix is a shape error; a traced value is rejected', () => {
    for (const [name, call] of rows) {
      const expected = plain(call(forms['nested array'] as Form))
      for (const [form, make] of Object.entries(forms))
        expect(plain(call(make)), `${name} from a ${form}`).toEqual(expected)
      const wrong: Form = {
        vector: () =>
          tensor([
            [1, 2],
            [3, 4],
          ]),
        matrix: () => tensor([1, 2, 3, 4]),
      }
      // tieCounts reads any rank in row-major order, as before.
      if (name !== 'tieCounts')
        expect(() => call(wrong), `${name} given the wrong rank`).toThrow(expect.objectContaining({ code: 'shape' }))
      const viaGrad = (): unknown =>
        grad((x: Value): Value => {
          const same: Form = { vector: () => x as TensorLike, matrix: () => x as TensorLike }
          call(same)
          return sum(x)
        })(tensor([1, 2, 3, 4]))
      expect(viaGrad, `${name} given a traced value`).toThrow(expect.objectContaining({ code: 'not-differentiable' }))
    }
  })
})
