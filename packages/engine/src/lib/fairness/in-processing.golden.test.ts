import { goldenSuite } from '../../testing/suite.ts'
import type { Estimator, Fitted, LearnData } from '../learn/index.ts'
import { lstsq } from '../numerics/linalg/index.ts'
import { asTensor, fromStorage, type Tensor, type TensorLike, tensor, toFlat } from '../tensor/index.ts'
import { exponentiatedGradientReduction, gridReduction, type ReductionConstraint } from './in-processing.ts'

/** The oracle's learner: weighted least squares on the 0/1 target, deciding 1 where the fit is at least 0.5. */
function rowsOf(x: TensorLike): number[][] {
  const t = asTensor(x)
  const [n = 0, d = 0] = t.shape
  const v = toFlat(t)
  return Array.from({ length: n }, (_, i) => Array.from({ length: d }, (_, j) => v[i * d + j] ?? 0))
}
const leastSquares = {
  name: 'leastSquaresClassifier',
  kind: 'classifier',
  hyper: {},
  fit(data: LearnData): Fitted {
    const rows = rowsOf(data.x)
    const y = toFlat(asTensor(data.y as TensorLike))
    const w = data.weights === undefined ? rows.map(() => 1) : Array.from(toFlat(asTensor(data.weights)))
    const a = rows.map((r, i) => [1, ...r].map((v) => v * Math.sqrt(w[i] ?? 0)))
    const b = rows.map((_, i) => [(y[i] ?? 0) * Math.sqrt(w[i] ?? 0)])
    const beta = toFlat(lstsq(a, b) as Tensor)
    return {
      estimator: 'leastSquaresClassifier',
      kind: 'classifier',
      params: [],
      predict: (x: TensorLike) => {
        const r = rowsOf(x)
        return fromStorage(
          Float64Array.from(r, (row) =>
            (beta[0] ?? 0) + row.reduce((s, v, j) => s + v * (beta[j + 1] ?? 0), 0) >= 0.5 ? 1 : 0,
          ),
          [r.length],
          'float64',
        )
      },
    } as unknown as Fitted
  },
} as unknown as Estimator

type Inputs = readonly (Tensor | undefined)[]
const dataOf = ([x, y, g]: Inputs) => ({
  x: x as Tensor,
  labels: Array.from(toFlat(y as Tensor)),
  groups: Array.from(toFlat(g as Tensor)),
})
const eg = (inputs: Inputs, o: Readonly<Record<string, unknown>>) =>
  exponentiatedGradientReduction(leastSquares, dataOf(inputs), {
    constraint: o['constraint'] as ReductionConstraint,
    epsilon: 0.02,
  })
const grid = (inputs: Inputs, o: Readonly<Record<string, unknown>>) =>
  gridReduction(leastSquares, dataOf(inputs), { constraint: o['constraint'] as ReductionConstraint, gridSize: 11 })

goldenSuite('exponentiated gradient and grid search against fairlearn', 'fairness/in_processing', {
  egWeights: (inputs, o) => tensor([...eg(inputs, o).weights]),
  egDecisions: (inputs, o) => tensor([...eg(inputs, o).decisions]),
  egBestIteration: (inputs, o) => eg(inputs, o).bestIteration,
  gridErrors: (inputs, o) => tensor(grid(inputs, o).map((p) => p.error)),
  gridViolations: (inputs, o) => tensor(grid(inputs, o).map((p) => p.violation)),
  gridBest: (inputs, o) => grid(inputs, o).findIndex((p) => p.best),
})
