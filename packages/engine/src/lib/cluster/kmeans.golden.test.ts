import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { lloydAlgorithm } from './kmeans.ts'

const fit = (x: Tensor, init: Tensor | undefined) => {
  const rows = toArray(init as Tensor) as number[][]
  return runAlgorithm(lloydAlgorithm(x, rows, { tol: 1e-10 }), { maxSteps: 300 }).state
}

goldenSuite('k-means with fixed initial centres against scikit-learn', 'methods/cluster/kmeans', {
  centres: ([x, init]) => tensor(fit(x, init).centres),
  labels: ([x, init]) => tensor(fit(x, init).labels),
  inertia: ([x, init]) => fit(x, init).inertia,
})
