import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { kMeans, lloydAlgorithm } from './kmeans.ts'

const fit = (x: Tensor, init: Tensor | undefined) => {
  const rows = toArray(init as Tensor) as number[][]
  return runAlgorithm(lloydAlgorithm(x, rows, { tol: 1e-10 }), { maxSteps: 300 }).state
}

goldenSuite('k-means with fixed initial centres against scikit-learn', 'methods/cluster/kmeans', {
  centres: ([x, init]) => tensor(fit(x, init).centres),
  labels: ([x, init]) => tensor(fit(x, init).labels),
  inertia: ([x, init]) => fit(x, init).inertia,
})

const fitEstimator = (x: Tensor, init: Tensor | undefined) =>
  kMeans
    .with({ nClusters: (init as Tensor).shape[0] ?? 1, init: toArray(init as Tensor) as number[][], tol: 1e-10 })
    .fit({ x })

goldenSuite('the KMeans estimator with fixed initial centres against scikit-learn', 'methods/cluster/kmeans', {
  centres: ([x, init]) => tensor(fitEstimator(x, init).params.centres),
  labels: ([x, init]) => tensor(fitEstimator(x, init).params.labels),
  inertia: ([x, init]) => fitEstimator(x, init).params.inertia,
})
