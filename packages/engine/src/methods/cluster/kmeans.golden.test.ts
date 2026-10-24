import { type Tensor, tensor, toArray } from '../../lib/tensor/index.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { kmeans } from './kmeans.ts'

const fit = (x: Tensor, init: Tensor | undefined) => {
  const rows = toArray(init as Tensor) as number[][]
  return kmeans({ nClusters: rows.length, init: rows, tol: 1e-10 }).fit({ x })
}

goldenSuite('k-means with fixed initial centres against scikit-learn', 'methods/cluster/kmeans', {
  centres: ([x, init]) => tensor(fit(x, init).params.centres),
  labels: ([x, init]) => tensor(fit(x, init).params.labels),
  inertia: ([x, init]) => fit(x, init).params.inertia,
})
