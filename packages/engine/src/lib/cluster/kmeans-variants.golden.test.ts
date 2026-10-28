import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import { elkanAlgorithm } from './kmeans-variants.ts'

const fit = (x: Tensor, init: Tensor | undefined) =>
  runAlgorithm(elkanAlgorithm(x, toArray(init as Tensor) as number[][], { tol: 1e-10 }), { maxSteps: 300 }).state

goldenSuite("Elkan's k-means with fixed initial centres against scikit-learn", 'methods/cluster/kmeans', {
  centres: ([x, init]) => tensor(fit(x, init).centres),
  labels: ([x, init]) => tensor(fit(x, init).labels),
  inertia: ([x, init]) => {
    const s = fit(x, init)
    const rows = toArray(x) as number[][]
    return rows.reduce(
      (t, r, i) => t + r.reduce((a, v, q) => a + (v - (s.centres[s.labels[i] ?? 0]?.[q] ?? 0)) ** 2, 0),
      0,
    )
  },
})
