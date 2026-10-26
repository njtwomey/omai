import { type Tensor, tensor } from '../../lib/tensor/index.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { localOutlierFactor } from './neighbours.ts'

goldenSuite('local outlier factors against scikit-learn', 'methods/anomaly/neighbours', {
  train: ([x], o) => tensor(localOutlierFactor({ nNeighbors: o['k'] as number }).fit({ x }).params.lof),
  novel: ([x, q], o) =>
    localOutlierFactor({ nNeighbors: o['k'] as number })
      .fit({ x })
      .scoreSamples(q as Tensor),
})
