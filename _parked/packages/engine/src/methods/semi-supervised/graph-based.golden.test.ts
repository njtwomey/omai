import { type Tensor, tensor } from '../../lib/tensor/index.ts'
import { learnGolden, num, str } from '../../testing/learn.ts'
import { labelPropagation, labelSpreading } from './graph-based.ts'

learnGolden('label propagation and spreading against scikit-learn', 'methods/semi_supervised/graph_based', {
  graph: (inputs, p) => {
    const [x, y] = inputs as readonly [Tensor, Tensor]
    const hyper = {
      kernel: str(p, 'kernel') as 'rbf' | 'knn',
      gamma: num(p, 'gamma'),
      nNeighbors: num(p, 'n_neighbors'),
      alpha: num(p, 'alpha'),
      maxIter: num(p, 'max_iter'),
    }
    const make = str(p, 'variant') === 'propagation' ? labelPropagation : labelSpreading
    return { distributions: tensor(make(hyper).fit({ x, y }).params.labelDistributions) }
  },
})
