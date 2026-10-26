import type { Metric } from '../../lib/numerics/geometry/index.ts'
import type { Tensor } from '../../lib/tensor/index.ts'
import { learnGolden, num, str } from '../../testing/learn.ts'
import { kNeighborsClassifier, kNeighborsRegressor, type NeighboursHyper, radiusNeighborsRegressor } from './knn.ts'

learnGolden('k-nearest neighbours against scikit-learn', 'methods/neighbours/knn', {
  knn: (inputs, p) => {
    const [x, y, t, q] = inputs as readonly [Tensor, Tensor, Tensor, Tensor]
    const metric = (str(p, 'metric') === 'manhattan' ? 'cityblock' : 'euclidean') as Metric
    const weights = str(p, 'weights') as NeighboursHyper['weights']
    const c = kNeighborsClassifier({ nNeighbors: 5, weights, metric }).fit({ x, y })
    const r = kNeighborsRegressor({ nNeighbors: 4, weights, metric }).fit({ x, y: t })
    return { proba: c.predictProba?.(q) as Tensor, predict: r.predict(q) }
  },
  radius: (inputs, p) => {
    const [x, t, q] = inputs as readonly [Tensor, Tensor, Tensor]
    return {
      predict: radiusNeighborsRegressor({ radius: num(p, 'radius') })
        .fit({ x, y: t })
        .predict(q),
    }
  },
})
