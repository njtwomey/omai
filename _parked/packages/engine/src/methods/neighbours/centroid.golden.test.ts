import { sliceAxis, type Tensor } from '../../lib/tensor/index.ts'
import { learnGolden, num } from '../../testing/learn.ts'
import { nearestCentroid } from './centroid.ts'

learnGolden('nearest centroids against scikit-learn', 'methods/neighbours/centroid', {
  centroid: (inputs, p) => {
    const [x, y] = inputs as readonly [Tensor, Tensor]
    const shrink = num(p, 'shrink')
    const f = nearestCentroid(shrink > 0 ? { shrinkThreshold: shrink } : {}).fit({ x, y })
    return { centroids: f.params.centroids, predict: f.predict(sliceAxis(x, 0, 0, 10)) }
  },
})
