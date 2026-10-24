import type { Tensor } from '../../lib/tensor/index.ts'
import { learnGolden } from '../../testing/learn.ts'
import { gaussianProcessRegressor } from './regression.ts'

learnGolden('Gaussian-process regression against scikit-learn', 'methods/gp/regression', {
  gpr: (inputs, p) => {
    const [x, y, q] = inputs as readonly [Tensor, Tensor, Tensor]
    const f = gaussianProcessRegressor({ alpha: 0.01, optimize: p['optimize'] === true, surfaceResolution: 0 }).fit({
      x,
      y,
    })
    return { lml: f.params.logMarginalLikelihood, mean: f.predict(q), std: f.predictStd(q) }
  },
})
