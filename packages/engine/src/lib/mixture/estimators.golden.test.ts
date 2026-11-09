import { learnGolden, str } from '../../testing/learn.ts'
import { type Tensor, tensor, toArray, toFlat } from '../tensor/index.ts'
import { gaussianMixture } from './estimators.ts'
import type { CovarianceType } from './gaussian.ts'

learnGolden('the GaussianMixture estimator against scikit-learn from given parameters', 'mixture/gaussian_estimator', {
  mixture: (inputs, p) => {
    const [x, held, weights, means, covs] = inputs as [Tensor, Tensor, Tensor, Tensor, Tensor]
    const covarianceType = str(p, 'covarianceType') as CovarianceType
    const f = gaussianMixture
      .with({
        nComponents: 3,
        covarianceType,
        tol: 1e-10,
        maxIter: 500,
        init: {
          weights: Array.from(toFlat(weights)),
          means: toArray(means) as number[][],
          covariances: toArray(covs) as number[][][],
        },
      })
      .fit({ x })
    return {
      weights: f.params.weights,
      means: f.params.means,
      covariances: tensor(f.params.covariances),
      predictProba: f.predictProba(held),
      labels: f.predict(held),
      scoreSamples: f.scoreSamples(held),
      score: f.score({ x: held }),
      bic: f.bic(held),
      aic: f.aic(held),
    }
  },
})
