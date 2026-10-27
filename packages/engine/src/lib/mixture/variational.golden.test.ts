import { learnGolden, num, str } from '../../testing/learn.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'
import type { CovarianceType } from './gaussian.ts'
import {
  expectedWeights,
  type VariationalPosterior,
  variationalMixtureAlgorithm,
  variationalPriors,
  variationalResponsibilities,
  type WeightPrior,
} from './variational.ts'

learnGolden('Bayesian Gaussian mixture by variational inference against scikit-learn', 'mixture/variational', {
  variational: (inputs, p) => {
    const [x, resp] = inputs as [Tensor, Tensor]
    const start = toArray(resp) as number[][]
    const covarianceType = str(p, 'covarianceType') as CovarianceType
    const weightPrior = str(p, 'weightPrior') as WeightPrior
    const priors = variationalPriors(x, start[0]?.length ?? 0, {
      covarianceType,
      weightConcentration: num(p, 'concentration'),
    })
    const out = runAlgorithm(variationalMixtureAlgorithm(x, start, priors, { covarianceType, weightPrior, tol: 0 }), {
      maxSteps: num(p, 'iterations'),
    })
    const post: VariationalPosterior = out.state.posterior
    return {
      weights: expectedWeights(post),
      means: post.means,
      covariances: tensor(post.covariances),
      degreesOfFreedom: post.degreesOfFreedom,
      meanPrecision: post.meanPrecision,
      concentration: tensor(post.concentration),
      lowerBound: out.state.lowerBound,
      predictProba: variationalResponsibilities(x, post).resp,
    }
  },
})
