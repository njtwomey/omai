import { learnGolden, num } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { tobit } from './tobit.ts'

learnGolden('Tobit regression against a statsmodels GenericLikelihoodModel', 'linear/tobit', {
  tobit: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const f = tobit({ lower: num(p, 'lower'), upper: num(p, 'upper') }).fit({ x, y })
    const m = f.params.covariance.length
    return {
      coef: f.params.coef,
      intercept: f.params.intercept,
      sigma: f.params.sigma,
      logLikelihood: f.params.logLikelihood,
      bse: Array.from({ length: m }, (_, j) => Math.sqrt(f.params.covariance[j]?.[j] ?? Number.NaN)),
    }
  },
})
