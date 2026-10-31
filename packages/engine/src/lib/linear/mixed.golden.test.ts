import { learnGolden, num } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { mixedModel } from './mixed.ts'

learnGolden('linear mixed models against statsmodels MixedLM', 'linear/mixed', {
  mixed: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const f = mixedModel({
      reml: num(p, 'reml') === 1,
      randomSlopes: num(p, 'slope') === 1 ? [0] : [],
      tol: 1e-8,
      maxIter: 300,
    }).fit({ x, y })
    const q = f.params.randomNames.length
    return {
      coef: f.params.coef,
      intercept: f.params.intercept,
      sigma2: f.params.sigma2,
      randomCovariance: f.params.randomCovariance,
      logLikelihood: f.params.logLikelihood,
      bse: f.params.coef
        .map((_, j) => Math.sqrt(f.params.fixedCovariance[j]?.[j] ?? Number.NaN))
        .concat(Math.sqrt(f.params.fixedCovariance[2]?.[2] ?? Number.NaN)),
      randomEffects: f.params.randomEffects.map((r) => r.slice(0, q)),
    }
  },
})
