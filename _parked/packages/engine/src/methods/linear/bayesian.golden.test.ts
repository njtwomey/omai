import { sliceAxis, type Tensor } from '../../lib/tensor/index.ts'
import { learnGolden } from '../../testing/learn.ts'
import { ardRegression, bayesianRidge } from './bayesian.ts'
import { isotonicRegression } from './isotonic.ts'

type In = readonly [Tensor, Tensor, Tensor?]

learnGolden('Bayesian linear models and isotonic regression against scikit-learn', 'methods/linear/bayesian', {
  bayesianRidge: (inputs) => {
    const [x, y] = inputs as In
    const f = bayesianRidge().fit({ x, y })
    const d = f.params.coef.length
    return {
      coef: f.params.coef,
      intercept: f.params.intercept,
      alpha: f.params.alpha,
      lambda: f.params.lambda[0] ?? 0,
      sigma: Array.from({ length: d }, (_, i) => f.params.sigma.slice(i * d, (i + 1) * d)),
      std: f.predictStd(sliceAxis(x, 0, 0, 5)),
    }
  },
  ard: (inputs) => {
    const [x, y] = inputs as In
    const f = ardRegression().fit({ x, y })
    return { coef: f.params.coef, intercept: f.params.intercept, alpha: f.params.alpha }
  },
  isotonic: (inputs, p) => {
    const [x, y, t] = inputs as In
    return {
      predict: isotonicRegression({ increasing: p['increasing'] === true })
        .fit({ x, y })
        .predict(t as Tensor),
    }
  },
})
