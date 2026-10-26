import type { Tensor } from '../../lib/tensor/index.ts'
import { learnGolden, num } from '../../testing/learn.ts'
import { type HuberParams, huberRegressor, quantileRegressor, theilSenRegressor } from './robust.ts'

type In = readonly [Tensor, Tensor]

learnGolden('robust regression against scikit-learn', 'methods/linear/robust', {
  huber: (inputs) => {
    const [x, y] = inputs as In
    const f = huberRegressor({ maxIter: 10000 }).fit({ x, y })
    return { coef: f.params.coef, intercept: f.params.intercept, scale: (f.params as HuberParams).sigma }
  },
  theilSen: (inputs) => {
    const [x, y] = inputs as In
    const f = theilSenRegressor().fit({ x, y })
    return { coef: f.params.coef, intercept: f.params.intercept }
  },
  quantile: (inputs, p) => {
    const [x, y] = inputs as In
    const f = quantileRegressor({ quantile: num(p, 'quantile'), alpha: num(p, 'alpha') }).fit({ x, y })
    return { coef: f.params.coef, intercept: f.params.intercept }
  },
})
