import { learnGolden, num } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { huber, quantileRegressor, theilSen } from './robust.ts'

learnGolden('robust regression against scikit-learn', 'linear/robust', {
  theilSen: (inputs) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const f = theilSen({ maxSubpopulation: 100000 }).fit({ x, y })
    return { coef: f.params.coef, intercept: f.params.intercept }
  },
  huber: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const f = huber({ alpha: num(p, 'alpha'), epsilon: num(p, 'epsilon'), tol: 1e-13, maxIter: 100000 }).fit({ x, y })
    return { coef: f.params.coef, intercept: f.params.intercept, scale: f.params.scale }
  },
  quantile: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const f = quantileRegressor({ quantile: num(p, 'quantile'), alpha: num(p, 'alpha') }).fit({ x, y })
    return { coef: f.params.coef, intercept: f.params.intercept }
  },
})
