import type { Tensor } from '../../lib/tensor/index.ts'
import { learnGolden, num } from '../../testing/learn.ts'
import { gammaRegressor, poissonRegressor, tweedieRegressor } from './glm.ts'

type In = readonly [Tensor, Tensor]
const fitted = (f: { readonly params: { readonly coef: number[]; readonly intercept: number } }) => ({
  coef: f.params.coef,
  intercept: f.params.intercept,
})

learnGolden('GLM regressors against scikit-learn and statsmodels', 'methods/linear/glm', {
  poisson: (inputs, p) => {
    const [x, y] = inputs as In
    return fitted(poissonRegressor({ alpha: num(p, 'alpha') }).fit({ x, y }))
  },
  gamma: (inputs, p) => {
    const [x, y] = inputs as In
    return fitted(gammaRegressor({ alpha: num(p, 'alpha') }).fit({ x, y }))
  },
  tweedie: (inputs, p) => {
    const [x, y] = inputs as In
    return fitted(tweedieRegressor({ alpha: num(p, 'alpha'), power: 1.5 }).fit({ x, y }))
  },
  poissonWald: (inputs) => {
    const [x, y] = inputs as In
    const f = poissonRegressor({ alpha: 0 }).fit({ x, y })
    const table = f.exposition[0]
    return {
      params: [...f.params.coef, f.params.intercept],
      bse: table?.kind === 'coefficientTable' ? (table.standardErrors ?? []) : [],
    }
  },
})
