import { learnGolden, num } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { toFlat } from '../tensor/index.ts'
import { expectileRegressor } from './expectile.ts'

const outputs = (f: ReturnType<ReturnType<typeof expectileRegressor>['fit']>): Record<string, number | number[]> => {
  const p = f.params.covariance.length
  return {
    coef: f.params.coef,
    intercept: f.params.intercept,
    bse: Array.from({ length: p }, (_, j) => Math.sqrt(f.params.covariance[j]?.[j] ?? Number.NaN)),
  }
}

learnGolden('expectile regression against scipy BFGS on the asymmetric least-squares loss', 'linear/expectile', {
  expectile: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    return outputs(expectileRegressor({ tau: num(p, 'tau'), alpha: num(p, 'alpha') }).fit({ x, y }))
  },
  expectileWeighted: (inputs, p) => {
    const [x, y, w] = inputs as [Tensor, Tensor, Tensor]
    return outputs(
      expectileRegressor({ tau: num(p, 'tau'), alpha: num(p, 'alpha') }).fit({ x, y, weights: Array.from(toFlat(w)) }),
    )
  },
})
