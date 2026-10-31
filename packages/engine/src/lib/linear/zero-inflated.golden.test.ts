import { learnGolden, num, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { type CountFamily, hurdle, zeroInflated } from './zero-inflated.ts'

const outputs = (f: ReturnType<ReturnType<typeof zeroInflated>['fit']>): Record<string, number | number[]> => ({
  coef: f.params.coef,
  intercept: f.params.intercept,
  zeroCoef: f.params.zeroCoef,
  zeroIntercept: f.params.zeroIntercept,
  alpha: f.params.alpha,
  logLikelihood: f.params.logLikelihood,
})

learnGolden('zero-inflated and hurdle count models against statsmodels', 'linear/zero_inflated', {
  zeroInflated: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const inflation = num(p, 'inflation') === 1 ? [0] : 'constant'
    return outputs(
      zeroInflated({ family: str(p, 'family') as CountFamily, inflation, tol: 1e-9, maxIter: 300 }).fit({ x, y }),
    )
  },
  hurdle: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    return outputs(
      hurdle({ family: str(p, 'family') as CountFamily, inflation: [0], tol: 1e-9, maxIter: 300 }).fit({ x, y }),
    )
  },
})
