import { learnGolden, num, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { lassoLarsIC } from './lars.ts'

learnGolden('LassoLarsIC against scikit-learn', 'linear/lasso_lars_ic', {
  ic: ([x, y, held], p) => {
    const noiseVariance = num(p, 'noiseVariance')
    const f = lassoLarsIC({
      criterion: str(p, 'criterion') as 'aic' | 'bic',
      fitIntercept: p['fitIntercept'] !== false,
      ...(noiseVariance < 0 ? {} : { noiseVariance }),
    }).fit({ x: x as Tensor, y: y as Tensor })
    return {
      coef: f.params.coef,
      intercept: f.params.intercept,
      alpha: f.params.alpha,
      criterion: f.params.criterion,
      noiseVariance: f.params.noiseVariance,
      predict: f.predict(held as Tensor),
    }
  },
})
