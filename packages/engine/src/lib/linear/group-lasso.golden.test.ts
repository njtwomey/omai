import { learnGolden, num } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { multiTaskLasso } from './group-lasso.ts'

learnGolden('multi-task lasso against scikit-learn', 'linear/group_lasso', {
  multiTask: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const f = multiTaskLasso({ alpha: num(p, 'alpha'), l1Ratio: num(p, 'l1Ratio'), tol: 1e-13, maxIter: 100000 }).fit({
      x,
      y,
    })
    return { coef: f.params.coef, intercept: f.params.intercept }
  },
})
