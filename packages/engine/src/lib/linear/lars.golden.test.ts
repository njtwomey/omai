import { learnGolden, num, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { lars, larsPath } from './lars.ts'

learnGolden('least angle regression against sklearn lars_path', 'linear/lars', {
  path: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const positive = num(p, 'positive', 0) === 1
    const path = larsPath({ x, y }, { method: str(p, 'method') as 'lar' | 'lasso', positive })
    // scikit-learn's last knot of a non-negative path is off the optimum (see the oracle case), so it is not compared.
    const keep = positive ? path.alphas.length - 1 : path.alphas.length
    return { alphas: path.alphas.slice(0, keep), coefs: path.coefs.slice(0, keep), order: path.order }
  },
  lassoLars: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const f = lars({ method: 'lasso', alpha: num(p, 'alpha'), positive: num(p, 'positive', 0) === 1 }).fit({ x, y })
    return { coef: f.params.coef, intercept: f.params.intercept }
  },
})
