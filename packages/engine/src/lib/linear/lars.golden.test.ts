import { learnGolden, num, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { lars, larsPath } from './lars.ts'

learnGolden('least angle regression against sklearn lars_path', 'linear/lars', {
  path: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const path = larsPath({ x, y }, { method: str(p, 'method') as 'lar' | 'lasso' })
    return { alphas: path.alphas, coefs: path.coefs, order: path.order }
  },
  lassoLars: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const f = lars({ method: 'lasso', alpha: num(p, 'alpha') }).fit({ x, y })
    return { coef: f.params.coef, intercept: f.params.intercept }
  },
})
