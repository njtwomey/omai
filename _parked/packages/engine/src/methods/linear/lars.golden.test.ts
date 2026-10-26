import type { Tensor } from '../../lib/tensor/index.ts'
import { learnGolden, str } from '../../testing/learn.ts'
import { larsPath, lassoLars } from './lars.ts'

learnGolden('least angle regression against scikit-learn', 'methods/linear/lars', {
  path: (inputs, p) => {
    const [x, y] = inputs as readonly [Tensor, Tensor]
    const r = larsPath({ x, y }, { method: str(p, 'method') as 'lar' | 'lasso' })
    return { alphas: r.alphas, coefs: r.coefs }
  },
  lassoLars: (inputs) => {
    const [x, y] = inputs as readonly [Tensor, Tensor]
    const f = lassoLars({ alpha: 0.05 }).fit({ x, y })
    return { coef: f.params.coef, intercept: f.params.intercept }
  },
})
