import { learnGolden, num } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { sur } from './sur.ts'

const equations = [[0], [1, 2], [0, 2]]

learnGolden('seemingly unrelated regressions against the stacked Kronecker GLS in numpy', 'linear/sur', {
  sur: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const f = sur({ equations, maxIter: num(p, 'maxIter'), tol: 0 }).fit({ x, y })
    const beta: number[] = []
    equations.forEach((cols, e) => {
      for (const j of cols) beta.push(f.params.coef[j]?.[e] ?? Number.NaN)
      beta.push(f.params.intercept[e] ?? Number.NaN)
    })
    return {
      beta,
      sigma: f.params.sigma,
      bse: f.params.covariance.map((r, j) => Math.sqrt(r[j] ?? Number.NaN)),
    }
  },
})
