import { learnGolden, num } from '../../testing/learn.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { totalLeastSquares } from './errors-in-variables.ts'

learnGolden(
  'total least squares against the orthogonal-distance objective minimised by BFGS',
  'linear/errors_in_variables',
  {
    totalLeastSquares: (inputs, p) => {
      const [x, y, sx] = inputs as [Tensor, Tensor, Tensor]
      const f = totalLeastSquares({ errorX: Array.from(toFlat(sx)), errorY: num(p, 'errorY') }).fit({ x, y })
      return { coef: f.params.coef, intercept: f.params.intercept }
    },
  },
)
