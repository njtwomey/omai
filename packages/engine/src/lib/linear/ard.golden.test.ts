import { learnGolden } from '../../testing/learn.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import { ardRegression } from './ard.ts'

learnGolden('ARDRegression against scikit-learn', 'linear/ard', {
  ard: ([x, y, held], p) => {
    const f = ardRegression({ tol: 1e-6, maxIter: 500, fitIntercept: p['fitIntercept'] !== false }).fit({
      x: x as Tensor,
      y: y as Tensor,
    })
    const kept = f.params.keep.map((k) => (k ? 1 : 0))
    return {
      coef: f.params.coef,
      intercept: f.params.intercept,
      alpha: f.params.alpha,
      lambda: f.params.lambda,
      sigma: f.params.sigma.length === 0 ? tensor([[]]) : tensor(f.params.sigma),
      keep: kept,
      nIter: f.params.iterations,
      score: f.params.logEvidence,
      predict: f.predict(held as Tensor),
      std: f.predictStd(held as Tensor),
    }
  },
})
