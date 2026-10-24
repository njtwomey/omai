import { learnGolden, leastSquares, num } from '../../../testing/learn.ts'
import { pinv } from '../../numerics/linalg/index.ts'
import { concat, matmul, ones, type Tensor, toFlat } from '../../tensor/index.ts'
import { requireVerb } from '../estimator.ts'
import { aic, aicc, bic, gaussianLogLikelihood, linearSmootherLoo } from './criteria.ts'

const design = (x: Tensor): Tensor => concat([ones([x.shape[0] ?? 0, 1]), x], 1) as Tensor

learnGolden('criteria against statsmodels and leave-one-out refits', 'learn/model_selection/criteria', {
  ols: ([x, y], p) => {
    const fitted = leastSquares().fit({ x: x as Tensor, y: y as Tensor })
    const pred = toFlat(requireVerb(fitted, 'predict', 'test')(x as Tensor))
    const yv = toFlat(y as Tensor)
    const rss = yv.reduce((s, v, i) => s + (v - (pred[i] ?? 0)) ** 2, 0)
    const n = yv.length
    const llf = gaussianLogLikelihood(rss, n)
    const k = num(p, 'k')
    return { rss, llf, aic: aic(llf, k), bic: bic(llf, k, n), aicc: aicc(llf, k, n) }
  },
  loo: ([x, y]) => {
    const a = design(x as Tensor)
    const hat = matmul(a, pinv(a))
    return { residuals: linearSmootherLoo(hat, y as Tensor).residuals }
  },
})
