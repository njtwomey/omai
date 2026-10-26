import { sliceAxis, type Tensor } from '../../lib/tensor/index.ts'
import { learnGolden, num, str } from '../../testing/learn.ts'
import { linearDiscriminantAnalysis, quadraticDiscriminantAnalysis } from './discriminant.ts'

learnGolden('discriminant analysis against scikit-learn', 'methods/discriminant/discriminant', {
  lda: (inputs, p) => {
    const [x, y] = inputs as readonly [Tensor, Tensor]
    const raw = p['shrinkage']
    const shrinkage = raw === 'none' || raw === undefined ? undefined : raw === 'auto' ? 'auto' : Number(raw)
    const f = linearDiscriminantAnalysis({
      solver: str(p, 'solver') as 'svd' | 'lsqr',
      ...(shrinkage === undefined ? {} : { shrinkage }),
    }).fit({ x, y })
    return {
      coef: f.params.coef,
      intercept: f.params.intercept,
      proba: f.predictProba(sliceAxis(x, 0, 0, 7)),
      ratio: f.params.explainedVarianceRatio,
    }
  },
  qda: (inputs, p) => {
    const [x, y] = inputs as readonly [Tensor, Tensor]
    return {
      proba: quadraticDiscriminantAnalysis({ regParam: num(p, 'reg') })
        .fit({ x, y })
        .predictProba(sliceAxis(x, 0, 0, 7)),
    }
  },
})
