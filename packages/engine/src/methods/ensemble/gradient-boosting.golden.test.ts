import { sliceAxis, type Tensor } from '../../lib/tensor/index.ts'
import { learnGolden, str } from '../../testing/learn.ts'
import { gradientBoostingClassifier, gradientBoostingRegressor, type RegressionLoss } from './gradient-boosting.ts'

type In = readonly [Tensor, Tensor]

learnGolden('gradient boosting against scikit-learn', 'methods/ensemble/gradient_boosting', {
  regressor: (inputs, p) => {
    const [x, y] = inputs as In
    const f = gradientBoostingRegressor({
      loss: str(p, 'loss') as RegressionLoss,
      nEstimators: 15,
      maxDepth: 2,
      learningRate: 0.3,
    }).fit({ x, y })
    return { predict: f.predict(sliceAxis(x, 0, 0, 8)) }
  },
  classifier: (inputs) => {
    const [x, y] = inputs as In
    const f = gradientBoostingClassifier({ nEstimators: 10, maxDepth: 2, learningRate: 0.5 }).fit({ x, y })
    return { proba: f.predictProba?.(sliceAxis(x, 0, 0, 8)) as Tensor }
  },
})
