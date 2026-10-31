import { gaussianNB, learnGolden, leastSquares } from '../../testing/learn.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { multiOutputClassifier, multiOutputRegressor, regressorChain } from './multioutput.ts'

learnGolden('multi-output wrappers against sklearn.multioutput', 'learn/multioutput', {
  regressor: (inputs) => {
    const [x, y, q] = inputs as [Tensor, Tensor, Tensor]
    return { predict: multiOutputRegressor(leastSquares()).fit({ x, y }).predict(q) }
  },
  regressorChain: (inputs, p) => {
    const [x, y, q] = inputs as [Tensor, Tensor, Tensor]
    const order = Array.from(toFlat(p['order'] as Tensor))
    return { predict: regressorChain(leastSquares(), { order }).fit({ x, y }).predict(q) }
  },
  classifier: (inputs) => {
    const [x, y, q] = inputs as [Tensor, Tensor, Tensor]
    const f = multiOutputClassifier(gaussianNB()).fit({ x, y })
    const each = (f.predictProbaEach as NonNullable<typeof f.predictProbaEach>)(q)
    return { predict: f.predict(q), proba0: each[0] as Tensor, proba1: each[1] as Tensor, proba2: each[2] as Tensor }
  },
})
