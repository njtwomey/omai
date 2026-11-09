import { learnGolden, num, str } from '../../testing/learn.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { dummyClassifier, dummyRegressor } from './dummy.ts'

const weightsOf = (p: Readonly<Record<string, unknown>>, w: Tensor | undefined): { weights?: Tensor } =>
  p['weighted'] === true && w !== undefined ? { weights: w } : {}

learnGolden('DummyClassifier and DummyRegressor against scikit-learn', 'learn/dummy', {
  classifier: (inputs, p) => {
    const [x, y, held, w] = inputs as [Tensor, Tensor, Tensor, Tensor]
    const strategy = str(p, 'strategy') as 'prior' | 'most_frequent' | 'constant'
    const f = dummyClassifier({ strategy, ...(strategy === 'constant' ? { constant: num(p, 'constant') } : {}) }).fit({
      x,
      y,
      ...weightsOf(p, w),
    })
    return { predict: f.predict(held), predictProba: f.predictProba(held), prior: f.params.prior }
  },
  regressor: (inputs, p) => {
    const [x, y, held, w] = inputs as [Tensor, Tensor, Tensor, Tensor]
    const strategy = str(p, 'strategy') as 'mean' | 'median' | 'quantile' | 'constant'
    const f = dummyRegressor({
      strategy,
      ...(strategy === 'quantile' ? { quantile: num(p, 'quantile') } : {}),
      ...(strategy === 'constant' ? { constant: num(p, 'constant') } : {}),
    }).fit({ x, y, ...weightsOf(p, w) })
    return { predict: toFlat(f.predict(held)) as unknown as number[] }
  },
})
