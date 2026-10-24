import { learnGolden, num, ridge, str } from '../../../testing/learn.ts'
import type { Tensor } from '../../tensor/index.ts'
import { iterativeImputer, knnImputer, missingIndicator, simpleImputer } from './imputation.ts'

type Strategy = 'mean' | 'median' | 'most_frequent' | 'constant'
type Order = 'ascending' | 'descending' | 'roman' | 'arabic'

learnGolden('imputers against sklearn.impute', 'learn/preprocessing/imputation', {
  simple: ([x, z], p) => {
    const f = simpleImputer({
      strategy: str(p, 'strategy') as Strategy,
      fillValue: num(p, 'fillValue', 0),
      addIndicator: p['addIndicator'] !== false,
      keepEmptyFeatures: p['keepEmptyFeatures'] === true,
    }).fit({ x: x as Tensor })
    const t = f.transform(z as Tensor)
    return { transform: t, ...(p['addIndicator'] === false ? {} : { inverse: f.inverseTransform(t) }) }
  },
  knn: ([x, z], p) => {
    const f = knnImputer({
      nNeighbors: num(p, 'nNeighbors'),
      weights: str(p, 'weights') as 'uniform' | 'distance',
      addIndicator: p['addIndicator'] === true,
    }).fit({ x: x as Tensor })
    return { transform: f.transform(z as Tensor), train: f.transform(x as Tensor) }
  },
  iterative: ([x, z], p) => {
    const f = iterativeImputer({
      estimator: ridge({ alpha: 1 }),
      maxIter: num(p, 'maxIter'),
      imputationOrder: str(p, 'order') as Order,
      initialStrategy: str(p, 'initialStrategy', 'mean') as Strategy,
      minValue: num(p, 'minValue', Number.NEGATIVE_INFINITY),
      maxValue: num(p, 'maxValue', Number.POSITIVE_INFINITY),
      skipComplete: p['skipComplete'] === true,
    }).fit({ x: x as Tensor })
    return { transform: f.transform(z as Tensor), nIter: f.params.nIter }
  },
  indicator: ([x, z], p) => ({
    transform: missingIndicator({ features: str(p, 'features') as 'all' | 'missing-only' })
      .fit({ x: x as Tensor })
      .transform(z as Tensor),
  }),
})
