import { learnGolden, num, str } from '../../../testing/learn.ts'
import { expm1, log1p, type Tensor } from '../../tensor/index.ts'
import {
  binarizer,
  functionTransformer,
  kBinsDiscretizer,
  polynomialFeatures,
  type SplineExtrapolation,
  splineFeatures,
} from './features.ts'

learnGolden('feature construction against sklearn.preprocessing', 'learn/preprocessing/features', {
  polynomial: ([x, z], p) => {
    const degree = num(p, 'degree')
    const f = polynomialFeatures({
      degree: typeof p['minDegree'] === 'number' ? [p['minDegree'], degree] : degree,
      interactionOnly: p['interactionOnly'] === true,
      includeBias: p['includeBias'] !== false,
    }).fit({ x: x as Tensor })
    return { transform: f.transform(z as Tensor), names: f.featureNamesOut().join(',') }
  },
  spline: ([x, z], p) => {
    const f = splineFeatures({
      nKnots: num(p, 'nKnots', 5),
      degree: num(p, 'degree'),
      knots: str(p, 'knots') as 'uniform' | 'quantile',
      extrapolation: str(p, 'extrapolation') as SplineExtrapolation,
      includeBias: p['includeBias'] === true,
    }).fit({ x: x as Tensor })
    return { transform: f.transform(z as Tensor), train: f.transform(x as Tensor) }
  },
  kbins: ([x, z], p) => {
    const f = kBinsDiscretizer({
      nBins: num(p, 'nBins', 4),
      strategy: str(p, 'strategy') as 'uniform' | 'quantile' | 'kmeans',
      encode: str(p, 'encode') as 'ordinal' | 'onehot',
      quantileMethod: str(p, 'quantileMethod', 'averaged_inverted_cdf') as 'linear' | 'averaged_inverted_cdf',
    }).fit({ x: x as Tensor })
    const t = f.transform(z as Tensor)
    return { transform: t, inverse: f.inverseTransform(t), edges: f.params.edges.flat() }
  },
  binarizer: ([x, z], p) => ({
    transform: binarizer({ threshold: num(p, 'threshold') })
      .fit({ x: x as Tensor })
      .transform(z as Tensor),
  }),
  function: ([x, z]) => {
    const f = functionTransformer({ func: (t) => log1p(t), inverseFunc: (t) => expm1(t) }).fit({ x: x as Tensor })
    return { transform: f.transform(z as Tensor), inverse: f.inverseTransform(z as Tensor) }
  },
})
