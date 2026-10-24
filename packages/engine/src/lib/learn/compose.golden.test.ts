import { learnGolden, leastSquares, ridge, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { columnTransformer, featureUnion, pipeline, transformedTargetRegressor } from './compose.ts'
import type { Fitted } from './estimator.ts'
import { kFold } from './model-selection/splitters.ts'
import {
  minMaxScaler,
  oneHotEncoder,
  polynomialFeatures,
  standardScaler,
  targetEncoder,
} from './preprocessing/index.ts'

learnGolden('composition against sklearn.pipeline and sklearn.compose', 'learn/compose', {
  polyRidge: ([x, y, z]) => {
    const f = pipeline([
      ['poly', polynomialFeatures()],
      ['scale', standardScaler()],
      ['ridge', ridge({ alpha: 0.5 })],
    ]).fit({ x: x as Tensor, y: y as Tensor })
    const transform = f.intermediate(z as Tensor)['scale'] as Tensor
    return {
      predict: f.predict?.(z as Tensor) as Tensor,
      score: f.score?.({ x: x as Tensor, y: y as Tensor }) ?? Number.NaN,
      transform,
      names:
        [f.namedSteps['poly'], f.namedSteps['scale']]
          .reduce<string[] | undefined>((names, s) => s?.featureNamesOut?.(names), undefined)
          ?.join(',') ?? '',
    }
  },
  columnsLinear: ([x, y, z]) => {
    const ct = columnTransformer([
      ['num', standardScaler(), [0, 2]],
      ['cat', oneHotEncoder(), [1]],
    ])
    const f = pipeline([
      ['ct', ct],
      ['lr', leastSquares()],
    ]).fit({ x: x as Tensor, y: y as Tensor })
    return { predict: f.predict?.(z as Tensor) as Tensor }
  },
  targetEncoded: ([x, y, z]) => {
    const f = pipeline([
      ['te', targetEncoder({ smooth: 1, targetType: 'continuous', cv: kFold(4) })],
      ['lr', leastSquares()],
    ]).fit({ x: x as Tensor, y: y as Tensor })
    return {
      predict: f.predict?.(z as Tensor) as Tensor,
      coef: (f.namedSteps['lr'] as Fitted<{ coef: number[] }>).params.coef,
    }
  },
  columns: ([x, z], p) => {
    const remainder = str(p, 'remainder')
    const f = columnTransformer(
      remainder === 'minmax'
        ? [['num', standardScaler(), [0]]]
        : [
            ['cat', oneHotEncoder(), [1]],
            ['mm', minMaxScaler(), [2]],
          ],
      { remainder: remainder === 'minmax' ? minMaxScaler() : (remainder as 'drop' | 'passthrough') },
    ).fit({ x: x as Tensor })
    return { transform: f.transform(z as Tensor), names: f.featureNamesOut().join(',') }
  },
  union: ([x, z]) => {
    const f = featureUnion(
      [
        ['z', standardScaler()],
        ['m', minMaxScaler()],
      ],
      { weights: { m: 2 } },
    ).fit({ x: x as Tensor })
    return { transform: f.transform(z as Tensor) }
  },
  transformedTarget: ([x, y, z], p) => {
    const est =
      str(p, 'kind') === 'func'
        ? transformedTargetRegressor({ regressor: leastSquares(), func: Math.log, inverseFunc: Math.exp })
        : transformedTargetRegressor({ regressor: leastSquares(), transformer: standardScaler() })
    return { predict: est.fit({ x: x as Tensor, y: y as Tensor }).predict?.(z as Tensor) as Tensor }
  },
})
