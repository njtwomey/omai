import { learnGolden, leastSquares, num, ridge, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import {
  chi2,
  fClassif,
  fRegression,
  mutualInfoClassif,
  mutualInfoRegression,
  type PValueControl,
  rRegression,
  selectByPValue,
  selectFromModel,
  selectKBest,
  selectPercentile,
  varianceThreshold,
} from './feature-selection.ts'
import { rfe, rfecv, sequentialFeatureSelector } from './feature-selection-wrappers.ts'
import { kFold } from './model-selection/splitters.ts'

const mask = (s: readonly boolean[]): number[] => s.map(Number)

learnGolden('feature selection against sklearn.feature_selection', 'learn/feature_selection', {
  fClassif: ([x, y]) => {
    const r = fClassif(x as Tensor, y as Tensor)
    return { scores: [...r.scores], pValues: [...(r.pValues ?? [])] }
  },
  fRegression: ([x, y]) => {
    const r = fRegression(x as Tensor, y as Tensor)
    return { scores: [...r.scores], pValues: [...(r.pValues ?? [])] }
  },
  rRegression: ([x, y]) => ({ scores: rRegression(x as Tensor, y as Tensor) }),
  chi2: ([x, y]) => {
    const r = chi2(x as Tensor, y as Tensor)
    return { scores: [...r.scores], pValues: [...(r.pValues ?? [])] }
  },
  mutualInfoClassif: ([x, y]) => ({ scores: mutualInfoClassif(x as Tensor, y as Tensor) }),
  mutualInfoRegression: ([x, y]) => ({ scores: mutualInfoRegression(x as Tensor, y as Tensor) }),
  variance: ([x], p) => ({ support: mask(varianceThreshold(num(p, 'threshold')).fit({ x: x as Tensor }).support) }),
  kBest: ([x, y], p) => ({
    support: mask(
      selectKBest({ scoreFunc: fRegression, k: num(p, 'k') }).fit({ x: x as Tensor, y: y as Tensor }).support,
    ),
  }),
  percentile: ([x, y], p) => ({
    support: mask(
      selectPercentile({ scoreFunc: fRegression, percentile: num(p, 'percentile') }).fit({
        x: x as Tensor,
        y: y as Tensor,
      }).support,
    ),
  }),
  pValue: ([x, y], p) => ({
    support: mask(
      selectByPValue(str(p, 'control') as PValueControl, { scoreFunc: fRegression }).fit({
        x: x as Tensor,
        y: y as Tensor,
      }).support,
    ),
  }),
  fromModel: ([x, y], p) => {
    const max = p['maxFeatures']
    const est =
      typeof max === 'number'
        ? selectFromModel({ estimator: ridge(), threshold: Number.NEGATIVE_INFINITY, maxFeatures: max })
        : selectFromModel({ estimator: ridge(), threshold: str(p, 'threshold') })
    return { support: mask(est.fit({ x: x as Tensor, y: y as Tensor }).support) }
  },
  rfe: ([x, y], p) => {
    const f = rfe({ estimator: leastSquares(), nFeaturesToSelect: num(p, 'nFeatures'), step: num(p, 'step') }).fit({
      x: x as Tensor,
      y: y as Tensor,
    })
    return { support: mask(f.support), ranking: f.params.ranking }
  },
  rfecv: ([x, y]) => {
    const f = rfecv({ estimator: leastSquares(), cv: kFold(3), scoring: 'r2' }).fit({ x: x as Tensor, y: y as Tensor })
    return {
      support: mask(f.support),
      mean: f.params.meanTestScore,
      std: f.params.stdTestScore,
      nFeatures: f.params.nFeatures,
    }
  },
  sequential: ([x, y], p) => {
    const f = sequentialFeatureSelector({
      estimator: leastSquares(),
      nFeaturesToSelect: 2,
      direction: str(p, 'direction') as 'forward' | 'backward',
      cv: kFold(3),
    }).fit({ x: x as Tensor, y: y as Tensor })
    return { support: mask(f.support) }
  },
})
