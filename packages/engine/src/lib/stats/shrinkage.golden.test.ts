import { goldenSuite } from '../../testing/suite.ts'
import { asTensor, type Tensor } from '../tensor/index.ts'
import * as S from './index.ts'

goldenSuite('empirical Bayes normal means against statsmodels combine_effects', 'stats/shrinkage', {
  empiricalBayes: ([y, variance]) => {
    const eb = S.empiricalBayesNormalMeans(y, { variances: variance })
    return asTensor([eb.priorVariance, eb.priorMean, ...Array.from(eb.posteriorMeans.data)]) as Tensor
  },
})
