import { learnGolden, num, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { pcr, reducedRankRegression } from './rank-regression.ts'

learnGolden('principal-component and reduced-rank regression against scikit-learn and numpy', 'decomposition/rank_regression', {
  pcr: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const f = pcr({ nComponents: num(p, 'nComponents'), scale: num(p, 'scale') === 1 }).fit({ x, y })
    return { coef: f.params.coef, intercept: f.params.intercept, explainedVariance: f.params.explainedVariance }
  },
  reducedRank: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const f = reducedRankRegression({
      rank: num(p, 'rank'),
      alpha: num(p, 'alpha'),
      weighting: str(p, 'weighting') as 'identity' | 'covariance',
    }).fit({ x, y })
    return { coef: f.params.coef, intercept: f.params.intercept }
  },
})
