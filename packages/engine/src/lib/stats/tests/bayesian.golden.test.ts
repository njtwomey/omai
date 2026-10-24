import { goldenSuite } from '../../../testing/suite.ts'
import { tensor } from '../../tensor/index.ts'
import * as T from './index.ts'

goldenSuite('Bayesian tests against pingouin and their integrals', 'stats/tests/bayesian', {
  ttest: (_, p) => {
    const ny = Number(p['ny'])
    return T.bayesFactorTTest(Number(p['t']), Number(p['nx']), {
      ...(ny > 0 ? { ny } : {}),
      paired: p['paired'] as boolean,
    }).statistic
  },
  binomial: (_, p) => T.bayesFactorBinomial(Number(p['k']), Number(p['n']), { p: Number(p['p']) }).statistic,
  pearson: (_, p) =>
    T.bayesFactorPearson(Number(p['r']), Number(p['n']), { method: (p['method'] as 'wetzels' | undefined) ?? 'ly' })
      .statistic,
  abTest: () => {
    const r = T.bayesianAbTest(20, 100, 30, 100)
    return tensor([r.statistic, Number(r.assumptions?.['expectedLossChoosingB'])])
  },
  hdi: ([x]) => tensor(T.highestDensityInterval(x, 0.9)),
})
