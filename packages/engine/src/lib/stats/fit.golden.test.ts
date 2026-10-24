import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as S from './index.ts'

goldenSuite('maximum-likelihood fits against scipy.stats', 'stats/fit', {
  fit: ([x], p) => {
    const f = S.fitDistribution(x, p['family'] as S.FitFamily)
    const v = Object.values(f.params)
    const se = Object.values(f.standardErrors)
    return tensor([...v, ...se, f.logLikelihood])
  },
  fitParams: ([x], p) => tensor(Object.values(S.fitDistribution(x, p['family'] as S.FitFamily).params)),
})
