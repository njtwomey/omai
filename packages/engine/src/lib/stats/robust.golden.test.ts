import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as S from './index.ts'

goldenSuite('robust statistics against scipy.stats and statsmodels', 'stats/robust', {
  trimMean: ([x], p) => S.trimMean(x, Number(p['p'])),
  trimboth: ([x], p) => S.trimboth(x, Number(p['p'])),
  winsorize: ([x], p) => S.winsorize(x, [Number(p['lo']), Number(p['hi'])]),
  winsorizedVariance: ([x], p) => S.winsorizedVariance(x, [Number(p['lo']), Number(p['hi'])]),
  mEstimate: ([x], p) => {
    const r = S.mEstimateLocation(x, { norm: p['norm'] as S.RobustNorm })
    return tensor([r.value, r.scale])
  },
})
