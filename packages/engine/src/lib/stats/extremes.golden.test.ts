import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as S from './index.ts'

goldenSuite('extreme-value fits against scipy.stats', 'stats/extremes', {
  gev: ([x]) => {
    const f = S.fitGev(x)
    return tensor([f.params.loc, f.params.scale, f.params.shape])
  },
  gevReturn: ([x], p) => S.returnLevel(S.fitGev(x), Number(p['period'])).value,
  gpd: ([x]) => {
    const f = S.fitGpd(x)
    return tensor([f.params.scale, f.params.shape])
  },
  gpdReturn: ([x], p) =>
    S.returnLevel(S.fitGpd(x), Number(p['period']), { kind: 'gpd', rate: Number(p['rate']) }).value,
  meanExcess: ([x, u]) => S.meanExcess(x, { thresholds: u }).meanExcess,
})
