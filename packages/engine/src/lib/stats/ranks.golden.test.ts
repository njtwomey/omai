import { goldenSuite } from '../../testing/suite.ts'
import * as S from './index.ts'

goldenSuite('ranks against scipy.stats', 'stats/ranks', {
  rankdata: ([x], p) => S.rankdata(x, { method: p['method'] as S.RankMethod }),
  tieCorrection: ([x]) => S.tieCorrection(x),
})
