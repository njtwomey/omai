import { goldenSuite, type Params } from '../../testing/suite.ts'
import type { Alternative } from '../base/index.ts'
import * as S from './index.ts'

const alt = (p: Params): Alternative => p['alternative'] as Alternative
const n = (p: Params, k: string): number => Number(p[k])

goldenSuite('power and sample size against statsmodels', 'stats/power', {
  ttest: (_, p) =>
    S.powerTTest({
      effectSize: n(p, 'd'),
      nobs: n(p, 'n'),
      kind: p['kind'] as S.TTestKind,
      ratio: p['ratio'] === undefined ? 1 : n(p, 'ratio'),
      alternative: alt(p),
    }),
  proportions: (_, p) =>
    S.powerProportions({ effectSize: n(p, 'd'), nobs: n(p, 'n'), ratio: n(p, 'ratio'), alternative: alt(p) }),
  sampleSizeTtest: () => S.sampleSizeTTest({ effectSize: 0.5, power: 0.8, kind: 'two-sample' }),
  sampleSizeTtestOne: () => S.sampleSizeTTest({ effectSize: 0.3, power: 0.9 }),
  sampleSizeProportions: () => S.sampleSizeProportions({ effectSize: 0.3, power: 0.8 }),
  anova: (_, p) => S.powerAnova({ effectSize: n(p, 'f'), nobs: n(p, 'n'), kGroups: n(p, 'k') }),
  sampleSizeAnova: () => S.sampleSizeAnova({ effectSize: 0.25, power: 0.8, kGroups: 3 }),
  chisquare: (_, p) => S.powerChisquare({ effectSize: n(p, 'w'), nobs: n(p, 'n'), nBins: n(p, 'bins') }),
  sampleSizeChisquare: () => S.sampleSizeChisquare({ effectSize: 0.3, power: 0.8, nBins: 4 }),
})
