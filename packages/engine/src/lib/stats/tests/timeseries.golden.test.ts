import { goldenSuite } from '../../../testing/suite.ts'
import { tensor } from '../../tensor/index.ts'
import * as T from './index.ts'

goldenSuite('time-series tests against statsmodels', 'stats/tests/timeseries', {
  ljungBox: ([x], p) => {
    const r = T.ljungBox(x, { lags: Number(p['lags']) })
    return tensor([r.statistic, r.pValue])
  },
  boxPierce: ([x], p) => {
    const r = T.ljungBox(x, { lags: Number(p['lags']), boxPierce: true })
    return tensor([r.statistic, r.pValue])
  },
  durbinWatson: ([x]) => T.durbinWatson(x).statistic,
  adfuller: ([x], p) => {
    const r = T.adfuller(x, {
      regression: p['regression'] as T.TrendTerms,
      ...(p['autolag'] === undefined
        ? { autolag: null, maxlag: Number(p['maxlag']) }
        : { autolag: p['autolag'] as 'AIC' }),
    })
    return tensor([r.statistic, r.pValue, Number(r.assumptions?.['usedlag'])])
  },
  kpss: ([x], p) => {
    const nl = p['nlags']
    const r = T.kpss(x, { regression: p['regression'] as 'c', nlags: typeof nl === 'number' ? nl : (nl as 'auto') })
    return tensor([r.statistic, r.pValue, Number(r.assumptions?.['nlags'])])
  },
  mackinnonp: (_, p) => T.mackinnonp(Number(p['stat']), p['regression'] as T.TrendTerms),
  mackinnoncrit: (_, p) => tensor(T.mackinnoncrit(80, p['regression'] as T.TrendTerms)),
  runsTest: ([x], p) => {
    const r = T.runsTest(x, { cutoff: p['cutoff'] as 'mean', correction: p['correction'] as boolean })
    return tensor([r.statistic, r.pValue])
  },
})
