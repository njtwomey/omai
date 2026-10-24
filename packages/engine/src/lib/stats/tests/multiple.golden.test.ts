import { goldenSuite } from '../../../testing/suite.ts'
import { tensor } from '../../tensor/index.ts'
import * as T from './index.ts'

goldenSuite('multiple testing against statsmodels and scipy', 'stats/tests/multiple', {
  adjust: ([p], o) => T.adjustPValues(p, o['method'] as T.CorrectionMethod),
  reject: ([p], o) =>
    tensor(T.multipleTests(p, { method: o['method'] as T.CorrectionMethod, alpha: 0.1 }).reject.map(Number)),
  combine: ([p], o) => {
    const r = T.combinePValues(p, { method: o['method'] as T.CombineMethod })
    return tensor([r.statistic, r.pValue])
  },
  combineWeighted: ([p, w]) => {
    const r = T.combinePValues(p, { method: 'stouffer', weights: w })
    return tensor([r.statistic, r.pValue])
  },
})
