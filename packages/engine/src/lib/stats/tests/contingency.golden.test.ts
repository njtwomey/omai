import { goldenSuite, type Params } from '../../../testing/suite.ts'
import type { Alternative, TestResult } from '../../base/index.ts'
import { type Tensor, tensor } from '../../tensor/index.ts'
import * as T from './index.ts'

const alt = (p: Params): Alternative => p['alternative'] as Alternative
const sp = (r: TestResult): Tensor => tensor([r.statistic, r.pValue])

goldenSuite('contingency-table tests against scipy.stats and statsmodels', 'stats/tests/contingency', {
  expectedFrequencies: ([t]) => T.expectedFrequencies(t),
  chi2Contingency: ([t], p) =>
    sp(
      T.chi2Contingency(t, {
        correction: p['correction'] as boolean,
        ...(p['lam'] === undefined ? {} : { lambda: p['lam'] as T.DivergenceLambda }),
      }),
    ),
  fisherExact: ([t], p) => sp(T.fisherExact(t, { alternative: alt(p) })),
  oddsRatio: ([t], p) => {
    const e = T.oddsRatio(t, { kind: p['kind'] as 'sample', alternative: alt(p), level: 0.9 })
    return tensor([e.value, e.interval?.[0] ?? Number.NaN, e.interval?.[1] ?? Number.NaN])
  },
  relativeRisk: () => {
    const e = T.relativeRisk(20, 80, 12, 90, { level: 0.9 })
    return tensor([e.value, e.interval?.[0] ?? Number.NaN, e.interval?.[1] ?? Number.NaN])
  },
  mcnemar: ([t], p) => sp(T.mcnemar(t, { exact: p['exact'] as boolean, correction: p['correction'] as boolean })),
  cochranQ: ([x]) => sp(T.cochranQ(x)),
})
