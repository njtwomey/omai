import { goldenSuite, type Params } from '../../../testing/suite.ts'
import type { Alternative, TestResult } from '../../base/index.ts'
import { normal } from '../../dists/index.ts'
import { type Tensor, tensor } from '../../tensor/index.ts'
import * as T from './index.ts'

const alt = (p: Params): Alternative => (p['alternative'] as Alternative | undefined) ?? 'two-sided'
const sp = (r: TestResult): Tensor => tensor([r.statistic, r.pValue])

goldenSuite('goodness-of-fit tests against scipy.stats and statsmodels', 'stats/tests/distribution', {
  kstest: ([x], p) =>
    sp(T.kstest(x, normal(), { alternative: alt(p), method: (p['method'] as T.KsMethod | undefined) ?? 'auto' })),
  andersonDarling: ([x], p) => sp(T.andersonDarling(x, { dist: p['dist'] as T.AndersonDist })),
  shapiro: ([x]) => sp(T.shapiro(x)),
  jarqueBera: ([x]) => sp(T.jarqueBera(x)),
  chisquare: ([x]) => sp(T.chisquare(x)),
  chisquareExpected: ([x, e], p) => sp(T.chisquare(x, { expected: e, ddof: Number(p['ddof']) })),
  powerDivergence: ([x, e], p) => sp(T.powerDivergence(x, { expected: e, lambda: p['lam'] as T.DivergenceLambda })),
  lilliefors: ([x]) => sp(T.lilliefors(x)),
  lillieforsStatistic: ([x]) => T.lilliefors(x, { dist: 'exp', nResamples: 10 }).statistic,
  cramervonmises: ([x]) => sp(T.cramervonmises(x, (u) => Math.min(1, Math.max(0, u)))),
})
