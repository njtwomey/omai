import { goldenSuite, type Params } from '../../../testing/suite.ts'
import type { Alternative } from '../../base/index.ts'
import { tensor } from '../../tensor/index.ts'
import * as T from './index.ts'

const alt = (p: Params): Alternative => p['alternative'] as Alternative
const n = (p: Params, k: string): number => Number(p[k])

goldenSuite('proportion tests against scipy.stats and statsmodels', 'stats/tests/proportions', {
  binomTest: (_, p) => T.binomTest(n(p, 'k'), n(p, 'n'), n(p, 'p'), { alternative: alt(p) }).pValue,
  binomTestCi: (_, p) =>
    tensor([...(T.binomTest(n(p, 'k'), n(p, 'n'), n(p, 'p'), { level: 0.9 }).estimate?.interval ?? [])]),
  proportionConfint: (_, p) =>
    tensor([
      ...(T.proportionConfint(n(p, 'k'), n(p, 'n'), { level: 0.9, method: p['method'] as T.ProportionInterval })
        .interval ?? []),
    ]),
  proportionsZtest1: (_, p) => {
    const r = T.proportionsZtest(14, 40, { value: 0.5, alternative: alt(p) })
    return tensor([r.statistic, r.pValue])
  },
  proportionsZtest2: (_, p) => {
    const r = T.proportionsZtest([30, 21], [60, 70], { alternative: alt(p) })
    return tensor([r.statistic, r.pValue])
  },
  proportionsDifferenceCi: (_, p) =>
    tensor([
      ...(T.proportionsDifference(30, 60, 21, 70, { level: 0.9, method: p['method'] as T.DifferenceInterval }).estimate
        ?.interval ?? []),
    ]),
  proportionsDifferenceTest: () => {
    const r = T.proportionsDifference(30, 60, 21, 70)
    return tensor([r.statistic, r.pValue])
  },
})
