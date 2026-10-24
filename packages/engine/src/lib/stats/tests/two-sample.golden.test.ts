import { goldenSuite, type Params } from '../../../testing/suite.ts'
import type { Alternative, TestResult } from '../../base/index.ts'
import { type Tensor, tensor } from '../../tensor/index.ts'
import * as T from './index.ts'

const alt = (p: Params): Alternative => (p['alternative'] as Alternative | undefined) ?? 'two-sided'
const sp = (r: TestResult): Tensor => tensor([r.statistic, r.pValue])

goldenSuite('two-sample tests against scipy.stats', 'stats/tests/two_sample', {
  ks2samp: ([x, y], p) =>
    sp(T.ks2samp(x, y, { alternative: alt(p), method: (p['method'] as 'asymp' | undefined) ?? 'auto' })),
  cramervonmises2samp: ([x, y], p) => sp(T.cramervonmises2samp(x, y, { method: p['method'] as 'exact' })),
  energyDistance: ([x, y]) => T.energyDistance(x, y),
  wassersteinDistance: ([x, y]) => T.wassersteinDistance(x, y),
  energyDistanceWeighted: ([x, y, wx, wy]) =>
    T.energyDistance(x, y, { uWeights: wx as Tensor, vWeights: wy as Tensor }),
  wassersteinDistanceWeighted: ([x, y, wx, wy]) =>
    T.wassersteinDistance(x, y, { uWeights: wx as Tensor, vWeights: wy as Tensor }),
  eppsSingleton: ([x, y]) => sp(T.eppsSingleton(x, y)),
  andersonKsamp: ([a, b, c]) => sp(T.andersonKsamp([a, b, c as Tensor])),
  energyTest: ([x, y]) => sp(T.energyTest(x, y)),
  wassersteinTest: ([x, y]) => sp(T.wassersteinTest(x, y)),
})
