import { goldenSuite } from '../../../testing/suite.ts'
import type { TestResult } from '../../base/index.ts'
import { type Tensor, tensor } from '../../tensor/index.ts'
import * as T from './index.ts'

const row = (r: TestResult): Tensor => {
  const a = r.assumptions ?? {}
  return tensor([
    r.pValue,
    Number(a['statisticLower']),
    Number(a['pLower']),
    Number(a['statisticUpper']),
    Number(a['pUpper']),
  ])
}

goldenSuite('equivalence tests against statsmodels', 'stats/tests/equivalence', {
  tostInd: ([x, y], p) => row(T.tost(x, y, { low: -0.6, upp: 0.6, equalVar: p['equalVar'] as boolean })),
  tostPaired: ([x, y]) => row(T.tost(x, y, { low: -0.3, upp: 0.3, paired: true })),
  tostOne: ([x]) => row(T.tost(x, undefined, { low: -0.2, upp: 0.5 })),
  tostProportion: () => {
    const r = T.tostProportion(52, 100, { low: 0.4, upp: 0.6 })
    return tensor([r.pValue, Number(r.assumptions?.['pLower']), Number(r.assumptions?.['pUpper'])])
  },
  tostProportions: (_, p) => {
    const r = T.tostProportions(60, 120, 58, 115, { low: -0.15, upp: 0.15, method: p['method'] as 'wald' })
    return tensor([r.statistic, r.pValue])
  },
})
