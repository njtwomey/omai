import { goldenSuite } from '../../../testing/suite.ts'
import { tensor } from '../../tensor/index.ts'
import * as T from './index.ts'

goldenSuite('kernel tests against their definitions', 'stats/tests/kernel', {
  medianHeuristic: ([x]) => T.medianHeuristic(x),
  mmdGamma: ([x, y]) => {
    const r = T.mmdTest(x, y, { null: 'gamma' })
    return tensor([r.statistic, r.pValue])
  },
  hsicGamma: ([x, y]) => {
    const r = T.hsicTest(x, y, { null: 'gamma' })
    return tensor([r.statistic, r.pValue])
  },
  mmd2Unbiased: ([x, y]) => T.mmd2(x, y, { unbiased: true }),
})
