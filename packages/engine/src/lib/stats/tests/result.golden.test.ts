import { goldenSuite } from '../../../testing/suite.ts'
import { type Tensor, tensor } from '../../tensor/index.ts'
import * as T from './index.ts'

const sp = (r: { statistic: number; pValue: number }): Tensor => tensor([r.statistic, r.pValue])

goldenSuite('equal-variance tests against scipy.stats', 'stats/tests/result', {
  levene: ([a, b, c], p) => sp(T.levene([a, b, c as Tensor], { center: p['center'] as T.SpreadCenter })),
  bartlett: ([a, b, c]) => sp(T.bartlett([a, b, c as Tensor])),
  fligner: ([a, b, c], p) => sp(T.fligner([a, b, c as Tensor], { center: p['center'] as T.SpreadCenter })),
})
