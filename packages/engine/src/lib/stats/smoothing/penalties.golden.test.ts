import { goldenSuite } from '../../../testing/suite.ts'
import { toFlat } from '../../tensor/index.ts'
import * as P from './index.ts'

goldenSuite('penalties against numpy and scipy', 'stats/smoothing/penalties', {
  difference: (_, p) => P.differencePenalty(Number(p['k']), Number(p['order'])),
  derivative: ([t], p) => P.derivativePenalty(t, Number(p['degree']), Number(p['order'])),
  sumToZero: ([X]) => P.sumToZeroConstraint(X),
  constraintNullSpace: ([C]) => P.constraintNullSpace(toFlat(C), C.shape[1] ?? 0),
})
