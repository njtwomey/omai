import { learnGolden, num, str } from '../../testing/learn.ts'
import type { Tensor } from '../tensor/index.ts'
import { type DiscontinuityKernel, regressionDiscontinuity } from './discontinuity.ts'

learnGolden('Sharp regression discontinuity against statsmodels weighted least squares', 'causal/discontinuity', {
  regressionDiscontinuity: (inputs, p) => {
    const [x, y] = inputs as [Tensor, Tensor]
    const r = regressionDiscontinuity(
      { running: x, outcome: y },
      {
        cutoff: num(p, 'cutoff'),
        bandwidth: num(p, 'bandwidth'),
        kernel: str(p, 'kernel') as DiscontinuityKernel,
        order: num(p, 'order'),
      },
    )
    return { value: r.effect.estimate.value, se: r.effect.estimate.standardError ?? Number.NaN }
  },
})
