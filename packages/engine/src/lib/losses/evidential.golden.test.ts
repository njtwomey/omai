import { goldenSuite } from '../../testing/suite.ts'
import type { Tensor } from '../tensor/index.ts'
import * as L from './index.ts'

goldenSuite('evidential losses against torch special functions and scipy Student-t', 'losses/evidential', {
  evidentialNll: ([e, y], p) => L.evidentialNll(e, y, p),
  normalInverseGammaNll: ([g, nu, a, b, y], p) => {
    const { lambda_, ...rest } = p as { lambda_?: number }
    return L.normalInverseGammaNll(g, nu as Tensor, a as Tensor, b as Tensor, y as Tensor, {
      ...rest,
      ...(lambda_ === undefined ? {} : { lambda: lambda_ }),
    })
  },
})
