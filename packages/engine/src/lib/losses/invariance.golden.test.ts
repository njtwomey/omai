import { goldenSuite } from '../../testing/suite.ts'
import * as L from './index.ts'

goldenSuite(
  'invariance penalties against torch autograd and the Deep CORAL and Fishr reference code',
  'losses/invariance',
  {
    irmPenalty: ([s, y], p) => {
      const { risk, ...rest } = p as { risk?: string; unbiased?: boolean }
      return L.irmPenalty(s, y, { ...rest, ...(risk === 'mse' ? { risk: L.mse } : {}) })
    },
    coral: ([a, b]) => L.coral(a, b),
    fishrPenalty: (gradients) => L.fishrPenalty(gradients.filter((g) => g !== undefined)),
  },
)
