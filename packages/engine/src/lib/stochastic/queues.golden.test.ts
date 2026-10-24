import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import { erlangB, erlangC, mg1, mmc } from './queues.ts'

goldenSuite('stochastic/queues against scipy and truncated birth-death chains', 'stochastic/queues', {
  'erlang-b': (_i, p) => erlangB(p['c'] as number, p['a'] as number),
  'erlang-c': (_i, p) => erlangC(p['c'] as number, p['a'] as number),
  mmc: (_i, p) => {
    const q = mmc(p['lam'] as number, p['mu'] as number, p['c'] as number)
    return tensor([q.L, q.Lq, q.p0])
  },
  mg1: (_i, p) => mg1(p['lam'] as number, { mean: p['mean'] as number, variance: p['variance'] as number }).Lq,
})
