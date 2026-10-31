import { learnGolden } from '../../testing/learn.ts'
import { type Tensor, toFlat } from '../tensor/index.ts'
import { rosenbaumBounds, rosenbaumBoundsBinary } from './index.ts'

const gammas = [1, 1.25, 1.5, 2, 3]

learnGolden('Rosenbaum bounds against scipy', 'causal/sensitivity', {
  signedRank: (inputs) => {
    const [t, c] = inputs as [Tensor, Tensor]
    const r = rosenbaumBounds(toFlat(t), toFlat(c), { gammas })
    return { pUpper: r.rows.map((x) => x.pUpper), pLower: r.rows.map((x) => x.pLower) }
  },
  mcnemar: (inputs) => {
    const [t, c] = inputs as [Tensor, Tensor]
    const r = rosenbaumBoundsBinary(toFlat(t), toFlat(c), { gammas })
    return { pUpper: r.rows.map((x) => x.pUpper), pLower: r.rows.map((x) => x.pLower) }
  },
})
