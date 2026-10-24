import { type Tensor, tensor, toArray, toFlat } from '../../lib/tensor/index.ts'
import { learnGolden, str } from '../../testing/learn.ts'
import { categoricalHMM, gaussianHMM } from './hmm.ts'

const rows = (t: Tensor): number[][] => toArray(t) as number[][]

learnGolden('Baum-Welch against hmmlearn', 'methods/models/hmm', {
  categorical: (inputs) => {
    const [x, start, trans, emission] = inputs as readonly Tensor[]
    const f = categoricalHMM({
      nComponents: 2,
      nIter: 8,
      tol: Number.NEGATIVE_INFINITY,
      init: {
        startprob: toFlat(start as Tensor),
        transmat: rows(trans as Tensor),
        emissionprob: rows(emission as Tensor),
      },
    }).fit({ x: x as Tensor })
    const p = f.params
    return {
      startprob: tensor(p.startprob),
      transmat: tensor(p.transmat),
      emissionprob: tensor(p.emissionprob ?? []),
      score: tensor(f.score({ x: x as Tensor })),
    }
  },
  gaussian: (inputs, o) => {
    const [y, start, trans, means, covs] = inputs as readonly Tensor[]
    const kind = str(o, 'kind') as 'diag' | 'full'
    const f = gaussianHMM({
      nComponents: 2,
      covarianceType: kind,
      nIter: 8,
      tol: Number.NEGATIVE_INFINITY,
      init: {
        startprob: toFlat(start as Tensor),
        transmat: rows(trans as Tensor),
        means: rows(means as Tensor),
        covars: toArray(covs as Tensor) as number[][][],
      },
    }).fit({ x: y as Tensor })
    const p = f.params
    const covars = p.covars ?? []
    return {
      startprob: tensor(p.startprob),
      transmat: tensor(p.transmat),
      means: tensor(p.means ?? []),
      covars: tensor(covars),
      score: tensor(f.score({ x: y as Tensor })),
    }
  },
})
