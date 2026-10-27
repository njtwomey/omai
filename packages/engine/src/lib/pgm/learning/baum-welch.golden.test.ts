import { learnGolden, str } from '../../../testing/learn.ts'
import { type Tensor, tensor, toArray, toFlat } from '../../tensor/index.ts'
import { runAlgorithm } from '../../trace/index.ts'
import { baumWelchAlgorithm, type HmmParameters, hmmDecode } from './baum-welch.ts'

const rows = (t: Tensor): number[][] => toArray(t) as number[][]

// Eight Baum-Welch iterations from fixed starting parameters, then the log likelihood of the fitted model; the fixture
// kept its path from before the algorithm moved into the core.
learnGolden('Baum-Welch against hmmlearn', 'methods/models/hmm', {
  categorical: (inputs) => {
    const [x, start, trans, emission] = inputs as readonly Tensor[]
    const sequence = Array.from(toFlat(x as Tensor))
    const from: HmmParameters = {
      kind: 'categorical',
      start: Array.from(toFlat(start as Tensor)),
      transition: rows(trans as Tensor),
      emission: rows(emission as Tensor),
    }
    const p = runAlgorithm(baumWelchAlgorithm([sequence], from, { tol: Number.NEGATIVE_INFINITY }), { maxSteps: 8 })
      .state.params
    if (p.kind !== 'categorical') throw new Error('expected categorical parameters')
    return {
      startprob: tensor(p.start),
      transmat: tensor(p.transition),
      emissionprob: tensor(p.emission),
      score: tensor(hmmDecode(p, sequence).logLikelihood),
    }
  },
  gaussian: (inputs, o) => {
    const [y, start, trans, means, covs] = inputs as readonly Tensor[]
    const sequence = rows(y as Tensor)
    const from: HmmParameters = {
      kind: 'gaussian',
      start: Array.from(toFlat(start as Tensor)),
      transition: rows(trans as Tensor),
      means: rows(means as Tensor),
      covariances: toArray(covs as Tensor) as number[][][],
    }
    const covarianceType = str(o, 'kind') as 'diag' | 'full'
    const p = runAlgorithm(baumWelchAlgorithm([sequence], from, { covarianceType, tol: Number.NEGATIVE_INFINITY }), {
      maxSteps: 8,
    }).state.params
    if (p.kind !== 'gaussian') throw new Error('expected Gaussian parameters')
    return {
      startprob: tensor(p.start),
      transmat: tensor(p.transition),
      means: tensor(p.means),
      covars: tensor(p.covariances),
      score: tensor(hmmDecode(p, sequence).logLikelihood),
    }
  },
})
