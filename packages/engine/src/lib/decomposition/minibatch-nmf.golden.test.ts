import { learnGolden, num, str } from '../../testing/learn.ts'
import { type Tensor, toArray } from '../tensor/index.ts'
import { type BetaLoss, miniBatchNmf } from './minibatch-nmf.ts'

learnGolden('MiniBatchNMF against scikit-learn from fixed initial factors', 'decomposition/minibatch_nmf', {
  minibatch: ([x, w0, h0, held], p) => {
    const betaLoss: BetaLoss = typeof p['betaLoss'] === 'string' ? (str(p, 'betaLoss') as BetaLoss) : num(p, 'betaLoss')
    const maxNoImprovement = num(p, 'maxNoImprovement')
    const f = miniBatchNmf
      .with({
        nComponents: (h0 as Tensor).shape[0] ?? 1,
        init: 'custom',
        wInit: toArray(w0 as Tensor) as number[][],
        hInit: toArray(h0 as Tensor) as number[][],
        betaLoss,
        batchSize: num(p, 'batchSize'),
        maxIter: num(p, 'maxIter'),
        tol: num(p, 'tol'),
        maxNoImprovement: maxNoImprovement < 0 ? null : maxNoImprovement,
        alphaW: num(p, 'alphaW'),
        alphaH: num(p, 'alphaH'),
        l1Ratio: num(p, 'l1Ratio'),
        forgetFactor: num(p, 'forgetFactor'),
        freshRestarts: p['freshRestarts'] === true,
        freshRestartsMaxIter: num(p, 'freshRestartsMaxIter'),
      })
      .fit({ x: x as Tensor })
    return {
      components: f.params.components,
      weights: f.params.weights,
      nSteps: f.params.nSteps,
      nIter: f.params.nIter,
      error: f.params.reconstructionErr,
      transform: f.transform(held as Tensor),
    }
  },
})
