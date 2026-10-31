import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import { type NmfLoss, type NmfSolver, nmf } from './nmf.ts'

const fit = (inputs: readonly (Tensor | undefined)[], loss: NmfLoss, solver: NmfSolver, maxIter: number) => {
  const [x, w, h] = inputs as Tensor[]
  return nmf
    .with({
      nComponents: 3,
      betaLoss: loss,
      solver,
      maxIter,
      tol: 0,
      init: 'custom',
      wInit: toArray(w as Tensor) as number[][],
      hInit: toArray(h as Tensor) as number[][],
    })
    .fit({ x: x as Tensor })
}

goldenSuite('NMF by multiplicative updates against scikit-learn', 'methods/decomposition/nmf', {
  h: (i, o) => tensor(fit(i, o['loss'] as NmfLoss, 'mu', 60).params.components),
  w: (i, o) => tensor(fit(i, o['loss'] as NmfLoss, 'mu', 60).params.weights),
  error: (i, o) => fit(i, o['loss'] as NmfLoss, 'mu', 60).params.reconstructionErr,
})

goldenSuite('NMF by HALS against the coordinate descent of scikit-learn', 'decomposition/nmf_hals', {
  h: (i, o) => tensor(fit(i, 'frobenius', 'hals', o['sweeps'] as number).params.components),
  w: (i, o) => tensor(fit(i, 'frobenius', 'hals', o['sweeps'] as number).params.weights),
  error: (i, o) => fit(i, 'frobenius', 'hals', o['sweeps'] as number).params.reconstructionErr,
})
