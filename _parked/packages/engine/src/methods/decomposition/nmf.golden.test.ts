import { type Tensor, tensor, toArray } from '../../lib/tensor/index.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { type NmfLoss, nmf } from './nmf.ts'

const fit = (inputs: readonly (Tensor | undefined)[], loss: unknown) => {
  const [x, w, h] = inputs as Tensor[]
  return nmf({
    nComponents: 3,
    betaLoss: loss as NmfLoss,
    maxIter: 60,
    tol: 0,
    init: { w: toArray(w as Tensor) as number[][], h: toArray(h as Tensor) as number[][] },
  }).fit({ x: x as Tensor })
}

goldenSuite('NMF by multiplicative updates against scikit-learn', 'methods/decomposition/nmf', {
  h: (i, o) => tensor(fit(i, o['loss']).params.components),
  w: (i, o) =>
    tensor(
      fit(i, o['loss']).exposition.flatMap((r) => (r.kind === 'components' ? [r.scores ?? []] : []))[0] as number[][],
    ),
  error: (i, o) => fit(i, o['loss']).params.reconstructionErr,
})
