import { geometry } from '../../lib/numerics/index.ts'
import { type Tensor, tensor, toArray } from '../../lib/tensor/index.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { isomap } from './isomap.ts'
import { locallyLinearEmbedding } from './lle.ts'
import { mds } from './mds.ts'
import { trustworthiness } from './quality.ts'
import { spectralEmbedding } from './spectral.ts'

const pd = (rows: number[][]) => geometry.pdist(rows)
const smacof = (x: Tensor, init: Tensor | undefined) =>
  mds({ init: toArray(init as Tensor) as number[][], maxIter: 300, eps: 1e-6 }).fit({ x })

goldenSuite('trustworthiness against scikit-learn', 'methods/manifold/quality', {
  trustworthiness: ([x, e], o) => trustworthiness(x, e, o['k'] as number),
})
goldenSuite('Isomap against scikit-learn', 'methods/manifold/isomap', {
  isomap: ([x], o) => pd(isomap({ nNeighbors: o['k'] as number }).fit({ x }).params.embedding),
})
goldenSuite('LLE against scikit-learn', 'methods/manifold/lle', {
  lle: ([x], o) => pd(locallyLinearEmbedding({ nNeighbors: o['k'] as number }).fit({ x }).params.embedding),
})
goldenSuite('spectral embedding against scikit-learn', 'methods/manifold/spectral', {
  rbf: ([x], o) => pd(spectralEmbedding({ affinity: 'rbf', gamma: o['gamma'] as number }).fit({ x }).params.embedding),
})
goldenSuite('SMACOF against scikit-learn', 'methods/manifold/mds', {
  embedding: ([x, init]) => tensor(smacof(x, init).params.embedding),
  stress: ([x, init]) => smacof(x, init).params.objective,
})
