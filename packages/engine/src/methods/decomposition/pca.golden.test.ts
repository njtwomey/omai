import { type Tensor, tensor } from '../../lib/tensor/index.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { incrementalPca, pca } from './pca.ts'

const fit = (x: Tensor, k: unknown) => pca({ nComponents: k as number }).fit({ x })
const inc = (x: Tensor, o: Readonly<Record<string, unknown>>) =>
  incrementalPca({ nComponents: o['k'] as number, batchSize: o['batch'] as number }).fit({ x }).params

goldenSuite('PCA against scikit-learn', 'methods/decomposition/pca', {
  components: ([x], o) => tensor(fit(x, o['k']).params.components),
  variance: ([x], o) => tensor(fit(x, o['k']).params.explainedVariance),
  ratio: ([x], o) => tensor(fit(x, o['k']).params.explainedVarianceRatio),
  noise: ([x], o) => fit(x, o['k']).params.noiseVariance,
  scores: ([x], o) => fit(x, o['k']).transform(x),
  score: ([x], o) => fit(x, o['k']).score({ x }),
  whitened: ([x]) => pca({ nComponents: 2, whiten: true }).fit({ x }).transform(x),
  ipcaComponents: ([x], o) => tensor(inc(x, o).components),
  ipcaVariance: ([x], o) => tensor(inc(x, o).explainedVariance),
  ipcaRatio: ([x], o) => tensor(inc(x, o).explainedVarianceRatio),
  ipcaNoise: ([x], o) => inc(x, o).noiseVariance,
})
