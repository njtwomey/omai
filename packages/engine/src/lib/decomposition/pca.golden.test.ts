import { type CaseFn, goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor } from '../tensor/index.ts'
import { incrementalPca, pca } from './pca.ts'

const inc = (x: Tensor, o: Readonly<Record<string, unknown>>) =>
  incrementalPca.with({ nComponents: o['k'] as number, batchSize: o['batch'] as number }).fit({ x }).params

const suite = (solver: 'full' | 'power'): Record<string, CaseFn> => {
  const fit = (x: Tensor, k: unknown) => pca.with({ nComponents: k as number, solver }).fit({ x })
  return {
    components: ([x], o) => tensor(fit(x, o['k']).params.components),
    variance: ([x], o) => tensor(fit(x, o['k']).params.explainedVariance),
    ratio: ([x], o) => tensor(fit(x, o['k']).params.explainedVarianceRatio),
    noise: ([x], o) => fit(x, o['k']).params.noiseVariance,
    scores: ([x], o) => fit(x, o['k']).transform(x),
    score: ([x], o) => fit(x, o['k']).score({ x }),
    whitened: ([x]) => pca.with({ nComponents: 2, whiten: true, solver }).fit({ x }).transform(x),
    ipcaComponents: ([x], o) => tensor(inc(x, o).components),
    ipcaVariance: ([x], o) => tensor(inc(x, o).explainedVariance),
    ipcaRatio: ([x], o) => tensor(inc(x, o).explainedVarianceRatio),
    ipcaNoise: ([x], o) => inc(x, o).noiseVariance,
  }
}

goldenSuite('PCA against scikit-learn', 'methods/decomposition/pca', suite('full'))
goldenSuite('PCA by power iteration against scikit-learn', 'methods/decomposition/pca', suite('power'))
