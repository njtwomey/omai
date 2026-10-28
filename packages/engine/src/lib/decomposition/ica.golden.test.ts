import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import { fastIca, type IcaContrast } from './ica.ts'

const fit = (x: Tensor, w0: Tensor | undefined, fun: unknown, algorithm: 'parallel' | 'deflation') =>
  fastIca
    .with({
      nComponents: 3,
      fun: fun as IcaContrast,
      wInit: toArray(w0 as Tensor) as number[][],
      maxIter: 400,
      algorithm,
    })
    .fit({ x })

goldenSuite('FastICA against scikit-learn', 'methods/decomposition/ica', {
  components: ([x, w0], o) => tensor(fit(x, w0, o['fun'], 'parallel').params.components),
  sources: ([x, w0], o) => fit(x, w0, o['fun'], 'parallel').transform(x),
  mixing: ([x, w0], o) => tensor(fit(x, w0, o['fun'], 'parallel').params.mixing),
})

goldenSuite('deflationary FastICA against scikit-learn', 'decomposition/ica_deflation', {
  components: ([x, w0], o) => tensor(fit(x, w0, o['fun'], 'deflation').params.components),
  sources: ([x, w0], o) => fit(x, w0, o['fun'], 'deflation').transform(x),
})
