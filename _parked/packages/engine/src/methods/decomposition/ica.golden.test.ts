import { type Tensor, tensor, toArray } from '../../lib/tensor/index.ts'
import { goldenSuite } from '../../testing/suite.ts'
import { fastIca, type IcaContrast } from './ica.ts'

const fit = (x: Tensor, w0: Tensor | undefined, fun: unknown) =>
  fastIca({ nComponents: 3, fun: fun as IcaContrast, wInit: toArray(w0 as Tensor) as number[][], maxIter: 400 }).fit({
    x,
  })

goldenSuite('FastICA against scikit-learn', 'methods/decomposition/ica', {
  components: ([x, w0], o) => tensor(fit(x, w0, o['fun']).params.components),
  sources: ([x, w0], o) => fit(x, w0, o['fun']).transform(x),
  mixing: ([x, w0], o) => tensor(fit(x, w0, o['fun']).params.mixing),
})
