import { goldenSuite } from '../../testing/suite.ts'
import * as V from './index.ts'

goldenSuite('total-variation denoising against scikit-image', 'vision/denoise', {
  denoiseTvChambolle: ([x], o) =>
    V.denoiseTvChambolle(x, {
      weight: o['weight'] as number,
      eps: o['eps'] as number,
      maxIter: o['maxIter'] as number,
    }),
})
