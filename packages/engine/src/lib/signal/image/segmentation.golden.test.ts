import { goldenSuite } from '../../../testing/suite.ts'
import { tensor } from '../../tensor/index.ts'
import * as I from './index.ts'

goldenSuite('segmentation against scikit-image', 'signal/image/segmentation', {
  thresholdOtsu: ([x], o) => tensor([I.thresholdOtsu(x, o['nbins'] as number)]),
  thresholdLi: ([x]) => tensor([I.thresholdLi(x)]),
  watershed: ([x, m, mask], o) =>
    I.watershed(x, m as never, {
      ...(o['connectivity'] === undefined ? {} : { connectivity: o['connectivity'] as 1 | 2 }),
      ...(mask === undefined ? {} : { mask }),
    }),
  felzenszwalb: ([x], o) =>
    I.felzenszwalb(x, {
      scale: o['scale'] as number,
      minSize: o['minSize'] as number,
      ...(o['sigma'] === undefined ? {} : { sigma: o['sigma'] as number }),
    }),
})
