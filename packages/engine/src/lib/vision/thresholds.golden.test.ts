import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as V from './index.ts'

goldenSuite('thresholds against OpenCV and scikit-image', 'vision/thresholds', {
  threshold: ([x], o) =>
    V.threshold(x, o['thresh'] as number, { maxValue: o['maxValue'] as number, type: o['type'] as V.ThresholdType })
      .image,
  thresholdTriangle: ([x]) => tensor([Math.round(V.thresholdTriangle(x) * 255)]),
  adaptiveThreshold: ([x], o) =>
    V.adaptiveThreshold(x, {
      method: o['method'] as 'mean' | 'gaussian',
      blockSize: o['blockSize'] as number,
      C: o['C'] as number,
      type: o['type'] as 'binary' | 'binary-inv',
    }),
  thresholdMultiOtsu: ([x], o) =>
    tensor(V.thresholdMultiOtsu(x, { classes: o['classes'] as number, nbins: o['nbins'] as number })),
  thresholdNiblack: ([x], o) => V.thresholdNiblack(x, { windowSize: o['windowSize'] as number, k: o['k'] as number }),
  thresholdSauvola: ([x], o) => V.thresholdSauvola(x, { windowSize: o['windowSize'] as number, k: o['k'] as number }),
})
