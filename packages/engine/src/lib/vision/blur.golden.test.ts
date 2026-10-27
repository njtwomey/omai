import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as V from './index.ts'

goldenSuite('smoothing filters against OpenCV', 'vision/blur', {
  getGaussianKernel: (_, o) => tensor(Array.from(V.getGaussianKernel(o['ksize'] as number, o['sigma'] as number))),
  gaussianBlur: ([x], o) =>
    V.gaussianBlur(x, o['ksize'] as [number, number], o['sigmaX'] as number, {
      ...(o['sigmaY'] === undefined ? {} : { sigmaY: o['sigmaY'] as number }),
      border: o['border'] as V.CvBorder,
    }),
  blur: ([x], o) => V.blur(x, o['ksize'] as [number, number], { border: o['border'] as V.CvBorder }),
  filter2D: ([x, k], o) => V.filter2D(x, k, { border: o['border'] as V.CvBorder }),
  medianBlur: ([x], o) => V.medianBlur(x, o['ksize'] as number),
})
