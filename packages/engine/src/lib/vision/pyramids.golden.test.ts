import { goldenSuite } from '../../testing/suite.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import * as V from './index.ts'

goldenSuite('pyramids against OpenCV', 'vision/pyramids', {
  pyrDown: ([x], o) => V.pyrDown(x, { border: o['border'] as V.CvBorder }),
  pyrUp: ([x], o) => V.pyrUp(x, o['size'] as [number, number] | undefined),
  buildPyramid: ([x], o) => tensor(V.buildPyramid(x, o['maxLevel'] as number).flatMap((l) => toFlat(l))),
  laplacianPyramid: ([x], o) => tensor(V.laplacianPyramid(x, o['levels'] as number).flatMap((l) => toFlat(l))),
})
