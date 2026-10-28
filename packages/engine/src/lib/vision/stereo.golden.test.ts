import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, toArray } from '../tensor/index.ts'
import * as V from './index.ts'

goldenSuite('disparity reprojection against OpenCV', 'vision/stereo', {
  reprojectImageTo3D: ([d, Q]) => V.reprojectImageTo3D(d, toArray(Q as Tensor) as number[][]),
})
