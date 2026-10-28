import { goldenSuite } from '../../testing/suite.ts'
import * as V from './index.ts'

goldenSuite('Haar-like features against scikit-image', 'vision/haar', {
  haarLikeFeature: ([ii], o) =>
    V.haarLikeFeature(ii, 0, 0, o['width'] as number, o['height'] as number, o['types'] as V.HaarType[]),
})
