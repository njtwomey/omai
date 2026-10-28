import { goldenSuite } from '../../testing/suite.ts'
import { tensor, zeros } from '../tensor/index.ts'
import * as V from './index.ts'

goldenSuite('FAST against OpenCV', 'vision/fast', {
  fast: ([x], o) => {
    const k = V.fast(x, { threshold: o['threshold'] as number, nonmaxSuppression: o['nonmaxSuppression'] as boolean })
    return k.length === 0 ? zeros([0, 3]) : tensor(k.map((p) => [p.x, p.y, p.response]))
  },
})
