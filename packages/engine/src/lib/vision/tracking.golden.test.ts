import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as V from './index.ts'

goldenSuite('window trackers against OpenCV', 'vision/tracking', {
  meanShift: ([p], o) => {
    const r = V.meanShift(p, o['window'] as V.Rect, { maxIter: 20, eps: 1 })
    return tensor([...r.window, r.iterations])
  },
  camShift: ([p], o) => {
    const r = V.camShift(p, o['window'] as V.Rect, { maxIter: 20, eps: 1 })
    return tensor([...r.box.center, ...r.box.size, r.box.angle, ...r.window])
  },
})
