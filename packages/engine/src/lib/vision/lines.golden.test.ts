import { goldenSuite } from '../../testing/suite.ts'
import { tensor } from '../tensor/index.ts'
import * as V from './index.ts'

goldenSuite('line segments against OpenCV', 'vision/lines', {
  lineSegmentDetector: ([x], o) => {
    const r = V.lineSegmentDetector(x, { validate: false, angleTolerance: o['angleTolerance'] as number })
    return tensor(r.lines.map((l, i) => [...l, r.widths[i] ?? 0]))
  },
})
