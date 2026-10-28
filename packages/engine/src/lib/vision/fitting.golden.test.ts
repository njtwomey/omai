import { goldenSuite } from '../../testing/suite.ts'
import { tensor, toArray, toFlat } from '../tensor/index.ts'
import * as V from './index.ts'

const points = (t: Parameters<typeof toArray>[0]): [number, number][] => toArray(t) as [number, number][]
const rect = (r: V.RotatedRect) => tensor([...r.center, ...r.size, r.angle])

goldenSuite('fitted shapes against OpenCV', 'vision/fitting', {
  minAreaRect: ([p]) => rect(V.minAreaRect(points(p))),
  boxPoints: ([b]) => {
    const [cx = 0, cy = 0, w = 0, h = 0, angle = 0] = toFlat(b)
    return tensor(V.boxPoints({ center: [cx, cy], size: [w, h], angle }).flat())
  },
  minEnclosingCircle: ([p]) => {
    const c = V.minEnclosingCircle(points(p))
    return tensor([...c.center, c.radius])
  },
  fitEllipse: ([p]) => rect(V.fitEllipse(points(p))),
  fitLine: ([p]) => tensor(V.fitLine(points(p))),
})
