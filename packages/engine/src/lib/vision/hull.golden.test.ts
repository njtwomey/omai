import { goldenSuite } from '../../testing/suite.ts'
import { tensor, toArray } from '../tensor/index.ts'
import * as V from './index.ts'

const points = (t: Parameters<typeof toArray>[0]): [number, number][] => toArray(t) as [number, number][]

goldenSuite('convex hulls against OpenCV', 'vision/hull', {
  convexHull: ([p], o) => tensor(V.convexHull(points(p), { clockwise: o['clockwise'] as boolean })),
  convexityDefects: ([c]) => {
    const pts = points(c)
    return tensor(V.convexityDefects(pts, V.convexHull(pts)).flatMap((d) => [d.start, d.end, d.farthest, d.depth]))
  },
})
