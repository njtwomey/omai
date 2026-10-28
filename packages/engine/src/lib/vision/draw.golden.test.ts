import { goldenSuite } from '../../testing/suite.ts'
import * as V from './index.ts'

const canvas = (): number[][] => Array.from({ length: 20 }, () => new Array<number>(24).fill(0))

goldenSuite('drawing against OpenCV', 'vision/draw', {
  line: (_, o) =>
    V.line(canvas(), o['p1'] as [number, number], o['p2'] as [number, number], 1, { lineType: o['lineType'] as 4 | 8 }),
  polylines: (_, o) => V.polylines(canvas(), o['points'] as [number, number][], true, 1),
  rectangle: (_, o) =>
    V.rectangle(canvas(), o['p1'] as [number, number], o['p2'] as [number, number], 1, {
      thickness: o['thickness'] as number,
    }),
})
