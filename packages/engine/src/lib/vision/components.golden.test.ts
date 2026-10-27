import { goldenSuite } from '../../testing/suite.ts'
import { tensor, toFlat } from '../tensor/index.ts'
import * as V from './index.ts'

goldenSuite('regions against OpenCV', 'vision/components', {
  connectedComponentsWithStats: ([x], o) => {
    const r = V.connectedComponentsWithStats(x, { connectivity: o['connectivity'] as 4 | 8 })
    return tensor([r.count, ...toFlat(r.labels), ...r.stats.flat(), ...r.centroids.flat()])
  },
  floodFill: ([x, barrier], o) => {
    const r = V.floodFill(x, o['seed'] as [number, number], 0.75, {
      loDiff: o['loDiff'] as number,
      upDiff: o['upDiff'] as number,
      connectivity: o['connectivity'] as 4 | 8,
      fixedRange: o['fixedRange'] as boolean,
      ...(barrier === undefined ? {} : { mask: barrier }),
    })
    return tensor([r.area, ...toFlat(r.image), ...toFlat(r.mask), ...r.rect])
  },
})
