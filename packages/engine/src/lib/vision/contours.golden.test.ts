import { goldenSuite } from '../../testing/suite.ts'
import { tensor, toArray } from '../tensor/index.ts'
import * as V from './index.ts'

const points = (t: Parameters<typeof toArray>[0]): [number, number][] => toArray(t) as [number, number][]

goldenSuite('contours against OpenCV', 'vision/contours', {
  findContours: ([x], o) => {
    const r = V.findContours(x, { mode: o['mode'] as V.RetrievalMode, method: o['method'] as V.ChainApproximation })
    const flat = [r.contours.length]
    for (const c of r.contours) flat.push(c.length, ...c.flat())
    if (r.contours.length > 0) flat.push(...r.hierarchy.flat())
    return tensor(flat)
  },
  approxPolyDP: ([c], o) => tensor(V.approxPolyDP(points(c), o['epsilon'] as number, o['closed'] as boolean)),
  arcLength: ([c], o) => tensor([V.arcLength(points(c), o['closed'] as boolean)]),
  contourArea: ([c], o) => tensor([V.contourArea(points(c), o['oriented'] as boolean)]),
  boundingRect: ([c]) => tensor(V.boundingRect(points(c))),
  isContourConvex: ([c]) => tensor([V.isContourConvex(points(c)) ? 1 : 0]),
  pointPolygonTest: ([c, q], o) =>
    tensor(points(q).map((p) => V.pointPolygonTest(points(c), p, o['measureDist'] as boolean))),
})
