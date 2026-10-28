import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import * as V from './index.ts'

const rows = (t: Tensor): number[][] => toArray(t) as number[][]
const pts = (t: Tensor): [number, number][] => toArray(t) as [number, number][]

goldenSuite('two-view geometry against OpenCV', 'vision/geometry', {
  getPerspectiveTransform: ([a, b]) => tensor(V.getPerspectiveTransform(a, b)),
  getAffineTransform: ([a, b]) => tensor(V.getAffineTransform(a, b)),
  perspectiveTransform: ([p, h]) => tensor(V.perspectiveTransform(p, rows(h))),
  findHomography: ([a, b], o) => {
    const r = V.findHomography(a, b, { method: o['method'] as 'all' | 'ransac' })
    return o['method'] === 'all' ? tensor(r.model) : tensor([...r.model.flat(), ...r.mask])
  },
  estimateAffine2D: ([a, b], o) => {
    const r = V.estimateAffine2D(a, b, { method: o['method'] as 'all' | 'ransac' })
    return o['method'] === 'all' ? tensor(r.model) : tensor([...r.model.flat(), ...r.mask])
  },
  findFundamentalMat: ([a, b], o) => {
    if (o['method'] === 'all') return tensor(V.findFundamentalMat(a, b, { method: 'all' }).model)
    return tensor(V.findFundamentalMat(a, b, { threshold: o['threshold'] as number, confidence: 0.999 }).mask)
  },
  computeCorrespondEpilines: ([p, f], o) => tensor(V.computeCorrespondEpilines(p, o['whichImage'] as 1 | 2, rows(f))),
  sampsonDistance: ([a, b, f]) => {
    const p = pts(a)
    const q = pts(b)
    return tensor(p.map((x, i) => V.sampsonDistance(x, q[i] as [number, number], rows(f as Tensor))))
  },
})
