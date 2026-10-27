import { goldenSuite } from '../../testing/suite.ts'
import { tensor, toArray } from '../tensor/index.ts'
import * as V from './index.ts'

const KEYS = [
  'm00',
  'm10',
  'm01',
  'm20',
  'm11',
  'm02',
  'm30',
  'm21',
  'm12',
  'm03',
  'mu20',
  'mu11',
  'mu02',
  'mu30',
  'mu21',
  'mu12',
  'mu03',
  'nu20',
  'nu11',
  'nu02',
  'nu30',
  'nu21',
  'nu12',
  'nu03',
] as const

const flat = (m: V.Moments) => tensor(KEYS.map((k) => m[k]))
const points = (t: Parameters<typeof toArray>[0]): [number, number][] => toArray(t) as [number, number][]

goldenSuite('moments against OpenCV', 'vision/moments', {
  moments: ([c]) => flat(V.moments(points(c))),
  huMoments: ([c]) => tensor(V.huMoments(V.moments(points(c)))),
  imageMoments: ([x], o) => flat(V.imageMoments(x, { binary: o['binary'] as boolean })),
  matchShapes: ([a, b], o) => tensor([V.matchShapes(points(a), points(b), o['method'] as 1 | 2 | 3)]),
})
