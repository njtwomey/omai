import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import * as V from './index.ts'

const rows = (t: Tensor): number[][] => toArray(t) as number[][]
const flatMatches = (ms: readonly V.DMatch[]): number[][] => ms.map((m) => [m.queryIdx, m.trainIdx, m.distance])

goldenSuite('descriptor matching against OpenCV', 'vision/matching', {
  matchDescriptors: ([a, b], o) =>
    tensor(
      flatMatches(
        V.matchDescriptors(rows(a), rows(b), {
          norm: o['norm'] as V.MatchNorm,
          crossCheck: o['crossCheck'] as boolean,
        }),
      ),
    ),
  knnMatch: ([a, b], o) =>
    tensor(V.knnMatch(rows(a), rows(b), o['k'] as number, o['norm'] as V.MatchNorm).flatMap(flatMatches)),
})
