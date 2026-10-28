import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toArray } from '../tensor/index.ts'
import * as V from './index.ts'

const rows = (t: Tensor): number[][] => toArray(t) as number[][]

goldenSuite('calibrated two-view geometry against OpenCV', 'vision/epipolar', {
  findEssentialMat: ([a, b, k]) => tensor(V.findEssentialMat(a, b, rows(k as Tensor)).model),
  recoverPose: ([e, a, b, k]) => {
    const r = V.recoverPose(rows(e), a, b as Tensor, rows(k as Tensor))
    return tensor([...r.R.flat(), ...r.t, r.good])
  },
  triangulatePoints: ([P1, P2, a, b]) => tensor(V.triangulatePoints(rows(P1), rows(P2), a as Tensor, b as Tensor)),
})
