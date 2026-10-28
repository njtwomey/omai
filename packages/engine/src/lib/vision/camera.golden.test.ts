import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toArray, toFlat } from '../tensor/index.ts'
import * as V from './index.ts'

const rows = (t: Tensor): number[][] => toArray(t) as number[][]
const vec3 = (t: Tensor): [number, number, number] => toFlat(t) as [number, number, number]

goldenSuite('the pinhole camera against OpenCV', 'vision/camera', {
  rodrigues: ([r]) => tensor(V.rodrigues(toFlat(r))),
  rodriguesVector: ([m]) => tensor(V.rodriguesVector(rows(m))),
  projectPoints: ([X, r, t, k, d]) =>
    tensor(V.projectPoints(X, { rvec: vec3(r), tvec: vec3(t as Tensor) }, rows(k as Tensor), toFlat(d as Tensor))),
  undistortPoints: ([p, k, d, R, P]) =>
    tensor(
      V.undistortPoints(p, rows(k), toFlat(d as Tensor), {
        ...(R === undefined ? {} : { R: rows(R) }),
        ...(P === undefined ? {} : { P: rows(P) }),
      }),
    ),
  initUndistortRectifyMap: ([k, d, R, newK], o) => {
    const m = V.initUndistortRectifyMap(rows(k), toFlat(d as Tensor), o['size'] as [number, number], {
      R: rows(R as Tensor),
      newCameraMatrix: rows(newK as Tensor),
    })
    return tensor([...toFlat(m.mapX), ...toFlat(m.mapY)])
  },
  solvePnP: ([X, x, k, d]) => {
    const p = V.solvePnP(X, x, rows(k as Tensor), toFlat(d as Tensor))
    return tensor([...p.rvec, ...p.tvec])
  },
})
