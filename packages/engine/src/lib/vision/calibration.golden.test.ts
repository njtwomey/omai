import { goldenSuite } from '../../testing/suite.ts'
import { type Tensor, tensor, toArray, toFlat } from '../tensor/index.ts'
import * as V from './index.ts'

const rows = (t: Tensor): number[][] => toArray(t) as number[][]

goldenSuite('calibration against OpenCV', 'vision/calibration', {
  calibrateCamera: ([obj, img], o) => {
    const O = toArray(obj) as number[][][]
    const I = toArray(img) as number[][][]
    const c = V.calibrateCamera(O, I, { fixK3: o['fixK3'] as boolean })
    const K = c.cameraMatrix
    return tensor([K[0]?.[0] ?? 0, K[1]?.[1] ?? 0, K[0]?.[2] ?? 0, K[1]?.[2] ?? 0, ...c.distortion, c.rms])
  },
  stereoRectify: ([k1, d1, k2, d2, R, t], o) => {
    const r = V.stereoRectify(
      rows(k1),
      toFlat(d1),
      rows(k2 as Tensor),
      toFlat(d2 as Tensor),
      o['size'] as [number, number],
      rows(R as Tensor),
      toFlat(t as Tensor) as [number, number, number],
    )
    return tensor([...r.R1.flat(), ...r.R2.flat()])
  },
})
