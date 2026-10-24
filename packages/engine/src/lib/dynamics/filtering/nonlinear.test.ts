import { describe, expect, it } from 'vitest'
import { key } from '../../random/index.ts'
import { matmul, reshape, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { kalmanFilter } from './kalman.ts'
import {
  ensembleKalmanFilter,
  extendedKalmanFilter,
  julierSigmaPoints,
  merweSigmaPoints,
  unscentedKalmanFilter,
  unscentedTransform,
} from './nonlinear.ts'

const F = [
  [1, 0.5],
  [0, 1],
]
const lin = {
  f: (x: Value): Value => reshape(matmul(tensor(F), reshape(x, [2, 1])), [2]),
  h: (x: Value): Value => reshape(matmul(tensor([[1, 0]]), reshape(x, [2, 1])), [1]),
  Q: [
    [0.01, 0],
    [0, 0.02],
  ],
  R: [[0.3]],
  x0: [0, 1],
  P0: [
    [1, 0],
    [0, 1],
  ],
}
const zs = [[0.4], [1.2], [1.4], [2.1], [2.4]]

describe('nonlinear Kalman filters', () => {
  it('reduce to the Kalman filter on a linear model', () => {
    const kf = toFlat(kalmanFilter({ F, H: [[1, 0]], Q: lin.Q, R: lin.R, x0: lin.x0, P0: lin.P0 }, zs).means)
    for (const r of [extendedKalmanFilter(lin, zs), unscentedKalmanFilter(lin, zs, { redraw: true })]) {
      toFlat(r.means).forEach((v, i) => {
        expect(Math.abs(v - (kf[i] ?? 0))).toBeLessThan(1e-9)
      })
    }
    const en = ensembleKalmanFilter(lin, zs, { size: 4000, key: key(3) })
    toFlat(en.means).forEach((v, i) => {
      expect(Math.abs(v - (kf[i] ?? 0))).toBeLessThan(0.05)
    })
  })

  it('sigma points reproduce the mean and covariance', () => {
    for (const pts of [merweSigmaPoints(2, 0.5, 2, 0), julierSigmaPoints(2)]) {
      const x = tensor([1, -2])
      const P = tensor([
        [2, 0.3],
        [0.3, 1],
      ])
      const s = pts.points(x, P)
      const ut = unscentedTransform(s, pts.wm, pts.wc)
      expect(toFlat(ut.x).map((v) => Math.round(v * 1e12) / 1e12)).toEqual([1, -2])
      toFlat(ut.P).forEach((v, i) => {
        expect(Math.abs(v - ([2, 0.3, 0.3, 1][i] ?? 0))).toBeLessThan(1e-12)
      })
    }
  })
})
