import { describe, it } from 'vitest'
import { type GoldenFn, runFixture } from '../../../testing/systems.ts'
import { add, mul, sin, sliceAxis, stack, sum, type Tensor, tensor, toFlat, type Value } from '../../tensor/index.ts'
import { binarySegmentation, pelt, type SegmentCost } from './changepoint.ts'
import { kalmanFilter, rtsSmoother } from './kalman.ts'
import { extendedKalmanFilter, merweSigmaPoints, unscentedKalmanFilter } from './nonlinear.ts'

const dt = 0.5
const cv = {
  F: [
    [1, dt],
    [0, 1],
  ],
  H: [[1, 0]],
  Q: [
    [(0.1 * dt ** 3) / 3, (0.1 * dt ** 2) / 2],
    [(0.1 * dt ** 2) / 2, 0.1 * dt],
  ],
  R: [[0.5]],
  x0: [0, 1],
  P0: [
    [1, 0],
    [0, 2],
  ],
}

const KALMAN: Readonly<Record<string, GoldenFn>> = {
  means: ([z]) => kalmanFilter(cv, z as Tensor).means,
  covariances: ([z]) => kalmanFilter(cv, z as Tensor).covariances,
  predicted: ([z]) => kalmanFilter(cv, z as Tensor).predictedMeans,
  smoothed: ([z]) => rtsSmoother(cv, kalmanFilter(cv, z as Tensor)).means,
  smoothed_cov: ([z]) => rtsSmoother(cv, kalmanFilter(cv, z as Tensor)).covariances,
  loglik: ([z]) => kalmanFilter(cv, z as Tensor).logLikelihood,
}

const at = (x: Value, i: number): Value => sum(sliceAxis(x, 0, i, i + 1))
const DT = 0.1
const pendulum = {
  f: (x: Value): Value => stack([add(at(x, 0), mul(at(x, 1), DT)), add(at(x, 1), mul(sin(at(x, 0)), -DT * 9.81))]),
  h: (x: Value): Value => stack([sin(at(x, 0))]),
  Q: [
    [1e-4, 0],
    [0, 1e-3],
  ],
  R: [[0.01]],
  x0: [0.5, 0],
  P0: [
    [0.1, 0],
    [0, 0.1],
  ],
}
const points = merweSigmaPoints(2, 0.3, 2, 1)

const NONLINEAR: Readonly<Record<string, GoldenFn>> = {
  ukf_means: ([z]) => unscentedKalmanFilter(pendulum, z as Tensor, { points }).means,
  ukf_covs: ([z]) => unscentedKalmanFilter(pendulum, z as Tensor, { points }).covariances,
  ekf_means: ([z]) => extendedKalmanFilter(pendulum, z as Tensor).means,
  ekf_covs: ([z]) => extendedKalmanFilter(pendulum, z as Tensor).covariances,
}

const CHANGEPOINT: Readonly<Record<string, GoldenFn>> = {
  pelt: ([y], p) => tensor(pelt(toFlat(y as Tensor), p['pen'] as number, { cost: p['cost'] as SegmentCost })),
  binseg_n: ([y], p) =>
    tensor(
      binarySegmentation(toFlat(y as Tensor), { changepoints: p['n'] as number }, { cost: p['cost'] as SegmentCost }),
    ),
  binseg_pen: ([y], p) =>
    tensor(
      binarySegmentation(toFlat(y as Tensor), { penalty: p['pen'] as number }, { cost: p['cost'] as SegmentCost }),
    ),
}

describe('dynamics/filtering against filterpy and ruptures', () => {
  describe('kalman', () => runFixture('dynamics/filtering/kalman', KALMAN, it))
  describe('nonlinear', () => runFixture('dynamics/filtering/nonlinear', NONLINEAR, it))
  describe('changepoint', () => runFixture('dynamics/filtering/changepoint', CHANGEPOINT, it))
})
