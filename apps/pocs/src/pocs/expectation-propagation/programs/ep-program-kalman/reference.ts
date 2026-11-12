// The reference EP is held against: the exact Kalman filter and Rauch-Tung-Striebel smoother of the same linear Gaussian model, from the
// engine's dynamics domain. A chain of linear Gaussian factors is a tree, so EP is exact there and must agree with it to rounding.
// Not part of the program shown.
import { filtering } from 'omai-engine/dynamics'
import { add, div, matmul, mul, reshape, sqrt, sub, take, tensor, toArray, toFlat, transpose } from 'omai-engine/tensor'
import { DT, type Setup, START, stepCovariance } from './program.ts'

/** The smoothed position and velocity at each step: mean, sd and the 5% and 95% quantiles. */
export interface Smoothed {
  readonly position: { mean: number[]; sd: number[]; lower: number[]; upper: number[] }
  readonly velocity: { mean: number[]; sd: number[]; lower: number[]; upper: number[] }
}

/** The exact posterior of the path given the readings. */
export function kalmanReference(y: readonly number[], { readingSd, push }: Setup): Smoothed {
  const F = [
    [1, DT],
    [0, 1],
  ]
  const Q = stepCovariance(push)
  // The engine's filter predicts before every update; one step earlier, so that the first prediction is the start N(mean, P0).
  const P0 = [
    [START.sd[0] ** 2, 0],
    [0, START.sd[1] ** 2],
  ]
  const model = {
    F,
    H: [[1, 0]],
    Q,
    R: [[readingSd ** 2]],
    x0: [0, 0],
    P0: [
      [0, 0],
      [0, 0],
    ],
  }
  const before = backOneStep(F, P0, Q, START.mean)
  const smoothed = filtering.rtsSmoother(
    { ...model, x0: before.mean, P0: before.covariance },
    filtering.kalmanFilter({ ...model, x0: before.mean, P0: before.covariance }, reshape(tensor(y), [-1, 1])),
  )
  const pick = (k: number) => {
    const mean = take(smoothed.means, k, 1)
    const sd = sqrt(take(take(smoothed.covariances, k, 1), k, 1))
    return {
      mean: toFlat(mean),
      sd: toFlat(sd),
      lower: toFlat(sub(mean, mul(sd, 1.645))),
      upper: toFlat(add(mean, mul(sd, 1.645))),
    }
  }
  return { position: pick(0), velocity: pick(1) }
}

/** The mean and covariance one step before the start: `F^-1 mean` and `F^-1 (P0 - Q) F^-T`, so that predicting from them gives the start. */
function backOneStep(F: number[][], P0: number[][], Q: number[][], mean: readonly number[]) {
  const [a, b, c, e] = [F[0]?.[0] as number, F[0]?.[1] as number, F[1]?.[0] as number, F[1]?.[1] as number]
  const inv = div(
    tensor([
      [e, -b],
      [-c, a],
    ]),
    a * e - b * c,
  )
  const left = matmul(inv, sub(tensor(P0), tensor(Q)))
  return { mean: toFlat(matmul(inv, tensor(mean))), covariance: toArray(matmul(left, transpose(inv))) as number[][] }
}
