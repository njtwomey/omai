// The reference EP is held against: the exact Kalman filter and Rauch-Tung-Striebel smoother of the same linear Gaussian model, from the
// engine's dynamics domain. A chain of linear Gaussian factors is a tree, so EP is exact there and must agree with it to rounding.
// Not part of the program shown.
import { filtering } from 'omai-engine/dynamics'
import { toArray } from 'omai-engine/tensor'
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
    filtering.kalmanFilter(
      { ...model, x0: before.mean, P0: before.covariance },
      y.map((v) => [v]),
    ),
  )
  const means = toArray(smoothed.means) as number[][]
  const covariances = toArray(smoothed.covariances) as number[][][]
  const pick = (k: 0 | 1) => {
    const mean = means.map((m) => m[k] as number)
    const sd = covariances.map((c) => Math.sqrt((c[k] as number[])[k] as number))
    return {
      mean,
      sd,
      lower: mean.map((m, t) => m - 1.645 * (sd[t] as number)),
      upper: mean.map((m, t) => m + 1.645 * (sd[t] as number)),
    }
  }
  return { position: pick(0), velocity: pick(1) }
}

/** The mean and covariance one step before the start: `F^-1 mean` and `F^-1 (P0 - Q) F^-T`, so that predicting from them gives the start. */
function backOneStep(F: number[][], P0: number[][], Q: number[][], mean: readonly number[]) {
  const [a, b, c, e] = [F[0]?.[0] as number, F[0]?.[1] as number, F[1]?.[0] as number, F[1]?.[1] as number]
  const det = a * e - b * c
  const inv = [
    [e / det, -b / det],
    [-c / det, a / det],
  ]
  const at = (m: number[][], i: number, j: number): number => (m[i] as number[])[j] as number
  const gap = [0, 1].map((i) => [0, 1].map((j) => at(P0, i, j) - at(Q, i, j)))
  const left = [0, 1].map((i) => [0, 1].map((j) => at(inv, i, 0) * at(gap, 0, j) + at(inv, i, 1) * at(gap, 1, j)))
  const covariance = [0, 1].map((i) =>
    [0, 1].map((j) => at(left, i, 0) * at(inv, j, 0) + at(left, i, 1) * at(inv, j, 1)),
  )
  return {
    mean: [
      at(inv, 0, 0) * (mean[0] as number) + at(inv, 0, 1) * (mean[1] as number),
      at(inv, 1, 0) * (mean[0] as number) + at(inv, 1, 1) * (mean[1] as number),
    ],
    covariance,
  }
}
