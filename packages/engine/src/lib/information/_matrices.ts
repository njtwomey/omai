// Matrices of channels and distortions, and checks of distributions, shared by channels.ts and rate-distortion.ts.
import { pairwiseSum, WhiteboxError } from '../base/index.ts'
import { type TensorLike, toFlat } from '../tensor/index.ts'
import { float } from './_shared.ts'

/** A discrete memoryless channel given by its transition matrix W[x][y] = P(y | x) (rows sum to one). */
export interface Channel {
  readonly w: Float64Array
  readonly nx: number
  readonly ny: number
}

export function matrixOf(m: TensorLike, name: string): Channel {
  const v = float(m)
  if (v.shape.length !== 2) throw new WhiteboxError('shape', `${name}: expected a matrix [rows, columns]`)
  const [nx = 0, ny = 0] = v.shape
  return { w: Float64Array.from(toFlat(v)), nx, ny }
}

export function channelOf(transition: TensorLike, name: string): Channel {
  const c = matrixOf(transition, name)
  for (let x = 0; x < c.nx; x++) {
    const row = c.w.subarray(x * c.ny, (x + 1) * c.ny)
    if (row.some((p) => !(p >= 0)) || Math.abs(pairwiseSum(row) - 1) > 1e-9)
      throw new WhiteboxError('value', `${name}: row ${x} of the transition matrix is not a distribution`)
  }
  return c
}

export function distribution(p: TensorLike, n: number, name: string): Float64Array {
  const xs = Float64Array.from(toFlat(float(p)))
  if (xs.length !== n || xs.some((v) => !(v >= 0)) || Math.abs(pairwiseSum(xs) - 1) > 1e-9)
    throw new WhiteboxError('value', `${name}: expected a distribution over ${n} values`)
  return xs
}

/** The binary entropy in nats of a number. */
export const h2 = (p: number): number => (p <= 0 || p >= 1 ? 0 : -p * Math.log(p) - (1 - p) * Math.log(1 - p))

export function checkProbability(p: number, name: string): void {
  if (!(p >= 0 && p <= 1)) throw new WhiteboxError('value', `${name}: the probability must lie in [0, 1]`)
}
