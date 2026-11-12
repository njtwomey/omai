// What the tests of the sites on a Gaussian vector (`multiclass-row`, `likelihood-pair-row`) share: the brute-force answer for a factor of
// linear coordinates `z = A w + b` of a Gaussian vector, by a fine grid over the standardised coordinates of the vector's own cavity (the
// tilted mean and covariance of `w`, the site as the quotient of their natural parameters by the cavity's, and the log mass), with none of
// the operator's own factorisation in it. Node only; not published.
import { spd } from '../lib/message-passing/numerics/_dense.ts'

/** The brute-force site of a factor of the coordinates of a vector, and its log mass under the normalised cavity. */
export interface BruteVectorSite {
  /** `[h, J]` of the site, as the operator returns it (the tilted natural parameters minus the cavity's, over the power). */
  readonly message: readonly number[]
  readonly logMass: number
}

/** Lower Cholesky factor of a symmetric positive-definite matrix, row-major. */
export function choleskyLower(a: readonly number[], n: number): number[] {
  const l = new Array<number>(n * n).fill(0)
  for (let i = 0; i < n; i++)
    for (let j = 0; j <= i; j++) {
      let s = a[i * n + j] as number
      for (let k = 0; k < j; k++) s -= (l[i * n + k] as number) * (l[j * n + k] as number)
      l[i * n + j] = i === j ? Math.sqrt(s) : s / (l[j * n + j] as number)
    }
  return l
}

/**
 * The brute-force site.
 *
 * @param cavity - The vector's cavity: its mean and covariance (row-major) and its natural parameters `[h, J]`.
 * @param matrix - `A`, row-major `K x D`.
 * @param offset - `b`, `K` entries.
 * @param logf - The log factor at the coordinates (before the power).
 * @param power - The power of the factor.
 * @param grid - Points per axis of the grid over the standardised coordinates (six standard deviations either side).
 * @returns The site and the log mass.
 */
export function bruteVectorSite(
  cavity: {
    readonly mean: readonly number[]
    readonly covariance: readonly number[]
    readonly eta: readonly number[]
  },
  matrix: readonly number[],
  offset: readonly number[],
  logf: (z: readonly number[]) => number,
  power: number,
  grid: number,
): BruteVectorSite {
  const d = cavity.mean.length
  const k = offset.length
  const l = choleskyLower(cavity.covariance, d)
  const span = 6
  const h = (2 * span) / grid
  const u = new Array<number>(d).fill(0)
  let mass = 0
  const m1 = new Array<number>(d).fill(0)
  const m2 = new Array<number>(d * d).fill(0)
  const w = new Array<number>(d).fill(0)
  const z = new Array<number>(k).fill(0)
  const walk = (i: number, logw: number): void => {
    if (i === d) {
      for (let a = 0; a < d; a++) {
        let s = cavity.mean[a] as number
        for (let b = 0; b <= a; b++) s += (l[a * d + b] as number) * (u[b] as number)
        w[a] = s
      }
      for (let a = 0; a < k; a++) {
        let s = offset[a] as number
        for (let b = 0; b < d; b++) s += (matrix[a * d + b] as number) * (w[b] as number)
        z[a] = s
      }
      const p = Math.exp(logw + power * logf(z))
      mass += p
      for (let a = 0; a < d; a++) {
        m1[a] = (m1[a] as number) + p * (w[a] as number)
        for (let b = 0; b < d; b++) m2[a * d + b] = (m2[a * d + b] as number) + p * (w[a] as number) * (w[b] as number)
      }
      return
    }
    for (let j = 0; j < grid; j++) {
      u[i] = -span + (j + 0.5) * h
      walk(i + 1, logw - 0.5 * (u[i] as number) ** 2 - 0.5 * Math.log(2 * Math.PI) + Math.log(h))
    }
  }
  walk(0, 0)
  const mean = m1.map((x) => x / mass)
  const covariance = m2.map((x, i) => x / mass - (mean[Math.floor(i / d)] as number) * (mean[i % d] as number))
  const precision = spd(covariance, d)?.inverse() as Float64Array
  const hTilt: number[] = []
  for (let i = 0; i < d; i++) {
    let s = 0
    for (let j = 0; j < d; j++) s += (precision[i * d + j] as number) * (mean[j] as number)
    hTilt.push(s)
  }
  return {
    message: [...hTilt, ...precision].map((x, i) => (x - (cavity.eta[i] as number)) / power),
    logMass: Math.log(mass),
  }
}
