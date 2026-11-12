// The joint EP site of a smooth factor of a few linear coordinates of one Gaussian vector: the factor is a function `f(z)` of
// `z = A w + b` (`K` coordinates, a constant matrix and offset of the vector `w`), and the site it sends back to `w` carries the
// correlations between the coordinates, which one Gaussian message per coordinate cannot. The cavity of `z` is `N(A m + b, A V A')` with
// full covariance; its rank (at most the smaller of `K` and the dimension of `w`, found by a pivoted Cholesky factor) is the number of
// coordinates the integral runs over, on the product grid of `_product.ts`, in the standardised coordinates `xi` of `z = mu + B xi`. The
// tilted Gaussian over the standard cavity gives the site in `xi` (`S^-1 a`, `S^-1 - I`), and `xi = G (w - m)` with `G = (B'B)^-1 B' A`
// carries it to `w`: precision `G' (S^-1 - I) G` and linear term `J m + G' S^-1 a`. The sites of the multiclass row (`links/`) and of the
// pair row (`likelihoods/`) are this with their own `f`.

import { etaOf } from '../core/args.ts'
import { ImproperMessageError, ParameterError } from '../core/errors.ts'
import { gaussianMvMoments } from '../core/families-vector.ts'
import type { Arg, Eta } from '../core/types.ts'
import { spd } from './_dense.ts'
import { MOST_PRODUCT, productTilt } from './_product.ts'

/** The factor of the coordinates of a vector, and how they are made from it. */
export interface VectorSite {
  /** The number of coordinates `K`. */
  readonly coordinates: number
  /** `A`, row-major `K x D`, and `b` (`K` entries). */
  readonly matrix: readonly number[]
  readonly offset: readonly number[]
  /** The log of the factor at the coordinates, already to its power. */
  readonly logf: (z: readonly number[]) => number
  /** The name of the operator, for the errors. */
  readonly name: string
}

/** What the integral found: the log mass of the factor under the normalised cavity of the vector, and the message to the vector for a factor to the power one. */
export interface VectorSiteTilt {
  readonly logMass: number
  /** `[h, J]` in natural parameters (the site of the factor as it was integrated, before any division by the power). */
  readonly site: Eta
}

/**
 * A factor `S = B B'` of a symmetric positive semi-definite matrix by pivoted Cholesky: the columns of `B` (`n x rank`, row-major) one
 * at a time, always on the largest remaining diagonal, until what is left is below `tol`. The rank is what the coordinates' cavity has.
 */
function pivotedFactor(s: readonly number[], n: number, tol: number): { readonly rank: number; readonly b: number[] } {
  const rest = Array.from({ length: n }, (_, i) => s[i * n + i] as number)
  const columns: number[][] = []
  for (let step = 0; step < n; step++) {
    let at = 0
    for (let i = 1; i < n; i++) if ((rest[i] as number) > (rest[at] as number)) at = i
    if (!((rest[at] as number) > tol)) break
    const column = Array.from({ length: n }, (_, i) => s[i * n + at] as number)
    for (const done of columns)
      for (let i = 0; i < n; i++) column[i] = (column[i] as number) - (done[i] as number) * (done[at] as number)
    const root = Math.sqrt(rest[at] as number)
    for (let i = 0; i < n; i++) {
      column[i] = (column[i] as number) / root
      rest[i] = (rest[i] as number) - (column[i] as number) ** 2
    }
    rest[at] = 0
    columns.push(column)
  }
  const rank = columns.length
  const b = new Array<number>(n * rank)
  for (const [j, column] of columns.entries()) for (let i = 0; i < n; i++) b[i * rank + j] = column[i] as number
  return { rank, b }
}

/**
 * The tilted message of a smooth factor of linear coordinates to the Gaussian vector they are made of.
 *
 * @param w - The cavity of the vector (a proper multivariate Gaussian message).
 * @param site - The factor and the coordinates.
 * @returns The log mass of the factor under the normalised cavity and the site message `[h, J]` it sends to the vector.
 * @throws ParameterError when the cavity of the coordinates has a rank past what the product grid reaches.
 * @throws ImproperMessageError when the vector's cavity or the tilted covariance is not positive definite.
 */
export function vectorSiteTilt(w: Arg, site: VectorSite): VectorSiteTilt {
  const { coordinates: k, matrix: a, offset: b, logf, name } = site
  const d = w.family.dim
  const { mean: m, covariance: v } = gaussianMvMoments(etaOf(w), d)
  // The coordinates' cavity: mean A m + b, covariance A V A'.
  const mu = Array.from({ length: k }, (_, i) => {
    let s = b[i] as number
    for (let e = 0; e < d; e++) s += (a[i * d + e] as number) * (m[e] as number)
    return s
  })
  const av = new Array<number>(k * d).fill(0)
  for (let i = 0; i < k; i++)
    for (let e = 0; e < d; e++) {
      let s = 0
      for (let f = 0; f < d; f++) s += (a[i * d + f] as number) * (v[f * d + e] as number)
      av[i * d + e] = s
    }
  const cov = new Array<number>(k * k).fill(0)
  for (let i = 0; i < k; i++)
    for (let l = 0; l <= i; l++) {
      let s = 0
      for (let e = 0; e < d; e++) s += (av[i * d + e] as number) * (a[l * d + e] as number)
      cov[i * k + l] = s
      cov[l * k + i] = s
    }
  let scale = 0
  for (let i = 0; i < k; i++) scale = Math.max(scale, cov[i * k + i] as number)
  const { rank: r, b: factor } = pivotedFactor(cov, k, 1e-10 * scale)
  const zeros = new Array<number>(d + d * d).fill(0)
  if (r === 0) return { logMass: logf(mu), site: zeros }
  if (r > MOST_PRODUCT)
    throw new ParameterError(
      `${name}: the cavity of the coordinates has rank ${r}, past the ${MOST_PRODUCT} the product grid reaches`,
    )
  // The coordinates are mu + B xi with xi standard normal under the cavity: the grid is over xi.
  const z = mu.slice()
  const t = productTilt(
    r,
    {
      mean: new Array<number>(r).fill(0),
      covariance: Array.from({ length: r * r }, (_, i) => (i % (r + 1) === 0 ? 1 : 0)),
    },
    (x) => {
      let s = -0.5 * r * Math.log(2 * Math.PI)
      for (let i = 0; i < r; i++) s -= 0.5 * (x[i] as number) ** 2
      return s
    },
    (x) => {
      for (let i = 0; i < k; i++) {
        let s = mu[i] as number
        for (let j = 0; j < r; j++) s += (factor[i * r + j] as number) * (x[j] as number)
        z[i] = s
      }
      return logf(z)
    },
  )
  // The site in the standardised coordinates: the tilted Gaussian over the standard cavity, (S^-1 a, S^-1 - I).
  const tiltedPrecision = spd(t.covariance, r)
  if (tiltedPrecision === undefined)
    throw new ImproperMessageError(`${name}: the tilted covariance of the coordinates is not positive definite`)
  const precision = tiltedPrecision.inverse()
  const linear = Array.from(tiltedPrecision.solve(t.mean))
  const lambda = Array.from(precision)
  for (let i = 0; i < r; i++) lambda[i * r + i] = (lambda[i * r + i] as number) - 1
  // xi = G (w - m) with G = (B'B)^-1 B' A: the site as a function of w is exp(linear' G (w - m) - (w - m)' G' lambda G (w - m) / 2).
  const btb = new Array<number>(r * r).fill(0)
  for (let i = 0; i < r; i++)
    for (let j = 0; j < r; j++) {
      let s = 0
      for (let c = 0; c < k; c++) s += (factor[c * r + i] as number) * (factor[c * r + j] as number)
      btb[i * r + j] = s
    }
  const gram = spd(btb, r)
  if (gram === undefined) throw new ImproperMessageError(`${name}: the coordinates have no full-rank factor`)
  const g = new Array<number>(r * d)
  for (let e = 0; e < d; e++) {
    const column = new Array<number>(r)
    for (let i = 0; i < r; i++) {
      let s = 0
      for (let c = 0; c < k; c++) s += (factor[c * r + i] as number) * (a[c * d + e] as number)
      column[i] = s
    }
    const solved = gram.solve(column)
    for (let i = 0; i < r; i++) g[i * d + e] = solved[i] as number
  }
  // J = G' lambda G, h = J m + G' linear.
  const lg = new Array<number>(r * d).fill(0)
  for (let i = 0; i < r; i++)
    for (let e = 0; e < d; e++) {
      let s = 0
      for (let j = 0; j < r; j++) s += (lambda[i * r + j] as number) * (g[j * d + e] as number)
      lg[i * d + e] = s
    }
  const j = new Array<number>(d * d).fill(0)
  for (let e = 0; e < d; e++)
    for (let f = 0; f <= e; f++) {
      let s = 0
      for (let i = 0; i < r; i++) s += (g[i * d + e] as number) * (lg[i * d + f] as number)
      j[e * d + f] = s
      j[f * d + e] = s
    }
  const h = new Array<number>(d).fill(0)
  for (let e = 0; e < d; e++) {
    let s = 0
    for (let f = 0; f < d; f++) s += (j[e * d + f] as number) * (m[f] as number)
    for (let i = 0; i < r; i++) s += (g[i * d + e] as number) * (linear[i] as number)
    h[e] = s
  }
  return { logMass: t.logMass, site: [...h, ...j] }
}
