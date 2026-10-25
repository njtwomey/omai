/**
 * Fitting copulas to data: pseudo-observations (column ranks scaled into the unit cube, so marginals need not be
 * assumed), the matrix of pairwise Kendall's tau, and `fitCopula`, which estimates an elliptical or Archimedean
 * copula's parameter by inverting Kendall's tau, as statsmodels does.
 */

import { WhiteboxError } from '../../base/index.ts'
import { eigh } from '../../numerics/linalg/index.ts'
import { brentq } from '../../numerics/roots/index.ts'
import {
  diagonal,
  div,
  expandDims,
  fromStorage,
  matmul,
  mul,
  primalOf,
  promoteFloat,
  sqrt,
  swapaxes,
  type Tensor,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import { inSupport } from '../distribution.ts'
import { claytonCopula, frankCopula, gumbelCopula } from './archimedean-copulas.ts'
import { type Copula, gaussianCopula, studentTCopula } from './copulas.ts'

/** Kendall's tau-b of two samples (O(n^2), with ties). */
function kendallTauB(x: readonly number[], y: readonly number[]): number {
  let concordant = 0
  let discordant = 0
  let tiesX = 0
  let tiesY = 0
  for (let i = 0; i < x.length; i++)
    for (let j = i + 1; j < x.length; j++) {
      const dx = Math.sign((x[i] ?? 0) - (x[j] ?? 0))
      const dy = Math.sign((y[i] ?? 0) - (y[j] ?? 0))
      if (dx === 0 && dy === 0) continue
      if (dx === 0) tiesX++
      else if (dy === 0) tiesY++
      else if (dx === dy) concordant++
      else discordant++
    }
  return (concordant - discordant) / Math.sqrt((concordant + discordant + tiesX) * (concordant + discordant + tiesY))
}

/** The columns of an `[n, d]` value as arrays. */
function columns(data: TensorLike): number[][] {
  const [x] = promoteFloat(data) as [Value]
  if (x.shape.length !== 2) throw new WhiteboxError('shape', 'copulas: data must be [n, d]')
  const [n, d] = x.shape as [number, number]
  const flat = toFlat(primalOf(x))
  return Array.from({ length: d }, (_, j) => Array.from({ length: n }, (_, i) => flat[i * d + j] ?? 0))
}

/**
 * Pseudo-observations: each column's ranks divided by n + 1 (ties averaged), mapping data to the unit cube for
 * copula fitting without assuming marginals.
 *
 * @param data - Observations `[n, d]`.
 * @returns Values `[n, d]` in (0, 1).
 * @example
 * ```ts
 * import { pseudoObservations } from 'whitebox-engine/dists'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(pseudoObservations([[3], [1], [2]])) // => [[0.75], [0.25], [0.5]]
 * ```
 */
export function pseudoObservations(data: TensorLike): Tensor {
  const cols = columns(data)
  const n = cols[0]?.length ?? 0
  const d = cols.length
  const out = new Float64Array(n * d)
  for (const [j, col] of cols.entries()) {
    const order = col.map((_, i) => i).sort((a, b) => (col[a] ?? 0) - (col[b] ?? 0))
    let start = 0
    while (start < n) {
      let end = start
      while (end + 1 < n && col[order[end + 1] ?? 0] === col[order[start] ?? 0]) end++
      const rank = (start + end) / 2 + 1
      for (let k = start; k <= end; k++) out[(order[k] ?? 0) * d + j] = rank / (n + 1)
      start = end + 1
    }
  }
  return fromStorage(out, [n, d], 'float64')
}

/**
 * The matrix of pairwise empirical Kendall's tau-b of the columns of a sample.
 *
 * @param data - Observations `[n, d]`.
 * @returns Kendall's taus `[d, d]` (ones on the diagonal).
 * @example
 * ```ts
 * import { kendallTauMatrix } from 'whitebox-engine/dists'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(kendallTauMatrix([[1, 3], [2, 2], [3, 1]])) // => [[1, -1], [-1, 1]]
 * ```
 */
export function kendallTauMatrix(data: TensorLike): Tensor {
  const cols = columns(data)
  const d = cols.length
  const out = new Float64Array(d * d)
  for (let i = 0; i < d; i++)
    for (let j = 0; j < d; j++)
      out[i * d + j] = i === j ? 1 : j < i ? (out[j * d + i] ?? 0) : kendallTauB(cols[i] ?? [], cols[j] ?? [])
  return fromStorage(out, [d, d], 'float64')
}

/** The copula families {@link fitCopula} knows. */
export type CopulaFamily = 'gaussian' | 'studentT' | 'clayton' | 'gumbel' | 'frank'

/** The nearest correlation matrix by clipping eigenvalues (at 1e-8) and rescaling to a unit diagonal. */
function repairCorrelation(r: Tensor): Value {
  const { eigenvalues, eigenvectors } = eigh(r)
  const clipped = toFlat(eigenvalues).map((v) => Math.max(v, 1e-8))
  const lam = promoteFloat(clipped)[0] as Value
  const rebuilt = matmul(mul(eigenvectors, lam), swapaxes(eigenvectors, 0, 1))
  const scale = sqrt(diagonal(rebuilt))
  return div(rebuilt, mul(expandDims(scale, -1), expandDims(scale, 0)))
}

/**
 * Fit a copula by inverting Kendall's tau (the method of moments statsmodels' `fit_corr_param` uses): elliptical
 * copulas take R_ij = sin(pi tau_ij / 2) (repaired to the nearest correlation matrix if needed), Archimedean ones the
 * parameter whose tau equals the average pairwise tau (theta = 2 tau / (1 - tau) for Clayton, 1 / (1 - tau) for
 * Gumbel, and a root of the Debye-function equation for Frank).
 *
 * @param family - `gaussian`, `studentT`, `clayton`, `gumbel` or `frank`.
 * @param data - Observations `[n, d]` (raw data or pseudo-observations: Kendall's tau is rank based).
 * @param options - `df` for the Student-t copula (default 4).
 * @returns The fitted copula.
 * @throws WhiteboxError with code `value` when the sample tau is outside the family's range.
 * @example
 * ```ts
 * import { fitCopula } from 'whitebox-engine/dists'
 * import { item } from 'whitebox-engine/tensor'
 * // Two concordant pairs and one discordant pair: tau = 1 / 3, so theta = 2 tau / (1 - tau) = 1.
 * Math.round(item(fitCopula('clayton', [[1, 1], [2, 3], [3, 2]]).params.theta) * 1e12) / 1e12 // => 1
 * ```
 */
export function fitCopula(family: CopulaFamily, data: TensorLike, options: { readonly df?: number } = {}): Copula {
  const tau = kendallTauMatrix(data)
  const d = tau.shape[0] ?? 0
  if (family === 'gaussian' || family === 'studentT') {
    const r = toFlat(tau).map((v, i) => (i % (d + 1) === 0 ? 1 : Math.sin((Math.PI * v) / 2)))
    const rt = fromStorage(Float64Array.from(r), [d, d], 'float64')
    const pd = toFlat(primalOf(inSupport({ kind: 'correlation' }, rt)))[0] === 1
    const corr = pd ? rt : repairCorrelation(rt)
    return family === 'gaussian' ? gaussianCopula(corr) : studentTCopula(corr, options.df ?? 4)
  }
  let total = 0
  for (let i = 0; i < d; i++) for (let j = i + 1; j < d; j++) total += toFlat(tau)[i * d + j] ?? 0
  const t = total / ((d * (d - 1)) / 2)
  switch (family) {
    case 'clayton':
      if (!(t > 0 && t < 1)) throw new WhiteboxError('value', `fitCopula: Clayton needs tau in (0, 1), got ${t}`)
      return claytonCopula((2 * t) / (1 - t), d)
    case 'gumbel':
      if (!(t >= 0 && t < 1)) throw new WhiteboxError('value', `fitCopula: Gumbel needs tau in [0, 1), got ${t}`)
      return gumbelCopula(1 / (1 - t), d)
    case 'frank': {
      if (!(Math.abs(t) < 1) || t === 0)
        throw new WhiteboxError('value', `fitCopula: Frank needs tau in (-1, 1) \\ {0}`)
      const f = (th: number): number => toFlat(frankCopula(th).kendallTau())[0] ?? 0
      const [lo, hi] = t > 0 ? [1e-6, 200] : [-200, -1e-6]
      return frankCopula(brentq((th) => f(th) - t, lo, hi).root, d)
    }
  }
}
