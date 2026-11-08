/**
 * Knockoff variable selection with false discovery rate control (Barber and Candes 2015; Candes, Fan, Janson and Lv
 * 2018). Gaussian model-X knockoffs are fake copies of the features that are exchangeable with the originals in
 * distribution yet independent of the response given them, so a feature that beats its own copy is evidence for it. This
 * file builds them for a Gaussian design (equicorrelated and coordinate-ascent scales, the latter a cheap alternative to
 * the semidefinite program), computes the knockoff statistics W_j (lasso coefficient difference, lasso signed maximum, or
 * the marginal correlation difference) and applies the knockoff and knockoff+ thresholds, which control the FDR at q.
 */

import { type Json, OmaiError } from '../base/index.ts'
import { cholesky, dgemm, eigh, solve } from '../numerics/linalg/index.ts'
import { lassoCoordinateDescent } from '../optim/sparse.ts'
import { type Key, key as makeKey, normal } from '../random/index.ts'
import { empiricalCovariance } from '../stats/index.ts'
import { fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm } from '../trace/index.ts'
import { matrixView, targetVector } from './validation.ts'

const flat = (v: Parameters<typeof primalOf>[0]): Float64Array => Float64Array.from(toFlat(primalOf(v)))
const square = (data: ArrayLike<number>, p: number): Tensor => fromStorage(Float64Array.from(data), [p, p], 'float64')

/** How the diagonal of D (the knockoff scale s) is chosen. */
export type KnockoffScale = 'equicorrelated' | 'coordinate' | ArrayLike<number>

/** Smallest eigenvalue of a symmetric matrix. */
function minEigenvalue(a: Float64Array, p: number): number {
  const ev = flat(eigh(square(a, p)).eigenvalues)
  return ev[0] ?? 0
}

/**
 * The equicorrelated knockoff scale (Barber and Candes 2015): s_j = min(2 lambda_min(R), 1) sigma_j^2 for the correlation matrix R,
 * the same for every feature, the largest uniform scale for which the knockoff covariance 2D - D Sigma^-1 D stays
 * positive semidefinite. It is cheap and conservative: when features are strongly correlated lambda_min is small and the
 * knockoffs are nearly the originals.
 *
 * @param sigma - The feature covariance `[p, p]`.
 * @param shrink - A factor below one that keeps the knockoff covariance positive definite (default 0.999).
 * @returns s, one entry per feature.
 * @concept knockoffs
 * @example
 * ```ts
 * import { equicorrelatedScale } from 'omai-engine/learn'
 * equicorrelatedScale([[1, 0.5], [0.5, 1]], 1) // => [1, 1]
 * ```
 */
export function equicorrelatedScale(sigma: TensorLike, shrink = 0.999): number[] {
  const m = matrixView(sigma, 'equicorrelatedScale')
  const p = m.n
  if (m.d !== p) throw new OmaiError('shape', 'equicorrelatedScale: sigma must be square')
  const sd = Array.from({ length: p }, (_, j) => Math.sqrt(m.data[j * p + j] ?? 0))
  const r = Float64Array.from(m.data, (v, k) => v / ((sd[Math.floor(k / p)] ?? 1) * (sd[k % p] ?? 1)))
  const s = Math.min(2 * minEigenvalue(r, p), 1) * shrink
  return sd.map((v) => s * v * v)
}

/** State of {@link knockoffScaleAlgorithm}. */
export interface KnockoffScaleState {
  readonly s: readonly number[]
  readonly sweep: number
  readonly change: number
}

/**
 * The coordinate-ascent knockoff scale as an algorithm with one sweep per step: an SDP-free approximation to the
 * semidefinite program that maximises sum_j s_j subject to 0 <= s_j <= sigma_j^2 and 2 Sigma - diag(s) positive
 * semidefinite. Each coordinate is raised to the largest value the constraint allows with the others held fixed,
 * 1 / (e_j' M^-1 e_j) for M the constraint matrix with s_j removed. The result is feasible at every sweep, starts from the
 * equicorrelated scale (so is never smaller in total) and keeps the knockoffs from collapsing onto the originals when a few
 * features are strongly correlated and the others are not.
 *
 * @param sigma - The feature covariance `[p, p]`.
 * @param options - Sweeps (default 20), tolerance on the largest change (default 1e-6) and the safety factor.
 * @returns The algorithm over {@link KnockoffScaleState}.
 * @concept knockoffs
 * @example
 * ```ts
 * import { knockoffScaleAlgorithm } from 'omai-engine/learn'
 * knockoffScaleAlgorithm([[1, 0.9], [0.9, 1]]).name // => 'knockoff-scale'
 * ```
 */
export function knockoffScaleAlgorithm(
  sigma: TensorLike,
  options: { readonly maxSweeps?: number; readonly tol?: number; readonly shrink?: number } = {},
): Algorithm<KnockoffScaleState> {
  const m = matrixView(sigma, 'knockoffScale')
  const p = m.n
  const shrink = options.shrink ?? 0.999
  const tol = options.tol ?? 1e-6
  const cap = Array.from({ length: p }, (_, j) => m.data[j * p + j] ?? 0)
  return {
    name: 'knockoff-scale',
    config: { p, maxSweeps: options.maxSweeps ?? 20 } as Json,
    schema: channels({
      metrics: {
        total: { description: 'sum of the scales (to be maximised)' },
        change: { description: 'largest change in a scale during the sweep' },
      },
      tensors: { s: { role: 'param', description: 'the knockoff scale' } },
    }),
    init: () => ({ s: equicorrelatedScale(sigma, shrink), sweep: 0, change: Number.POSITIVE_INFINITY }),
    step: (st) => {
      const s = [...st.s]
      let change = 0
      for (let j = 0; j < p; j++) {
        // M = 2 Sigma - diag(s) with s_j set to 0; the largest feasible s_j is 1 / (M^-1)_jj.
        const mm = Float64Array.from(m.data, (v) => 2 * v)
        for (let i = 0; i < p; i++) if (i !== j) mm[i * p + i] = (mm[i * p + i] ?? 0) - (s[i] ?? 0)
        const e = new Float64Array(p)
        e[j] = 1
        const col = flat(solve(square(mm, p), fromStorage(e, [p], 'float64')))
        const best = Math.min(cap[j] ?? 0, 1 / (col[j] ?? Number.POSITIVE_INFINITY)) * shrink
        const next = Math.max(s[j] ?? 0, best)
        change = Math.max(change, Math.abs(next - (s[j] ?? 0)))
        s[j] = next
      }
      return { s, sweep: st.sweep + 1, change }
    },
    done: (st) => (st.change < tol ? 'converged' : undefined),
    observe: (st) => ({
      metrics: { total: st.s.reduce((a, b) => a + b, 0), change: Number.isFinite(st.change) ? st.change : 0 },
      tensors: { s: { role: 'param', value: st.s } },
    }),
    serialise: (st) => ({ s: [...st.s], sweep: st.sweep, change: Number.isFinite(st.change) ? st.change : -1 }),
    deserialise: (j) => {
      const o = j as unknown as KnockoffScaleState
      return { ...o, change: o.change < 0 ? Number.POSITIVE_INFINITY : o.change }
    },
  }
}

/**
 * The knockoff scale by a named method, or the given vector.
 *
 * @param sigma - The feature covariance.
 * @param scale - `'equicorrelated'` (default), `'coordinate'` (the ascent of {@link knockoffScaleAlgorithm}) or a vector.
 * @returns s.
 * @concept knockoffs
 * @example
 * ```ts
 * import { knockoffScales } from 'omai-engine/learn'
 * knockoffScales([[1, 0], [0, 1]], 'coordinate').map((v) => Math.round(v * 1000) / 1000) // => [0.999, 0.999]
 * ```
 */
export function knockoffScales(sigma: TensorLike, scale: KnockoffScale = 'equicorrelated'): number[] {
  if (typeof scale !== 'string') return Array.from(scale)
  if (scale === 'equicorrelated') return equicorrelatedScale(sigma)
  const r = runAlgorithm(knockoffScaleAlgorithm(sigma), { maxSteps: 20 })
  return [...r.state.s]
}

/** Options of {@link gaussianKnockoffs}. */
export interface GaussianKnockoffOptions {
  /** The feature covariance (default: the sample covariance, which is only valid when n is much larger than p). */
  readonly sigma?: TensorLike
  /** The feature mean (default the column means). */
  readonly mean?: ArrayLike<number>
  /** The scale s (default equicorrelated). */
  readonly scale?: KnockoffScale
  /** The key for the knockoff draw (default `key(0)`). */
  readonly key?: Key
}

/** Gaussian model-X knockoffs of a design. */
export interface GaussianKnockoffs {
  /** The knockoff copies `[n, p]`. */
  readonly xTilde: Tensor
  /** The scale s. */
  readonly s: readonly number[]
  /** The covariance used, `[p, p]`. */
  readonly sigma: Tensor
  readonly mean: readonly number[]
}

/**
 * Gaussian model-X knockoffs (Candes et al. 2018, equation 3.2): for rows x ~ N(mu, Sigma), the copy x~ given x is
 * N(mu + (x - mu)(I - Sigma^-1 D), 2D - D Sigma^-1 D) with D = diag(s). Then (x, x~) is exchangeable under swapping any
 * subset of coordinates with their copies, which is what the knockoff filter's FDR control rests on; the control is exact when
 * Sigma is the true covariance and approximate when it is estimated.
 *
 * @param x - The design `[n, p]`.
 * @param options - Covariance, mean, scale and key.
 * @returns The knockoffs, the scale and the covariance used.
 * @throws OmaiError with code `value` when the scale makes the knockoff covariance indefinite.
 * @concept knockoffs
 * @example
 * ```ts
 * import { gaussianKnockoffs } from 'omai-engine/learn'
 * gaussianKnockoffs([[0, 1], [1, 0], [2, 3], [3, 2], [4, 6]], { sigma: [[2, 1], [1, 2]] }).xTilde.shape // => [5, 2]
 * ```
 */
export function gaussianKnockoffs(x: TensorLike, options: GaussianKnockoffOptions = {}): GaussianKnockoffs {
  const m = matrixView(x, 'gaussianKnockoffs')
  const { n, d: p } = m
  const mean = Array.from(
    options.mean ??
      Array.from({ length: p }, (_, j) => {
        let a = 0
        for (let i = 0; i < n; i++) a += (m.data[i * p + j] ?? 0) / n
        return a
      }),
  )
  const sigmaData =
    options.sigma === undefined ? flat(empiricalCovariance(x)) : matrixView(options.sigma, 'gaussianKnockoffs').data
  const sigma = square(sigmaData, p)
  const s = knockoffScales(sigma, options.scale ?? 'equicorrelated')
  if (s.length !== p) throw new OmaiError('shape', 'gaussianKnockoffs: the scale needs one entry per feature')
  // A = Sigma^-1 D; V = 2D - D A (symmetric positive semidefinite).
  const dMat = new Float64Array(p * p)
  s.forEach((v, j) => {
    dMat[j * p + j] = v
  })
  const a = flat(solve(sigma, square(dMat, p)))
  const da = dgemm(dMat, a, p, p, p)
  const v = Float64Array.from(dMat, (dv, k) => 2 * dv - (da[k] ?? 0))
  // Symmetrise against rounding, then factor with a small jitter if the scale sits on the boundary.
  for (let i = 0; i < p; i++)
    for (let j = i + 1; j < p; j++) {
      const avg = ((v[i * p + j] ?? 0) + (v[j * p + i] ?? 0)) / 2
      v[i * p + j] = avg
      v[j * p + i] = avg
    }
  let chol: Float64Array | undefined
  for (const jitter of [0, 1e-10, 1e-8, 1e-6]) {
    try {
      const vv = Float64Array.from(v)
      for (let i = 0; i < p; i++) vv[i * p + i] = (vv[i * p + i] ?? 0) + jitter * (1 + (vv[i * p + i] ?? 0))
      chol = flat(cholesky(square(vv, p)))
      break
    } catch {
      // Try a larger jitter.
    }
  }
  if (chol === undefined)
    throw new OmaiError('value', 'gaussianKnockoffs: the scale makes the knockoff covariance indefinite')
  // Conditional mean: mu + (x - mu)(I - A).
  const centred = Float64Array.from(m.data, (val, k) => val - (mean[k % p] ?? 0))
  const shift = dgemm(centred, a, n, p, p)
  const z = Float64Array.from(toFlat(normal(options.key ?? makeKey(0), [n * p])))
  const noise = dgemm(z, chol, n, p, p, { transposeB: true })
  const out = new Float64Array(n * p)
  for (let k = 0; k < n * p; k++) out[k] = (mean[k % p] ?? 0) + (centred[k] ?? 0) - (shift[k] ?? 0) + (noise[k] ?? 0)
  return { xTilde: fromStorage(out, [n, p], 'float64'), s, sigma, mean }
}

/** Which knockoff statistic W_j to compute. */
export type KnockoffStatistic = 'lcd' | 'lsm' | 'correlation'

/** Options of {@link knockoffStatistics}. */
export interface KnockoffStatisticOptions {
  /** `'lcd'` (default): lasso coefficient difference at a cross-validated penalty; `'lsm'`: lasso signed maximum along the path; `'correlation'`: |x_j'y| - |x~_j'y|. */
  readonly statistic?: KnockoffStatistic
  /** For `'lcd'`: the lasso penalty alpha in the scale (1/2n)|y - Xb|^2 + alpha |b|_1 (default chosen by 5-fold cross-validation). */
  readonly alpha?: number
  /** For `'lsm'` and the cross-validation: the number of penalties on the path (default 50). */
  readonly nAlphas?: number
}

/** Standardise the columns of [X, X~] to unit norm after centring, and centre y. */
function augmentedDesign(
  x: Float64Array,
  xt: Float64Array,
  y: Float64Array,
  n: number,
  p: number,
): { readonly design: Float64Array; readonly yc: Float64Array } {
  const design = new Float64Array(n * 2 * p)
  for (let block = 0; block < 2; block++) {
    const src = block === 0 ? x : xt
    for (let j = 0; j < p; j++) {
      let mean = 0
      for (let i = 0; i < n; i++) mean += (src[i * p + j] ?? 0) / n
      let ss = 0
      for (let i = 0; i < n; i++) ss += ((src[i * p + j] ?? 0) - mean) ** 2
      const scale = Math.sqrt(ss / n) || 1
      for (let i = 0; i < n; i++) design[i * 2 * p + block * p + j] = ((src[i * p + j] ?? 0) - mean) / scale
    }
  }
  let ym = 0
  for (const v of y) ym += v / n
  return { design, yc: Float64Array.from(y, (v) => v - ym) }
}

/** The lasso penalties on a geometric path from alpha_max down by a factor of 1000. */
function alphaPath(design: Float64Array, y: Float64Array, n: number, k: number, nAlphas: number): number[] {
  let amax = 0
  for (let j = 0; j < k; j++) {
    let c = 0
    for (let i = 0; i < n; i++) c += (design[i * k + j] ?? 0) * (y[i] ?? 0)
    amax = Math.max(amax, Math.abs(c) / n)
  }
  return Array.from({ length: nAlphas }, (_, i) => amax * 1e-3 ** (i / (nAlphas - 1)))
}

function lassoAt(
  design: Float64Array,
  y: Float64Array,
  n: number,
  k: number,
  alpha: number,
  x0?: Float64Array,
): Float64Array {
  const r = lassoCoordinateDescent(fromStorage(design, [n, k], 'float64'), y, alpha * n, {
    tol: 1e-9,
    maxIterations: 500,
    ...(x0 === undefined ? {} : { x0 }),
  })
  return Float64Array.from(toFlat(r.coef))
}

/** The penalty minimising the 5-fold cross-validation error of the lasso on the augmented design. */
function cvAlpha(design: Float64Array, y: Float64Array, n: number, k: number, nAlphas: number): number {
  const alphas = alphaPath(design, y, n, k, nAlphas)
  const folds = 5
  const err = new Float64Array(nAlphas)
  for (let f = 0; f < folds; f++) {
    const train: number[] = []
    const test: number[] = []
    for (let i = 0; i < n; i++) (i % folds === f ? test : train).push(i)
    const dt = new Float64Array(train.length * k)
    train.forEach((i, r) => {
      dt.set(design.subarray(i * k, (i + 1) * k), r * k)
    })
    const yt = Float64Array.from(train, (i) => y[i] ?? 0)
    let warm: Float64Array | undefined
    alphas.forEach((alpha, a) => {
      warm = lassoAt(dt, yt, train.length, k, alpha, warm)
      for (const i of test) {
        let pred = 0
        for (let j = 0; j < k; j++) pred += (design[i * k + j] ?? 0) * (warm[j] ?? 0)
        err[a] = (err[a] ?? 0) + (pred - (y[i] ?? 0)) ** 2
      }
    })
  }
  let best = 0
  for (let a = 1; a < nAlphas; a++) if ((err[a] ?? 0) < (err[best] ?? 0)) best = a
  return alphas[best] ?? 0
}

/**
 * The knockoff statistics W_j, which are large and positive for features that matter more than their copies and
 * symmetric about zero for null features (the flip-sign property the filter needs). The lasso is fitted on the
 * standardised augmented design [X, X~]. `lcd` is the lasso coefficient difference |b_j| - |b~_j| at one penalty;
 * `lsm` is the lasso signed maximum Z_j - Z~_j, where Z is the largest penalty at which the coefficient is nonzero
 * (computed along a path with warm starts); `correlation` is the marginal difference |x_j'y| - |x~_j'y|.
 *
 * @param x - The design `[n, p]`.
 * @param xTilde - Its knockoffs `[n, p]`.
 * @param y - The response `[n]`.
 * @param options - The statistic and its penalty.
 * @returns W, one entry per feature.
 * @concept knockoffs
 * @example
 * ```ts
 * import { knockoffStatistics } from 'omai-engine/learn'
 * knockoffStatistics([[1, 0], [2, 1], [3, 0], [4, 1]], [[0, 1], [1, 0], [0, 3], [1, 2]], [1, 2, 3, 4], { statistic: 'correlation' }).length // => 2
 * ```
 */
export function knockoffStatistics(
  x: TensorLike,
  xTilde: TensorLike,
  y: TensorLike,
  options: KnockoffStatisticOptions = {},
): number[] {
  const m = matrixView(x, 'knockoffStatistics')
  const mt = matrixView(xTilde, 'knockoffStatistics', { nFeatures: m.d })
  const yv = targetVector(y, 'knockoffStatistics', m.n)
  const { n, d: p } = m
  if (mt.n !== n) throw new OmaiError('shape', 'knockoffStatistics: x and its knockoffs differ in rows')
  const { design, yc } = augmentedDesign(m.data, mt.data, yv, n, p)
  const kind = options.statistic ?? 'lcd'
  const k = 2 * p
  if (kind === 'correlation') {
    return Array.from({ length: p }, (_, j) => {
      let a = 0
      let b = 0
      for (let i = 0; i < n; i++) {
        a += (design[i * k + j] ?? 0) * (yc[i] ?? 0)
        b += (design[i * k + p + j] ?? 0) * (yc[i] ?? 0)
      }
      return Math.abs(a) - Math.abs(b)
    })
  }
  const nAlphas = options.nAlphas ?? 50
  if (kind === 'lcd') {
    const alpha = options.alpha ?? cvAlpha(design, yc, n, k, nAlphas)
    const b = lassoAt(design, yc, n, k, alpha)
    return Array.from({ length: p }, (_, j) => Math.abs(b[j] ?? 0) - Math.abs(b[p + j] ?? 0))
  }
  const alphas = alphaPath(design, yc, n, k, nAlphas)
  const entry = new Float64Array(k)
  let warm: Float64Array | undefined
  for (const alpha of alphas) {
    warm = lassoAt(design, yc, n, k, alpha, warm)
    for (let j = 0; j < k; j++) if ((entry[j] ?? 0) === 0 && (warm[j] ?? 0) !== 0) entry[j] = alpha
  }
  return Array.from({ length: p }, (_, j) => {
    const z = entry[j] ?? 0
    const zt = entry[p + j] ?? 0
    return Math.max(z, zt) * (z > zt ? 1 : z < zt ? -1 : 0)
  })
}

/**
 * The knockoff data-dependent threshold: T = min {t in {|W_j|} : (offset + #{j : W_j <= -t}) / max(1, #{j : W_j >= t}) <= q},
 * or infinity if none. With offset 1 (knockoff+) selecting {j : W_j >= T} controls the FDR at q; with offset 0 (the
 * original knockoff) it controls the modified FDR.
 *
 * @param w - The statistics.
 * @param q - The target FDR.
 * @param plus - Whether to use knockoff+ (default true).
 * @returns The threshold.
 * @concept knockoffs
 * @example
 * ```ts
 * import { knockoffThreshold } from 'omai-engine/learn'
 * knockoffThreshold([5, 4, 3, 2, 1, -0.5, 0.2, 6, 7, 8, 9, 10], 0.2) // => 0.2
 * ```
 */
export function knockoffThreshold(w: ArrayLike<number>, q: number, plus = true): number {
  if (!(q > 0 && q < 1)) throw new OmaiError('value', 'knockoffThreshold: q must be in (0, 1)')
  const offset = plus ? 1 : 0
  const candidates = [...new Set(Array.from(w, Math.abs).filter((v) => v > 0))].sort((a, b) => a - b)
  for (const t of candidates) {
    let neg = 0
    let pos = 0
    for (let j = 0; j < w.length; j++) {
      if ((w[j] ?? 0) <= -t) neg++
      if ((w[j] ?? 0) >= t) pos++
    }
    if ((offset + neg) / Math.max(1, pos) <= q) return t
  }
  return Number.POSITIVE_INFINITY
}

/** Options of {@link knockoffFilter}. */
export interface KnockoffFilterOptions extends GaussianKnockoffOptions, KnockoffStatisticOptions {
  /** The target FDR (default 0.1). */
  readonly q?: number
  /** Use knockoff+ (default true). */
  readonly plus?: boolean
}

/** The outcome of the knockoff filter. */
export interface KnockoffResult {
  /** Selected feature indices. */
  readonly selected: readonly number[]
  readonly statistics: readonly number[]
  readonly threshold: number
  readonly knockoffs: GaussianKnockoffs
}

/**
 * The model-X knockoff filter: build Gaussian knockoffs, compute the statistics W and select the features with
 * W_j >= T, the knockoff(+) threshold, so that the expected fraction of false discoveries is at most q (exactly for a
 * known Gaussian design and any response model).
 *
 * @param x - The design `[n, p]`.
 * @param y - The response `[n]`.
 * @param options - q, knockoff+, the statistic, covariance, scale and key.
 * @returns The selection, W, the threshold and the knockoffs.
 * @concept knockoffs
 * @example
 * ```ts
 * import { knockoffFilter } from 'omai-engine/learn'
 * import { key, normal } from 'omai-engine/random'
 * import { toFlat } from 'omai-engine/tensor'
 * const z = toFlat(normal(key(1), [200 * 4]))
 * const x = Array.from({ length: 200 }, (_, i) => [z[4 * i] ?? 0, z[4 * i + 1] ?? 0, z[4 * i + 2] ?? 0, z[4 * i + 3] ?? 0])
 * const y = x.map((r, i) => 3 * ((r[0] ?? 0) + (r[1] ?? 0) + (r[2] ?? 0)) + 0.1 * Math.sin(i))
 * knockoffFilter(x, y, { sigma: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], q: 0.5, statistic: 'correlation' }).selected.includes(0) // => true
 * ```
 */
export function knockoffFilter(x: TensorLike, y: TensorLike, options: KnockoffFilterOptions = {}): KnockoffResult {
  const q = options.q ?? 0.1
  const knockoffs = gaussianKnockoffs(x, options)
  const w = knockoffStatistics(x, knockoffs.xTilde, y, options)
  const threshold = knockoffThreshold(w, q, options.plus ?? true)
  return { selected: w.flatMap((v, j) => (v >= threshold ? [j] : [])), statistics: w, threshold, knockoffs }
}
