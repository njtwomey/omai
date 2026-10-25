/**
 * Sparse approximation and recovery, writing a signal as a combination of few atoms of a dictionary: matching pursuit
 * and orthogonal matching pursuit, CoSaMP and iterative hard thresholding for compressed sensing, basis pursuit by
 * ADMM, and the lasso and basis pursuit denoising by coordinate descent (as scikit-learn's `Lasso`), the iterative ones
 * as traced algorithms.
 */

import { argmaxOf, type Json, MdmlError } from '../base/index.ts'
import { cholesky, choSolve, daxpy, ddot, dgemv, dnrm2, lstsq } from '../numerics/linalg/index.ts'
import { asTensor, fromStorage, primalOf, type Tensor, type TensorLike, toFlat } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, type RunStatus, runAlgorithm } from '../trace/index.ts'
import { admmAlgorithm } from './admm.ts'

type Vec = Float64Array

/** A dense matrix as row-major data with its dimensions. */
interface Matrix {
  readonly data: Vec
  readonly rows: number
  readonly cols: number
}

function matrix(a: TensorLike, name: string): Matrix {
  const t = primalOf(asTensor(a)) as Tensor
  if (t.shape.length !== 2) throw new MdmlError('shape', `${name}: the dictionary must be a matrix`)
  return { data: Float64Array.from(t.data), rows: t.shape[0] ?? 0, cols: t.shape[1] ?? 0 }
}

const vector = (y: TensorLike): Vec => Float64Array.from(toFlat(primalOf(asTensor(y))))
const column = (d: Matrix, j: number): Vec =>
  Float64Array.from({ length: d.rows }, (_, i) => d.data[i * d.cols + j] ?? 0)
const columns = (d: Matrix, idx: readonly number[]): Tensor => {
  const out = new Float64Array(d.rows * idx.length)
  for (let i = 0; i < d.rows; i++)
    for (const [k, j] of idx.entries()) out[i * idx.length + k] = d.data[i * d.cols + j] ?? 0
  return fromStorage(out, [d.rows, idx.length], 'float64')
}
const correlations = (d: Matrix, r: Vec): Vec => dgemv(d.data, d.rows, d.cols, r, true)

/** Least squares on a support: argmin |D_S c - y|, as dense coefficients of length `cols`. */
function supportLeastSquares(d: Matrix, y: Vec, support: readonly number[]): Vec {
  const x = new Float64Array(d.cols)
  if (support.length === 0) return x
  const c = toFlat(primalOf(lstsq(columns(d, support), fromStorage(Float64Array.from(y), [y.length], 'float64'))))
  for (const [k, j] of support.entries()) x[j] = c[k] ?? 0
  return x
}

/** The indices of the `k` largest |v| (ties to the smaller index). */
function topK(v: ArrayLike<number>, k: number): number[] {
  return Array.from({ length: v.length }, (_, i) => i)
    .sort((a, b) => Math.abs(v[b] ?? 0) - Math.abs(v[a] ?? 0) || a - b)
    .slice(0, Math.max(0, k))
}

/** Result of a pursuit. */
export interface SparseResult {
  /** The coefficients `[k]` (one per atom). */
  readonly coef: Tensor
  /** The selected atoms, in selection order. */
  readonly support: readonly number[]
  readonly residualNorm: number
  readonly iterations: number
  readonly status: RunStatus
  readonly converged: boolean
  /** The residual norm after each iteration. */
  readonly history: readonly number[]
}

/** State of the greedy pursuits. */
export interface PursuitState {
  readonly x: Vec
  readonly residual: Vec
  readonly support: readonly number[]
  readonly residualNorm: number
  readonly history: readonly number[]
  readonly stop?: { readonly status: 'converged' | 'failed'; readonly message: string }
}

const PURSUIT_SCHEMA = channels({
  metrics: {
    residualNorm: { description: 'norm of the residual y - D x', direction: 'lower' },
    supportSize: { description: 'number of selected atoms' },
  },
  tensors: { x: { role: 'param', description: 'coefficients' } },
  messages: { select: { description: 'the atom selected in this iteration' } },
})

const observePursuit = (s: PursuitState) => ({
  metrics: { residualNorm: s.residualNorm, supportSize: s.support.length },
  tensors: { x: { role: 'param' as const, value: s.x } },
  messages: s.support.length > 0 ? [{ kind: 'select', payload: s.support.at(-1) ?? -1 }] : [],
})

const initial = (d: Matrix, y: Vec): PursuitState => {
  const n = dnrm2(y)
  return { x: new Float64Array(d.cols), residual: Float64Array.from(y), support: [], residualNorm: n, history: [n] }
}

function finish(
  algorithm: Algorithm<PursuitState>,
  run: Partial<Omit<RunOptions, 'maxSteps'>>,
  maxSteps: number,
): SparseResult {
  const result = runAlgorithm(algorithm, { ...run, maxSteps })
  const s = result.state
  return {
    coef: fromStorage(Float64Array.from(s.x), [s.x.length], 'float64'),
    support: s.support,
    residualNorm: s.residualNorm,
    iterations: result.steps,
    status: result.status,
    converged: result.status === 'converged',
    history: s.history,
  }
}

/** Options of the greedy pursuits. */
export interface PursuitOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** The number of atoms to select (default max(floor(k / 10), 1), sklearn's default for OMP). */
  readonly nNonzeroCoefs?: number
  /** Stop once the squared residual norm falls to this (sklearn's tol; overrides nNonzeroCoefs). */
  readonly tol?: number
  readonly maxIterations?: number
}

/**
 * Matching pursuit (Mallat and Zhang 1993) as a traced algorithm: pick the atom most correlated with the residual and
 * subtract its projection; atoms may be picked again.
 *
 * @param dictionary - The dictionary `[m, k]`, atoms as columns.
 * @param y - The signal `[m]`.
 * @param options - Stopping rule (atoms or squared residual).
 * @returns The algorithm.
 * @concept sparse-approximation
 * @example
 * ```ts
 * import { matchingPursuitAlgorithm } from 'mdml-engine/optim'
 * matchingPursuitAlgorithm([[1, 0], [0, 1]], [1, 2]).name // => 'matchingPursuit'
 * ```
 */
export function matchingPursuitAlgorithm(
  dictionary: TensorLike,
  y: TensorLike,
  options: PursuitOptions = {},
): Algorithm<PursuitState> {
  const d = matrix(dictionary, 'matchingPursuit')
  const yv = vector(y)
  const norms = Float64Array.from({ length: d.cols }, (_, j) => ddot(column(d, j), column(d, j)))
  const tol = options.tol
  return {
    name: 'matchingPursuit',
    config: { tol: tol ?? null } as Json,
    schema: PURSUIT_SCHEMA,
    init: () => initial(d, yv),
    step: (s) => {
      const c = correlations(d, s.residual)
      const j = argmaxOf(Float64Array.from(c, (v, i) => Math.abs(v) / Math.sqrt(Math.max(norms[i] ?? 0, 1e-300))))
      const coef = (c[j] ?? 0) / Math.max(norms[j] ?? 0, 1e-300)
      const x = Float64Array.from(s.x)
      x[j] = (x[j] ?? 0) + coef
      const residual = daxpy(-coef, column(d, j), s.residual)
      const rn = dnrm2(residual)
      const support = s.support.includes(j) ? s.support : [...s.support, j]
      const stop =
        tol !== undefined && rn * rn <= tol
          ? ({ status: 'converged', message: 'residual below tolerance' } as const)
          : options.nNonzeroCoefs !== undefined && support.length >= options.nNonzeroCoefs
            ? ({ status: 'converged', message: 'requested number of atoms selected' } as const)
            : undefined
      return { x, residual, support, residualNorm: rn, history: [...s.history, rn], ...(stop ? { stop } : {}) }
    },
    done: (s) => s.stop?.status,
    observe: observePursuit,
  }
}

/**
 * Approximate y by a sparse combination of atoms by matching pursuit.
 *
 * @param dictionary - The dictionary `[m, k]`.
 * @param y - The signal.
 * @param options - Squared-residual tolerance, iteration cap (default 10 k) and run options.
 * @returns The coefficients, support and residual history.
 * @concept sparse-approximation
 * @example
 * ```ts
 * import { matchingPursuit } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(matchingPursuit([[1, 0], [0, 1]], [1, 2], { tol: 1e-20 }).coef) // => [1, 2]
 * ```
 */
export function matchingPursuit(dictionary: TensorLike, y: TensorLike, options: PursuitOptions = {}): SparseResult {
  const { nNonzeroCoefs: _n, tol: _t, maxIterations, ...run } = options
  const d = matrix(dictionary, 'matchingPursuit')
  return finish(matchingPursuitAlgorithm(dictionary, y, options), run, maxIterations ?? 10 * d.cols)
}

/**
 * Orthogonal matching pursuit (Pati et al. 1993; sklearn's Cholesky OMP) as a traced algorithm: pick the atom most
 * correlated with the residual, then refit all selected atoms by least squares (an updated Cholesky factor of their
 * Gram matrix). Stops after `nNonzeroCoefs` atoms or once the squared residual falls to `tol`.
 *
 * @param dictionary - The dictionary `[m, k]`, atoms as columns.
 * @param y - The signal `[m]`.
 * @param options - Number of atoms or squared-residual tolerance.
 * @returns The algorithm.
 * @concept sparse-approximation
 * @example
 * ```ts
 * import { orthogonalMatchingPursuitAlgorithm } from 'mdml-engine/optim'
 * orthogonalMatchingPursuitAlgorithm([[1, 0], [0, 1]], [1, 2]).name // => 'orthogonalMatchingPursuit'
 * ```
 */
export function orthogonalMatchingPursuitAlgorithm(
  dictionary: TensorLike,
  y: TensorLike,
  options: PursuitOptions = {},
): Algorithm<PursuitState> {
  const d = matrix(dictionary, 'orthogonalMatchingPursuit')
  const yv = vector(y)
  const tol = options.tol
  const maxFeatures = tol !== undefined ? d.cols : (options.nNonzeroCoefs ?? Math.max(Math.floor(d.cols / 10), 1))
  const alpha = correlations(d, yv)
  const eps = 2.220446049250313e-16
  return {
    name: 'orthogonalMatchingPursuit',
    config: { nNonzeroCoefs: maxFeatures, tol: tol ?? null } as Json,
    schema: PURSUIT_SCHEMA,
    init: () => {
      const s = initial(d, yv)
      if (tol !== undefined && s.residualNorm ** 2 <= tol)
        return { ...s, stop: { status: 'converged', message: 'residual below tolerance' } }
      return s
    },
    step: (s) => {
      const c = correlations(d, s.residual)
      const lam = argmaxOf(Float64Array.from(c, Math.abs))
      if (s.support.includes(lam) || (alpha[lam] ?? 0) ** 2 < eps)
        return {
          ...s,
          stop: { status: 'converged', message: 'ended early: atom already selected (linear dependence)' },
        }
      const support = [...s.support, lam]
      const ds = columns(d, support)
      const gram = new Float64Array(support.length * support.length)
      for (const [a, i] of support.entries())
        for (const [b, j] of support.entries()) gram[a * support.length + b] = ddot(column(d, i), column(d, j))
      let coefs: number[]
      try {
        const l = cholesky(fromStorage(gram, [support.length, support.length], 'float64'))
        coefs = toFlat(
          primalOf(
            choSolve(
              l,
              fromStorage(
                Float64Array.from(support, (j) => alpha[j] ?? 0),
                [support.length],
                'float64',
              ),
            ),
          ),
        )
      } catch {
        return { ...s, stop: { status: 'converged', message: 'ended early: selected atoms are linearly dependent' } }
      }
      if (coefs.some((v) => !Number.isFinite(v)))
        return { ...s, stop: { status: 'converged', message: 'ended early: selected atoms are linearly dependent' } }
      const x = new Float64Array(d.cols)
      for (const [k, j] of support.entries()) x[j] = coefs[k] ?? 0
      const fitted = dgemv(ds.data as Vec, d.rows, support.length, Float64Array.from(coefs))
      const residual = daxpy(-1, fitted, yv)
      const rn = dnrm2(residual)
      const stop =
        tol !== undefined && rn * rn <= tol
          ? ({ status: 'converged', message: 'residual below tolerance' } as const)
          : support.length >= maxFeatures
            ? ({ status: 'converged', message: 'requested number of atoms selected' } as const)
            : undefined
      return { x, residual, support, residualNorm: rn, history: [...s.history, rn], ...(stop ? { stop } : {}) }
    },
    done: (s) => s.stop?.status,
    observe: observePursuit,
  }
}

/**
 * Sparse approximation by orthogonal matching pursuit (sklearn.linear_model.OrthogonalMatchingPursuit without an
 * intercept).
 *
 * @param dictionary - The dictionary `[m, k]`.
 * @param y - The signal.
 * @param options - Number of atoms (default max(floor(k / 10), 1)) or squared-residual tolerance, and run options.
 * @returns The coefficients, support and residual history.
 * @concept sparse-approximation
 * @example
 * ```ts
 * import { orthogonalMatchingPursuit } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(orthogonalMatchingPursuit([[1, 0, 1], [0, 1, 1]], [2, 0], { nNonzeroCoefs: 1 }).coef) // => [2, 0, 0]
 * ```
 */
export function orthogonalMatchingPursuit(
  dictionary: TensorLike,
  y: TensorLike,
  options: PursuitOptions = {},
): SparseResult {
  const { nNonzeroCoefs: _n, tol: _t, maxIterations, ...run } = options
  const d = matrix(dictionary, 'orthogonalMatchingPursuit')
  return finish(orthogonalMatchingPursuitAlgorithm(dictionary, y, options), run, maxIterations ?? d.cols + 1)
}

/** Options of {@link cosamp} and {@link iterativeHardThresholding}. */
export interface SparseRecoveryOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** Stop once the residual norm falls to this, or once it stops changing (default 1e-10 relative to |y|). */
  readonly tol?: number
  readonly maxIterations?: number
}

/**
 * CoSaMP (Needell and Tropp 2009) as a traced algorithm: merge the 2s atoms most correlated with the residual into the
 * current support, solve least squares on the merged set, and keep its s largest coefficients.
 *
 * @param dictionary - The measurement matrix `[m, k]`.
 * @param y - The measurements.
 * @param sparsity - The target sparsity s.
 * @param options - Tolerance.
 * @returns The algorithm.
 * @concept compressed-sensing
 * @example
 * ```ts
 * import { cosampAlgorithm } from 'mdml-engine/optim'
 * cosampAlgorithm([[1, 0], [0, 1]], [1, 0], 1).name // => 'cosamp'
 * ```
 */
export function cosampAlgorithm(
  dictionary: TensorLike,
  y: TensorLike,
  sparsity: number,
  options: SparseRecoveryOptions = {},
): Algorithm<PursuitState> {
  const d = matrix(dictionary, 'cosamp')
  const yv = vector(y)
  const tol = (options.tol ?? 1e-10) * Math.max(dnrm2(yv), 1e-300)
  return {
    name: 'cosamp',
    config: { sparsity, tol } as Json,
    schema: PURSUIT_SCHEMA,
    init: () => initial(d, yv),
    step: (s) => {
      const proxy = correlations(d, s.residual)
      const omega = topK(proxy, 2 * sparsity)
      const current = Array.from(s.x.keys()).filter((j) => (s.x[j] ?? 0) !== 0)
      const merged = [...new Set([...omega, ...current])].sort((a, b) => a - b)
      const b = supportLeastSquares(d, yv, merged)
      const keep = topK(b, sparsity).sort((p, q) => p - q)
      const x = new Float64Array(d.cols)
      for (const j of keep) x[j] = b[j] ?? 0
      const residual = daxpy(-1, dgemv(d.data, d.rows, d.cols, x), yv)
      const rn = dnrm2(residual)
      const stalled = Math.abs(rn - s.residualNorm) <= 1e-14 * Math.max(1, s.residualNorm)
      const stop =
        rn <= tol
          ? ({ status: 'converged', message: 'residual below tolerance' } as const)
          : stalled
            ? ({ status: 'converged', message: 'residual stopped decreasing' } as const)
            : undefined
      return { x, residual, support: keep, residualNorm: rn, history: [...s.history, rn], ...(stop ? { stop } : {}) }
    },
    done: (s) => s.stop?.status,
    observe: observePursuit,
  }
}

/**
 * Recover an s-sparse vector from measurements y = D x by CoSaMP.
 *
 * @param dictionary - The measurement matrix `[m, k]`.
 * @param y - The measurements.
 * @param sparsity - The sparsity s.
 * @param options - Tolerance, iteration cap (default 100) and run options.
 * @returns The coefficients and residual history.
 * @concept compressed-sensing
 * @example
 * ```ts
 * import { cosamp } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(cosamp([[1, 0, 1], [0, 1, 1]], [0, 3], 1).coef).map((v) => Math.round(v * 1e8) / 1e8) // => [0, 3, 0]
 * ```
 */
export function cosamp(
  dictionary: TensorLike,
  y: TensorLike,
  sparsity: number,
  options: SparseRecoveryOptions = {},
): SparseResult {
  const { tol: _t, maxIterations, ...run } = options
  return finish(cosampAlgorithm(dictionary, y, sparsity, options), run, maxIterations ?? 100)
}

/** Options of {@link iterativeHardThresholding}. */
export interface IhtOptions extends SparseRecoveryOptions {
  /** Normalised IHT (Blumensath and Davies 2010, the default) or a fixed step. */
  readonly normalized?: boolean
  /** The fixed step mu (default 1; |D|_2 < 1 is required for the plain method). */
  readonly stepSize?: number
}

/**
 * Iterative hard thresholding as a traced algorithm: x+ = H_s(x + mu D^T (y - D x)), with the normalised step of
 * Blumensath and Davies (2010) (mu = |g_S|^2 / |D_S g_S|^2 on the current support, halved when the support changes and
 * the step is too long), which needs no scaling of D.
 *
 * @param dictionary - The measurement matrix `[m, k]`.
 * @param y - The measurements.
 * @param sparsity - The sparsity s.
 * @param options - Normalisation, step and tolerance.
 * @returns The algorithm.
 * @concept compressed-sensing
 * @example
 * ```ts
 * import { iterativeHardThresholdingAlgorithm } from 'mdml-engine/optim'
 * iterativeHardThresholdingAlgorithm([[1, 0], [0, 1]], [1, 0], 1).name // => 'iht'
 * ```
 */
export function iterativeHardThresholdingAlgorithm(
  dictionary: TensorLike,
  y: TensorLike,
  sparsity: number,
  options: IhtOptions = {},
): Algorithm<PursuitState> {
  const d = matrix(dictionary, 'iterativeHardThresholding')
  const yv = vector(y)
  const normalized = options.normalized ?? true
  const tol = (options.tol ?? 1e-10) * Math.max(dnrm2(yv), 1e-300)
  const apply = (x: Vec): Vec => dgemv(d.data, d.rows, d.cols, x)
  const threshold = (v: Vec): { x: Vec; support: number[] } => {
    const support = topK(v, sparsity).sort((a, b) => a - b)
    const x = new Float64Array(v.length)
    for (const j of support) x[j] = v[j] ?? 0
    return { x, support }
  }
  return {
    name: 'iht',
    config: { sparsity, normalized, tol } as Json,
    schema: PURSUIT_SCHEMA,
    init: () => initial(d, yv),
    step: (s) => {
      const g = correlations(d, s.residual)
      let mu = options.stepSize ?? 1
      let next = threshold(daxpy(mu, g, s.x))
      if (normalized) {
        const support = s.support.length > 0 ? s.support : topK(g, sparsity)
        const gs = new Float64Array(d.cols)
        for (const j of support) gs[j] = g[j] ?? 0
        const dg = apply(gs)
        mu = ddot(gs, gs) / Math.max(ddot(dg, dg), 1e-300)
        next = threshold(daxpy(mu, g, s.x))
        const changed = next.support.join(',') !== [...support].sort((a, b) => a - b).join(',')
        for (let tries = 0; changed && tries < 60; tries++) {
          const dx = daxpy(-1, s.x, next.x)
          const ddx = apply(dx)
          const omega = (1 - 0.01) * (ddot(dx, dx) / Math.max(ddot(ddx, ddx), 1e-300))
          if (mu <= omega) break
          mu /= 2
          next = threshold(daxpy(mu, g, s.x))
        }
      }
      const residual = daxpy(-1, apply(next.x), yv)
      const rn = dnrm2(residual)
      const stalled = Math.abs(rn - s.residualNorm) <= 1e-15 * Math.max(1, s.residualNorm)
      const stop =
        rn <= tol
          ? ({ status: 'converged', message: 'residual below tolerance' } as const)
          : stalled
            ? ({ status: 'converged', message: 'residual stopped decreasing' } as const)
            : undefined
      return {
        x: next.x,
        residual,
        support: next.support,
        residualNorm: rn,
        history: [...s.history, rn],
        ...(stop ? { stop } : {}),
      }
    },
    done: (s) => s.stop?.status,
    observe: observePursuit,
  }
}

/**
 * Recover an s-sparse vector by (normalised) iterative hard thresholding.
 *
 * @param dictionary - The measurement matrix `[m, k]`.
 * @param y - The measurements.
 * @param sparsity - The sparsity s.
 * @param options - Normalisation, step, tolerance, iteration cap (default 1000) and run options.
 * @returns The coefficients and residual history.
 * @concept compressed-sensing
 * @example
 * ```ts
 * import { iterativeHardThresholding } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(iterativeHardThresholding([[1, 0, 1], [0, 1, 1]], [0, 3], 1).coef).map((v) => Math.round(v * 1e8) / 1e8) // => [0, 3, 0]
 * ```
 */
export function iterativeHardThresholding(
  dictionary: TensorLike,
  y: TensorLike,
  sparsity: number,
  options: IhtOptions = {},
): SparseResult {
  const { tol: _t, maxIterations, normalized: _n, stepSize: _s, ...run } = options
  return finish(iterativeHardThresholdingAlgorithm(dictionary, y, sparsity, options), run, maxIterations ?? 1000)
}

/** Result of {@link basisPursuit} and {@link basisPursuitDenoising}. */
export interface L1Result {
  readonly coef: Tensor
  readonly objective: number
  readonly iterations: number
  readonly converged: boolean
}

/**
 * Basis pursuit min |x|_1 subject to D x = y (Chen, Donoho and Saunders 1998), by ADMM: the x-update projects onto
 * {D x = y} (a cached factor of D D^T), the z-update soft-thresholds.
 *
 * @param dictionary - The matrix `[m, k]`, m < k, of full row rank.
 * @param y - The measurements.
 * @param options - Penalty (default 1), tolerances (default 1e-10), iteration cap (default 20000) and run options.
 * @returns The minimum-L1 solution.
 * @concept compressed-sensing
 * @example
 * ```ts
 * import { basisPursuit } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(basisPursuit([[1, 1, 2]], [2]).coef).map((v) => Math.round(v * 1e6) / 1e6) // => [0, 0, 1]
 * ```
 */
export function basisPursuit(
  dictionary: TensorLike,
  y: TensorLike,
  options: { readonly rho?: number; readonly tol?: number; readonly maxIterations?: number } & Partial<
    Omit<RunOptions, 'maxSteps' | 'start'>
  > = {},
): L1Result {
  const { rho, tol, maxIterations, ...run } = options
  const d = matrix(dictionary, 'basisPursuit')
  const yv = vector(y)
  const ddt = new Float64Array(d.rows * d.rows)
  for (let i = 0; i < d.rows; i++)
    for (let j = 0; j < d.rows; j++) {
      let s = 0
      for (let k = 0; k < d.cols; k++) s += (d.data[i * d.cols + k] ?? 0) * (d.data[j * d.cols + k] ?? 0)
      ddt[i * d.rows + j] = s
    }
  const l = cholesky(fromStorage(ddt, [d.rows, d.rows], 'float64'))
  const project = (v: Vec): Vec => {
    const r = daxpy(-1, yv, dgemv(d.data, d.rows, d.cols, v))
    const w = Float64Array.from(toFlat(primalOf(choSolve(l, fromStorage(r, [r.length], 'float64')))))
    return daxpy(-1, dgemv(d.data, d.rows, d.cols, w, true), v)
  }
  const soft = (v: Vec, t: number): Vec => Float64Array.from(v, (e) => Math.sign(e) * Math.max(Math.abs(e) - t, 0))
  const alg = admmAlgorithm(
    {
      xUpdate: (z, u) => project(daxpy(-1, u, z)),
      zUpdate: (ax, u, r) => soft(daxpy(1, u, ax), 1 / r),
      objective: (_x, z) => z.reduce((a, b) => a + Math.abs(b), 0),
    },
    {
      x0: new Float64Array(d.cols),
      rho: rho ?? 1,
      absTol: tol ?? 1e-10,
      relTol: tol ?? 1e-10,
    },
  )
  const result = runAlgorithm(alg, { ...run, maxSteps: maxIterations ?? 20000 })
  const x = project(result.state.z)
  return {
    coef: fromStorage(x, [x.length], 'float64'),
    objective: x.reduce((a, b) => a + Math.abs(b), 0),
    iterations: result.steps,
    converged: result.status === 'converged',
  }
}

/** Options of {@link lassoCoordinateDescent}. */
export interface LassoCoordinateDescentOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** sklearn's tol: stop once the duality gap falls to tol |y|^2 (checked when the largest update is small). */
  readonly tol?: number
  readonly maxIterations?: number
  /** A starting point (warm start). */
  readonly x0?: ArrayLike<number>
  /** The L2 weight of the elastic net (default 0, the lasso). */
  readonly l2?: number
}

/** State of {@link lassoCoordinateDescentAlgorithm}. */
export interface LassoCoordinateDescentState {
  readonly x: Vec
  readonly residual: Vec
  readonly gap: number
  readonly maxUpdate: number
  readonly objective: number
}

/**
 * Cyclic coordinate descent for the lasso (and elastic net) min |D x - y|^2 / 2 + lambda |x|_1 + (l2 / 2) |x|^2, as
 * sklearn's `enet_coordinate_descent`: exact coordinate minimisation by soft thresholding, the duality gap as the
 * stopping test. Records the objective, the gap and the largest update of each sweep.
 *
 * @param dictionary - The design `[m, k]`.
 * @param y - The response.
 * @param lambda - The L1 weight.
 * @param options - Tolerance, warm start and L2 weight.
 * @returns The algorithm.
 * @concept lasso
 * @example
 * ```ts
 * import { lassoCoordinateDescentAlgorithm } from 'mdml-engine/optim'
 * lassoCoordinateDescentAlgorithm([[1, 0], [0, 1]], [1, 0], 0.1).name // => 'lassoCd'
 * ```
 */
export function lassoCoordinateDescentAlgorithm(
  dictionary: TensorLike,
  y: TensorLike,
  lambda: number,
  options: LassoCoordinateDescentOptions = {},
): Algorithm<LassoCoordinateDescentState> {
  const d = matrix(dictionary, 'lassoCoordinateDescent')
  const yv = vector(y)
  const l2 = options.l2 ?? 0
  const tol = (options.tol ?? 1e-4) * ddot(yv, yv)
  const norms = Float64Array.from({ length: d.cols }, (_, j) => ddot(column(d, j), column(d, j)))
  const objective = (x: Vec, r: Vec): number => {
    let l1 = 0
    for (const v of x) l1 += Math.abs(v)
    return ddot(r, r) / 2 + lambda * l1 + (l2 / 2) * ddot(x, x)
  }
  const dualityGap = (x: Vec, r: Vec): number => {
    // sklearn's gap for the elastic net (in the scaling |r|^2 / 2 + lambda |x|_1 + l2 |x|^2 / 2).
    const xta = daxpy(-l2, x, correlations(d, r))
    let dualNorm = 0
    for (const v of xta) dualNorm = Math.max(dualNorm, Math.abs(v))
    const rNorm2 = ddot(r, r)
    const wNorm2 = ddot(x, x)
    let constant = 1
    let gap: number
    if (dualNorm > lambda) {
      constant = lambda / dualNorm
      gap = 0.5 * rNorm2 * (1 + constant * constant)
    } else gap = rNorm2
    let l1 = 0
    for (const v of x) l1 += Math.abs(v)
    gap += lambda * l1 - constant * ddot(r, yv) + 0.5 * l2 * (1 + constant * constant) * wNorm2
    return gap
  }
  return {
    name: 'lassoCd',
    config: { lambda, l2, tol } as Json,
    schema: channels({
      metrics: {
        objective: { description: 'lasso objective', direction: 'lower' },
        gap: { description: 'duality gap' },
        maxUpdate: { description: 'largest coordinate change in the sweep' },
      },
      tensors: { x: { role: 'param', description: 'coefficients' } },
    }),
    init: () => {
      const x = Float64Array.from(options.x0 ?? new Float64Array(d.cols))
      const r = daxpy(-1, dgemv(d.data, d.rows, d.cols, x), yv)
      return { x, residual: r, gap: Infinity, maxUpdate: Infinity, objective: objective(x, r) }
    },
    step: (s) => {
      const x = Float64Array.from(s.x)
      const r = Float64Array.from(s.residual)
      let maxUpdate = 0
      let maxW = 0
      for (let j = 0; j < d.cols; j++) {
        const nj = norms[j] ?? 0
        if (nj === 0) continue
        const old = x[j] ?? 0
        let rho = 0
        for (let i = 0; i < d.rows; i++) rho += (d.data[i * d.cols + j] ?? 0) * (r[i] ?? 0)
        rho += nj * old
        const next = (Math.sign(rho) * Math.max(Math.abs(rho) - lambda, 0)) / (nj + l2)
        if (next !== old) {
          for (let i = 0; i < d.rows; i++) r[i] = (r[i] ?? 0) - (d.data[i * d.cols + j] ?? 0) * (next - old)
          x[j] = next
        }
        maxUpdate = Math.max(maxUpdate, Math.abs(next - old))
        maxW = Math.max(maxW, Math.abs(next))
      }
      const small = maxW === 0 || maxUpdate / maxW < (options.tol ?? 1e-4)
      const gap = small ? dualityGap(x, r) : Infinity
      return { x, residual: r, gap, maxUpdate, objective: objective(x, r) }
    },
    done: (s) => (s.gap <= tol ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { objective: s.objective, gap: s.gap, maxUpdate: s.maxUpdate },
      tensors: { x: { role: 'param', value: s.x } },
    }),
  }
}

/**
 * The lasso by coordinate descent (sklearn.linear_model.Lasso with alpha = lambda / m and no intercept).
 *
 * @param dictionary - The design `[m, k]`.
 * @param y - The response.
 * @param lambda - The L1 weight.
 * @param options - Tolerance (default 1e-4, as sklearn), warm start, L2 weight, iteration cap (default 1000).
 * @returns The coefficients.
 * @concept lasso
 * @example
 * ```ts
 * import { lassoCoordinateDescent } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(lassoCoordinateDescent([[1, 0], [0, 1]], [3, 0.5], 1).coef) // => [2, 0]
 * ```
 */
export function lassoCoordinateDescent(
  dictionary: TensorLike,
  y: TensorLike,
  lambda: number,
  options: LassoCoordinateDescentOptions = {},
): L1Result {
  const { tol: _t, x0: _x, l2: _l, maxIterations, ...run } = options
  const result = runAlgorithm(lassoCoordinateDescentAlgorithm(dictionary, y, lambda, options), {
    ...run,
    maxSteps: maxIterations ?? 1000,
  })
  const s = result.state
  return {
    coef: fromStorage(Float64Array.from(s.x), [s.x.length], 'float64'),
    objective: s.objective,
    iterations: result.steps,
    converged: result.status === 'converged',
  }
}

/**
 * Basis pursuit denoising min |D x - y|^2 / 2 + lambda |x|_1 (the lasso of signal processing), by coordinate descent.
 *
 * @param dictionary - The dictionary `[m, k]`.
 * @param y - The signal.
 * @param lambda - The L1 weight.
 * @param options - As {@link lassoCoordinateDescent} (tolerance default 1e-10 here).
 * @returns The coefficients.
 * @concept compressed-sensing
 * @example
 * ```ts
 * import { basisPursuitDenoising } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(basisPursuitDenoising([[1, 0], [0, 1]], [3, -0.5], 1).coef) // => [2, 0]
 * ```
 */
export function basisPursuitDenoising(
  dictionary: TensorLike,
  y: TensorLike,
  lambda: number,
  options: LassoCoordinateDescentOptions = {},
): L1Result {
  return lassoCoordinateDescent(dictionary, y, lambda, { tol: 1e-10, ...options })
}
