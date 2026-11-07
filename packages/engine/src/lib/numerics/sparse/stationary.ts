/**
 * Stationary iterations for linear systems A x = b: Jacobi (with an optional damping, the smoother of multigrid),
 * Gauss-Seidel and successive over-relaxation (SOR), each one sweep per step as a traced algorithm whose state
 * serialises, plus the iteration matrix of each, its spectral radius (the asymptotic rate), and the optimal SOR
 * relaxation for a consistently ordered matrix.
 */

import { type Json, MdmlError } from '../../base/index.ts'
import { fromStorage, primalOf, promoteFloat, type Tensor, type TensorLike, toFlat } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { solveKernel } from '../linalg/_kernels.ts'
import { dnrm2 as norm } from '../linalg/blas.ts'
import { eigvals } from '../linalg/eig.ts'
import { csrOf, residualNormOf, rowsOf, sweep } from './_rows.ts'
import { type CsrMatrix, toDense } from './csr.ts'
import type { KrylovResult } from './iterative.ts'

/** The stationary methods: Jacobi, Gauss-Seidel (a forward sweep) and SOR (a forward sweep relaxed by omega). */
export type StationaryMethod = 'jacobi' | 'gauss-seidel' | 'sor'

/** Options of the stationary iterations. */
export interface StationaryOptions {
  /** The splitting (default `'gauss-seidel'`). */
  readonly method?: StationaryMethod
  /**
   * The relaxation: SOR's omega (default 1.5), or the damping of Jacobi (default 1; 2/3 smooths a Poisson system best).
   * Gauss-Seidel is SOR with omega 1 and ignores this.
   */
  readonly omega?: number
  /** The starting iterate (default zeros). */
  readonly x0?: ArrayLike<number>
  /** Stop when the residual norm is at most `max(rtol |b|, atol)` (defaults 1e-8 and 0). */
  readonly rtol?: number
  readonly atol?: number
}

/** The state of a stationary iteration: the iterate and the residual norm after each sweep (the first is the start's). */
export interface StationaryState {
  readonly x: readonly number[]
  readonly residualNorm: number
  readonly history: readonly number[]
}

const SCHEMA = channels({
  metrics: { residualNorm: { description: 'norm of the residual b - A x', direction: 'lower' } },
  tensors: { x: { role: 'state', description: 'current iterate' } },
})

function checkOmega(method: StationaryMethod, omega: number): void {
  if (!(omega > 0 && Number.isFinite(omega))) throw new MdmlError('value', 'stationary iterations need omega > 0')
  if (method === 'sor' && omega >= 2) throw new MdmlError('value', 'SOR diverges for omega outside (0, 2)')
}

/**
 * A stationary iteration as a traced algorithm: one sweep per step (Jacobi from the old iterate, Gauss-Seidel and
 * SOR in place from the newest values). The state is the iterate, the residual norm and the residual history, all
 * plain numbers, so a run can be paused, serialised and resumed.
 *
 * @param a - The matrix (CSR or dense, with a nonzero diagonal).
 * @param b - The right-hand side.
 * @param options - Method, relaxation, start and tolerances.
 * @returns The algorithm, named for the method.
 * @example
 * ```ts
 * import { stationaryAlgorithm } from 'mdml-engine/numerics/sparse'
 * stationaryAlgorithm([[4, 1], [1, 3]], [1, 2], { method: 'jacobi' }).name // => 'jacobi'
 * ```
 */
export function stationaryAlgorithm(
  a: CsrMatrix | TensorLike,
  b: TensorLike,
  options: StationaryOptions = {},
): Algorithm<StationaryState> {
  const rows = rowsOf(a)
  const bv = Float64Array.from(toFlat(primalOf(promoteFloat(b)[0] as Tensor)))
  if (bv.length !== rows.n) throw new MdmlError('shape', 'stationary iterations need b with one entry per row')
  const method = options.method ?? 'gauss-seidel'
  const omega = options.omega ?? (method === 'sor' ? 1.5 : 1)
  checkOmega(method, omega)
  const rtol = options.rtol ?? 1e-8
  const atol = options.atol ?? 0
  const tol = Math.max(rtol * norm(bv), atol)
  const x0 = options.x0 === undefined ? new Array<number>(rows.n).fill(0) : Array.from(options.x0)
  if (x0.length !== rows.n) throw new MdmlError('shape', 'stationary iterations need x0 with one entry per row')
  return {
    name: method,
    config: { method, omega: method === 'gauss-seidel' ? 1 : omega, rtol, atol } as Json,
    schema: SCHEMA,
    init: () => {
      const rn = residualNormOf(rows, x0, bv)
      return { x: x0, residualNorm: rn, history: [rn] }
    },
    step: (s) => {
      const x = sweep(rows, bv, s.x, method, omega)
      const rn = residualNormOf(rows, x, bv)
      return { x, residualNorm: rn, history: [...s.history, rn] }
    },
    done: (s) => (s.residualNorm <= tol ? 'converged' : Number.isFinite(s.residualNorm) ? undefined : 'diverged'),
    observe: (s) => ({
      metrics: { residualNorm: s.residualNorm },
      tensors: { x: { role: 'state' as const, value: s.x } },
    }),
    serialise: (s) => ({ x: [...s.x], residualNorm: s.residualNorm, history: [...s.history] }) as Json,
    deserialise: (j) => {
      const o = j as { x: number[]; residualNorm: number; history: number[] }
      return { x: o.x, residualNorm: o.residualNorm, history: o.history }
    },
  }
}

/** Run options of the stationary solvers. */
export type StationaryRun = StationaryOptions &
  Partial<Omit<RunOptions, 'maxSteps'>> & { readonly maxIterations?: number }

/**
 * Solve A x = b by a stationary iteration, to a residual tolerance or an iteration cap (default 100 n).
 *
 * @param a - The matrix.
 * @param b - The right-hand side.
 * @param options - Method, relaxation, start, tolerances, iteration cap and trace options.
 * @returns The solution with its residual history.
 * @example
 * ```ts
 * import { stationarySolve } from 'mdml-engine/numerics/sparse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(stationarySolve([[4, 1], [1, 3]], [1, 2], { method: 'gauss-seidel', rtol: 1e-12 }).x).map((v) => Math.round(v * 1e8) / 1e8) // => [0.09090909, 0.63636364]
 * ```
 */
export function stationarySolve(a: CsrMatrix | TensorLike, b: TensorLike, options: StationaryRun = {}): KrylovResult {
  const { maxIterations, method, omega, x0, rtol, atol, ...run } = options
  const algorithm = stationaryAlgorithm(a, b, {
    ...(method === undefined ? {} : { method }),
    ...(omega === undefined ? {} : { omega }),
    ...(x0 === undefined ? {} : { x0 }),
    ...(rtol === undefined ? {} : { rtol }),
    ...(atol === undefined ? {} : { atol }),
  })
  const n = rowsOf(a).n
  const result = runAlgorithm(algorithm, { ...run, maxSteps: maxIterations ?? 100 * n })
  return {
    x: fromStorage(Float64Array.from(result.state.x), [n], 'float64'),
    iterations: result.steps,
    converged: result.status === 'converged',
    status: result.status,
    residualNorm: result.state.residualNorm,
    history: result.state.history,
  }
}

/**
 * Jacobi iteration (damped when `omega` is not 1) for A x = b.
 *
 * @param a - The matrix.
 * @param b - The right-hand side.
 * @param options - Damping, start, tolerances, iteration cap and trace options.
 * @returns The solution with its residual history.
 * @example
 * ```ts
 * import { jacobi } from 'mdml-engine/numerics/sparse'
 * jacobi([[4, 1], [1, 3]], [1, 2]).converged // => true
 * ```
 */
export function jacobi(
  a: CsrMatrix | TensorLike,
  b: TensorLike,
  options: Omit<StationaryRun, 'method'> = {},
): KrylovResult {
  return stationarySolve(a, b, { ...options, method: 'jacobi' })
}

/**
 * Gauss-Seidel iteration (forward sweeps) for A x = b.
 *
 * @param a - The matrix.
 * @param b - The right-hand side.
 * @param options - Start, tolerances, iteration cap and trace options.
 * @returns The solution with its residual history.
 * @example
 * ```ts
 * import { gaussSeidel } from 'mdml-engine/numerics/sparse'
 * gaussSeidel([[4, 1], [1, 3]], [1, 2]).converged // => true
 * ```
 */
export function gaussSeidel(
  a: CsrMatrix | TensorLike,
  b: TensorLike,
  options: Omit<StationaryRun, 'method' | 'omega'> = {},
): KrylovResult {
  return stationarySolve(a, b, { ...options, method: 'gauss-seidel' })
}

/**
 * Successive over-relaxation for A x = b: a Gauss-Seidel sweep moved `omega` of the way (1 < omega < 2 over-relaxes).
 *
 * @param a - The matrix.
 * @param b - The right-hand side.
 * @param options - The relaxation `omega` (default 1.5), start, tolerances, iteration cap and trace options.
 * @returns The solution with its residual history.
 * @example
 * ```ts
 * import { sor } from 'mdml-engine/numerics/sparse'
 * sor([[4, 1], [1, 3]], [1, 2], { omega: 1.1 }).converged // => true
 * ```
 */
export function sor(
  a: CsrMatrix | TensorLike,
  b: TensorLike,
  options: Omit<StationaryRun, 'method'> = {},
): KrylovResult {
  return stationarySolve(a, b, { ...options, method: 'sor' })
}

/**
 * The iteration matrix G of a stationary method, x_{k+1} = G x_k + c: damped Jacobi I - omega D^-1 A, and
 * (D + omega L)^-1 ((1 - omega) D - omega U) for SOR (Gauss-Seidel at omega 1), with A = L + D + U.
 *
 * @param a - The matrix.
 * @param options - Method and relaxation (as for the algorithm).
 * @returns The dense iteration matrix.
 * @example
 * ```ts
 * import { iterationMatrix } from 'mdml-engine/numerics/sparse'
 * import { toArray } from 'mdml-engine/tensor'
 * toArray(iterationMatrix([[2, 1], [1, 2]], { method: 'jacobi' })) // => [[0, -0.5], [-0.5, 0]]
 * ```
 */
export function iterationMatrix(
  a: CsrMatrix | TensorLike,
  options: Pick<StationaryOptions, 'method' | 'omega'> = {},
): Tensor {
  const rows = rowsOf(a)
  const n = rows.n
  const method = options.method ?? 'gauss-seidel'
  const omega = method === 'gauss-seidel' ? 1 : (options.omega ?? (method === 'sor' ? 1.5 : 1))
  checkOmega(method, omega)
  const dense = Float64Array.from(toFlat(primalOf(toDense(csrOf(a)))))
  const g = new Float64Array(n * n)
  if (method === 'jacobi') {
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        g[i * n + j] = (i === j ? 1 : 0) - (omega * (dense[i * n + j] ?? 0)) / (dense[i * n + i] ?? 1)
    return fromStorage(g, [n, n], 'float64')
  }
  // (D + omega L) G = (1 - omega) D - omega U, solved column by column.
  const lhs = new Float64Array(n * n)
  const rhs = new Float64Array(n * n)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const aij = dense[i * n + j] ?? 0
      if (j < i) lhs[i * n + j] = omega * aij
      else if (j === i) {
        lhs[i * n + j] = aij
        rhs[i * n + j] = (1 - omega) * aij
      } else rhs[i * n + j] = -omega * aij
    }
  return fromStorage(solveKernel(lhs, rhs, n, n), [n, n], 'float64')
}

/**
 * The spectral radius of a stationary method's iteration matrix: the asymptotic factor by which each sweep shrinks the
 * error (the method converges for every start exactly when it is below 1).
 *
 * @param a - The matrix.
 * @param options - Method and relaxation (as for the algorithm).
 * @returns The largest eigenvalue magnitude of the iteration matrix.
 * @example
 * ```ts
 * import { stationarySpectralRadius } from 'mdml-engine/numerics/sparse'
 * stationarySpectralRadius([[2, 1], [1, 2]], { method: 'jacobi' }) // => 0.5
 * ```
 */
export function stationarySpectralRadius(
  a: CsrMatrix | TensorLike,
  options: Pick<StationaryOptions, 'method' | 'omega'> = {},
): number {
  const { re, im } = eigvals(iterationMatrix(a, options))
  let rho = 0
  for (let i = 0; i < re.data.length; i++) rho = Math.max(rho, Math.hypot(re.data[i] ?? 0, im.data[i] ?? 0))
  return rho
}

/**
 * The SOR relaxation that minimises the spectral radius of a consistently ordered matrix (Young): with the Jacobi
 * spectral radius rho, omega = 2 / (1 + sqrt(1 - rho^2)). The model Poisson problems are consistently ordered.
 *
 * @param jacobiRadius - The spectral radius of the Jacobi iteration matrix, below 1.
 * @returns The optimal omega in [1, 2).
 * @example
 * ```ts
 * import { sorOptimalOmega } from 'mdml-engine/numerics/sparse'
 * sorOptimalOmega(0) // => 1
 * ```
 */
export function sorOptimalOmega(jacobiRadius: number): number {
  if (!(jacobiRadius >= 0 && jacobiRadius < 1))
    throw new MdmlError('value', 'sorOptimalOmega needs a Jacobi radius in [0, 1)')
  return 2 / (1 + Math.sqrt(1 - jacobiRadius * jacobiRadius))
}
