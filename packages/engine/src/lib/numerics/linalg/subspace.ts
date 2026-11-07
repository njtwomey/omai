/**
 * Subspace (block power) iteration for the dominant eigenspace of a symmetric matrix as a traced algorithm: multiply a
 * block of k vectors by the matrix, re-orthonormalise by QR, and read the eigenvalue estimates off the projected k by
 * k matrix (the Rayleigh-Ritz values). One block product per step; the state serialises.
 */

import { type Json, MdmlError, maxOf } from '../../base/index.ts'
import { type Key, normal } from '../../random/index.ts'
import { concreteOrThrow, fromStorage, promoteFloat, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { eighKernel, qrKernel } from './_kernels.ts'
import { dnrm2 } from './blas.ts'

/** A symmetric matrix given as a dense matrix or implicitly by its order and its product with a vector. */
export type SymmetricOperator = TensorLike | { readonly n: number; readonly matvec: (x: Float64Array) => Float64Array }

/** Options of the subspace iteration. */
export interface SubspaceOptions {
  /** The starting block, `[n, k]` (default: Gaussian from the run's key). */
  readonly x0?: TensorLike
  /** Stop when every Ritz pair's residual norm is at most `tol` times the largest Ritz value's magnitude (default 1e-10). */
  readonly tol?: number
}

/**
 * The state of a subspace iteration: the orthonormal block and its product with the matrix (row-major `n x k`), the
 * Ritz values by decreasing magnitude with the residual norm `|A y - theta y|` of each Ritz pair, and the iteration.
 */
export interface SubspaceState {
  readonly q: readonly number[]
  readonly aq: readonly number[]
  readonly ritzValues: readonly number[]
  readonly residuals: readonly number[]
  readonly iteration: number
}

interface Op {
  readonly n: number
  readonly matvec: (x: Float64Array) => Float64Array
}

function operatorOf(a: SymmetricOperator): Op {
  if (typeof a === 'object' && a !== null && 'matvec' in a && 'n' in a) return a as Op
  const m = concreteOrThrow(promoteFloat(a as TensorLike)[0] as Tensor, 'subspaceIteration')
  if (m.shape.length !== 2 || m.shape[0] !== m.shape[1])
    throw new MdmlError('shape', 'subspaceIteration needs a square matrix')
  const n = m.shape[0] ?? 0
  const data = m.data
  return {
    n,
    matvec: (x) => {
      const y = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        let s = 0
        for (let j = 0; j < n; j++) s += (data[i * n + j] ?? 0) * (x[j] ?? 0)
        y[i] = s
      }
      return y
    },
  }
}

/** A times a row-major `n x k` block, column by column. */
function blockProduct(op: Op, block: ArrayLike<number>, k: number): Float64Array {
  const n = op.n
  const out = new Float64Array(n * k)
  for (let c = 0; c < k; c++) {
    const col = Float64Array.from({ length: n }, (_, i) => block[i * k + c] ?? 0)
    const y = op.matvec(col)
    for (let i = 0; i < n; i++) out[i * k + c] = y[i] ?? 0
  }
  return out
}

/** The Rayleigh-Ritz values of an orthonormal block and its product, and the residual norms of the Ritz pairs. */
function ritz(q: Float64Array, aq: Float64Array, n: number, k: number): { values: number[]; residuals: number[] } {
  const t = new Float64Array(k * k)
  for (let i = 0; i < k; i++)
    for (let j = 0; j < k; j++) {
      let s = 0
      for (let r = 0; r < n; r++) s += (q[r * k + i] ?? 0) * (aq[r * k + j] ?? 0)
      t[i * k + j] = s
    }
  const { w, v } = eighKernel(t, k)
  const order = Array.from({ length: k }, (_, i) => i).sort((a, b) => Math.abs(w[b] ?? 0) - Math.abs(w[a] ?? 0))
  const values = order.map((i) => w[i] ?? 0)
  const residuals = order.map((col) => {
    const res = new Float64Array(n)
    for (let r = 0; r < n; r++) {
      let ay = 0
      let y = 0
      for (let j = 0; j < k; j++) {
        const c = v[j * k + col] ?? 0
        ay += (aq[r * k + j] ?? 0) * c
        y += (q[r * k + j] ?? 0) * c
      }
      res[r] = ay - (w[col] ?? 0) * y
    }
    return dnrm2(res)
  })
  return { values, residuals }
}

const SCHEMA = channels({
  metrics: {
    maxResidual: { description: 'largest residual norm of a Ritz pair', direction: 'lower' },
    leadingRitzValue: { description: 'the Ritz value of largest magnitude' },
  },
  tensors: {
    ritzValues: { role: 'state', description: 'the Ritz values, by decreasing magnitude' },
    residuals: { role: 'state', description: 'the residual norm of each Ritz pair' },
    basis: { role: 'state', description: 'the orthonormal block, n by k' },
  },
})

/**
 * Subspace iteration as a traced algorithm: Z = A Q, Q = qr(Z), one block product per step. The k-dimensional span of
 * Q converges to the dominant invariant subspace at the rate |lambda_{k+1} / lambda_k|, and the i-th Ritz value to
 * lambda_i at the square of the rate |lambda_{k+1} / lambda_i| (the Rayleigh-Ritz values are second-order accurate in the vector error; the leading ones settle first).
 *
 * @param a - A symmetric matrix, or `{ n, matvec }`.
 * @param k - The block size (the number of eigenpairs sought).
 * @param options - The starting block and the tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { subspaceIterationAlgorithm } from 'mdml-engine/numerics/linalg'
 * subspaceIterationAlgorithm([[3, 0], [0, 1]], 1).name // => 'subspace-iteration'
 * ```
 */
export function subspaceIterationAlgorithm(
  a: SymmetricOperator,
  k: number,
  options: SubspaceOptions = {},
): Algorithm<SubspaceState> {
  const op = operatorOf(a)
  const n = op.n
  if (!Number.isInteger(k) || k < 1 || k > n)
    throw new MdmlError('value', 'subspaceIteration needs a block size in 1..n')
  const tol = options.tol ?? 1e-10
  const start = (key: Key): Float64Array => {
    if (options.x0 !== undefined) {
      const x = concreteOrThrow(promoteFloat(options.x0)[0] as Tensor, 'subspaceIteration')
      if (x.shape.length !== 2 || x.shape[0] !== n || x.shape[1] !== k)
        throw new MdmlError('shape', `subspaceIteration: x0 must be [${n}, ${k}]`)
      return Float64Array.from(x.data)
    }
    return Float64Array.from(normal(key, [n, k]).data)
  }
  const orthonormalise = (z: Float64Array): Float64Array => qrKernel(z, n, k).q
  const stateOf = (q: Float64Array, iteration: number): SubspaceState => {
    const aq = blockProduct(op, q, k)
    const { values, residuals } = ritz(q, aq, n, k)
    return { q: Array.from(q), aq: Array.from(aq), ritzValues: values, residuals, iteration }
  }
  return {
    name: 'subspace-iteration',
    config: { n, k, tol } as Json,
    schema: SCHEMA,
    init: (key) => stateOf(orthonormalise(start(key)), 0),
    step: (s) => stateOf(orthonormalise(Float64Array.from(s.aq)), s.iteration + 1),
    done: (s) => {
      const scale = Math.max(Math.abs(s.ritzValues[0] ?? 0), Number.MIN_VALUE)
      return maxOf(s.residuals) <= tol * scale ? 'converged' : undefined
    },
    observe: (s) => ({
      metrics: { maxResidual: maxOf(s.residuals), leadingRitzValue: s.ritzValues[0] ?? Number.NaN },
      tensors: {
        ritzValues: { role: 'state' as const, value: s.ritzValues },
        residuals: { role: 'state' as const, value: s.residuals },
        basis: { role: 'state' as const, value: s.q, shape: [n, k] },
      },
    }),
    serialise: (s) =>
      ({
        q: [...s.q],
        aq: [...s.aq],
        ritzValues: [...s.ritzValues],
        residuals: [...s.residuals],
        iteration: s.iteration,
      }) as Json,
    deserialise: (j) => j as unknown as SubspaceState,
  }
}

/** The result of {@link subspaceIteration}. */
export interface SubspaceResult {
  /** The Ritz values by decreasing magnitude. */
  readonly values: readonly number[]
  /** The orthonormal basis of the dominant subspace, `[n, k]`. */
  readonly basis: Tensor
  readonly iterations: number
  readonly converged: boolean
  /** The largest Ritz-pair residual norm after each iteration (the first is the start's). */
  readonly residualHistory: readonly number[]
}

/**
 * The k dominant eigenvalues (by magnitude) of a symmetric matrix and an orthonormal basis of their invariant
 * subspace, by subspace iteration.
 *
 * @param a - A symmetric matrix, or `{ n, matvec }`.
 * @param k - The block size.
 * @param options - Starting block, tolerance, iteration cap (default 2000), seed or key.
 * @returns The Ritz values, the basis and the convergence history.
 * @example
 * ```ts
 * import { subspaceIteration } from 'mdml-engine/numerics/linalg'
 * subspaceIteration([[4, 1, 0], [1, 3, 0], [0, 0, 0.5]], 2).values.map((v) => Math.round(v * 1e6) / 1e6) // => [4.618034, 2.381966]
 * ```
 */
export function subspaceIteration(
  a: SymmetricOperator,
  k: number,
  options: SubspaceOptions & Partial<Omit<RunOptions, 'maxSteps'>> & { readonly maxIterations?: number } = {},
): SubspaceResult {
  const { x0, tol, maxIterations, ...run } = options
  const algorithm = subspaceIterationAlgorithm(a, k, {
    ...(x0 === undefined ? {} : { x0 }),
    ...(tol === undefined ? {} : { tol }),
  })
  const history: number[] = []
  const watched: Algorithm<SubspaceState> = {
    ...algorithm,
    init: (key) => {
      const s = algorithm.init(key)
      history.push(maxOf(s.residuals))
      return s
    },
    step: (s, ctx) => {
      const next = algorithm.step(s, ctx)
      history.push(maxOf(next.residuals))
      return next
    },
  }
  const result = runAlgorithm(watched, { ...run, maxSteps: maxIterations ?? 2000 })
  const n = operatorOf(a).n
  return {
    values: result.state.ritzValues,
    basis: fromStorage(Float64Array.from(result.state.q), [n, k], 'float64'),
    iterations: result.steps,
    converged: result.status === 'converged',
    residualHistory: history,
  }
}
