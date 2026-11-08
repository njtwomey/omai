/**
 * One-sided Jacobi SVD as a traced algorithm: the columns of A are rotated in pairs until they are mutually
 * orthogonal, when their norms are the singular values, the normalised columns are U, and the accumulated rotations are
 * V. A step is one rotation or one full sweep of the cyclic order; the state serialises.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { concreteOrThrow, fromStorage, promoteFloat, type Tensor, type TensorLike } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, type RunStatus, runAlgorithm } from '../../trace/index.ts'

/** Options of the one-sided Jacobi iteration. */
export interface JacobiSvdOptions {
  /** A step is one plane rotation (`'rotation'`) or one cyclic sweep over all column pairs (`'sweep'`, default). */
  readonly granularity?: 'rotation' | 'sweep'
  /** A pair is orthogonal when `|a_p . a_q| <= tol |a_p| |a_q|` (default 1e-15, a few ulps). */
  readonly tol?: number
}

/**
 * The state of the iteration: the working matrix `A V` (`m x n`, row-major, whose columns approach `U S`), the
 * accumulated rotations `V` (`n x n`), the number of full sweeps made, the next column pair in the cyclic order, and
 * the largest cosine between two columns (the off-diagonal measure: zero when the columns are orthogonal).
 */
export interface JacobiSvdState {
  readonly u: readonly number[]
  readonly v: readonly number[]
  readonly sweeps: number
  readonly pair: readonly [number, number]
  readonly rotationsInSweep: number
  readonly maxCosine: number
  readonly finished: boolean
}

function columnStats(u: ArrayLike<number>, m: number, n: number, p: number, q: number): [number, number, number] {
  let alpha = 0
  let beta = 0
  let gamma = 0
  for (let i = 0; i < m; i++) {
    const up = u[i * n + p] ?? 0
    const uq = u[i * n + q] ?? 0
    alpha += up * up
    beta += uq * uq
    gamma += up * uq
  }
  return [alpha, beta, gamma]
}

function maxCosineOf(u: ArrayLike<number>, m: number, n: number): number {
  let worst = 0
  for (let p = 0; p < n - 1; p++)
    for (let q = p + 1; q < n; q++) {
      const [alpha, beta, gamma] = columnStats(u, m, n, p, q)
      if (alpha > 0 && beta > 0) worst = Math.max(worst, Math.abs(gamma) / Math.sqrt(alpha * beta))
    }
  return worst
}

const SCHEMA = channels({
  metrics: {
    maxCosine: { description: 'largest cosine between two columns (the off-diagonal measure)', direction: 'lower' },
    sweeps: { description: 'full cyclic sweeps made' },
  },
  tensors: {
    columns: { role: 'state', description: 'the working matrix A V, whose columns become U S' },
    singularValues: { role: 'state', description: 'the column norms' },
  },
})

/**
 * The singular values of a state: the norms of the working columns, in column order.
 *
 * @param state - The state of {@link jacobiSvdAlgorithm}.
 * @param m - Rows of the matrix.
 * @param n - Columns of the matrix.
 * @returns The `n` column norms.
 * @example
 * ```ts
 * import { jacobiSingularValues } from 'omai-engine/numerics/linalg'
 * const state = { u: [3, 0, 0, 4], v: [1, 0, 0, 1], sweeps: 0, pair: [0, 1], rotationsInSweep: 0, maxCosine: 0, finished: false } as const
 * jacobiSingularValues(state, 2, 2) // => [3, 4]
 * ```
 */
export function jacobiSingularValues(state: JacobiSvdState, m: number, n: number): number[] {
  return Array.from({ length: n }, (_, c) => {
    let s = 0
    for (let i = 0; i < m; i++) s += (state.u[i * n + c] ?? 0) ** 2
    return Math.sqrt(s)
  })
}

/**
 * One-sided Jacobi (Hestenes) SVD as a traced algorithm for a matrix with at least as many rows as columns. Each
 * rotation of a column pair (p, q) makes them orthogonal; the sweeps cycle through all pairs and the largest cosine
 * between columns falls quadratically at the end. The singular values emerge as column norms and keep their relative
 * accuracy even when they span many orders of magnitude.
 *
 * @param a - The matrix `[m, n]` with `m >= n` (transpose a wide matrix).
 * @param options - Granularity and tolerance.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { jacobiSvdAlgorithm } from 'omai-engine/numerics/linalg'
 * jacobiSvdAlgorithm([[3, 0], [0, 1], [0, 0]]).name // => 'jacobi-svd'
 * ```
 */
export function jacobiSvdAlgorithm(a: TensorLike, options: JacobiSvdOptions = {}): Algorithm<JacobiSvdState> {
  const t = concreteOrThrow(promoteFloat(a)[0] as Tensor, 'jacobiSvdAlgorithm')
  if (t.shape.length !== 2) throw new OmaiError('shape', 'jacobiSvdAlgorithm needs a matrix')
  const [m, n] = t.shape as [number, number]
  if (m < n) throw new OmaiError('shape', 'jacobiSvdAlgorithm needs at least as many rows as columns')
  const tol = options.tol ?? 1e-15
  const granularity = options.granularity ?? 'sweep'
  const identity = Array.from({ length: n * n }, (_, i) => (i % (n + 1) === 0 ? 1 : 0))
  const a0 = Array.from(t.data)
  return {
    name: 'jacobi-svd',
    config: { m, n, tol, granularity } as Json,
    schema: SCHEMA,
    init: () => ({
      u: a0,
      v: identity,
      sweeps: 0,
      pair: [0, 1],
      rotationsInSweep: 0,
      maxCosine: n > 1 ? maxCosineOf(a0, m, n) : 0,
      finished: n < 2,
    }),
    step: (s) => {
      const u = Array.from(s.u)
      const v = Array.from(s.v)
      let [p, q] = s.pair
      let sweeps = s.sweeps
      let rotations = s.rotationsInSweep
      // In sweep granularity, run until the pair pointer wraps; otherwise one pair.
      const pairs = granularity === 'sweep' ? (n * (n - 1)) / 2 : 1
      let finished = false
      for (let done = 0; done < pairs; done++) {
        const [alpha, beta, gamma] = columnStats(u, m, n, p, q)
        if (gamma !== 0 && Math.abs(gamma) > tol * Math.sqrt(alpha * beta)) {
          rotations += 1
          const zeta = (beta - alpha) / (2 * gamma)
          const tan = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta))
          const c = 1 / Math.sqrt(1 + tan * tan)
          const sn = c * tan
          for (let i = 0; i < m; i++) {
            const up = u[i * n + p] ?? 0
            const uq = u[i * n + q] ?? 0
            u[i * n + p] = c * up - sn * uq
            u[i * n + q] = sn * up + c * uq
          }
          for (let i = 0; i < n; i++) {
            const vp = v[i * n + p] ?? 0
            const vq = v[i * n + q] ?? 0
            v[i * n + p] = c * vp - sn * vq
            v[i * n + q] = sn * vp + c * vq
          }
        }
        q += 1
        if (q >= n) {
          p += 1
          q = p + 1
        }
        if (p >= n - 1) {
          // A sweep is complete: finished when it needed no rotation.
          sweeps += 1
          if (rotations === 0) finished = true
          rotations = 0
          p = 0
          q = 1
        }
      }
      return { u, v, sweeps, pair: [p, q], rotationsInSweep: rotations, maxCosine: maxCosineOf(u, m, n), finished }
    },
    done: (s) => (s.finished ? 'converged' : undefined),
    observe: (s) => ({
      metrics: { maxCosine: s.maxCosine, sweeps: s.sweeps },
      tensors: {
        columns: { role: 'state' as const, value: s.u, shape: [m, n] },
        singularValues: { role: 'state' as const, value: jacobiSingularValues(s, m, n) },
      },
    }),
    serialise: (s) => ({ ...s, u: [...s.u], v: [...s.v], pair: [...s.pair] }) as Json,
    deserialise: (j) => j as unknown as JacobiSvdState,
  }
}

/** The result of {@link jacobiSvd}. */
export interface JacobiSvdResult {
  readonly u: Tensor
  /** Singular values, descending. */
  readonly s: readonly number[]
  readonly vt: Tensor
  readonly sweeps: number
  readonly converged: boolean
  /** How the run ended (`converged`, `completed` at the sweep cap, ...). */
  readonly status: RunStatus
  /** The largest column cosine after each step (the first is the start's). */
  readonly history: readonly number[]
}

/**
 * The reduced SVD A = U diag(s) V^T of a matrix with at least as many rows as columns by the traced one-sided Jacobi
 * iteration, with the history of the off-diagonal measure.
 *
 * @param a - The matrix `[m, n]`, `m >= n`.
 * @param options - Granularity, tolerance, step cap (default 60 sweeps' worth) and trace options.
 * @returns U, the singular values (descending), V^T and the history.
 * @example
 * ```ts
 * import { jacobiSvd } from 'omai-engine/numerics/linalg'
 * jacobiSvd([[3, 0], [0, 1], [0, 0]]).s // => [3, 1]
 * ```
 */
export function jacobiSvd(
  a: TensorLike,
  options: JacobiSvdOptions & Partial<Omit<RunOptions, 'maxSteps'>> & { readonly maxSteps?: number } = {},
): JacobiSvdResult {
  const { granularity, tol, maxSteps, ...run } = options
  const algorithm = jacobiSvdAlgorithm(a, {
    ...(granularity === undefined ? {} : { granularity }),
    ...(tol === undefined ? {} : { tol }),
  })
  const t = promoteFloat(a)[0] as Tensor
  const [m, n] = t.shape as [number, number]
  const history: number[] = []
  const watched: Algorithm<JacobiSvdState> = {
    ...algorithm,
    init: (key) => {
      const s = algorithm.init(key)
      history.push(s.maxCosine)
      return s
    },
    step: (s, ctx) => {
      const next = algorithm.step(s, ctx)
      history.push(next.maxCosine)
      return next
    },
  }
  const pairs = Math.max(1, (n * (n - 1)) / 2)
  const cap = maxSteps ?? (granularity === 'rotation' ? 60 * pairs : 60)
  const result = runAlgorithm(watched, { ...run, maxSteps: cap })
  const state = result.state
  const norms = jacobiSingularValues(state, m, n)
  const order = norms.map((_, i) => i).sort((i, j) => (norms[j] ?? 0) - (norms[i] ?? 0))
  const u = new Float64Array(m * n)
  const vt = new Float64Array(n * n)
  order.forEach((src, dst) => {
    const sv = norms[src] ?? 0
    for (let i = 0; i < m; i++) u[i * n + dst] = sv > 0 ? (state.u[i * n + src] ?? 0) / sv : 0
    for (let i = 0; i < n; i++) vt[dst * n + i] = state.v[i * n + src] ?? 0
  })
  return {
    u: fromStorage(u, [m, n], 'float64'),
    s: order.map((i) => norms[i] ?? 0),
    vt: fromStorage(vt, [n, n], 'float64'),
    sweeps: state.sweeps,
    converged: result.status === 'converged',
    status: result.status,
    history,
  }
}
