/**
 * Regularising iterations for ill-posed least-squares problems, where stopping early plays the part of the
 * regularisation parameter: Landweber, CGLS (conjugate gradients on the normal equations) and LSQR (Golub-Kahan
 * bidiagonalisation), as algorithms, with `regularise` to run one and collect residual, solution and error curves.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import { primalOf, promoteFloat, type TensorLike, toFlat, type Value } from '../../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../trace/index.ts'
import { dgemv, ddot as dot, dnrm2 as norm } from '../linalg/index.ts'

/** A dense matrix with products A x and A^T y. */
function operator(a: TensorLike): {
  m: number
  n: number
  mv: (x: Float64Array) => Float64Array
  rmv: (y: Float64Array) => Float64Array
  norm2: number
} {
  const [av] = promoteFloat(a) as [Value]
  if (av.shape.length !== 2) throw new OmaiError('shape', 'regularising iterations need a matrix')
  const [m = 0, n = 0] = av.shape
  const d = toFlat(primalOf(av))
  const mv = (x: Float64Array): Float64Array => dgemv(d, m, n, x)
  const rmv = (y: Float64Array): Float64Array => dgemv(d, m, n, y, true)
  // The largest singular value squared, by power iteration on A^T A (for Landweber's step bound).
  let v: Float64Array = Float64Array.from({ length: n }, (_, i) => 1 + Math.sin(i))
  let lambda = 0
  for (let it = 0; it < 100; it++) {
    const w = rmv(mv(v))
    const nw = norm(w)
    if (nw === 0) break
    const next = dot(v, w) / dot(v, v)
    v = w.map((x) => x / nw)
    if (Math.abs(next - lambda) <= 1e-10 * next) {
      lambda = next
      break
    }
    lambda = next
  }
  return { m, n, mv, rmv, norm2: lambda }
}

/** State of a regularising iteration. */
export interface RegularisingState {
  readonly x: Float64Array
  readonly residualNorm: number
  readonly solutionNorm: number
  /** |x - x_true| when a true solution was given (semi-convergence). */
  readonly error: number
  readonly internal: Readonly<Record<string, unknown>>
}

/** Options of the regularising iterations. */
export interface RegularisingOptions {
  /** The true solution, to record the error and show semi-convergence. */
  readonly xTrue?: ArrayLike<number>
  /** Stop when the residual norm falls below this (the discrepancy principle as a stopping rule). */
  readonly stopResidual?: number
}

const SCHEMA = channels({
  metrics: {
    residualNorm: { description: '|A x - b|' },
    solutionNorm: { description: '|x|' },
    error: { description: '|x - x_true|, when known' },
  },
  tensors: { x: { role: 'state', description: 'current iterate' } },
})

function stateOf(
  x: Float64Array,
  r: Float64Array,
  xTrue: Float64Array | undefined,
  internal: Record<string, unknown>,
): RegularisingState {
  return {
    x,
    residualNorm: norm(r),
    solutionNorm: norm(x),
    error: xTrue === undefined ? Number.NaN : norm(x.map((v, i) => v - (xTrue[i] ?? 0))),
    internal,
  }
}

const observeState = (s: RegularisingState) => ({
  metrics: {
    residualNorm: s.residualNorm,
    solutionNorm: s.solutionNorm,
    ...(Number.isNaN(s.error) ? {} : { error: s.error }),
  },
  tensors: { x: { role: 'state' as const, value: s.x } },
})

/**
 * The Landweber iteration x <- x + omega A^T (b - A x), omega < 2 / |A|^2 (default 1 / |A|^2): a regularising method
 * whose iteration count plays the part of the parameter. Error first falls, then rises as noise is fitted
 * (semi-convergence).
 *
 * @param a - Matrix.
 * @param b - Data.
 * @param options - Step omega, true solution and residual stopping level.
 * @returns The algorithm.
 * @concept semi-convergence
 * @example
 * ```ts
 * import { landweberAlgorithm } from 'omai-engine/numerics/inverse'
 * landweberAlgorithm([[1, 0], [0, 2]], [1, 2]).name // => 'landweber'
 * ```
 */
export function landweberAlgorithm(
  a: TensorLike,
  b: TensorLike,
  options: RegularisingOptions & { readonly omega?: number } = {},
): Algorithm<RegularisingState> {
  const op = operator(a)
  const bv = Float64Array.from(toFlat(primalOf(promoteFloat(b)[0] as Value)))
  const omega = options.omega ?? 1 / op.norm2
  const xTrue = options.xTrue === undefined ? undefined : Float64Array.from(options.xTrue)
  return {
    name: 'landweber',
    config: { omega } as Json,
    schema: SCHEMA,
    init: () => stateOf(new Float64Array(op.n), bv, xTrue, {}),
    step: (s) => {
      const r = bv.map((v, i) => v - (op.mv(s.x)[i] ?? 0))
      const g = op.rmv(r)
      const x = s.x.map((v, i) => v + omega * (g[i] ?? 0))
      return stateOf(
        x,
        bv.map((v, i) => v - (op.mv(x)[i] ?? 0)),
        xTrue,
        {},
      )
    },
    done: (s) =>
      options.stopResidual !== undefined && s.residualNorm <= options.stopResidual ? 'converged' : undefined,
    observe: observeState,
  }
}

/**
 * CGLS: conjugate gradients on the normal equations A^T A x = A^T b without forming them, a fast regularising
 * iteration (each step is optimal in the Krylov space of A^T A).
 *
 * @param a - Matrix.
 * @param b - Data.
 * @param options - True solution and residual stopping level.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { cglsAlgorithm } from 'omai-engine/numerics/inverse'
 * cglsAlgorithm([[1, 0], [0, 2]], [1, 2]).name // => 'cgls'
 * ```
 */
export function cglsAlgorithm(
  a: TensorLike,
  b: TensorLike,
  options: RegularisingOptions = {},
): Algorithm<RegularisingState> {
  const op = operator(a)
  const bv = Float64Array.from(toFlat(primalOf(promoteFloat(b)[0] as Value)))
  const xTrue = options.xTrue === undefined ? undefined : Float64Array.from(options.xTrue)
  return {
    name: 'cgls',
    config: {} as Json,
    schema: SCHEMA,
    init: () => {
      const s0 = op.rmv(bv)
      return stateOf(new Float64Array(op.n), bv, xTrue, { r: bv, s: s0, p: s0, gamma: dot(s0, s0) })
    },
    step: (st) => {
      const { r, p, gamma } = st.internal as { r: Float64Array; s: Float64Array; p: Float64Array; gamma: number }
      if (gamma === 0) return st
      const q = op.mv(p)
      const alpha = gamma / dot(q, q)
      const x = st.x.map((v, i) => v + alpha * (p[i] ?? 0))
      const rNext = r.map((v, i) => v - alpha * (q[i] ?? 0))
      const sNext = op.rmv(rNext)
      const gammaNext = dot(sNext, sNext)
      const pNext = sNext.map((v, i) => v + (gammaNext / gamma) * (p[i] ?? 0))
      return stateOf(x, rNext, xTrue, { r: rNext, s: sNext, p: pNext, gamma: gammaNext })
    },
    done: (s) =>
      options.stopResidual !== undefined && s.residualNorm <= options.stopResidual
        ? 'converged'
        : (s.internal as { gamma: number }).gamma === 0
          ? 'converged'
          : undefined,
    observe: observeState,
  }
}

/**
 * LSQR (Paige and Saunders): Golub-Kahan bidiagonalisation with QR updates, mathematically equivalent to CGLS but more
 * stable; the standard iterative least-squares and regularisation method (scipy.sparse.linalg.lsqr without damping).
 *
 * @param a - Matrix.
 * @param b - Data.
 * @param options - True solution and residual stopping level.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { lsqrAlgorithm } from 'omai-engine/numerics/inverse'
 * lsqrAlgorithm([[1, 0], [0, 2]], [1, 2]).name // => 'lsqr'
 * ```
 */
export function lsqrAlgorithm(
  a: TensorLike,
  b: TensorLike,
  options: RegularisingOptions = {},
): Algorithm<RegularisingState> {
  const op = operator(a)
  const bv = Float64Array.from(toFlat(primalOf(promoteFloat(b)[0] as Value)))
  const xTrue = options.xTrue === undefined ? undefined : Float64Array.from(options.xTrue)
  type L = { u: Float64Array; v: Float64Array; w: Float64Array; phiBar: number; rhoBar: number; alpha: number }
  return {
    name: 'lsqr',
    config: {} as Json,
    schema: SCHEMA,
    init: () => {
      const beta = norm(bv)
      const u = bv.map((x) => x / (beta || 1))
      let v = op.rmv(u)
      const alpha = norm(v)
      v = v.map((x) => x / (alpha || 1))
      const internal: L = { u, v, w: Float64Array.from(v), phiBar: beta, rhoBar: alpha, alpha }
      return {
        x: new Float64Array(op.n),
        residualNorm: beta,
        solutionNorm: 0,
        error: xTrue === undefined ? Number.NaN : norm(xTrue),
        internal,
      }
    },
    step: (s) => {
      const st = s.internal as L
      // Bidiagonalisation: beta u = A v - alpha u, alpha v = A^T u - beta v.
      let u = op.mv(st.v).map((x, i) => x - st.alpha * (st.u[i] ?? 0))
      const beta = norm(u)
      u = u.map((x) => x / (beta || 1))
      let v = op.rmv(u).map((x, i) => x - beta * (st.v[i] ?? 0))
      const alpha = norm(v)
      v = v.map((x) => x / (alpha || 1))
      // Plane rotation eliminating beta.
      const rho = Math.hypot(st.rhoBar, beta)
      const c = st.rhoBar / rho
      const sn = beta / rho
      const theta = sn * alpha
      const rhoBar = -c * alpha
      const phi = c * st.phiBar
      const phiBar = sn * st.phiBar
      const x = s.x.map((xi, i) => xi + (phi / rho) * (st.w[i] ?? 0))
      const w = v.map((vi, i) => vi - (theta / rho) * (st.w[i] ?? 0))
      return {
        x,
        residualNorm: Math.abs(phiBar),
        solutionNorm: norm(x),
        error: xTrue === undefined ? Number.NaN : norm(x.map((xi, i) => xi - (xTrue[i] ?? 0))),
        internal: { u, v, w, phiBar, rhoBar, alpha } satisfies L,
      }
    },
    done: (s) =>
      options.stopResidual !== undefined && s.residualNorm <= options.stopResidual ? 'converged' : undefined,
    observe: observeState,
  }
}

/** Result of a regularising iteration. */
export interface RegularisingResult {
  readonly x: number[]
  readonly iterations: number
  readonly residualNorms: readonly number[]
  readonly solutionNorms: readonly number[]
  /** Errors per iteration when a true solution was given. */
  readonly errors: readonly number[]
}

/**
 * Run a regularising iteration for a number of steps, collecting the residual, solution and error curves.
 *
 * @param algorithm - From {@link landweberAlgorithm}, {@link cglsAlgorithm} or {@link lsqrAlgorithm}.
 * @param iterations - Number of steps.
 * @param run - Trace options.
 * @returns The final iterate and the curves.
 * @example
 * ```ts
 * import { cglsAlgorithm, regularise } from 'omai-engine/numerics/inverse'
 * regularise(cglsAlgorithm([[1, 0], [0, 2]], [1, 2]), 2).x.map((v) => Math.round(v * 1e12) / 1e12) // => [1, 1]
 * ```
 */
export function regularise(
  algorithm: Algorithm<RegularisingState>,
  iterations: number,
  run: Partial<Omit<RunOptions, 'maxSteps'>> = {},
): RegularisingResult {
  const residualNorms: number[] = []
  const solutionNorms: number[] = []
  const errors: number[] = []
  const wrapped: Algorithm<RegularisingState> = {
    ...algorithm,
    step: (s, ctx) => {
      const next = algorithm.step(s, ctx)
      residualNorms.push(next.residualNorm)
      solutionNorms.push(next.solutionNorm)
      errors.push(next.error)
      return next
    },
  }
  const result = runAlgorithm(wrapped, { ...run, maxSteps: iterations })
  return { x: [...result.state.x], iterations: result.steps, residualNorms, solutionNorms, errors }
}
