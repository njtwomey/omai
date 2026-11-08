/**
 * Matrix equations, scipy.linalg style: Sylvester and Lyapunov equations (continuous and discrete) through their
 * Kronecker forms, the matrix sign function, and the continuous and discrete algebraic Riccati equations that LQR
 * control and Kalman filtering rest on.
 */

import { type Json, OmaiError } from '../../base/index.ts'
import {
  abs,
  add,
  concat,
  div,
  item,
  kron,
  matmul,
  max,
  mul,
  neg,
  primalOf,
  promoteFloat,
  reshape,
  sliceAxis,
  sub,
  sum,
  type TensorLike,
  toFlat,
  type Value,
} from '../../tensor/index.ts'
import {
  type Algorithm,
  type Contract,
  defineContract,
  numberFromJson,
  numberToJson,
  runAlgorithm,
  scalar,
  stateFromJson,
  stateToJson,
  tensor,
} from '../../trace/index.ts'
import { identity, mT, namingSingular, squareSize, sym } from './_matrix.ts'
import { slogdet } from './basics.ts'
import { inv, solve } from './lu.ts'
import { lstsq } from './qr.ts'
import { svdvals } from './svd.ts'

const norm1 = (a: Value): number => item(primalOf(max(sum(abs(a), 0))))

/**
 * Solve the Sylvester equation A X + X B = Q (scipy.linalg.solve_sylvester) through its Kronecker form
 * (A kron I + I kron B^T) vec(X) = vec(Q) (row-major vec): straightforward and differentiable, for moderate sizes.
 *
 * @param a - `[m, m]`.
 * @param b - `[n, n]`.
 * @param q - `[m, n]`.
 * @returns X `[m, n]`.
 * @concept matrix-equations
 * @example
 * ```ts
 * import { solveSylvester } from 'omai-engine/numerics/linalg'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(solveSylvester([[1]], [[2]], [[6]])) // => [[2]]
 * ```
 */
export function solveSylvester(a: TensorLike, b: TensorLike, q: TensorLike): Value {
  const [av, bv, qv] = promoteFloat(a, b, q) as [Value, Value, Value]
  const m = squareSize(av.shape, 'solveSylvester')
  const n = squareSize(bv.shape, 'solveSylvester')
  const system = add(kron(av, identity(n, av)), kron(identity(m, av), mT(bv)))
  return reshape(solve(system, reshape(qv, [m * n])), [m, n])
}

/**
 * Solve the continuous Lyapunov equation A X + X A^T = Q (scipy.linalg.solve_continuous_lyapunov).
 *
 * @param a - `[n, n]`.
 * @param q - `[n, n]`.
 * @returns X.
 * @example
 * ```ts
 * import { solveContinuousLyapunov } from 'omai-engine/numerics/linalg'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(solveContinuousLyapunov([[-1]], [[-2]])) // => [[1]]
 * ```
 */
export function solveContinuousLyapunov(a: TensorLike, q: TensorLike): Value {
  const [av] = promoteFloat(a) as [Value]
  return solveSylvester(av, mT(av), q)
}

/**
 * Solve the discrete Lyapunov equation A X A^T - X + Q = 0 (scipy.linalg.solve_discrete_lyapunov) through
 * (I - A kron A) vec(X) = vec(Q).
 *
 * @param a - `[n, n]`.
 * @param q - `[n, n]`.
 * @returns X.
 * @example
 * ```ts
 * import { solveDiscreteLyapunov } from 'omai-engine/numerics/linalg'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(solveDiscreteLyapunov([[0.5]], [[0.75]])) // => [[1]]
 * ```
 */
export function solveDiscreteLyapunov(a: TensorLike, q: TensorLike): Value {
  const [av, qv] = promoteFloat(a, q) as [Value, Value]
  const n = squareSize(av.shape, 'solveDiscreteLyapunov')
  const system = sub(identity(n * n, av), kron(av, av))
  return reshape(solve(system, reshape(qv, [n * n])), [n, n])
}

/**
 * The matrix sign function (scipy.linalg.signm) by Newton's iteration Z <- (c Z + (c Z)^{-1}) / 2 with determinant
 * scaling c = |det Z|^{-1/n}, for matrices with no eigenvalues on the imaginary axis: sign(A) has the eigenvalues of A
 * mapped to +1 (right half-plane) or -1 (left), and (I - sign(A)) / 2 projects onto A's stable invariant subspace.
 * Built from inverses, so differentiable.
 *
 * @param a - Square matrix `[n, n]`.
 * @returns sign(A).
 * @throws OmaiError with code `convergence` when the iteration does not settle (eigenvalues on or near the axis).
 * @concept matrix-functions
 * @example
 * ```ts
 * import { signm } from 'omai-engine/numerics/linalg'
 * import { toArray } from 'omai-engine/tensor'
 * toArray(signm([[-2, 0], [0, 3]])) // => [[-1, 0], [0, 1]]
 * ```
 */
export function signm(a: TensorLike): Value {
  const [z0] = promoteFloat(a) as [Value]
  const n = squareSize(z0.shape, 'signm')
  let z = z0
  return namingSingular('signm', 'A is singular (it has an eigenvalue on the imaginary axis)', () => {
    for (let i = 0; i < 100; i++) {
      // Determinant scaling c = |det Z|^{-1/n} speeds the early iterations.
      const c = Math.exp(-item(primalOf(slogdet(z).logabsdet)) / n)
      const next = mul(add(mul(z, c), div(inv(z), c)), 0.5)
      const change = norm1(sub(next, z)) / Math.max(norm1(next), 1e-300)
      z = next
      if (change <= 1e-14) return z
    }
    throw new OmaiError('convergence', 'signm: the Newton iteration did not converge')
  })
}

/**
 * Solve the continuous algebraic Riccati equation A^T X + X A - X B R^{-1} B^T X + Q = 0 for the stabilising X
 * (scipy.linalg.solve_continuous_are), by the matrix sign function of the Hamiltonian [[A, -G], [-Q, -A^T]] (Newton's
 * iteration with determinant scaling) and a least-squares solve; differentiable.
 *
 * @param a - `[n, n]`.
 * @param b - `[n, m]`.
 * @param q - `[n, n]` (symmetric).
 * @param r - `[m, m]` (symmetric positive definite).
 * @returns X (symmetric).
 * @throws OmaiError with code `convergence` when the sign iteration does not settle or there is no stabilising solution.
 * @concept matrix-equations
 * @example
 * ```ts
 * import { solveCare } from 'omai-engine/numerics/linalg'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(solveCare([[0]], [[1]], [[1]], [[1]])) - 1) < 1e-12 // => true
 * ```
 */
export function solveCare(a: TensorLike, b: TensorLike, q: TensorLike, r: TensorLike): Value {
  const [av, bv, qv, rv] = promoteFloat(a, b, q, r) as [Value, Value, Value, Value]
  const n = squareSize(av.shape, 'solveCare')
  const g = matmul(bv, solve(rv, mT(bv)))
  let z: Value
  try {
    z = signm(concat([concat([av, neg(g)], 1), concat([neg(qv), neg(mT(av))], 1)], 0))
  } catch (e) {
    if (e instanceof OmaiError && e.code === 'convergence') {
      throw new OmaiError('convergence', 'solveCare: the sign iteration did not converge')
    }
    throw e
  }
  const block = (r0: number, c0: number): Value => sliceAxis(sliceAxis(z, 0, r0, r0 + n), 1, c0, c0 + n)
  const eye = identity(n, av)
  // [W12; W22 + I] X = -[W11 + I; W21].
  const lhs = concat([block(0, n), add(block(n, n), eye)], 0)
  const rhs = neg(concat([add(block(0, 0), eye), block(n, 0)], 0))
  // Without a stabilising solution the stable invariant subspace has fewer than n dimensions: lhs then lacks full column
  // rank (its singular values are those of the sign matrix's stable projector), and any least-squares answer is spurious.
  const sv = toFlat(primalOf(svdvals(lhs)))
  if (!((sv[n - 1] ?? 0) > (sv[0] ?? 0) * 2 * n * Number.EPSILON))
    throw new OmaiError(
      'convergence',
      'solveCare: no stabilising solution (the stable invariant subspace is deficient)',
    )
  const x = lstsq(lhs, rhs)
  if (!(norm1(sub(matmul(lhs, x), rhs)) <= 1e-6 * Math.max(1, norm1(rhs))))
    throw new OmaiError(
      'convergence',
      'solveCare: no stabilising solution (the stable invariant subspace is deficient)',
    )
  return sym(x)
}

/** The paths the Riccati and spline algorithms log, as a type. */
export type EquationPath = 'dare/residual' | 'dare/iterate' | 'dare/solution'

/** What the doubling iteration for the discrete Riccati equation logs: the relative change, the iteration and the iterate. */
export const dareContract: Contract<unknown, EquationPath> = defineContract({
  'dare/residual': scalar({ description: 'the relative change of H in this doubling step', direction: 'lower' }),
  'dare/iterate': scalar({ description: 'doubling steps taken' }),
  'dare/solution': tensor({ role: 'state', shape: ['n', 'n'], description: 'the current H, flat (it converges to X)' }),
})

/**
 * The state of {@link solveDareAlgorithm}: the doubling iterates A_k, G_k, H_k (H_k converges to the solution) and the
 * relative change of the last step. Under `grad` the matrices carry the trace; they serialise when concrete.
 */
export interface DareState {
  readonly ak: Value
  readonly gk: Value
  readonly hk: Value
  /** The relative change of H in the last step (Infinity before the first). */
  readonly change: number
  readonly iterations: number
}

/** Options of {@link solveDareAlgorithm}. */
export interface DareOptions {
  /** Stop when H changes by less than this relative amount (default 1e-14). */
  readonly tol?: number
  /** Most doubling steps (default 100). */
  readonly maxIterations?: number
}

/**
 * The structure-preserving doubling algorithm for the discrete algebraic Riccati equation as a stepped algorithm: each
 * step squares the pair (A_k, G_k, H_k) -> (A_k W^-1 A_k, G_k + A_k W^-1 G_k A_k^T, H_k + A_k^T H_k W^-1 A_k) with
 * W = I + G_k H_k, so the horizon doubles and H_k reaches the stabilising solution quadratically. It logs the relative
 * change, the iteration and the iterate, and is done (`converged`) when the change is below `tol`, `failed` after
 * `maxIterations`. Differentiable like `solveDare`; the state serialises when its matrices are concrete.
 *
 * @param a - `[n, n]`.
 * @param b - `[n, m]`.
 * @param q - `[n, n]`.
 * @param r - `[m, m]`.
 * @param options - Tolerance and iteration cap.
 * @returns The algorithm.
 * @example
 * ```ts
 * import { solveDareAlgorithm } from 'omai-engine/numerics/linalg'
 * import { runAlgorithm } from 'omai-engine/trace'
 * runAlgorithm(solveDareAlgorithm([[1]], [[1]], [[1]], [[1]]), { maxSteps: 100 }).status // => 'converged'
 * ```
 */
export function solveDareAlgorithm(
  a: TensorLike,
  b: TensorLike,
  q: TensorLike,
  r: TensorLike,
  options: DareOptions = {},
): Algorithm<DareState, EquationPath> {
  const [av, bv, qv, rv] = promoteFloat(a, b, q, r) as [Value, Value, Value, Value]
  const n = squareSize(av.shape, 'solveDare')
  const tol = options.tol ?? 1e-14
  const maxIterations = options.maxIterations ?? 100
  return {
    name: 'numerics/dare',
    config: { n, tol, maxIterations },
    init: () => ({
      ak: av,
      gk: matmul(bv, solve(rv, mT(bv))),
      hk: qv,
      change: Number.POSITIVE_INFINITY,
      iterations: 0,
    }),
    step: (s, ctx) => {
      const eye = identity(n, av)
      const { ak, gk, hk } = s
      const w = add(eye, matmul(gk, hk))
      const wa = solve(w, ak)
      const nextA = matmul(ak, wa)
      const nextG = add(gk, matmul(ak, solve(w, matmul(gk, mT(ak)))))
      const nextH = add(hk, matmul(mT(ak), matmul(hk, wa)))
      const change = norm1(sub(nextH, hk)) / Math.max(norm1(nextH), 1e-300)
      ctx.log('dare/residual', change)
      ctx.log('dare/iterate', s.iterations + 1)
      if (ctx.wants('dare/solution')) ctx.log('dare/solution', toFlat(primalOf(nextH)))
      return { ak: nextA, gk: nextG, hk: nextH, change, iterations: s.iterations + 1 }
    },
    done: (s) => (s.change <= tol ? 'converged' : s.iterations >= maxIterations ? 'failed' : undefined),
    serialise: (s) => ({
      ak: stateToJson(s.ak),
      gk: stateToJson(s.gk),
      hk: stateToJson(s.hk),
      change: numberToJson(s.change),
      iterations: s.iterations,
    }),
    deserialise: (j) => {
      const o = j as unknown as Record<string, Json>
      return {
        ak: stateFromJson(o['ak'] as Json) as Value,
        gk: stateFromJson(o['gk'] as Json) as Value,
        hk: stateFromJson(o['hk'] as Json) as Value,
        change: numberFromJson(o['change']),
        iterations: o['iterations'] as number,
      }
    },
  }
}

/**
 * Solve the discrete algebraic Riccati equation X = A^T X A - A^T X B (R + B^T X B)^{-1} B^T X A + Q for the
 * stabilising X (scipy.linalg.solve_discrete_are), by the structure-preserving doubling algorithm
 * ({@link solveDareAlgorithm} run to the end); differentiable.
 *
 * @param a - `[n, n]`.
 * @param b - `[n, m]`.
 * @param q - `[n, n]`.
 * @param r - `[m, m]`.
 * @returns X (symmetric).
 * @throws OmaiError with code `convergence` when the doubling does not converge (the return is a bare matrix; use
 *   {@link solveDareAlgorithm} for a run that ends with a status).
 * @example
 * ```ts
 * import { solveDare } from 'omai-engine/numerics/linalg'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(solveDare([[1]], [[1]], [[1]], [[1]])) - (1 + Math.sqrt(5)) / 2) < 1e-12 // => true
 * ```
 */
export function solveDare(a: TensorLike, b: TensorLike, q: TensorLike, r: TensorLike): Value {
  const run = runAlgorithm(solveDareAlgorithm(a, b, q, r), { maxSteps: 100 })
  if (run.status !== 'converged')
    throw new OmaiError('convergence', 'solveDare: the doubling iteration did not converge')
  return sym(run.state.hk)
}
