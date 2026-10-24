import { WhiteboxError } from '../../base/index.ts'
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
  type Value,
} from '../../tensor/index.ts'
import { identity, mT, squareSize, sym } from './_matrix.ts'
import { slogdet } from './basics.ts'
import { inv, solve } from './lu.ts'
import { lstsq } from './qr.ts'

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
 * import { solveSylvester } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { solveContinuousLyapunov } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * import { solveDiscreteLyapunov } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
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
 * @throws WhiteboxError with code `convergence` when the iteration does not settle (eigenvalues on or near the axis).
 * @concept matrix-functions
 * @example
 * ```ts
 * import { signm } from 'whitebox-engine/numerics/linalg'
 * import { toArray } from 'whitebox-engine/tensor'
 * toArray(signm([[-2, 0], [0, 3]])) // => [[-1, 0], [0, 1]]
 * ```
 */
export function signm(a: TensorLike): Value {
  const [z0] = promoteFloat(a) as [Value]
  const n = squareSize(z0.shape, 'signm')
  let z = z0
  for (let i = 0; i < 100; i++) {
    // Determinant scaling c = |det Z|^{-1/n} speeds the early iterations.
    const c = Math.exp(-item(primalOf(slogdet(z).logabsdet)) / n)
    const next = mul(add(mul(z, c), div(inv(z), c)), 0.5)
    const change = norm1(sub(next, z)) / Math.max(norm1(next), 1e-300)
    z = next
    if (change <= 1e-14) return z
  }
  throw new WhiteboxError('convergence', 'signm: the Newton iteration did not converge')
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
 * @concept matrix-equations
 * @example
 * ```ts
 * import { solveCare } from 'whitebox-engine/numerics/linalg'
 * import { item } from 'whitebox-engine/tensor'
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
    if (e instanceof WhiteboxError && e.code === 'convergence') {
      throw new WhiteboxError('convergence', 'solveCare: the sign iteration did not converge')
    }
    throw e
  }
  const block = (r0: number, c0: number): Value => sliceAxis(sliceAxis(z, 0, r0, r0 + n), 1, c0, c0 + n)
  const eye = identity(n, av)
  // [W12; W22 + I] X = -[W11 + I; W21].
  const lhs = concat([block(0, n), add(block(n, n), eye)], 0)
  const rhs = neg(concat([add(block(0, 0), eye), block(n, 0)], 0))
  return sym(lstsq(lhs, rhs))
}

/**
 * Solve the discrete algebraic Riccati equation X = A^T X A - A^T X B (R + B^T X B)^{-1} B^T X A + Q for the
 * stabilising X (scipy.linalg.solve_discrete_are), by the structure-preserving doubling algorithm; differentiable.
 *
 * @param a - `[n, n]`.
 * @param b - `[n, m]`.
 * @param q - `[n, n]`.
 * @param r - `[m, m]`.
 * @returns X (symmetric).
 * @example
 * ```ts
 * import { solveDare } from 'whitebox-engine/numerics/linalg'
 * import { item } from 'whitebox-engine/tensor'
 * Math.abs(item(solveDare([[1]], [[1]], [[1]], [[1]])) - (1 + Math.sqrt(5)) / 2) < 1e-12 // => true
 * ```
 */
export function solveDare(a: TensorLike, b: TensorLike, q: TensorLike, r: TensorLike): Value {
  const [av, bv, qv, rv] = promoteFloat(a, b, q, r) as [Value, Value, Value, Value]
  const n = squareSize(av.shape, 'solveDare')
  const eye = identity(n, av)
  let ak = av
  let gk = matmul(bv, solve(rv, mT(bv)))
  let hk = qv
  for (let i = 0; i < 100; i++) {
    const w = add(eye, matmul(gk, hk))
    const wa = solve(w, ak)
    const nextA = matmul(ak, wa)
    const nextG = add(gk, matmul(ak, solve(w, matmul(gk, mT(ak)))))
    const nextH = add(hk, matmul(mT(ak), matmul(hk, wa)))
    const change = norm1(sub(nextH, hk)) / Math.max(norm1(nextH), 1e-300)
    ak = nextA
    gk = nextG
    hk = nextH
    if (change <= 1e-14) return sym(hk)
  }
  throw new WhiteboxError('convergence', 'solveDare: the doubling iteration did not converge')
}
