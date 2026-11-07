// The statements of the six SVM duals as the quadratic programs the SMO solvers take: the signed Gram matrix, the linear
// term, the labels, the bounds and the feasible start of each model. `optim`'s run-to-the-end functions (`svcDual` and
// the others of `smo.ts` and `smo-nu.ts`) and `svm`'s steppable duals both build their problem here, so a fix to a
// start or a bound reaches both. Not exported from the barrel.

import { fromStorage, type Tensor } from '../tensor/index.ts'
import type { SmoNuProblem } from './smo-nu.ts'
import type { SmoProblem } from './smo.ts'

/** Row-major m x m matrix `entry(i, j)` as a tensor. */
function matrixOf(m: number, entry: (i: number, j: number) => number): Tensor {
  const data = new Float64Array(m * m)
  for (let i = 0; i < m; i++) for (let j = 0; j < m; j++) data[i * m + j] = entry(i, j)
  return fromStorage(data, [m, m], 'float64')
}

/** The labels of the 2n variables (alpha, alpha*) of a regression dual: +1 for the first block, -1 for the second. */
function regressionLabels(n: number): Float64Array {
  return Float64Array.from({ length: 2 * n }, (_, i) => (i < n ? 1 : -1))
}

/** The Gram matrix of the regression dual over (alpha, alpha*): `y_i y_j K` on the indices taken modulo n. */
function regressionGram(K: Float64Array, n: number, y: Float64Array): Tensor {
  return matrixOf(2 * n, (i, j) => (y[i] ?? 1) * (y[j] ?? 1) * (K[(i % n) * n + (j % n)] ?? 0))
}

/** The number of samples of a flat square Gram matrix. */
function sampleCount(K: Float64Array): number {
  return Math.round(Math.sqrt(K.length))
}

/** A feasible start filling the variables in order up to `bound` until `total` is placed (one-class and SVDD). */
function fillStart(n: number, total: number, bound: number): Float64Array {
  const alpha = new Float64Array(n)
  let remaining = total
  for (let i = 0; i < n && remaining > 0; i++) {
    alpha[i] = Math.min(bound, remaining)
    remaining -= alpha[i] ?? 0
  }
  return alpha
}

/**
 * The C-SVC dual: minimise 1/2 a^T (y y^T * K) a - 1^T a subject to y^T a = 0 and 0 <= a_i <= C_i.
 *
 * @param K - The flat n x n Gram matrix.
 * @param y - Labels, +1 or -1.
 * @param C - The penalty: one, or one per sample.
 * @returns The problem for {@link smoAlgorithm}.
 */
export function svcProblem(K: Float64Array, y: Float64Array, C: SmoProblem['C']): SmoProblem {
  const n = y.length
  return {
    Q: matrixOf(n, (i, j) => (y[i] ?? 1) * (y[j] ?? 1) * (K[i * n + j] ?? 0)),
    p: new Float64Array(n).fill(-1),
    y,
    C,
  }
}

/**
 * The epsilon-SVR dual over the 2n variables (alpha, alpha*): Q = [K, -K; -K, K], p = [epsilon - t; epsilon + t].
 *
 * @param K - The flat n x n Gram matrix.
 * @param t - The regression targets.
 * @param C - The penalty: one, or one per sample.
 * @param epsilon - Half-width of the insensitive tube.
 * @returns The problem for {@link smoAlgorithm}.
 */
export function svrProblem(K: Float64Array, t: Float64Array, C: number | Float64Array, epsilon: number): SmoProblem {
  const n = t.length
  const y = regressionLabels(n)
  return {
    Q: regressionGram(K, n, y),
    p: Float64Array.from({ length: 2 * n }, (_, i) => (i < n ? epsilon - (t[i] ?? 0) : epsilon + (t[i - n] ?? 0))),
    y,
    C: Float64Array.from({ length: 2 * n }, (_, i) => (typeof C === 'number' ? C : (C[i % n] ?? 0))),
  }
}

/**
 * The one-class dual: minimise 1/2 a^T K a subject to sum a = nu n and 0 <= a_i <= 1, from the start that fills the
 * variables in order.
 *
 * @param K - The flat n x n Gram matrix.
 * @param nu - In (0, 1].
 * @returns The problem for {@link smoAlgorithm}.
 */
export function oneClassProblem(K: Float64Array, nu: number): SmoProblem {
  const n = sampleCount(K)
  return {
    Q: fromStorage(Float64Array.from(K), [n, n], 'float64'),
    p: new Float64Array(n),
    y: new Float64Array(n).fill(1),
    C: 1,
    alpha0: fillStart(n, nu * n, 1),
  }
}

/**
 * The SVDD dual: minimise a^T K a - sum a_i K_ii subject to sum a = 1 and 0 <= a_i <= C (Q = 2K, p = -diag K), from the
 * start that fills the variables in order.
 *
 * @param K - The flat n x n Gram matrix.
 * @param C - The penalty, at least 1/n.
 * @returns The problem for {@link smoAlgorithm}.
 */
export function svddProblem(K: Float64Array, C: number): SmoProblem {
  const n = sampleCount(K)
  return {
    Q: fromStorage(
      Float64Array.from(K, (v) => 2 * v),
      [n, n],
      'float64',
    ),
    p: Float64Array.from({ length: n }, (_, i) => -(K[i * n + i] ?? 0)),
    y: new Float64Array(n).fill(1),
    C,
    alpha0: fillStart(n, 1, C),
  }
}

/**
 * The nu-SVC dual of libsvm's Solver_NU: the C-SVC matrix with no linear term, bound 1 and the given start.
 *
 * @param K - The flat n x n Gram matrix.
 * @param y - Labels, +1 or -1.
 * @param alpha0 - The feasible start (`nuSvcStart`).
 * @returns The problem for {@link nuSmoAlgorithm}.
 */
export function nuSvcProblem(K: Float64Array, y: Float64Array, alpha0: Float64Array): SmoNuProblem {
  const n = y.length
  return {
    Q: matrixOf(n, (i, j) => (y[i] ?? 1) * (y[j] ?? 1) * (K[i * n + j] ?? 0)),
    p: new Float64Array(n),
    y,
    C: 1,
    alpha0,
  }
}

/**
 * The nu-SVR dual over (alpha, alpha*): the regression matrix, p = [-t; t], the common bound C and the given start.
 *
 * @param K - The flat n x n Gram matrix.
 * @param t - The regression targets.
 * @param C - The penalty.
 * @param alpha0 - The feasible start (`nuSvrStart`).
 * @returns The problem for {@link nuSmoAlgorithm}.
 */
export function nuSvrProblem(K: Float64Array, t: Float64Array, C: number, alpha0: Float64Array): SmoNuProblem {
  const n = t.length
  const y = regressionLabels(n)
  return {
    Q: regressionGram(K, n, y),
    p: Float64Array.from({ length: 2 * n }, (_, i) => (i < n ? -(t[i] ?? 0) : (t[i - n] ?? 0))),
    y,
    C,
    alpha0,
  }
}
