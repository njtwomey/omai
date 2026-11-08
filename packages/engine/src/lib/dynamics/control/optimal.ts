/**
 * Linear-quadratic optimal control and estimation, as in python-control: the continuous and discrete LQR gains (`lqr`,
 * `dlqr`) that minimise a quadratic state and input cost, the steady-state Kalman estimator gains (`lqe`, `dlqe`), and
 * their combination into an LQG output-feedback controller. Finite-horizon LQR is solved by the backward Riccati
 * recursion or by integrating the Riccati differential equation, giving time-varying gains.
 */

import { OmaiError } from '../../base/index.ts'
import { eigvals, solve, solveCare, solveDare, symmetricPart } from '../../numerics/linalg/index.ts'
import { type StateSpace, ss } from '../../systems/index.ts'
import {
  add,
  asTensor,
  matmul,
  neg,
  primalOf,
  reshape,
  stack,
  sub,
  type TensorLike,
  transpose,
  type Value,
  zeros,
} from '../../tensor/index.ts'
import { numbersOf } from '../ode/_vec.ts'
import { solveIvp } from '../ode/solve-ivp.ts'

/** The result of a regulator design (python-control's (K, S, E)). */
export interface RegulatorSolution {
  /** The gain: u = -K x. */
  readonly K: Value
  /** The Riccati solution (the cost-to-go matrix). */
  readonly S: Value
  /** The closed-loop eigenvalues of A - B K. */
  readonly E: { readonly re: readonly number[]; readonly im: readonly number[] }
}

function closedLoop(a: Value, b: Value, k: Value): { re: number[]; im: number[] } {
  const ev = eigvals(primalOf(sub(a, matmul(b, k))))
  return { re: Array.from(numbersOf(ev.re)), im: Array.from(numbersOf(ev.im)) }
}

function crossTerm(n: number, m: number, cross: TensorLike | undefined): Value {
  return cross === undefined ? zeros([n, m]) : asTensor(cross)
}

/**
 * The continuous-time linear-quadratic regulator (python-control's `lqr`): minimise the integral of
 * x^T Q x + u^T R u + 2 x^T N u subject to x' = A x + B u. S solves the algebraic Riccati equation (with the cross
 * term folded in: A - B R^-1 N^T and Q - N R^-1 N^T) and K = R^-1 (B^T S + N^T). Differentiable in every matrix.
 *
 * @param a - A `[n, n]`.
 * @param b - B `[n, m]`.
 * @param q - Q `[n, n]` (positive semidefinite).
 * @param r - R `[m, m]` (positive definite).
 * @param n - The cross weight N `[n, m]` (default 0).
 * @returns K, S and the closed-loop eigenvalues.
 * @concept linear-quadratic-regulator
 * @index function Continuous-time linear-quadratic regulator.
 * @example
 * ```ts
 * import { control } from 'omai-engine/dynamics'
 * import { item } from 'omai-engine/tensor'
 * // x' = u with q = r = 1: K = 1.
 * Math.abs(item(control.lqr([[0]], [[1]], [[1]], [[1]]).K) - 1) < 1e-10 // => true
 * ```
 */
export function lqr(a: TensorLike, b: TensorLike, q: TensorLike, r: TensorLike, n?: TensorLike): RegulatorSolution {
  const A = asTensor(a)
  const B = asTensor(b)
  const Q = asTensor(q)
  const R = asTensor(r)
  const N = crossTerm(A.shape[0] ?? 0, B.shape[1] ?? 0, n)
  const RiNt = solve(R, transpose(N))
  const S = solveCare(sub(A, matmul(B, RiNt)), B, sub(Q, matmul(N, RiNt)), R)
  const K = solve(R, add(matmul(transpose(B), S), transpose(N)))
  return { K, S, E: closedLoop(A, B, K) }
}

/**
 * The discrete-time linear-quadratic regulator (python-control's `dlqr`): minimise the sum of
 * x^T Q x + u^T R u + 2 x^T N u subject to x[k+1] = A x[k] + B u[k]; K = (R + B^T S B)^-1 (B^T S A + N^T).
 *
 * @param a - A.
 * @param b - B.
 * @param q - Q.
 * @param r - R.
 * @param n - The cross weight N (default 0).
 * @returns K, S and the closed-loop eigenvalues.
 * @concept linear-quadratic-regulator
 * @example
 * ```ts
 * import { control } from 'omai-engine/dynamics'
 * import { item } from 'omai-engine/tensor'
 * Number(item(control.dlqr([[1]], [[1]], [[1]], [[1]]).K).toFixed(12)) // => 0.61803398875
 * ```
 */
export function dlqr(a: TensorLike, b: TensorLike, q: TensorLike, r: TensorLike, n?: TensorLike): RegulatorSolution {
  const A = asTensor(a)
  const B = asTensor(b)
  const Q = asTensor(q)
  const R = asTensor(r)
  const N = crossTerm(A.shape[0] ?? 0, B.shape[1] ?? 0, n)
  const RiNt = solve(R, transpose(N))
  const S = solveDare(sub(A, matmul(B, RiNt)), B, sub(Q, matmul(N, RiNt)), R)
  const Bt = transpose(B)
  const K = solve(add(R, matmul(Bt, matmul(S, B))), add(matmul(Bt, matmul(S, A)), transpose(N)))
  return { K, S, E: closedLoop(A, B, K) }
}

/** The result of an estimator design (python-control's (L, P, E)). */
export interface EstimatorSolution {
  /** The estimator gain. */
  readonly L: Value
  /** The steady-state error covariance. */
  readonly P: Value
  /** The eigenvalues of A - L C. */
  readonly E: { readonly re: readonly number[]; readonly im: readonly number[] }
}

/**
 * The continuous-time linear-quadratic estimator, the steady-state Kalman-Bucy filter (python-control's `lqe`) for
 * x' = A x + G w, y = C x + v with E[w w^T] = QN, E[v v^T] = RN, E[w v^T] = NN: L = (P C^T + G NN) RN^-1, P from the
 * dual Riccati equation.
 *
 * @param a - A.
 * @param g - G (process-noise input).
 * @param c - C.
 * @param qn - QN.
 * @param rn - RN.
 * @param nn - NN (default 0).
 * @returns L, P and the estimator eigenvalues.
 * @concept kalman-filter
 * @example
 * ```ts
 * import { control } from 'omai-engine/dynamics'
 * import { item } from 'omai-engine/tensor'
 * Math.abs(item(control.lqe([[0]], [[1]], [[1]], [[1]], [[1]]).L) - 1) < 1e-10 // => true
 * ```
 */
export function lqe(
  a: TensorLike,
  g: TensorLike,
  c: TensorLike,
  qn: TensorLike,
  rn: TensorLike,
  nn?: TensorLike,
): EstimatorSolution {
  const A = asTensor(a)
  const G = asTensor(g)
  const C = asTensor(c)
  const Q = matmul(G, matmul(asTensor(qn), transpose(G)))
  const NN = nn === undefined ? zeros([Q.shape[0] ?? 0, C.shape[0] ?? 0]) : matmul(G, asTensor(nn))
  const sol = lqr(transpose(A), transpose(C), Q, rn, NN)
  const L = transpose(sol.K)
  return { L, P: sol.S, E: closedLoop(A, L, C) }
}

/**
 * The discrete-time linear-quadratic estimator (python-control's `dlqe`): the steady-state Kalman filter gain
 * L = A P C^T (C P C^T + RN)^-1 of x[k+1] = A x + G w, y = C x + v, with P from the discrete Riccati equation.
 *
 * @param a - A.
 * @param g - G.
 * @param c - C.
 * @param qn - QN.
 * @param rn - RN.
 * @returns L, P and the estimator eigenvalues.
 * @concept kalman-filter
 * @example
 * ```ts
 * import { control } from 'omai-engine/dynamics'
 * control.dlqe([[1]], [[1]], [[1]], [[1]], [[1]]).E.re.length // => 1
 * ```
 */
export function dlqe(a: TensorLike, g: TensorLike, c: TensorLike, qn: TensorLike, rn: TensorLike): EstimatorSolution {
  const A = asTensor(a)
  const G = asTensor(g)
  const C = asTensor(c)
  const Q = matmul(G, matmul(asTensor(qn), transpose(G)))
  const P = solveDare(transpose(A), transpose(C), Q, rn)
  const Ct = transpose(C)
  const L = transpose(solve(add(matmul(C, matmul(P, Ct)), asTensor(rn)), matmul(C, matmul(P, transpose(A)))))
  return { L, P, E: closedLoop(A, L, C) }
}

/**
 * The linear-quadratic-Gaussian controller (python-control's `lqg`): an LQR gain on the Kalman-filter estimate, as
 * the dynamic controller x_hat' = (A - B K - L C) x_hat + L y, u = -K x_hat.
 *
 * @param sys - The plant (continuous or discrete state space).
 * @param qx - State weight Q.
 * @param ru - Input weight R.
 * @param qn - Process-noise covariance (on the inputs, G = B).
 * @param rn - Measurement-noise covariance.
 * @returns The controller as a state space (input y, output u), with the gains.
 * @concept lqg
 * @example
 * ```ts
 * import { control } from 'omai-engine/dynamics'
 * import { ss } from 'omai-engine/systems'
 * control.lqg(ss([[0, 1], [0, 0]], [[0], [1]], [[1, 0]], [[0]]), [[1, 0], [0, 1]], [[1]], [[1]], [[1]]).controller.a.shape // => [2, 2]
 * ```
 */
export function lqg(
  sys: StateSpace,
  qx: TensorLike,
  ru: TensorLike,
  qn: TensorLike,
  rn: TensorLike,
): { readonly controller: StateSpace; readonly K: Value; readonly L: Value } {
  const discrete = sys.dt !== 0
  const reg = discrete ? dlqr(sys.a, sys.b, qx, ru) : lqr(sys.a, sys.b, qx, ru)
  const est = discrete ? dlqe(sys.a, sys.b, sys.c, qn, rn) : lqe(sys.a, sys.b, sys.c, qn, rn)
  const ac = sub(sub(sys.a, matmul(sys.b, reg.K)), matmul(est.L, sys.c))
  const m = sys.b.shape[1] ?? 0
  const p = sys.c.shape[0] ?? 0
  return { controller: ss(ac, est.L, neg(reg.K), zeros([m, p]), { dt: sys.dt }), K: reg.K, L: est.L }
}

/** A finite-horizon LQR solution: time-varying gains and cost-to-go matrices. */
export interface FiniteHorizonLqr {
  /** K_0 .. K_{N-1}: u_k = -K_k x_k. */
  readonly gains: readonly Value[]
  /** P_0 .. P_N (P_N = Qf). */
  readonly costToGo: readonly Value[]
}

/**
 * The backward Riccati recursion of the finite-horizon discrete LQR: P_N = Qf, K_k = (R + B^T P B)^-1 B^T P A,
 * P_k = Q + A^T P (A - B K_k). Built from operations, so the gains are differentiable in the model and weights.
 *
 * @param a - A.
 * @param b - B.
 * @param q - Q.
 * @param r - R.
 * @param qf - Terminal weight Qf.
 * @param horizon - N.
 * @returns The gains and cost-to-go matrices.
 * @throws OmaiError with code `value` for a negative horizon.
 * @concept riccati-equation
 * @example
 * ```ts
 * import { control } from 'omai-engine/dynamics'
 * control.riccatiRecursion([[1]], [[1]], [[1]], [[1]], [[0]], 3).gains.length // => 3
 * ```
 */
export function riccatiRecursion(
  a: TensorLike,
  b: TensorLike,
  q: TensorLike,
  r: TensorLike,
  qf: TensorLike,
  horizon: number,
): FiniteHorizonLqr {
  if (!Number.isInteger(horizon) || horizon < 0) throw new OmaiError('value', 'riccatiRecursion: horizon >= 0')
  const A = asTensor(a)
  const B = asTensor(b)
  const Q = asTensor(q)
  const R = asTensor(r)
  let P: Value = asTensor(qf)
  const gains: Value[] = []
  const costToGo: Value[] = [P]
  const Bt = transpose(B)
  const At = transpose(A)
  for (let k = horizon - 1; k >= 0; k--) {
    const K = solve(add(R, matmul(Bt, matmul(P, B))), matmul(Bt, matmul(P, A)))
    P = add(Q, matmul(At, matmul(P, sub(A, matmul(B, K)))))
    P = symmetricPart(P)
    gains.unshift(K)
    costToGo.unshift(P)
  }
  return { gains, costToGo }
}

/**
 * The finite-horizon continuous LQR by integrating the Riccati differential equation
 * -P' = A^T P + P A - P B R^-1 B^T P + Q backwards from P(T) = Qf; the gain at time t is K(t) = R^-1 B^T P(t).
 *
 * @param a - A.
 * @param b - B.
 * @param q - Q.
 * @param r - R.
 * @param qf - Qf.
 * @param horizon - T.
 * @param times - Times in [0, T] at which to report P (default [0]).
 * @returns P(t) `[nt, n, n]` and K(t) `[nt, m, n]`.
 * @concept riccati-equation
 * @example
 * ```ts
 * import { control } from 'omai-engine/dynamics'
 * import { item } from 'omai-engine/tensor'
 * // Over a long horizon P(0) approaches the algebraic solution (1 for x' = u, q = r = 1).
 * Math.abs(item(control.riccatiOde([[0]], [[1]], [[1]], [[1]], [[0]], 20).P) - 1) < 1e-6 // => true
 * ```
 */
export function riccatiOde(
  a: TensorLike,
  b: TensorLike,
  q: TensorLike,
  r: TensorLike,
  qf: TensorLike,
  horizon: number,
  times: readonly number[] = [0],
): { readonly P: Value; readonly K: Value } {
  const A = asTensor(a)
  const B = asTensor(b)
  const Q = asTensor(q)
  const R = asTensor(r)
  const n = A.shape[0] ?? 0
  const BRB = matmul(B, solve(R, transpose(B)))
  const field = (_t: number, p: Value): Value => {
    const P = reshape(p, [n, n])
    const rhs = add(sub(add(matmul(transpose(A), P), matmul(P, A)), matmul(P, matmul(BRB, P))), Q)
    // dP/dt = -(A^T P + P A - P B R^-1 B^T P + Q)
    return reshape(neg(rhs), [n * n])
  }
  const ordered = [...times].sort((x, y) => y - x)
  const sol = solveIvp(
    field,
    [horizon, Math.min(0, ordered[ordered.length - 1] ?? 0)],
    reshape(asTensor(qf), [n * n]),
    {
      method: 'Radau',
      rtol: 1e-10,
      atol: 1e-12,
      tEval: ordered,
    },
  )
  const byTime = new Map(sol.t.map((t, i) => [t, sol.ys[i] as Value]))
  const Ps = times.map((t) => reshape(byTime.get(t) ?? (sol.ys.at(-1) as Value), [n, n]))
  const Ks = Ps.map((P) => solve(R, matmul(transpose(B), P)))
  return { P: Ps.length === 1 ? (Ps[0] as Value) : stack(Ps), K: Ks.length === 1 ? (Ks[0] as Value) : stack(Ks) }
}
