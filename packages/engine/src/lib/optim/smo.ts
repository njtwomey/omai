/**
 * Sequential minimal optimisation (SMO, Platt; libsvm's solver): the dual quadratic program of a support vector machine
 * solved by repeatedly optimising a pair of variables that violate the optimality conditions, as a traced algorithm.
 * Includes the C-SVC and epsilon-SVR duals, the one-class and support vector data description duals, which return the
 * solution as a kernel expansion.
 */

import { MdmlError } from '../base/index.ts'
import type { Tensor, TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { matrixFrom, vectorFrom, vectorTensor } from './_programming.ts'
import { oneClassProblem, svcProblem, svddProblem, svrProblem } from './_smo-problems.ts'
import { TAU, updatePair } from './_smo-update.ts'

/**
 * The quadratic program SMO solves (libsvm's general form): minimise 1/2 alpha^T Q alpha + p^T alpha subject to
 * y^T alpha = delta and 0 <= alpha_i <= C_i, with labels y_i = +1 or -1. The equality's value delta is that of the
 * starting point.
 */
export interface SmoProblem {
  /** Symmetric positive semidefinite n x n matrix (Q_ij = y_i y_j K_ij for the SVM dual). */
  readonly Q: TensorLike
  readonly p: TensorLike
  /** Labels, +1 or -1. */
  readonly y: TensorLike
  /** Upper bounds: one for all, or one per variable. */
  readonly C: number | TensorLike
  /** A feasible start (default 0). */
  readonly alpha0?: TensorLike
}

/** Options of {@link smoAlgorithm} and {@link smo}. */
export interface SmoOptions extends Partial<Omit<RunOptions, 'maxSteps' | 'start'>> {
  /** Stop when the maximal violation m(alpha) - M(alpha) falls below this (libsvm's eps, default 1e-3). */
  readonly tol?: number
  /** Working-set selection: second order (Fan, Chen and Lin's WSS3, default) or the maximal violating pair. */
  readonly selection?: 'second-order' | 'first-order'
  /** Temporarily remove bounded variables unlikely to change (default true). */
  readonly shrinking?: boolean
  readonly maxIterations?: number
}

/** The state of {@link smoAlgorithm}. */
export interface SmoState {
  readonly alpha: Float64Array
  /** The gradient Q alpha + p (current on the active set; complete after a reconstruction). */
  readonly grad: Float64Array
  /** Variables currently considered (all unless shrunk). */
  readonly active: readonly number[]
  /** m(alpha) - M(alpha), the maximal KKT violation (at the last selection). */
  readonly violation: number
  readonly iterations: number
  /** Iterations until the next shrinking pass. */
  readonly counter: number
  readonly unshrunk: boolean
  /** The last working pair. */
  readonly pair?: readonly [number, number]
  readonly finished: boolean
}

function squareFrom(x: TensorLike, name: string): { readonly data: Float64Array; readonly n: number } {
  const t = vectorFrom(x)
  const n = Math.round(Math.sqrt(t.length))
  if (n * n !== t.length) throw new MdmlError('shape', `${name} must be square`)
  return { data: t, n }
}

/** Result of {@link smo}. */
export interface SmoResult {
  readonly alpha: Tensor
  /** libsvm's rho: the decision function is sum_i alpha_i y_i K(x_i, x) - rho. */
  readonly rho: number
  /** -rho, the intercept. */
  readonly bias: number
  readonly objective: number
  readonly iterations: number
  readonly converged: boolean
  /** Indices with alpha > 0. */
  readonly support: readonly number[]
}

/**
 * Sequential minimal optimisation (Platt; libsvm's solver) as a traced algorithm: each step picks a working pair (i, j)
 * that violates the KKT conditions, by maximal violation for i and, for j, the largest second-order decrease of the
 * objective (Fan, Chen and Lin 2005, WSS3) or the maximal violating pair; solves the two-variable subproblem in closed
 * form, clipped to the box and the equality; and updates the gradient. Shrinking periodically drops bounded variables
 * whose gradients make them unlikely to move, and the gradient is reconstructed before declaring optimality. Records
 * the objective, the violation, the active-set size and the support size, and each working pair.
 *
 * @param problem - Q, p, labels, bounds and start.
 * @param options - Tolerance, selection rule and shrinking.
 * @returns The algorithm.
 * @concept support-vector-machines
 * @reference R.-E. Fan, P.-H. Chen and C.-J. Lin, Working set selection using second order information for training
 * support vector machines (2005).
 * @example
 * ```ts
 * import { smoAlgorithm } from 'mdml-engine/optim'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const alg = smoAlgorithm({ Q: [[1, 0], [0, 1]], p: [-1, -1], y: [1, -1], C: 10 })
 * Array.from(runAlgorithm(alg, { maxSteps: 10 }).state.alpha) // => [1, 1]
 * ```
 */
export function smoAlgorithm(problem: SmoProblem, options: SmoOptions = {}): Algorithm<SmoState> {
  const p = vectorFrom(problem.p)
  const n = p.length
  const Q = matrixFrom(problem.Q, n, 'Q').data
  const y = vectorFrom(problem.y)
  if (y.length !== n || y.some((v) => v !== 1 && v !== -1)) throw new MdmlError('value', 'smo: y must be +1 or -1')
  const C = typeof problem.C === 'number' ? new Float64Array(n).fill(problem.C) : vectorFrom(problem.C)
  const eps = options.tol ?? 1e-3
  const secondOrder = options.selection !== 'first-order'
  const shrinking = options.shrinking ?? true
  const QD = Float64Array.from({ length: n }, (_, i) => Q[i * n + i] ?? 0)
  const upper = (a: Float64Array, i: number): boolean => (a[i] ?? 0) >= (C[i] ?? 0)
  const lower = (a: Float64Array, i: number): boolean => (a[i] ?? 0) <= 0
  const fullGradient = (a: Float64Array): Float64Array => {
    const g = Float64Array.from(p)
    for (let j = 0; j < n; j++) {
      const aj = a[j] ?? 0
      if (aj === 0) continue
      for (let i = 0; i < n; i++) g[i] = (g[i] ?? 0) + (Q[i * n + j] ?? 0) * aj
    }
    return g
  }
  const select = (
    a: Float64Array,
    g: Float64Array,
    active: readonly number[],
  ): { i: number; j: number; violation: number } => {
    let gmax = -Infinity
    let gmax2 = -Infinity
    let iBest = -1
    let jBest = -1
    let best = Infinity
    for (const t of active) {
      if ((y[t] ?? 1) === 1) {
        if (!upper(a, t) && -(g[t] ?? 0) >= gmax) {
          gmax = -(g[t] ?? 0)
          iBest = t
        }
      } else if (!lower(a, t) && (g[t] ?? 0) >= gmax) {
        gmax = g[t] ?? 0
        iBest = t
      }
    }
    const i = iBest
    for (const t of active) {
      const yt = y[t] ?? 1
      if (yt === 1 ? lower(a, t) : upper(a, t)) continue
      const gt = yt === 1 ? (g[t] ?? 0) : -(g[t] ?? 0)
      if (gt >= gmax2) gmax2 = gt
      const diff = gmax + gt
      if (diff <= 0 || i < 0) continue
      let score: number
      if (secondOrder) {
        const quad = (QD[i] ?? 0) + (QD[t] ?? 0) - 2 * (y[i] ?? 1) * yt * (Q[i * n + t] ?? 0)
        score = -(diff * diff) / (quad > 0 ? quad : TAU)
      } else score = -gt
      if (score <= best) {
        best = score
        jBest = t
      }
    }
    return { i, j: jBest, violation: gmax + gmax2 }
  }
  const shrink = (s: SmoState): SmoState => {
    let g1 = -Infinity
    let g2 = -Infinity
    for (const t of s.active) {
      const gt = s.grad[t] ?? 0
      if ((y[t] ?? 1) === 1) {
        if (!upper(s.alpha, t)) g1 = Math.max(g1, -gt)
        if (!lower(s.alpha, t)) g2 = Math.max(g2, gt)
      } else {
        if (!upper(s.alpha, t)) g1 = Math.max(g1, gt)
        if (!lower(s.alpha, t)) g2 = Math.max(g2, -gt)
      }
    }
    let grad = s.grad
    let active = s.active
    let unshrunk = s.unshrunk
    if (!unshrunk && g1 + g2 <= eps * 10) {
      unshrunk = true
      grad = fullGradient(s.alpha)
      active = Array.from({ length: n }, (_, i) => i)
    }
    const shrunk = (t: number): boolean => {
      const gt = grad[t] ?? 0
      const yt = y[t] ?? 1
      if (upper(s.alpha, t)) return yt === 1 ? -gt > g1 : -gt > g2
      if (lower(s.alpha, t)) return yt === 1 ? gt > g2 : gt > g1
      return false
    }
    return { ...s, grad, active: active.filter((t) => !shrunk(t)), unshrunk }
  }
  return {
    name: 'smo',
    config: { tol: eps, selection: secondOrder ? 'second-order' : 'first-order', shrinking },
    schema: channels({
      metrics: {
        objective: { description: 'dual objective 1/2 a^T Q a + p^T a', direction: 'lower' },
        violation: { description: 'maximal KKT violation m(a) - M(a)' },
        activeSize: { description: 'variables not shrunk' },
        support: { description: 'variables with alpha > 0' },
      },
      tensors: { alpha: { role: 'param', description: 'the dual variables' } },
      messages: { workingSet: { description: 'the working pair (i, j) of a step' } },
    }),
    init: () => {
      const alpha = problem.alpha0 === undefined ? new Float64Array(n) : vectorFrom(problem.alpha0)
      return {
        alpha,
        grad: fullGradient(alpha),
        active: Array.from({ length: n }, (_, i) => i),
        violation: Infinity,
        iterations: 0,
        counter: Math.min(n, 1000) + 1,
        unshrunk: false,
        finished: false,
      }
    },
    step: (state) => {
      let s = state
      let counter = s.counter - 1
      if (counter === 0) {
        counter = Math.min(n, 1000)
        if (shrinking) s = shrink(s)
      }
      let pick = select(s.alpha, s.grad, s.active)
      if (pick.violation < eps || pick.j < 0) {
        // Optimal on the active set: reconstruct the gradient and check every variable.
        const grad = fullGradient(s.alpha)
        const all = Array.from({ length: n }, (_, i) => i)
        s = { ...s, grad, active: all }
        pick = select(s.alpha, grad, all)
        if (pick.violation < eps || pick.j < 0) {
          const { pair: _last, ...rest } = s
          return { ...rest, violation: pick.violation, counter, finished: true }
        }
        counter = 1
      }
      const { i, j } = pick
      const a = Float64Array.from(s.alpha)
      const oldI = a[i] ?? 0
      const oldJ = a[j] ?? 0
      updatePair(
        a,
        i,
        j,
        { yi: y[i] ?? 1, yj: y[j] ?? 1 },
        { gi: s.grad[i] ?? 0, gj: s.grad[j] ?? 0 },
        { qij: Q[i * n + j] ?? 0, qdi: QD[i] ?? 0, qdj: QD[j] ?? 0 },
        { ci: C[i] ?? 0, cj: C[j] ?? 0 },
      )
      const di = (a[i] ?? 0) - oldI
      const dj = (a[j] ?? 0) - oldJ
      const grad = Float64Array.from(s.grad)
      for (const t of s.active) grad[t] = (grad[t] ?? 0) + (Q[t * n + i] ?? 0) * di + (Q[t * n + j] ?? 0) * dj
      return { ...s, alpha: a, grad, violation: pick.violation, iterations: s.iterations + 1, counter, pair: [i, j] }
    },
    done: (s) => (s.finished ? 'converged' : undefined),
    serialise: (s) => stateToJson(s),
    deserialise: (j) => stateFromJson(j) as SmoState,
    observe: (s) => {
      let objective = 0
      for (let i = 0; i < n; i++) objective += 0.5 * (s.alpha[i] ?? 0) * ((s.grad[i] ?? 0) + (p[i] ?? 0))
      return {
        metrics: {
          objective,
          ...(Number.isFinite(s.violation) ? { violation: s.violation } : {}),
          activeSize: s.active.length,
          support: s.alpha.reduce((c, v) => c + (v > 0 ? 1 : 0), 0),
        },
        tensors: { alpha: { role: 'param', value: s.alpha } },
        ...(s.pair === undefined ? {} : { messages: [{ kind: 'workingSet', payload: [s.pair[0], s.pair[1]] }] }),
      }
    },
  }
}

/**
 * libsvm's rho of a state of the dual: the mean of y_i G_i over the free variables, else the middle of the feasible
 * interval. The decision function of a support vector machine is the kernel expansion minus rho.
 *
 * @param alpha - The dual variables.
 * @param grad - The gradient Q alpha + p (complete, as in a finished state).
 * @param y - Labels, +1 or -1.
 * @param C - Upper bounds, one per variable.
 * @returns rho.
 * @concept support-vector-machines
 * @example
 * ```ts
 * import { smoRho } from 'mdml-engine/optim'
 * smoRho(Float64Array.of(0.5, 0.5), Float64Array.of(-0.5, 0.5), Float64Array.of(1, -1), Float64Array.of(1, 1)) // => -0.5
 * ```
 */
export function smoRho(alpha: Float64Array, grad: Float64Array, y: Float64Array, C: Float64Array): number {
  let ub = Infinity
  let lb = -Infinity
  let free = 0
  let sum = 0
  for (let i = 0; i < alpha.length; i++) {
    const yg = (y[i] ?? 1) * (grad[i] ?? 0)
    const a = alpha[i] ?? 0
    if (a >= (C[i] ?? 0)) {
      if ((y[i] ?? 1) === -1) ub = Math.min(ub, yg)
      else lb = Math.max(lb, yg)
    } else if (a <= 0) {
      if ((y[i] ?? 1) === 1) ub = Math.min(ub, yg)
      else lb = Math.max(lb, yg)
    } else {
      free += 1
      sum += yg
    }
  }
  return free > 0 ? sum / free : (ub + lb) / 2
}

/**
 * Solve a box- and equality-constrained QP of the SVM dual family by sequential minimal optimisation.
 *
 * @param problem - Q, p, labels, bounds and start.
 * @param options - Tolerance, selection rule, shrinking, iteration cap (default 10^7 as libsvm, capped by 100 n^2) and
 * run options.
 * @returns alpha, rho and the bias, the objective, iterations and the support.
 * @concept support-vector-machines
 * @example
 * ```ts
 * import { smo } from 'mdml-engine/optim'
 * const r = smo({ Q: [[1, 0], [0, 1]], p: [-1, -1], y: [1, -1], C: 10 })
 * [r.objective, r.rho] // => [-1, 0]
 * ```
 */
export function smo(problem: SmoProblem, options: SmoOptions = {}): SmoResult {
  const { tol: _t, selection: _s, shrinking: _k, maxIterations, ...run } = options
  const n = vectorFrom(problem.p).length
  const r = runAlgorithm(smoAlgorithm(problem, options), {
    ...run,
    maxSteps: maxIterations ?? Math.min(10_000_000, Math.max(10_000, 100 * n * n)),
  })
  const s = r.state
  const p = vectorFrom(problem.p)
  const y = vectorFrom(problem.y)
  const C = typeof problem.C === 'number' ? new Float64Array(n).fill(problem.C) : vectorFrom(problem.C)
  const Q = matrixFrom(problem.Q, n, 'Q').data
  const grad = Float64Array.from(p)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) grad[i] = (grad[i] ?? 0) + (Q[i * n + j] ?? 0) * (s.alpha[j] ?? 0)
  let objective = 0
  for (let i = 0; i < n; i++) objective += 0.5 * (s.alpha[i] ?? 0) * ((grad[i] ?? 0) + (p[i] ?? 0))
  const rho = smoRho(s.alpha, grad, y, C)
  return {
    alpha: vectorTensor(s.alpha),
    rho,
    bias: -rho,
    objective,
    iterations: s.iterations,
    converged: r.status === 'converged',
    support: Array.from(s.alpha).flatMap((v, i) => (v > 0 ? [i] : [])),
  }
}

/** Result of the duals of this file: the dual solution as a kernel expansion. */
export interface SvmDualResult extends SmoResult {
  /** The expansion coefficients (y_i alpha_i for classification, alpha_i - alpha*_i for regression), one per sample. */
  readonly coef: Tensor
}

/**
 * The C-support-vector-classification dual (libsvm's C-SVC): minimise 1/2 a^T (y y^T * K) a - 1^T a subject to
 * y^T a = 0, 0 <= a_i <= C_i, by {@link smo}. The decision function is sum_i coef_i K(x_i, x) + bias.
 *
 * @param kernel - The n x n kernel (Gram) matrix of the training samples.
 * @param labels - +1 or -1.
 * @param C - The penalty: one, or one per sample (class weights scale it).
 * @param options - SMO options.
 * @returns alpha, coef = y alpha, rho, bias and the support.
 * @concept support-vector-machines
 * @example
 * ```ts
 * import { svcDual } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * const K = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
 * toArray(svcDual(K, [1, -1, -1], 10, { tol: 1e-12 }).coef).map((v) => Math.round(v * 1e6) / 1e6) // => [1.333333, -0.666667, -0.666667]
 * ```
 */
export function svcDual(
  kernel: TensorLike,
  labels: TensorLike,
  C: number | TensorLike,
  options: SmoOptions = {},
): SvmDualResult {
  const y = vectorFrom(labels)
  const n = y.length
  const K = matrixFrom(kernel, n, 'kernel').data
  const r = smo(svcProblem(K, y, C), options)
  return { ...r, coef: vectorTensor(Float64Array.from(r.alpha.data, (a, i) => a * (y[i] ?? 1))) }
}

/**
 * The epsilon-support-vector-regression dual (libsvm's epsilon-SVR) over 2n variables (alpha, alpha*): Q is
 * [K, -K; -K, K], p = [epsilon - t; epsilon + t], labels [+1; -1], solved by {@link smo}. The prediction is
 * sum_i coef_i K(x_i, x) + bias with coef = alpha - alpha*.
 *
 * @param kernel - The n x n kernel matrix.
 * @param targets - Regression targets.
 * @param C - The penalty.
 * @param epsilon - Half-width of the insensitive tube.
 * @param options - SMO options.
 * @returns The 2n alphas, coef, rho, bias and the support (in the 2n variables).
 * @concept support-vector-machines
 * @example
 * ```ts
 * import { svrDual } from 'mdml-engine/optim'
 * const r = svrDual([[1, 0], [0, 1]], [1, -1], 10, 0.1)
 * Math.round(r.bias * 1e9) / 1e9 // => 0
 * ```
 */
export function svrDual(
  kernel: TensorLike,
  targets: TensorLike,
  C: number,
  epsilon: number,
  options: SmoOptions = {},
): SvmDualResult {
  const t = vectorFrom(targets)
  const n = t.length
  const K = matrixFrom(kernel, n, 'kernel').data
  const r = smo(svrProblem(K, t, C, epsilon), options)
  const a = r.alpha.data
  return { ...r, coef: vectorTensor(Float64Array.from({ length: n }, (_, i) => (a[i] ?? 0) - (a[i + n] ?? 0))) }
}

/**
 * The one-class support vector machine dual (Schoelkopf et al., libsvm's ONE_CLASS): minimise 1/2 a^T K a subject to
 * sum a = nu n and 0 <= a_i <= 1, by {@link smo}. The score of a point is sum_i a_i K(x_i, x) - rho, positive for
 * points the learnt region contains; at most a fraction nu of the training points lie outside it and at least nu are
 * support vectors.
 *
 * @param kernel - The n x n kernel matrix.
 * @param nu - In (0, 1].
 * @param options - SMO options.
 * @returns alpha, coef = alpha, rho, bias and the support.
 * @throws MdmlError `domain` for nu outside (0, 1].
 * @concept support-vector-machines
 * @reference B. Schoelkopf, J. Platt, J. Shawe-Taylor, A. Smola and R. Williamson, Estimating the support of a
 * high-dimensional distribution (2001).
 * @example
 * ```ts
 * import { oneClassDual } from 'mdml-engine/optim'
 * import { toArray } from 'mdml-engine/tensor'
 * const r = oneClassDual([[1, 0, 0], [0, 1, 0], [0, 0, 1]], 0.5, { tol: 1e-12 })
 * toArray(r.coef).map((v) => Math.round(v * 1e6) / 1e6) // => [0.5, 0.5, 0.5]
 * ```
 */
export function oneClassDual(kernel: TensorLike, nu: number, options: SmoOptions = {}): SvmDualResult {
  const K = squareFrom(kernel, 'kernel')
  if (!(nu > 0 && nu <= 1)) throw new MdmlError('domain', `oneClassDual: nu must be in (0, 1], got ${nu}`)
  const r = smo(oneClassProblem(K.data, nu), options)
  return { ...r, coef: r.alpha }
}

/** Result of {@link svddDual}. */
export interface SvddResult extends SvmDualResult {
  /** The squared radius R^2 of the enclosing ball in feature space. */
  readonly radiusSquared: number
}

/**
 * The support vector data description dual (Tax and Duin): the smallest ball in feature space enclosing the data up
 * to slack, max sum a_i K_ii - a^T K a subject to sum a = 1 and 0 <= a_i <= C, by {@link smo} on Q = 2K,
 * p = -diag(K). The centre is sum a_i phi(x_i); the score of a point is R^2 - |phi(x) - centre|^2 =
 * 2 sum a_i K(x_i, x) - K(x, x) - rho, positive inside the ball.
 *
 * @param kernel - The n x n kernel matrix.
 * @param C - The penalty on points outside the ball, at least 1/n (1 is the hard ball).
 * @param options - SMO options.
 * @returns alpha, coef = alpha, rho, the squared radius and the support.
 * @throws MdmlError `domain` for C < 1/n.
 * @concept support-vector-machines
 * @reference D. Tax and R. Duin, Support vector data description (2004).
 * @example
 * ```ts
 * import { svddDual } from 'mdml-engine/optim'
 * const r = svddDual([[1, 0], [0, 1]], 1, { tol: 1e-12 })
 * [r.alpha.data[0], r.alpha.data[1], r.radiusSquared] // => [0.5, 0.5, 0.5]
 * ```
 */
export function svddDual(kernel: TensorLike, C: number, options: SmoOptions = {}): SvddResult {
  const K = squareFrom(kernel, 'kernel')
  const n = K.n
  if (!(C * n >= 1 - 1e-12)) throw new MdmlError('domain', `svddDual: C must be at least 1/n = ${1 / n}, got ${C}`)
  const r = smo(svddProblem(K.data, C), options)
  let quad = 0
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) quad += (r.alpha.data[i] ?? 0) * (K.data[i * n + j] ?? 0) * (r.alpha.data[j] ?? 0)
  return { ...r, coef: r.alpha, radiusSquared: quad - r.rho }
}
