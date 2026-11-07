/**
 * The nu variants of the support vector machine duals (libsvm's Solver_NU): the same sequential minimal optimisation
 * with two equality constraints, so that the working pair is chosen within one class and both sums are preserved.
 * Includes the nu-SVC and nu-SVR duals, which return the solution as a kernel expansion like their C counterparts.
 */

import { MdmlError } from '../base/index.ts'
import type { TensorLike } from '../tensor/index.ts'
import { type Algorithm, channels, runAlgorithm, stateFromJson, stateToJson } from '../trace/index.ts'
import { finiteVectorFrom, matrixFrom, vectorFrom, vectorTensor } from './_programming.ts'
import { nuSvcProblem, nuSvrProblem } from './_smo-problems.ts'
import { TAU, updatePair } from './_smo-update.ts'
import type { SmoOptions, SmoState, SvmDualResult } from './smo.ts'

/**
 * The program Solver_NU solves: minimise 1/2 alpha^T Q alpha + p^T alpha subject to 0 <= alpha_i <= C, y^T alpha = delta
 * and sum_{y_i = +1} alpha_i = sum_{y_i = -1} alpha_i, starting from a feasible `alpha0`.
 */
export interface SmoNuProblem {
  /** Symmetric positive semidefinite matrix (signed: Q_ij = y_i y_j K_ij for the SVM duals). */
  readonly Q: TensorLike
  readonly p: TensorLike
  /** Labels, +1 or -1. */
  readonly y: TensorLike
  /** The common upper bound. */
  readonly C: number
  /** A feasible start (see {@link nuSvcStart} and {@link nuSvrStart}). */
  readonly alpha0: TensorLike
}

/**
 * The feasible start of the nu-SVC dual (libsvm): each class gets total weight nu n / 2, filled in order up to the
 * bound 1.
 *
 * @param labels - +1 or -1.
 * @param nu - In (0, 1], at most 2 min(n+, n-) / n for the problem to be feasible.
 * @returns The start alpha0.
 * @throws MdmlError `domain` when nu is infeasible for the class sizes.
 * @concept support-vector-machines
 * @example
 * ```ts
 * import { nuSvcStart } from 'mdml-engine/optim'
 * Array.from(nuSvcStart([1, 1, -1, -1], 0.5)) // => [1, 0, 1, 0]
 * ```
 */
export function nuSvcStart(labels: TensorLike, nu: number): Float64Array {
  const y = vectorFrom(labels)
  const n = y.length
  const positives = y.reduce((c, v) => c + (v > 0 ? 1 : 0), 0)
  if (!(nu > 0 && nu <= 1)) throw new MdmlError('domain', `nuSvc: nu must be in (0, 1], got ${nu}`)
  if (nu * n > 2 * Math.min(positives, n - positives) + 1e-9)
    throw new MdmlError('domain', `nuSvc: nu = ${nu} is infeasible for ${positives} and ${n - positives} samples`)
  let sumPos = (nu * n) / 2
  let sumNeg = (nu * n) / 2
  const alpha = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    if ((y[i] ?? 1) > 0) {
      alpha[i] = Math.min(1, sumPos)
      sumPos -= alpha[i] ?? 0
    } else {
      alpha[i] = Math.min(1, sumNeg)
      sumNeg -= alpha[i] ?? 0
    }
  }
  return alpha
}

/**
 * The feasible start of the nu-SVR dual over 2n variables (libsvm): the same weight min(C n nu / 2, C) filled into
 * alpha and alpha* alike.
 *
 * @param n - Samples.
 * @param C - The penalty.
 * @param nu - In (0, 1].
 * @returns The start alpha0 `[2n]`.
 * @concept support-vector-machines
 * @example
 * ```ts
 * import { nuSvrStart } from 'mdml-engine/optim'
 * Array.from(nuSvrStart(2, 1, 1)) // => [1, 0, 1, 0]
 * ```
 */
export function nuSvrStart(n: number, C: number, nu: number): Float64Array {
  if (!(nu > 0 && nu <= 1)) throw new MdmlError('domain', `nuSvr: nu must be in (0, 1], got ${nu}`)
  let sum = (C * nu * n) / 2
  const alpha = new Float64Array(2 * n)
  for (let i = 0; i < n; i++) {
    const a = Math.min(sum, C)
    alpha[i] = a
    alpha[i + n] = a
    sum -= a
  }
  return alpha
}

/** r1 and r2 of libsvm's calculate_rho for the nu solver: the offsets of the two classes. */
function classOffsets(alpha: Float64Array, grad: Float64Array, y: Float64Array, C: number): readonly [number, number] {
  const free = [0, 0]
  const upper = [Infinity, Infinity]
  const lower = [-Infinity, -Infinity]
  const sum = [0, 0]
  for (let i = 0; i < alpha.length; i++) {
    const c = (y[i] ?? 1) > 0 ? 0 : 1
    const a = alpha[i] ?? 0
    const g = grad[i] ?? 0
    if (a >= C) lower[c] = Math.max(lower[c] ?? -Infinity, g)
    else if (a <= 0) upper[c] = Math.min(upper[c] ?? Infinity, g)
    else {
      free[c] = (free[c] ?? 0) + 1
      sum[c] = (sum[c] ?? 0) + g
    }
  }
  const offset = (c: number): number =>
    (free[c] ?? 0) > 0 ? (sum[c] ?? 0) / (free[c] ?? 1) : ((upper[c] ?? 0) + (lower[c] ?? 0)) / 2
  return [offset(0), offset(1)]
}

/**
 * The offsets of a state of the nu solver: libsvm's rho = (r1 - r2) / 2 (the decision function is the expansion minus
 * rho, before the rescaling of nu-SVC) and the margin r = (r1 + r2) / 2.
 *
 * @param alpha - The dual variables.
 * @param grad - The gradient Q alpha + p.
 * @param y - Labels.
 * @param C - The common upper bound.
 * @returns The offset rho and the margin r.
 * @concept support-vector-machines
 * @example
 * ```ts
 * import { nuRho } from 'mdml-engine/optim'
 * nuRho(Float64Array.of(0.5, 0.5), Float64Array.of(1, 0.5), Float64Array.of(1, -1), 1) // => { rho: 0.25, margin: 0.75 }
 * ```
 */
export function nuRho(
  alpha: Float64Array,
  grad: Float64Array,
  y: Float64Array,
  C: number,
): { readonly rho: number; readonly margin: number } {
  const [r1, r2] = classOffsets(alpha, grad, y, C)
  return { rho: (r1 - r2) / 2, margin: (r1 + r2) / 2 }
}

/**
 * The nu solver (libsvm's Solver_NU) as a traced algorithm: each step picks the class whose maximal violator and the
 * second-order best partner give the largest decrease, and optimises that pair, which shares a class, so that both
 * equalities hold throughout. The state is that of {@link smoAlgorithm} (no shrinking, so the gradient is always
 * complete); it stops when the larger of the two classes' violations falls below `tol`.
 *
 * @param problem - Q, p, labels, bound and feasible start.
 * @param options - Tolerance (default 1e-3).
 * @returns The algorithm.
 * @throws MdmlError with code `value` when p or Q is NaN or infinite (bounds may be infinite, not NaN).
 * @concept support-vector-machines
 * @reference C.-C. Chang and C.-J. Lin, Training nu-support vector classifiers: theory and algorithms (2001).
 * @example
 * ```ts
 * import { nuSmoAlgorithm, nuSvcStart } from 'mdml-engine/optim'
 * import { runAlgorithm } from 'mdml-engine/trace'
 * const y = [1, 1, -1, -1]
 * const K = [[1, 0.5, 0.1, 0], [0.5, 1, 0, 0.1], [0.1, 0, 1, 0.5], [0, 0.1, 0.5, 1]]
 * const Q = K.map((row, i) => row.map((v, j) => v * (y[i] as number) * (y[j] as number)))
 * const alg = nuSmoAlgorithm({ Q, p: [0, 0, 0, 0], y, C: 1, alpha0: nuSvcStart(y, 0.5) })
 * runAlgorithm(alg, { maxSteps: 100 }).status // => 'converged'
 * ```
 */
export function nuSmoAlgorithm(problem: SmoNuProblem, options: Pick<SmoOptions, 'tol'> = {}): Algorithm<SmoState> {
  const p = finiteVectorFrom(problem.p, 'p')
  const n = p.length
  const Q = matrixFrom(problem.Q, n, 'Q').data
  const y = vectorFrom(problem.y)
  if (y.length !== n || y.some((v) => v !== 1 && v !== -1)) throw new MdmlError('value', 'smoNu: y must be +1 or -1')
  const C = problem.C
  const eps = options.tol ?? 1e-3
  const QD = Float64Array.from({ length: n }, (_, i) => Q[i * n + i] ?? 0)
  const all = Array.from({ length: n }, (_, i) => i)
  const upper = (a: Float64Array, i: number): boolean => (a[i] ?? 0) >= C
  const lower = (a: Float64Array, i: number): boolean => (a[i] ?? 0) <= 0
  const gradientOf = (a: Float64Array): Float64Array => {
    const g = Float64Array.from(p)
    for (let j = 0; j < n; j++) {
      const aj = a[j] ?? 0
      if (aj === 0) continue
      for (let i = 0; i < n; i++) g[i] = (g[i] ?? 0) + (Q[i * n + j] ?? 0) * aj
    }
    return g
  }
  const select = (a: Float64Array, g: Float64Array): { i: number; j: number; violation: number } => {
    let gmaxp = -Infinity
    let gmaxn = -Infinity
    let ip = -1
    let inn = -1
    for (let t = 0; t < n; t++) {
      if ((y[t] ?? 1) > 0) {
        if (!upper(a, t) && -(g[t] ?? 0) >= gmaxp) {
          gmaxp = -(g[t] ?? 0)
          ip = t
        }
      } else if (!lower(a, t) && (g[t] ?? 0) >= gmaxn) {
        gmaxn = g[t] ?? 0
        inn = t
      }
    }
    let gmaxp2 = -Infinity
    let gmaxn2 = -Infinity
    let jmin = -1
    let best = Infinity
    for (let t = 0; t < n; t++) {
      const positive = (y[t] ?? 1) > 0
      if (positive ? lower(a, t) : upper(a, t)) continue
      const gt = g[t] ?? 0
      const i = positive ? ip : inn
      if (positive) gmaxp2 = Math.max(gmaxp2, gt)
      else gmaxn2 = Math.max(gmaxn2, -gt)
      const diff = positive ? gmaxp + gt : gmaxn - gt
      if (diff <= 0 || i < 0) continue
      const quad = (QD[i] ?? 0) + (QD[t] ?? 0) - 2 * (Q[i * n + t] ?? 0)
      const score = -(diff * diff) / (quad > 0 ? quad : TAU)
      if (score <= best) {
        best = score
        jmin = t
      }
    }
    const violation = Math.max(gmaxp + gmaxp2, gmaxn + gmaxn2)
    const j = jmin
    const i = j < 0 ? -1 : (y[j] ?? 1) > 0 ? ip : inn
    return { i, j, violation }
  }
  return {
    name: 'smo-nu',
    config: { tol: eps },
    schema: channels({
      metrics: {
        objective: { description: 'dual objective 1/2 a^T Q a + p^T a', direction: 'lower' },
        violation: { description: 'the larger violation of the two classes' },
        activeSize: { description: 'variables considered (all)' },
        support: { description: 'variables with alpha > 0' },
      },
      tensors: { alpha: { role: 'param', description: 'the dual variables' } },
      messages: { workingSet: { description: 'the working pair (i, j) of a step, from one class' } },
    }),
    init: () => {
      const alpha = vectorFrom(problem.alpha0)
      if (alpha.length !== n) throw new MdmlError('shape', 'smoNu: alpha0 must have one entry per variable')
      return {
        alpha,
        grad: gradientOf(alpha),
        active: all,
        violation: Infinity,
        iterations: 0,
        counter: 0,
        unshrunk: true,
        finished: false,
      }
    },
    step: (s) => {
      const pick = select(s.alpha, s.grad)
      if (pick.violation < eps || pick.j < 0) {
        const { pair: _last, ...rest } = s
        return { ...rest, violation: pick.violation, finished: true }
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
        { ci: C, cj: C },
      )
      const di = (a[i] ?? 0) - oldI
      const dj = (a[j] ?? 0) - oldJ
      const grad = Float64Array.from(s.grad)
      for (let t = 0; t < n; t++) grad[t] = (grad[t] ?? 0) + (Q[t * n + i] ?? 0) * di + (Q[t * n + j] ?? 0) * dj
      return { ...s, alpha: a, grad, violation: pick.violation, iterations: s.iterations + 1, pair: [i, j] }
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
          activeSize: n,
          support: s.alpha.reduce((c, v) => c + (v > 0 ? 1 : 0), 0),
        },
        tensors: { alpha: { role: 'param', value: s.alpha } },
        ...(s.pair === undefined ? {} : { messages: [{ kind: 'workingSet', payload: [s.pair[0], s.pair[1]] }] }),
      }
    },
  }
}

/** Result of {@link nuSvcDual} and {@link nuSvrDual}. */
export interface NuDualResult extends SvmDualResult {
  /** The margin r of the nu solver: nu-SVC divides alpha and rho by it, so the margin of the scaled problem is 1. */
  readonly margin: number
}

function solveNu(
  problem: SmoNuProblem,
  options: SmoOptions,
): { alpha: Float64Array; grad: Float64Array; iterations: number; converged: boolean; y: Float64Array } {
  const n = vectorFrom(problem.p).length
  const r = runAlgorithm(nuSmoAlgorithm(problem, options), {
    ...options,
    maxSteps: options.maxIterations ?? Math.min(10_000_000, Math.max(10_000, 100 * n * n)),
  })
  return {
    alpha: r.state.alpha,
    grad: r.state.grad,
    iterations: r.state.iterations,
    converged: r.status === 'converged',
    y: vectorFrom(problem.y),
  }
}

/**
 * The nu-support-vector-classification dual (libsvm's NU_SVC): minimise 1/2 a^T (y y^T * K) a subject to y^T a = 0,
 * sum a = nu n and 0 <= a_i <= 1, by {@link nuSmoAlgorithm}, then rescaled by the margin r so that the result is a
 * C-SVC solution with C = 1 / r. The decision function is sum_i coef_i K(x_i, x) + bias. nu bounds the fraction of
 * margin errors above and the fraction of support vectors below.
 *
 * @param kernel - The n x n kernel matrix.
 * @param labels - +1 or -1.
 * @param nu - In (0, 1], feasible for the class sizes.
 * @param options - SMO options.
 * @returns alpha (scaled), coef = y alpha, rho, bias, the margin and the support.
 * @concept support-vector-machines
 * @reference C.-C. Chang and C.-J. Lin, Training nu-support vector classifiers: theory and algorithms (2001).
 * @example
 * ```ts
 * import { nuSvcDual } from 'mdml-engine/optim'
 * const r = nuSvcDual([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], [1, 1, -1, -1], 0.5, { tol: 1e-12 })
 * [r.support.length, Math.round(r.margin * 1e6) / 1e6] // => [4, 0.5]
 * ```
 */
export function nuSvcDual(kernel: TensorLike, labels: TensorLike, nu: number, options: SmoOptions = {}): NuDualResult {
  const y = vectorFrom(labels)
  const n = y.length
  const K = matrixFrom(kernel, n, 'kernel').data
  const s = solveNu(nuSvcProblem(K, y, nuSvcStart(y, nu)), options)
  const { rho, margin } = nuRho(s.alpha, s.grad, y, 1)
  const alpha = s.alpha.map((a) => a / margin)
  let objective = 0
  for (let i = 0; i < n; i++) objective += 0.5 * (s.alpha[i] ?? 0) * (s.grad[i] ?? 0)
  return {
    alpha: vectorTensor(alpha),
    coef: vectorTensor(alpha.map((a, i) => a * (y[i] ?? 1))),
    rho: rho / margin,
    bias: -rho / margin,
    margin,
    objective: objective / (margin * margin),
    iterations: s.iterations,
    converged: s.converged,
    status: s.converged ? 'converged' : 'completed',
    support: Array.from(alpha).flatMap((v, i) => (v > 0 ? [i] : [])),
  }
}

/**
 * The nu-support-vector-regression dual (libsvm's NU_SVR) over 2n variables (alpha, alpha*): minimise
 * 1/2 (a - a*)^T K (a - a*) - t^T (a - a*) subject to sum (a - a*) = 0, sum (a + a*) = C n nu and 0 <= a, a* <= C, by
 * {@link nuSmoAlgorithm}. The tube width epsilon is found by the solver (it is the margin r); nu bounds the fraction
 * of points outside the tube above and the fraction of support vectors below.
 *
 * @param kernel - The n x n kernel matrix.
 * @param targets - Regression targets.
 * @param C - The penalty.
 * @param nu - In (0, 1].
 * @param options - SMO options.
 * @returns The 2n alphas, coef = a - a*, rho, bias, the tube half-width (`margin`) and the support.
 * @concept support-vector-machines
 * @example
 * ```ts
 * import { nuSvrDual } from 'mdml-engine/optim'
 * const r = nuSvrDual([[1, 0], [0, 1]], [1, -1], 10, 1, { tol: 1e-12 })
 * r.converged // => true
 * ```
 */
export function nuSvrDual(
  kernel: TensorLike,
  targets: TensorLike,
  C: number,
  nu: number,
  options: SmoOptions = {},
): NuDualResult {
  const t = vectorFrom(targets)
  const n = t.length
  const K = matrixFrom(kernel, n, 'kernel').data
  const problem = nuSvrProblem(K, t, C, nuSvrStart(n, C, nu))
  const s = solveNu(problem, options)
  const m = 2 * n
  const p = vectorFrom(problem.p)
  const { rho, margin } = nuRho(s.alpha, s.grad, s.y, C)
  let objective = 0
  for (let i = 0; i < m; i++) objective += 0.5 * (s.alpha[i] ?? 0) * ((s.grad[i] ?? 0) + (p[i] ?? 0))
  return {
    alpha: vectorTensor(s.alpha),
    coef: vectorTensor(Float64Array.from({ length: n }, (_, i) => (s.alpha[i] ?? 0) - (s.alpha[i + n] ?? 0))),
    rho,
    bias: -rho,
    margin,
    objective,
    iterations: s.iterations,
    converged: s.converged,
    status: s.converged ? 'converged' : 'completed',
    support: Array.from(s.alpha).flatMap((v, i) => (v > 0 ? [i] : [])),
  }
}
