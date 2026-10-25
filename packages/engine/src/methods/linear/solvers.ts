/**
 * The solvers of the penalised GLM core: direct least squares (Cholesky, QR or SVD, including ridge for many penalties
 * from one SVD), elastic-net coordinate descent with duality gaps, iteratively reweighted least squares (Fisher
 * scoring) and quasi-Newton (L-BFGS, or OWL-QN for an L1 penalty), each traced. `chooseSolver` picks one for a problem
 * and says why.
 */

import { type Json, WhiteboxError } from '../../lib/base/index.ts'
import type { DispersionFamily, Link } from '../../lib/dists/families/index.ts'
import { collecting, type MatrixView } from '../../lib/learn/index.ts'
import { cholesky, choSolve, ddot, lstsq, svd } from '../../lib/numerics/linalg/index.ts'
import { lassoCoordinateDescentAlgorithm, lbfgs, owlqn } from '../../lib/optim/index.ts'
import {
  add,
  fromStorage,
  matmul,
  mul,
  primalOf,
  square,
  sum,
  type Tensor,
  toFlat,
  type Value,
} from '../../lib/tensor/index.ts'
import { type Algorithm, channels, type RunOptions, runAlgorithm } from '../../lib/trace/index.ts'
import { centre, type Design, interceptFor, matrixTensor, vectorTensor } from './_design.ts'

/** The solvers of the penalised GLM core. */
export type LinearSolver = 'auto' | 'cholesky' | 'qr' | 'svd' | 'irls' | 'coordinate-descent' | 'lbfgs' | 'owlqn'

/**
 * A penalised GLM problem: minimise sum_i w_i d(y_i, mu_i) / (2 sum_i w_i) + alpha l1Ratio |beta|_1 +
 * alpha (1 - l1Ratio) / 2 |beta|^2 with mu = g^{-1}(x beta + b) and the intercept b unpenalised (scikit-learn's
 * scaling of the elastic net and of its GLMs).
 */
export interface GlmProblem {
  readonly design: Design
  readonly family: DispersionFamily
  readonly link: Link
  readonly alpha: number
  readonly l1Ratio: number
  readonly fitIntercept: boolean
}

/** A solved GLM problem. */
export interface GlmSolution {
  readonly coef: Float64Array
  readonly intercept: number
  readonly solver: Exclude<LinearSolver, 'auto'>
  readonly iterations: number
  readonly converged: boolean
  /** The penalised objective at the solution. */
  readonly objective: number
  /** The penalised objective after each iteration (one entry for direct solvers). */
  readonly history: readonly number[]
  /** The duality gap at the end (coordinate descent). */
  readonly gap?: number
  /** The iterates, the start first, with their objective (IRLS): for recording the model through training. */
  readonly path?: readonly GlmIterate[]
}

/** One iterate of an iterative GLM solve. */
export interface GlmIterate {
  readonly coef: readonly number[]
  readonly intercept: number
  readonly objective: number
}

/** Options of {@link solveGlm}. */
export interface SolveOptions extends Partial<Pick<RunOptions, 'key' | 'tracer' | 'policy' | 'parent'>> {
  readonly solver?: LinearSolver
  readonly maxIter?: number
  readonly tol?: number
  /** A warm start: coefficients (and intercept). */
  readonly start?: { readonly coef: ArrayLike<number>; readonly intercept: number }
}

const sumOf = (v: ArrayLike<number>): number => {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i] ?? 0
  return s
}

const isLeastSquares = (p: GlmProblem): boolean => p.family.name === 'gaussian' && p.link.name === 'identity'

/**
 * The solver the core picks for a problem, and why: least squares without an L1 penalty is a direct solve (Cholesky
 * of the normal equations when there are more rows than columns, else the SVD); least squares with an L1 penalty is
 * coordinate descent; any other family is IRLS (with coordinate descent inside it for an L1 penalty, as glmnet).
 *
 * @param problem - The problem.
 * @returns The solver and the reason.
 */
export function chooseSolver(problem: GlmProblem): {
  readonly solver: Exclude<LinearSolver, 'auto'>
  readonly reason: string
} {
  const l1 = problem.alpha * problem.l1Ratio > 0
  if (isLeastSquares(problem)) {
    if (l1) return { solver: 'coordinate-descent', reason: 'least squares with an L1 penalty: coordinate descent' }
    return problem.design.x.n > problem.design.x.d
      ? { solver: 'cholesky', reason: 'least squares, more rows than columns: Cholesky of the normal equations' }
      : { solver: 'svd', reason: 'least squares, at least as many columns as rows: minimum-norm SVD solve' }
  }
  return l1
    ? { solver: 'irls', reason: `${problem.family.name} with an L1 penalty: IRLS with coordinate descent inside` }
    : { solver: 'irls', reason: `${problem.family.name} family: iteratively reweighted least squares` }
}

/** The penalised objective of a problem at given coefficients. */
export function glmObjective(problem: GlmProblem, coef: ArrayLike<number>, intercept: number): number {
  const { x, y, w } = problem.design
  const eta = new Float64Array(x.n)
  for (let i = 0; i < x.n; i++) eta[i] = intercept + ddot(x.data.subarray(i * x.d, (i + 1) * x.d), coef)
  const mu = problem.link.inverse(vectorTensor(eta))
  const deviance = toFlat(primalOf(problem.family.deviance(vectorTensor(y), mu, vectorTensor(w))))[0] ?? 0
  let l1 = 0
  let l2 = 0
  for (let j = 0; j < coef.length; j++) {
    l1 += Math.abs(coef[j] ?? 0)
    l2 += (coef[j] ?? 0) ** 2
  }
  return (
    deviance / (2 * sumOf(w)) + problem.alpha * problem.l1Ratio * l1 + (problem.alpha * (1 - problem.l1Ratio) * l2) / 2
  )
}

/**
 * Penalised least squares by a direct method: min |sqrt(w) (y - x beta - b)|^2 / 2 + lambda |beta|^2 / 2, on the
 * weighted, centred design, by Cholesky of the normal equations, QR (of the design augmented with sqrt(lambda) I) or
 * the SVD (minimum norm when rank deficient).
 *
 * @param design - Features, targets and weights.
 * @param lambda - The ridge weight on the unscaled sum of squares.
 * @param fitIntercept - Whether to fit an intercept.
 * @param solver - The direct method.
 * @returns Coefficients and intercept.
 */
export function leastSquaresSolve(
  design: Design,
  lambda: number,
  fitIntercept: boolean,
  solver: 'cholesky' | 'qr' | 'svd',
): { readonly coef: Float64Array; readonly intercept: number } {
  const c = centre(design, fitIntercept)
  const { n, d } = design.x
  let coef: Float64Array
  if (solver === 'svd') coef = svdRidge(c.x, n, d, c.y, [lambda])[0] ?? new Float64Array(d)
  else if (solver === 'qr') {
    const rows = lambda > 0 ? n + d : n
    const a = new Float64Array(rows * d)
    a.set(c.x)
    const b = new Float64Array(rows)
    b.set(c.y)
    if (lambda > 0) for (let j = 0; j < d; j++) a[(n + j) * d + j] = Math.sqrt(lambda)
    coef = Float64Array.from(toFlat(primalOf(lstsq(matrixTensor(a, rows, d), vectorTensor(b)))))
  } else {
    const g = new Float64Array(d * d)
    const r = new Float64Array(d)
    for (let i = 0; i < n; i++) {
      const row = c.x.subarray(i * d, (i + 1) * d)
      const yi = c.y[i] ?? 0
      for (let a = 0; a < d; a++) {
        r[a] = (r[a] ?? 0) + (row[a] ?? 0) * yi
        for (let b = 0; b <= a; b++) g[a * d + b] = (g[a * d + b] ?? 0) + (row[a] ?? 0) * (row[b] ?? 0)
      }
    }
    for (let a = 0; a < d; a++) {
      g[a * d + a] = (g[a * d + a] ?? 0) + lambda
      for (let b = 0; b < a; b++) g[b * d + a] = g[a * d + b] ?? 0
    }
    try {
      coef = Float64Array.from(toFlat(primalOf(choSolve(cholesky(matrixTensor(g, d, d)), vectorTensor(r)))))
      if (coef.some((v) => !Number.isFinite(v))) throw new WhiteboxError('value', 'singular')
    } catch {
      // A singular Gram matrix (collinear features): fall back to the minimum-norm SVD solution.
      coef = svdRidge(c.x, n, d, c.y, [lambda])[0] ?? new Float64Array(d)
    }
  }
  return { coef, intercept: fitIntercept ? interceptFor(c, coef) : 0 }
}

/** The reduced SVD of a row-major matrix as flat arrays: u `[n, k]`, s `[k]`, vh `[k, d]`. */
export function svdOf(
  x: Float64Array,
  n: number,
  d: number,
): { readonly u: number[]; readonly s: number[]; readonly vh: number[] } {
  const { u, s, vh } = svd(matrixTensor(x, n, d))
  return { u: toFlat(primalOf(u)), s: toFlat(primalOf(s)), vh: toFlat(primalOf(vh)) }
}

/**
 * Ridge coefficients for several penalties from one SVD of the (centred, weighted) design: beta(lambda) =
 * V diag(s / (s^2 + lambda)) U^T y, singular values below rcond s_max dropped (the minimum-norm solution at 0).
 *
 * @param x - The design, row-major `[n, d]`.
 * @param n - Rows.
 * @param d - Columns.
 * @param y - Targets.
 * @param lambdas - Penalties.
 * @returns One coefficient vector per penalty.
 */
export function svdRidge(
  x: Float64Array,
  n: number,
  d: number,
  y: Float64Array,
  lambdas: readonly number[],
): Float64Array[] {
  const { u: uu, s: ss, vh: vv } = svdOf(x, n, d)
  const k = ss.length
  const uty = new Float64Array(k)
  for (let a = 0; a < k; a++) {
    let t = 0
    for (let i = 0; i < n; i++) t += (uu[i * k + a] ?? 0) * (y[i] ?? 0)
    uty[a] = t
  }
  const smax = ss[0] ?? 0
  const rcond = Math.max(n, d) * 2.220446049250313e-16 * smax
  return lambdas.map((lambda) => {
    const coef = new Float64Array(d)
    for (let a = 0; a < k; a++) {
      const sa = ss[a] ?? 0
      if (lambda === 0 && sa <= rcond) continue
      const f = (sa / (sa * sa + lambda)) * (uty[a] ?? 0)
      for (let j = 0; j < d; j++) coef[j] = (coef[j] ?? 0) + f * (vv[a * d + j] ?? 0)
    }
    return coef
  })
}

/**
 * The elastic net by coordinate descent (scikit-learn's `enet_coordinate_descent`, from the core's
 * `lassoCoordinateDescentAlgorithm`) on the weighted, centred design, traced: objective, duality gap and the largest
 * update per sweep.
 *
 * @param design - Features, targets and weights.
 * @param alpha - The penalty (scikit-learn's per-sample scaling).
 * @param l1Ratio - The L1 share.
 * @param fitIntercept - Whether to fit an intercept.
 * @param options - Iterations (default 1000), tolerance (default 1e-4, scaled by |y|^2 as scikit-learn), warm start and
 * run options.
 * @returns The solution with the final duality gap.
 */
export function coordinateDescentSolve(
  design: Design,
  alpha: number,
  l1Ratio: number,
  fitIntercept: boolean,
  options: SolveOptions = {},
): GlmSolution {
  const c = centre(design, fitIntercept)
  const { n, d } = design.x
  const s = sumOf(design.w)
  const algorithm = lassoCoordinateDescentAlgorithm(matrixTensor(c.x, n, d), vectorTensor(c.y), alpha * l1Ratio * s, {
    tol: options.tol ?? 1e-4,
    l2: alpha * (1 - l1Ratio) * s,
    ...(options.start === undefined ? {} : { x0: options.start.coef }),
  })
  const tracked = collecting(algorithm, (state) => state.objective / s)
  const r = runAlgorithm(tracked.algorithm, { ...runOf(options), maxSteps: options.maxIter ?? 1000 })
  const coef = Float64Array.from(r.state.x)
  return {
    coef,
    intercept: fitIntercept ? interceptFor(c, coef) : 0,
    solver: 'coordinate-descent',
    iterations: r.steps,
    converged: r.status === 'converged',
    objective: r.state.objective / s,
    history: tracked.history.slice(1),
    gap: r.state.gap / s,
  }
}

const runOf = (options: SolveOptions): Pick<RunOptions, 'key' | 'tracer' | 'policy' | 'parent'> => ({
  ...(options.key === undefined ? {} : { key: options.key }),
  ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
  ...(options.policy === undefined ? {} : { policy: options.policy }),
  ...(options.parent === undefined ? {} : { parent: options.parent }),
})

/** The state of {@link irlsAlgorithm}. */
export interface IrlsState {
  readonly coef: Float64Array
  readonly intercept: number
  readonly objective: number
  readonly change: number
  readonly halvings: number
}

/**
 * Iteratively reweighted least squares (Fisher scoring) for a penalised GLM, as a traced algorithm: each step forms
 * the working weights w / (g'(mu)^2 V(mu)) and working response eta + (y - mu) g'(mu) from the family, solves the
 * penalised weighted least-squares problem (directly for a ridge penalty, by coordinate descent for an L1 penalty, as
 * glmnet's proximal Newton step), and halves the step while the penalised objective increases. Records the objective,
 * its change, the step halvings and the coefficients.
 *
 * @param problem - The problem.
 * @param options - Tolerance on the largest relative coefficient change (default 1e-10) and a warm start.
 * @returns The algorithm.
 */
export function irlsAlgorithm(problem: GlmProblem, options: SolveOptions = {}): Algorithm<IrlsState> {
  const { x, y, w } = problem.design
  const s = sumOf(w)
  const tol = options.tol ?? 1e-10
  const l1 = problem.alpha * problem.l1Ratio
  const yT = vectorTensor(y)
  const etaOf = (coef: ArrayLike<number>, b: number): Float64Array => {
    const eta = new Float64Array(x.n)
    for (let i = 0; i < x.n; i++) eta[i] = b + ddot(x.data.subarray(i * x.d, (i + 1) * x.d), coef)
    return eta
  }
  return {
    name: 'irls',
    config: {
      family: problem.family.name,
      link: problem.link.name,
      alpha: problem.alpha,
      l1Ratio: problem.l1Ratio,
    } as Json,
    schema: channels({
      metrics: {
        objective: { description: 'penalised deviance / (2 sum w)', direction: 'lower' },
        change: { description: 'largest coefficient change, relative' },
        halvings: { description: 'step halvings' },
      },
      tensors: { coef: { role: 'param', description: 'coefficients' } },
    }),
    init: () => {
      if (options.start !== undefined) {
        const coef = Float64Array.from(options.start.coef)
        return {
          coef,
          intercept: options.start.intercept,
          objective: glmObjective(problem, coef, options.start.intercept),
          change: Infinity,
          halvings: 0,
        }
      }
      return { coef: new Float64Array(x.d), intercept: 0, objective: Infinity, change: Infinity, halvings: 0 }
    },
    step: (state) => {
      const first = !Number.isFinite(state.objective)
      const eta = first
        ? Float64Array.from(toFlat(primalOf(problem.link.link(problem.family.startingMu(yT)))))
        : etaOf(state.coef, state.intercept)
      const etaT = vectorTensor(eta)
      const mu = problem.link.inverse(etaT)
      const muFlat = toFlat(primalOf(mu))
      const ww = Float64Array.from(toFlat(primalOf(problem.family.irlsWeights(mu, problem.link, vectorTensor(w)))))
      const gp = toFlat(primalOf(problem.link.derivative(mu)))
      const z = Float64Array.from(eta, (e, i) => e + ((y[i] ?? 0) - (muFlat[i] ?? 0)) * (gp[i] ?? 0))
      const working: Design = { x, y: z, w: ww }
      const solved =
        l1 > 0
          ? coordinateDescentSolve(working, (problem.alpha * s) / sumOf(ww), problem.l1Ratio, problem.fitIntercept, {
              tol: 1e-10,
              maxIter: 1000,
              start: { coef: state.coef, intercept: state.intercept },
            })
          : leastSquaresSolve(working, problem.alpha * (1 - problem.l1Ratio) * s, problem.fitIntercept, 'cholesky')
      let coef = solved.coef
      let intercept = solved.intercept
      let objective = glmObjective(problem, coef, intercept)
      let halvings = 0
      while (!first && !(objective <= state.objective + 1e-12 * Math.abs(state.objective)) && halvings < 30) {
        halvings++
        coef = Float64Array.from(coef, (v, j) => 0.5 * (v + (state.coef[j] ?? 0)))
        intercept = 0.5 * (intercept + state.intercept)
        objective = glmObjective(problem, coef, intercept)
      }
      // The largest coefficient change relative to the coefficients' size (intercept included).
      let delta = Math.abs(intercept - state.intercept)
      let size = Math.abs(intercept)
      for (let j = 0; j < coef.length; j++) {
        delta = Math.max(delta, Math.abs((coef[j] ?? 0) - (state.coef[j] ?? 0)))
        size = Math.max(size, Math.abs(coef[j] ?? 0))
      }
      const change = first ? Infinity : delta / (1 + size)
      return { coef, intercept, objective, change, halvings }
    },
    done: (state) => (state.change <= tol ? 'converged' : undefined),
    observe: (state) => ({
      metrics: { objective: state.objective, change: state.change, halvings: state.halvings },
      tensors: { coef: { role: 'param', value: state.coef } },
    }),
  }
}

/**
 * A penalised GLM by L-BFGS (smooth penalties) or OWL-QN (an L1 penalty), from the core optimisers with gradients by
 * automatic differentiation of the family's deviance: the gradient-based alternative to IRLS, traced by the optimiser.
 *
 * @param problem - The problem.
 * @param options - Iterations, gradient tolerance and run options.
 * @returns The solution.
 */
export function quasiNewtonSolve(problem: GlmProblem, options: SolveOptions = {}): GlmSolution {
  const { x, y, w } = problem.design
  const s = sumOf(w)
  const d = x.d
  const cols = d + (problem.fitIntercept ? 1 : 0)
  const aug = new Float64Array(x.n * cols)
  for (let i = 0; i < x.n; i++) {
    for (let j = 0; j < d; j++) aug[i * cols + j] = x.data[i * d + j] ?? 0
    if (problem.fitIntercept) aug[i * cols + d] = 1
  }
  const a = matrixTensor(aug, x.n, cols)
  const mask = vectorTensor(Array.from({ length: cols }, (_, j) => (j < d ? 1 : 0)))
  const yT = vectorTensor(y)
  const wT = vectorTensor(w)
  const l2 = problem.alpha * (1 - problem.l1Ratio)
  const l1 = problem.alpha * problem.l1Ratio
  const f = (p: Value): Value => {
    const mu = problem.link.inverse(matmul(a, p))
    const dev = mul(problem.family.deviance(yT, mu, wT), 1 / (2 * s))
    return l2 > 0 ? add(dev, mul(sum(square(mul(p, mask))), l2 / 2)) : dev
  }
  const x0 = new Float64Array(cols)
  if (options.start !== undefined) {
    x0.set(Array.from(options.start.coef).slice(0, d))
    if (problem.fitIntercept) x0[d] = options.start.intercept
  }
  const history: number[] = []
  const run = { ...runOf(options), maxIterations: options.maxIter ?? 1000, gradTol: options.tol ?? 1e-10 }
  const r =
    l1 > 0
      ? owlqn(f, x0, { ...run, l1: Array.from({ length: cols }, (_, j) => (j < d ? l1 : 0)), ftol: 1e-14 })
      : lbfgs(f, x0, { ...run, ftol: 1e-15 })
  const p = toFlat(r.x as Tensor)
  const coef = Float64Array.from(p.slice(0, d))
  const intercept = problem.fitIntercept ? (p[d] ?? 0) : 0
  history.push(r.fun)
  return {
    coef,
    intercept,
    solver: l1 > 0 ? 'owlqn' : 'lbfgs',
    iterations: r.steps,
    converged: r.converged,
    objective: glmObjective(problem, coef, intercept),
    history,
  }
}

/**
 * Solve a penalised GLM with the chosen (or automatically chosen) solver.
 *
 * @param problem - The problem.
 * @param options - Solver, iterations, tolerance, warm start and run options.
 * @returns The solution.
 * @throws WhiteboxError `value` when a direct least-squares solver is asked for a problem it cannot solve.
 */
export function solveGlm(problem: GlmProblem, options: SolveOptions = {}): GlmSolution {
  const solver =
    options.solver === undefined || options.solver === 'auto' ? chooseSolver(problem).solver : options.solver
  const l1 = problem.alpha * problem.l1Ratio > 0
  const s = sumOf(problem.design.w)
  if (solver === 'cholesky' || solver === 'qr' || solver === 'svd') {
    if (!isLeastSquares(problem) || l1)
      throw new WhiteboxError('value', `solveGlm: ${solver} solves least squares without an L1 penalty only`)
    const r = leastSquaresSolve(problem.design, problem.alpha * s, problem.fitIntercept, solver)
    const objective = glmObjective(problem, r.coef, r.intercept)
    return { ...r, solver, iterations: 1, converged: true, objective, history: [objective] }
  }
  if (solver === 'coordinate-descent') {
    if (!isLeastSquares(problem)) return irls(problem, options)
    return coordinateDescentSolve(problem.design, problem.alpha, problem.l1Ratio, problem.fitIntercept, options)
  }
  if (solver === 'lbfgs' || solver === 'owlqn') return quasiNewtonSolve(problem, options)
  return irls(problem, options)
}

function irls(problem: GlmProblem, options: SolveOptions): GlmSolution {
  const tracked = collecting(
    irlsAlgorithm(problem, options),
    (state): GlmIterate => ({ coef: Array.from(state.coef), intercept: state.intercept, objective: state.objective }),
  )
  const r = runAlgorithm(tracked.algorithm, { ...runOf(options), maxSteps: options.maxIter ?? 100 })
  return {
    coef: r.state.coef,
    intercept: r.state.intercept,
    solver: 'irls',
    iterations: r.steps,
    converged: r.status === 'converged',
    objective: r.state.objective,
    history: tracked.history.slice(1).map((h) => h.objective),
    path: [...tracked.history],
  }
}

/** The augmented design [x, 1] as a tensor (for covariance and inference). */
export function augmented(x: MatrixView, fitIntercept: boolean): Tensor {
  const cols = x.d + (fitIntercept ? 1 : 0)
  const out = new Float64Array(x.n * cols)
  for (let i = 0; i < x.n; i++) {
    for (let j = 0; j < x.d; j++) out[i * cols + j] = x.data[i * x.d + j] ?? 0
    if (fitIntercept) out[i * cols + x.d] = 1
  }
  return fromStorage(out, [x.n, cols], 'float64')
}
