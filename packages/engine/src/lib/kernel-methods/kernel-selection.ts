/**
 * Choosing the kernel and the ridge of kernel ridge regression from the data: the three classical criteria as
 * differentiable functions of the Gram matrix and the ridge (`logEvidence`, the log marginal likelihood of kernel ridge
 * read as a Gaussian process; `looScore`, the exact leave-one-out error; `gcvScore`, generalised cross-validation), the
 * criterion as a smooth problem over the log-parameterised kernel hyperparameters and the log ridge
 * (`kernelCriterionProblem`, whose gradient is automatic differentiation through the Gram matrix), a tuner that
 * minimises it with any solver of `smoothSolvers` (`tuneKernelRidge`) and a grid search over the same criteria
 * (`gridKernelRidge`).
 */

import { MdmlError } from '../base/index.ts'
import { type Kernel, theta as thetaOf, withTheta } from '../kernels/index.ts'
import { cholesky, choSolve, inv } from '../numerics/linalg/index.ts'
import { type SmoothProblem, type SolverReached, smoothSolverConfig, smoothSolvers } from '../optim/index.ts'
import { type Key, key as makeKey } from '../random/index.ts'
import type { SpaceValue } from '../space/index.ts'
import {
  add,
  diagonal,
  div,
  exp,
  eye,
  log,
  matmul,
  mul,
  primalOf,
  promoteFloat,
  reshape,
  sliceAxis,
  square,
  sub,
  sum,
  type TensorLike,
  toFlat,
  type Value,
} from '../tensor/index.ts'
import { runAlgorithm } from '../trace/index.ts'

/** What a kernel is selected by: the log evidence (maximised), the leave-one-out error or the GCV score (minimised). */
export type SelectionCriterion = 'evidence' | 'loo' | 'gcv'

function target(y: TensorLike, who: string): Value {
  const t = promoteFloat(y)[0] as Value
  if (t.shape.length !== 1) throw new MdmlError('shape', `${who}: y must be [n], got [${t.shape.join(', ')}]`)
  return t
}

function gramOf(k: TensorLike, n: number, who: string): Value {
  const g = promoteFloat(k)[0] as Value
  if (g.shape.length !== 2 || g.shape[0] !== n || g.shape[1] !== n) {
    throw new MdmlError('shape', `${who}: expected a Gram matrix [${n}, ${n}], got [${g.shape.join(', ')}]`)
  }
  return g
}

const ridged = (g: Value, n: number, alpha: number | Value): Value => add(g, mul(eye(n), alpha))

/**
 * The log evidence of kernel ridge regression read as Gaussian-process regression with the kernel and noise variance
 * `alpha`: `-1/2 y'(K + alpha I)^(-1) y - 1/2 log |K + alpha I| - n/2 log 2 pi`. Differentiable in `k` and `alpha`.
 *
 * @param k - Gram matrix `[n, n]` (possibly traced).
 * @param y - Targets `[n]`.
 * @param alpha - Noise variance, or ridge (default 1).
 * @returns The log evidence, a scalar.
 * @throws MdmlError `shape` for mismatched inputs.
 * @concept kernel-selection
 * @reference C. E. Rasmussen and C. K. I. Williams, Gaussian Processes for Machine Learning (2006), equation 2.30.
 * @example
 * ```ts
 * import { logEvidence } from 'mdml-engine/kernel-methods'
 * import { item } from 'mdml-engine/tensor'
 * Math.round(item(logEvidence([[1]], [1], 1)) * 1e6) / 1e6 // => -1.515512
 * ```
 */
export function logEvidence(k: TensorLike, y: TensorLike, alpha: number | Value = 1): Value {
  const yv = target(y, 'logEvidence')
  const n = yv.shape[0] ?? 0
  const l = cholesky(ridged(gramOf(k, n, 'logEvidence'), n, alpha))
  const a = choSolve(l, yv)
  const quad = sum(mul(yv, a))
  const logdet = mul(sum(log(diagonal(l))), 2)
  return sub(mul(add(quad, logdet), -0.5), 0.5 * n * Math.log(2 * Math.PI))
}

/**
 * The exact leave-one-out mean squared error of kernel ridge regression as a differentiable function of the Gram matrix
 * and the ridge: with `A = K + alpha I` the leave-one-out residual of observation `i` is `(A^(-1) y)_i / (A^(-1))_ii`.
 *
 * @param k - Gram matrix `[n, n]` (possibly traced).
 * @param y - Targets `[n]`.
 * @param alpha - Ridge (default 1).
 * @returns The mean of the squared leave-one-out residuals, a scalar.
 * @concept kernel-selection
 * @reference S. An, W. Liu and S. Venkatesh, Fast cross-validation algorithms for least squares support vector machine
 *   and kernel ridge regression (Pattern Recognition 40, 2007).
 * @example
 * ```ts
 * import { looScore } from 'mdml-engine/kernel-methods'
 * import { item } from 'mdml-engine/tensor'
 * item(looScore([[2, 1], [1, 2]], [3, 3], 1)) // => 4
 * ```
 */
export function looScore(k: TensorLike, y: TensorLike, alpha: number | Value = 1): Value {
  const yv = target(y, 'looScore')
  const n = yv.shape[0] ?? 0
  const ai = inv(ridged(gramOf(k, n, 'looScore'), n, alpha))
  const c = reshape(matmul(ai, reshape(yv, [n, 1])), [n])
  return div(sum(square(div(c, diagonal(ai)))), n)
}

/**
 * Generalised cross-validation (Golub, Heath and Wahba) of kernel ridge regression: the training error scaled by the
 * remaining degrees of freedom, `n |(I - H) y|^2 / tr(I - H)^2` with the smoother `H = K (K + alpha I)^(-1)`, which
 * replaces each leverage in the leave-one-out error by their average. Differentiable in `k` and `alpha`.
 *
 * @param k - Gram matrix `[n, n]` (possibly traced).
 * @param y - Targets `[n]`.
 * @param alpha - Ridge (default 1).
 * @returns The GCV score, a scalar.
 * @concept kernel-selection
 * @reference G. H. Golub, M. Heath and G. Wahba, Generalized cross-validation as a method for choosing a good ridge
 *   parameter (Technometrics 21, 1979).
 * @example
 * ```ts
 * import { gcvScore } from 'mdml-engine/kernel-methods'
 * import { item } from 'mdml-engine/tensor'
 * item(gcvScore([[1, 0], [0, 1]], [1, 1], 1)) // => 1
 * ```
 */
export function gcvScore(k: TensorLike, y: TensorLike, alpha: number | Value = 1): Value {
  const yv = target(y, 'gcvScore')
  const n = yv.shape[0] ?? 0
  const ai = inv(ridged(gramOf(k, n, 'gcvScore'), n, alpha))
  const c = reshape(matmul(ai, reshape(yv, [n, 1])), [n])
  return div(mul(sum(square(c)), n), square(sum(diagonal(ai))))
}

/**
 * A criterion as a quantity to minimise: the negative log evidence, the leave-one-out error or the GCV score.
 *
 * @param criterion - Which criterion.
 * @param k - Gram matrix `[n, n]`.
 * @param y - Targets `[n]`.
 * @param alpha - Ridge.
 * @returns The scalar to minimise.
 * @concept kernel-selection
 * @example
 * ```ts
 * import { criterionValue } from 'mdml-engine/kernel-methods'
 * import { item } from 'mdml-engine/tensor'
 * item(criterionValue('loo', [[2, 1], [1, 2]], [3, 3], 1)) // => 4
 * ```
 */
export function criterionValue(
  criterion: SelectionCriterion,
  k: TensorLike,
  y: TensorLike,
  alpha: number | Value = 1,
): Value {
  if (criterion === 'evidence') return mul(logEvidence(k, y, alpha), -1)
  return criterion === 'loo' ? looScore(k, y, alpha) : gcvScore(k, y, alpha)
}

/** What the kernel criterion problem and the tuners take. */
export interface KernelTuneOptions {
  /** The criterion (default `evidence`). */
  readonly criterion?: SelectionCriterion
  /** The ridge to start from, or to hold when `tuneAlpha` is false (default 1). */
  readonly alpha?: number
  /** Whether the ridge is tuned with the kernel hyperparameters (default true). */
  readonly tuneAlpha?: boolean
  /** The solver, a value of `smoothSolvers.space` (default L-BFGS). */
  readonly solver?: SpaceValue
  /** The most solver steps (default 200). */
  readonly maxSteps?: number
  readonly key?: Key
}

/**
 * The selection criterion as a smooth problem for the solver slot. The unknowns are the log-parameterised kernel
 * hyperparameters (`theta(kernel)` order) followed by the log ridge when it is tuned; the objective is the negative log
 * evidence, the leave-one-out error or the GCV score of the Gram matrix at those values, and its gradient is automatic
 * differentiation through the kernel.
 *
 * @param kernel - The kernel, whose current hyperparameters are the start.
 * @param x - Inputs `[n, d]`.
 * @param y - Targets `[n]`.
 * @param options - The criterion, the starting ridge and whether it is tuned.
 * @returns The problem.
 * @throws MdmlError `value` for a non-positive ridge.
 * @concept kernel-selection
 * @example
 * ```ts
 * import { kernelCriterionProblem } from 'mdml-engine/kernel-methods'
 * import { rbfKernel } from 'mdml-engine/kernels'
 * kernelCriterionProblem(rbfKernel(), [[0], [1], [2]], [0, 1, 0]).x0.length // => 2
 * ```
 */
export function kernelCriterionProblem(
  kernel: Kernel,
  x: TensorLike,
  y: TensorLike,
  options: KernelTuneOptions = {},
): SmoothProblem {
  const alpha = options.alpha ?? 1
  if (!(alpha > 0)) throw new MdmlError('value', `kernelCriterionProblem: alpha must be positive, got ${alpha}`)
  const tuneAlpha = options.tuneAlpha ?? true
  const criterion = options.criterion ?? 'evidence'
  const start = Array.from(toFlat(primalOf(thetaOf(kernel))))
  const p = start.length
  const yv = target(y, 'kernelCriterionProblem')
  return {
    objective: (t: unknown): Value => {
      const v = promoteFloat(t as Value)[0] as Value
      const k = p === 0 ? kernel : withTheta(kernel, sliceAxis(v, 0, 0, p))
      const a = tuneAlpha ? exp(reshape(sliceAxis(v, 0, p, p + 1), [])) : alpha
      return criterionValue(criterion, k.k(x), yv, a)
    },
    x0: tuneAlpha ? [...start, Math.log(alpha)] : start,
  }
}

/** A tuned kernel ridge: the kernel and ridge at the optimum of a criterion. */
export interface KernelTuned {
  /** The kernel with the tuned hyperparameters. */
  readonly kernel: Kernel
  readonly alpha: number
  /** The log-parameterised hyperparameters (`theta(kernel)` order). */
  readonly theta: number[]
  readonly criterion: SelectionCriterion
  /** The criterion at the optimum: the negative log evidence, or the error to minimise. */
  readonly value: number
  readonly steps: number
  readonly converged: boolean
}

/**
 * Tune the kernel hyperparameters and the ridge of kernel ridge regression by minimising a selection criterion with a
 * solver of the slot (L-BFGS by default; any of `smoothSolvers` reaches the same optimum).
 *
 * @param kernel - The kernel, whose hyperparameters are the start.
 * @param x - Inputs `[n, d]`.
 * @param y - Targets `[n]`.
 * @param options - The criterion, the starting ridge, the solver and the step budget.
 * @returns The tuned kernel and ridge.
 * @concept kernel-selection
 * @example
 * ```ts
 * import { tuneKernelRidge } from 'mdml-engine/kernel-methods'
 * import { rbfKernel } from 'mdml-engine/kernels'
 * const x = [[0], [0.5], [1], [1.5], [2], [2.5], [3]]
 * const y = [0, 0.48, 0.84, 1, 0.91, 0.6, 0.14]
 * tuneKernelRidge(rbfKernel(), x, y, { criterion: 'loo', alpha: 0.1 }).alpha < 0.1 // => true
 * ```
 */
export function tuneKernelRidge(
  kernel: Kernel,
  x: TensorLike,
  y: TensorLike,
  options: KernelTuneOptions = {},
): KernelTuned {
  const problem = kernelCriterionProblem(kernel, x, y, options)
  const solver = options.solver ?? smoothSolverConfig({ kind: 'lbfgs' })
  const key = options.key ?? makeKey(0)
  const run = runAlgorithm(smoothSolvers.make(problem, solver, key), { key, maxSteps: options.maxSteps ?? 200 })
  const r = run.state as SolverReached
  const p = (problem.x0 as ArrayLike<number>).length - ((options.tuneAlpha ?? true) ? 1 : 0)
  const at = Array.from(r.x)
  const tuned = p === 0 ? kernel : withTheta(kernel, at.slice(0, p))
  return {
    kernel: tuned,
    alpha: (options.tuneAlpha ?? true) ? Math.exp(at[p] ?? 0) : (options.alpha ?? 1),
    theta: at.slice(0, p),
    criterion: options.criterion ?? 'evidence',
    value: r.value,
    steps: run.steps,
    converged: run.status === 'converged',
  }
}

/** The grids of {@link gridKernelRidge}. */
export interface KernelGrid {
  /** Log-parameterised values to try for each kernel hyperparameter (`theta(kernel)` order); one array per hyperparameter. */
  readonly theta: readonly (readonly number[])[]
  /** Ridge values to try. */
  readonly alpha: readonly number[]
}

/** One point of a grid search. */
export interface GridPoint {
  readonly theta: number[]
  readonly alpha: number
  readonly value: number
}

/** The outcome of {@link gridKernelRidge}. */
export interface KernelGridResult {
  readonly best: GridPoint
  readonly table: readonly GridPoint[]
}

/**
 * Grid search over the kernel hyperparameters and the ridge for a selection criterion: every combination of the grids
 * is scored and the best returned with the whole table.
 *
 * @param kernel - The kernel (its structure; the grid supplies the log hyperparameters).
 * @param x - Inputs `[n, d]`.
 * @param y - Targets `[n]`.
 * @param grid - Values per hyperparameter and for the ridge.
 * @param criterion - The criterion (default `evidence`).
 * @returns The best point and the table.
 * @throws MdmlError `shape` when the grid has not one array per hyperparameter.
 * @concept kernel-selection
 * @example
 * ```ts
 * import { gridKernelRidge } from 'mdml-engine/kernel-methods'
 * import { rbfKernel } from 'mdml-engine/kernels'
 * const r = gridKernelRidge(rbfKernel(), [[0], [1], [2], [3]], [0, 1, 1, 0], { theta: [[-1, 0, 1]], alpha: [0.01, 1] }, 'loo')
 * r.table.length // => 6
 * ```
 */
export function gridKernelRidge(
  kernel: Kernel,
  x: TensorLike,
  y: TensorLike,
  grid: KernelGrid,
  criterion: SelectionCriterion = 'evidence',
): KernelGridResult {
  const p = (primalOf(thetaOf(kernel)).shape[0] ?? 0) as number
  if (grid.theta.length !== p) {
    throw new MdmlError(
      'shape',
      `gridKernelRidge: the kernel has ${p} hyperparameters but the grid has ${grid.theta.length}`,
    )
  }
  const yv = target(y, 'gridKernelRidge')
  const table: GridPoint[] = []
  const walk = (i: number, chosen: number[]): void => {
    if (i < p) {
      for (const v of grid.theta[i] ?? []) walk(i + 1, [...chosen, v])
      return
    }
    const k = (p === 0 ? kernel : withTheta(kernel, chosen)).k(x)
    for (const alpha of grid.alpha) {
      table.push({
        theta: chosen,
        alpha,
        value: toFlat(primalOf(criterionValue(criterion, k, yv, alpha)))[0] ?? Number.NaN,
      })
    }
  }
  walk(0, [])
  const finite = table.filter((t) => Number.isFinite(t.value))
  const best = finite.reduce<GridPoint | undefined>((b, t) => (b === undefined || t.value < b.value ? t : b), undefined)
  if (best === undefined) throw new MdmlError('value', 'gridKernelRidge: no grid point gave a finite criterion')
  return { best, table }
}
